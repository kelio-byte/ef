"""Euler 采样 + 分支维护 (Euler-Beam).

维护最多 n_branches 条并行 Euler 轨迹。核心特性：

- **共享前向**: 所有活跃分支每步拼成一个 batch，单次模型前向。
- **独立采样**: 每个分支用自己的随机种子独立采样编辑（保留 Euler 的 0~N 编辑/步灵活性）。
- **去重加权**: 产生相同 token 序列的分支合并权重。高共识 → 高权重。
- **Top-K 剪枝**: 去重后保留 (-path_log_p, weight) 最高的 n_branches 条分支。
- **分裂补充**: 当不同结果不足 n_branches 时，从最高 rank 分支分裂出新分支（新种子）维持探索多样性。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from edit_flows.core.rate_scale import apply_rate_parameterization, get_rate_scale
from edit_flows.core.scheduler import KappaScheduler
from edit_flows.sampling.euler import (
    _compute_model_time,
    _sample_edit_actions,
    get_adaptive_h,
)
from edit_flows.sampling.ops import apply_ins_del_operations
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


# ---------------------------------------------------------------------------
# 分支状态
# ---------------------------------------------------------------------------


class _BranchState:
    """单条分支的轻量可变状态。"""

    __slots__ = ("x_t", "weight", "path_log_p", "t", "seed")

    def __init__(
        self,
        x_t: Tensor,         # (1, L) 当前 token 序列
        weight: float = 1.0, # 共识权重（多分支汇聚 = 高权重）
        path_log_p: float = 0.0,  # 编辑路径累计 log-prob（≤0，越大越好）
        t: float = 0.0,      # 当前连续时间
        seed: int = 0,       # 随机种子
    ):
        self.x_t = x_t
        self.weight = weight
        self.path_log_p = path_log_p
        self.t = t
        self.seed = seed

    def clone(self) -> _BranchState:
        return _BranchState(
            x_t=self.x_t.clone(),
            weight=self.weight,
            path_log_p=self.path_log_p,
            t=self.t,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _branch_sort_key(br: _BranchState) -> Tuple[float, float]:
    """分支排序键: 有编辑的分支优先, 共识度高的优先。

    path_log_p ≤ 0。无编辑的分支 path_log_p = 0；有编辑的 path_log_p < 0
    → -path_log_p > 0。用 -path_log_p 做主键保证编辑过的分支排在空闲分支前面，
    用 weight 做次键打破平局（共识度高的优先）。
    """
    return (-br.path_log_p, br.weight)


def _token_key(x_t: Tensor, pad_token: int, bos_token: int) -> Tuple[int, ...]:
    """可哈希的 token 序列标识（排除 PAD 和 BOS）。"""
    return tuple(int(t) for t in x_t[0].tolist()
                 if int(t) not in (pad_token, bos_token))


def _step_log_p(
    actions: dict,
    log_rates_eff: Tensor,    # (1, L, 3)
    log_ins_probs: Tensor,    # (1, L, V)
    log_sub_probs: Tensor,    # (1, L, V)
    adapt_h: float,
) -> float:
    """计算本步所有触发编辑的累计 log-prob。

    每个触发的编辑:
        INS/SUB: log(1 - exp(-h·λ)) + log(token_prob)
        DEL:     log(1 - exp(-h·λ))
    """
    rates = torch.exp(log_rates_eff[0])  # (L, 3)
    lp = 0.0
    eps = 1e-12

    # INSERT
    ins_mask = actions["ins_mask"][0]
    if ins_mask.any():
        for pos in ins_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            lam = rates[pos, 0].item()
            tok_id = int(actions["ins_tokens"][0, pos].item())
            tok_p = torch.exp(log_ins_probs[0, pos, tok_id]).item()
            lp += math.log(max(1.0 - math.exp(-adapt_h * lam), eps))
            lp += math.log(max(tok_p, eps))

    # SUBSTITUTE
    sub_mask = actions["sub_mask"][0]
    if sub_mask.any():
        for pos in sub_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            lam = rates[pos, 1].item()
            tok_id = int(actions["sub_tokens"][0, pos].item())
            tok_p = torch.exp(log_sub_probs[0, pos, tok_id]).item()
            lp += math.log(max(1.0 - math.exp(-adapt_h * lam), eps))
            lp += math.log(max(tok_p, eps))

    # DELETE
    del_mask = actions["del_mask"][0]
    if del_mask.any():
        for pos in del_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            lam = rates[pos, 2].item()
            lp += math.log(max(1.0 - math.exp(-adapt_h * lam), eps))

    return lp


def _apply_edits(
    x_t: Tensor,          # (1, L)
    actions: dict,
    max_seq_len: int,
    pad_token: int,
) -> Tensor:
    """对单条序列应用采样的编辑, 返回 (1, L')。"""
    x_next = x_t.clone()
    x_next[actions["sub_mask"]] = actions["sub_tokens"][actions["sub_mask"]]
    x_next = apply_ins_del_operations(
        x_next,
        actions["ins_mask"],
        actions["del_mask"],
        actions["ins_tokens"],
        max_seq_len=max_seq_len,
        pad_token=pad_token,
    )
    return x_next


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_euler_beam(
    model,
    x_0: Tensor,                     # (B, L_0) — 含 BOS, PAD 填充
    scheduler: KappaScheduler,
    n_branches: int = 5,             # 每条样本维护的并行分支数
    n_steps: int = 100,              # Euler 步数
    max_seq_len: int = 512,          # 序列最大长度
    pad_token: int = PAD_TOKEN,
    bos_token: int = BOS_TOKEN,
    use_rate_reparam: bool = False,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
    time_input: str = "t",
    train_scheduler: Optional[KappaScheduler] = None,
    event_prob_mode: str = "poisson",
    use_origin_mask: bool = False,
    base_seed: int = 42,             # 基础随机种子 (分支 k 得到 base_seed + k)
    record_trajectory: bool = False, # 未实现, 预留
    record_all_events: bool = False, # 未实现, 预留
    x_1: Optional[Tensor] = None,    # 未使用, 预留 (与 sample_euler 接口对齐)
    vocab_size: Optional[int] = None, # 未使用, 预留
) -> Tensor:
    """Euler 采样 + 分支维护。

    Args:
        model: EditFlowsTransformer 模型。
        x_0: (B, L_0) 初始序列。
        n_branches: 每条样本最多维护的并行分支数。
        n_steps: Euler 步数 (与 sample_euler 一致)。
        base_seed: 基础随机种子。

    Returns:
        x_final: (B, L_out) 每条样本的最优分支, PAD 填充到等长。
    """
    device = next(model.parameters()).device
    B = x_0.shape[0]
    default_h = 1.0 / n_steps

    # ── 初始化: 每条样本创建 n_branches 条分支 ──
    all_branches: List[List[_BranchState]] = []
    for b in range(B):
        branches = []
        for k in range(n_branches):
            branches.append(_BranchState(
                x_t=x_0[b:b + 1].to(device),
                t=0.0,
                seed=base_seed + b * n_branches + k,
            ))
        all_branches.append(branches)

    # ── 主循环: n_steps 步 Euler 推进 ──
    for step in range(n_steps):
        # 1. 收集所有未完成的分支 (t < 1.0)
        flat: List[Tuple[int, int, _BranchState]] = []
        for b in range(B):
            for k, br in enumerate(all_branches[b]):
                if br.t < 1.0:
                    flat.append((b, k, br))

        if not flat:
            break

        N = len(flat)
        max_L = max(s.x_t.shape[1] for _, _, s in flat)

        # 2. 构建批量 tensor (单次 GPU 传输)
        x_batch = torch.full((N, max_L), pad_token, dtype=torch.long, device=device)
        t_vals = torch.zeros(N, 1, device=device)
        for i, (_, _, s) in enumerate(flat):
            L = s.x_t.shape[1]
            x_batch[i, :L] = s.x_t
            t_vals[i, 0] = s.t

        # 3. 单次模型前向 (所有分支共享)
        x_pad_mask = x_batch == pad_token
        t_model = _compute_model_time(t_vals, scheduler, time_input, train_scheduler)
        log_rates, log_ins_probs, log_sub_probs = model(
            x_batch, t_model, x_pad_mask,
        )

        # 4. 速率修正 (与 sample_euler 完全一致)
        if not use_rate_reparam and train_scheduler is not None and \
           scheduler.name != train_scheduler.name:
            k_sample = get_rate_scale(
                t_vals, scheduler, clamp_kappa=clamp_kappa, clamp_max=clamp_max,
            )
            k_train = get_rate_scale(
                t_model, train_scheduler, clamp_kappa=clamp_kappa, clamp_max=clamp_max,
            )
            log_correction = torch.log(
                k_sample / k_train.clamp_min(1e-2)
            ).unsqueeze(1)
            log_rates = log_rates + log_correction

        log_rates_eff = apply_rate_parameterization(
            log_rates, t_vals, scheduler,
            use_rate_reparam=use_rate_reparam,
            clamp_kappa=clamp_kappa, clamp_max=clamp_max,
        )

        # 5. 批量采样 (一次 GPU 调用覆盖所有分支)
        N_br = len(flat)
        max_L_br = max(s.x_t.shape[1] for _, _, s in flat)
        V = log_ins_probs.shape[-1]

        x_br = torch.full((N_br, max_L_br), pad_token, dtype=torch.long, device=device)
        lr_br = torch.full((N_br, max_L_br, 3), -1e9, device=device, dtype=log_rates_eff.dtype)
        lip_br = torch.full((N_br, max_L_br, V), -1e9, device=device, dtype=log_ins_probs.dtype)
        lsp_br = torch.full((N_br, max_L_br, V), -1e9, device=device, dtype=log_sub_probs.dtype)
        branch_t_vals = torch.zeros(N_br, 1, device=device)

        for i, (_, _, s) in enumerate(flat):
            L = s.x_t.shape[1]
            x_br[i, :L] = s.x_t
            lr_br[i, :L] = log_rates_eff[i, :L]
            lip_br[i, :L] = log_ins_probs[i, :L]
            lsp_br[i, :L] = log_sub_probs[i, :L]
            branch_t_vals[i, 0] = s.t

        # 批量计算自适应步长 + 批量随机采样
        adapt_h_br = get_adaptive_h(default_h, branch_t_vals, scheduler)
        torch.manual_seed(base_seed + step)
        actions_batch = _sample_edit_actions(
            x_br, lr_br, lip_br, lsp_br, adapt_h_br,
            pad_token=pad_token, event_prob_mode=event_prob_mode,
        )

        # 6. 逐分支: 提取动作、计分、应用编辑 (CPU 轻量循环)
        new_branches: Dict[int, List[_BranchState]] = {b: [] for b in range(B)}

        for i, (b, k, s) in enumerate(flat):
            L = s.x_t.shape[1]

            # 从批量结果中提取本分支的动作
            actions = {
                "ins_mask":    actions_batch["ins_mask"][i:i + 1, :L],
                "sub_mask":    actions_batch["sub_mask"][i:i + 1, :L],
                "del_mask":    actions_batch["del_mask"][i:i + 1, :L],
                "ins_tokens":  actions_batch["ins_tokens"][i:i + 1, :L],
                "sub_tokens":  actions_batch["sub_tokens"][i:i + 1, :L],
            }

            step_lp = _step_log_p(
                actions,
                log_rates_eff[i:i + 1, :L],
                log_ins_probs[i:i + 1, :L],
                log_sub_probs[i:i + 1, :L],
                adapt_h_br[i].item(),
            )

            x_next = _apply_edits(s.x_t, actions, max_seq_len, pad_token)

            new_branches[b].append(_BranchState(
                x_t=x_next,
                weight=s.weight,
                path_log_p=s.path_log_p + step_lp,
                t=s.t + adapt_h_br[i].item(),
                seed=s.seed,
            ))

        # 7. 逐样本: 去重、排序、剪枝、分裂
        for b in range(B):
            candidates = new_branches[b]

            # 7a. 合并相同 token 序列的分支
            # 仅在不同分支汇聚到相同结果时合并。如果所有分支都相同（尚未发散），
            # 不合并 —— 保留所有分支的种子多样性以便后续发散。
            merged: Dict[Tuple[int, ...], _BranchState] = {}
            for br in candidates:
                key = _token_key(br.x_t, pad_token, bos_token)
                if key in merged:
                    merged[key].weight += br.weight
                    if _branch_sort_key(br) > _branch_sort_key(merged[key]):
                        merged[key].path_log_p = br.path_log_p
                        merged[key].t = br.t
                        merged[key].seed = br.seed
                else:
                    merged[key] = br

            if len(merged) == 1:
                # 全相同 → 保留原始分支（种子多样性）
                ranked = candidates
            else:
                ranked = sorted(merged.values(), key=_branch_sort_key, reverse=True)
            all_branches[b] = ranked[:n_branches]

            # 7b. 不足 n_branches 条时从最高 rank 分支分裂补充
            while len(all_branches[b]) < n_branches:
                parent_idx = len(all_branches[b]) % max(len(all_branches[b]), 1)
                parent = all_branches[b][parent_idx]
                all_branches[b].append(_BranchState(
                    x_t=parent.x_t.clone(),
                    weight=parent.weight * 0.5,
                    path_log_p=parent.path_log_p,
                    t=parent.t,
                    seed=parent.seed + 10000 + len(all_branches[b]),
                ))

    # ── 返回每条样本的最优分支 ──
    results: List[Tensor] = []
    for b in range(B):
        best = max(all_branches[b], key=_branch_sort_key)
        results.append(best.x_t)

    out_len = max(r.shape[1] for r in results)
    out = torch.full((B, out_len), pad_token, dtype=torch.long, device=device)
    for b, r in enumerate(results):
        out[b, :r.shape[1]] = r

    return out
