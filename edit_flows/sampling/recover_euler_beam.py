"""Euler 采样 + 分支维护 (Euler-Beam, optimized).

在不改变主要采样、排序和去重流程的前提下，优化以下热点：

1. 直接复用 x_batch / t_vals / 模型输出，不再二次构造并复制大 Tensor。
2. 一次性批量计算所有分支的 step log-prob，避免逐编辑位置 .item()/.tolist()。
3. 一次性批量应用 SUBSTITUTE / INSERT / DELETE，避免逐分支启动小型 CUDA kernel。
4. 将 step score、下一时刻和有效长度合并为一次 GPU -> CPU 传输。
5. 使用 torch.inference_mode() 减少推理开销。
6. 将所有分支的序列一次性传到 CPU 并批量生成去重 key，避免逐分支 .tolist()。
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
        x_t: Tensor,                  # (1, L) 当前 token 序列
        weight: float = 1.0,          # 共识权重
        path_log_p: float = 0.0,      # 编辑路径累计 log-prob
        t: float = 0.0,               # 当前连续时间
        seed: int = 0,                # 随机种子
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
    """保持原代码的排序逻辑。"""
    return (-br.path_log_p, br.weight)


def _step_log_p_batch(
    actions: dict,
    log_rates_eff: Tensor,   # (N, L, 3)
    log_ins_probs: Tensor,   # (N, L, V)
    log_sub_probs: Tensor,   # (N, L, V)
    adapt_h: Tensor,         # (N, 1) 或 (N,)
) -> Tensor:
    """批量复现原 `_step_log_p` 的评分，返回每条分支的分数 `(N,)`。

    该函数刻意保持原评分语义：只累计已经触发的编辑事件概率，
    不额外加入未发生事件的 survival probability。
    """
    eps = 1e-12
    log_eps = math.log(eps)

    rates = torch.exp(log_rates_eff)                       # (N, L, 3)
    h = adapt_h.reshape(-1, 1, 1).to(rates.dtype)         # (N, 1, 1)

    # log(1 - exp(-h * lambda))；expm1 在 h*lambda 很小时更稳定。
    event_log_p = torch.log(
        (-torch.expm1(-h * rates)).clamp_min(eps)
    )                                                      # (N, L, 3)

    vocab_size = log_ins_probs.shape[-1]

    # 未触发编辑的位置可能使用 -1 等哨兵 token；先 clamp，随后只在 mask=True
    # 的位置使用 gather 结果。
    ins_token_ids = actions["ins_tokens"].clamp(0, vocab_size - 1)
    sub_token_ids = actions["sub_tokens"].clamp(0, vocab_size - 1)

    ins_token_log_p = log_ins_probs.gather(
        dim=-1,
        index=ins_token_ids.unsqueeze(-1),
    ).squeeze(-1).clamp_min(log_eps)

    sub_token_log_p = log_sub_probs.gather(
        dim=-1,
        index=sub_token_ids.unsqueeze(-1),
    ).squeeze(-1).clamp_min(log_eps)

    zeros = torch.zeros_like(event_log_p[:, :, 0])

    ins_score = torch.where(
        actions["ins_mask"],
        event_log_p[:, :, 0] + ins_token_log_p,
        zeros,
    )
    sub_score = torch.where(
        actions["sub_mask"],
        event_log_p[:, :, 1] + sub_token_log_p,
        zeros,
    )
    del_score = torch.where(
        actions["del_mask"],
        event_log_p[:, :, 2],
        zeros,
    )

    return ins_score.sum(dim=1) + sub_score.sum(dim=1) + del_score.sum(dim=1)


def _apply_edits_batch(
    x_batch: Tensor,       # (N, L)
    actions: dict,
    max_seq_len: int,
    pad_token: int,
) -> Tensor:
    """一次性对整个 batch 应用编辑。"""
    x_next = x_batch.clone()

    sub_mask = actions["sub_mask"]
    x_next[sub_mask] = actions["sub_tokens"][sub_mask]

    return apply_ins_del_operations(
        x_next,
        actions["ins_mask"],
        actions["del_mask"],
        actions["ins_tokens"],
        max_seq_len=max_seq_len,
        pad_token=pad_token,
    )


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


@torch.inference_mode()
def sample_euler_beam(
    model,
    x_0: Tensor,                     # (B, L_0) — 含 BOS, PAD 填充
    scheduler: KappaScheduler,
    n_branches: int = 5,
    n_steps: int = 100,
    max_seq_len: int = 512,
    pad_token: int = PAD_TOKEN,
    bos_token: int = BOS_TOKEN,
    use_rate_reparam: bool = False,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
    time_input: str = "t",
    train_scheduler: Optional[KappaScheduler] = None,
    event_prob_mode: str = "poisson",
    use_origin_mask: bool = False,
    base_seed: int = 42,
    record_trajectory: bool = False,
    record_all_events: bool = False,
    x_1: Optional[Tensor] = None,
    vocab_size: Optional[int] = None,
) -> Tensor:
    """Euler 采样 + 分支维护的性能优化版本。

    主要优化不改变模型前向次数、随机采样调用和分支维护流程；只减少
    重复 Tensor 分配、逐分支 CUDA kernel 和 GPU/CPU 同步。
    """
    device = next(model.parameters()).device
    x_0_device = x_0.to(device)

    B = x_0_device.shape[0]
    default_h = 1.0 / n_steps

    # 初始化：每条样本创建 n_branches 条分支。
    all_branches: List[List[_BranchState]] = []
    for b in range(B):
        branches = []
        for k in range(n_branches):
            branches.append(
                _BranchState(
                    x_t=x_0_device[b:b + 1],
                    t=0.0,
                    seed=base_seed + b * n_branches + k,
                )
            )
        all_branches.append(branches)

    for step in range(n_steps):
        # 1. 收集所有未完成的分支。
        flat: List[Tuple[int, int, _BranchState]] = []
        for b in range(B):
            for k, branch in enumerate(all_branches[b]):
                if branch.t < 1.0:
                    flat.append((b, k, branch))

        if not flat:
            break

        N = len(flat)
        branch_lengths_list = [state.x_t.shape[1] for _, _, state in flat]
        max_L = max(branch_lengths_list)

        # 2. 构建一次模型输入 batch，同时记录每条分支自身 Tensor 的宽度。
        x_batch = torch.full(
            (N, max_L),
            pad_token,
            dtype=torch.long,
            device=device,
        )
        t_vals = torch.empty(N, 1, dtype=torch.float, device=device)

        for i, (_, _, state) in enumerate(flat):
            length = state.x_t.shape[1]
            x_batch[i, :length] = state.x_t[0]
            t_vals[i, 0] = state.t

        # 3. 所有分支共享一次模型前向。
        x_pad_mask = x_batch == pad_token
        t_model = _compute_model_time(
            t_vals,
            scheduler,
            time_input,
            train_scheduler,
        )
        log_rates, log_ins_probs, log_sub_probs = model(
            x_batch,
            t_model,
            x_pad_mask,
        )

        # 4. 速率修正。
        if (
            not use_rate_reparam
            and train_scheduler is not None
            and scheduler.name != train_scheduler.name
        ):
            k_sample = get_rate_scale(
                t_vals,
                scheduler,
                clamp_kappa=clamp_kappa,
                clamp_max=clamp_max,
            )
            k_train = get_rate_scale(
                t_model,
                train_scheduler,
                clamp_kappa=clamp_kappa,
                clamp_max=clamp_max,
            )
            log_correction = torch.log(
                k_sample / k_train.clamp_min(1e-2)
            ).unsqueeze(1)
            log_rates = log_rates + log_correction

        log_rates_eff = apply_rate_parameterization(
            log_rates,
            t_vals,
            scheduler,
            use_rate_reparam=use_rate_reparam,
            clamp_kappa=clamp_kappa,
            clamp_max=clamp_max,
        )

        # 5. 保持旧代码的 padding 行为：超出每条分支自身 Tensor 宽度的位置
        # 设为 -1e9。这里直接原地修改已有输出，不再复制 N*L*V 大 Tensor。
        branch_lengths = torch.tensor(
            branch_lengths_list,
            dtype=torch.long,
            device=device,
        )
        positions = torch.arange(max_L, device=device).unsqueeze(0)
        outside_branch = positions >= branch_lengths.unsqueeze(1)  # (N, L)

        outside_branch_3d = outside_branch.unsqueeze(-1)
        log_rates_eff.masked_fill_(outside_branch_3d, -1e9)
        log_ins_probs.masked_fill_(outside_branch_3d, -1e9)
        log_sub_probs.masked_fill_(outside_branch_3d, -1e9)

        # 6. 直接复用 x_batch、t_vals 和模型输出进行采样。
        adapt_h_br = get_adaptive_h(default_h, t_vals, scheduler)

        torch.manual_seed(base_seed + step)
        actions_batch = _sample_edit_actions(
            x_batch,
            log_rates_eff,
            log_ins_probs,
            log_sub_probs,
            adapt_h_br,
            pad_token=pad_token,
            event_prob_mode=event_prob_mode,
        )

        # 7. 整个 batch 一次性评分和应用编辑。
        step_log_ps = _step_log_p_batch(
            actions_batch,
            log_rates_eff,
            log_ins_probs,
            log_sub_probs,
            adapt_h_br,
        )
        x_next_batch = _apply_edits_batch(
            x_batch,
            actions_batch,
            max_seq_len=max_seq_len,
            pad_token=pad_token,
        )

        next_times = t_vals.squeeze(-1) + adapt_h_br.reshape(-1)
        next_lengths = (x_next_batch != pad_token).sum(dim=1)

        # 将循环中需要的三个标量一次性传到 CPU。
        metadata_cpu = torch.stack(
            (
                step_log_ps,
                next_times.to(step_log_ps.dtype),
                next_lengths.to(step_log_ps.dtype),
            ),
            dim=1,
        ).cpu().tolist()

        # 根据本 batch 的实际最大有效长度，只传输去重所需的序列区域。
        # 相比逐分支 x_t.tolist()，这里只进行一次批量 GPU -> CPU 传输。
        max_next_length = max(
            max(int(row[2]), 1)
            for row in metadata_cpu
        )
        x_next_rows_cpu = x_next_batch[:, :max_next_length].cpu().tolist()

        branch_keys: List[Tuple[int, ...]] = []
        for row, metadata in zip(x_next_rows_cpu, metadata_cpu):
            next_length = max(int(metadata[2]), 1)
            branch_keys.append(
                tuple(
                    token
                    for token in row[:next_length]
                    if token not in (pad_token, bos_token)
                )
            )

        # 8. Python 循环只负责创建轻量状态对象，并附带预先生成的 key。
        new_branches: Dict[
            int,
            List[Tuple[_BranchState, Tuple[int, ...]]],
        ] = {b: [] for b in range(B)}

        for i, (b, _k, state) in enumerate(flat):
            step_lp, next_t, next_length_float = metadata_cpu[i]
            next_length = max(int(next_length_float), 1)

            # clone 使每个分支拥有独立、紧凑的存储，避免小切片长期持有
            # 整个 x_next_batch 的底层显存。
            x_next = x_next_batch[i:i + 1, :next_length].clone()
            branch = _BranchState(
                x_t=x_next,
                weight=state.weight,
                path_log_p=state.path_log_p + float(step_lp),
                t=float(next_t),
                seed=state.seed,
            )
            new_branches[b].append((branch, branch_keys[i]))

        # 9. 保留原来的去重、排序、剪枝和分裂逻辑。
        for b in range(B):
            candidates = new_branches[b]

            merged: Dict[Tuple[int, ...], _BranchState] = {}
            for branch, key in candidates:
                if key in merged:
                    merged[key].weight += branch.weight
                    if _branch_sort_key(branch) > _branch_sort_key(merged[key]):
                        merged[key].path_log_p = branch.path_log_p
                        merged[key].t = branch.t
                        merged[key].seed = branch.seed
                else:
                    merged[key] = branch

            if len(merged) == 1:
                ranked = [branch for branch, _key in candidates]
            else:
                ranked = sorted(
                    merged.values(),
                    key=_branch_sort_key,
                    reverse=True,
                )

            all_branches[b] = ranked[:n_branches]

            while len(all_branches[b]) < n_branches:
                parent_idx = len(all_branches[b]) % max(
                    len(all_branches[b]),
                    1,
                )
                parent = all_branches[b][parent_idx]
                all_branches[b].append(
                    _BranchState(
                        x_t=parent.x_t.clone(),
                        weight=parent.weight * 0.5,
                        path_log_p=parent.path_log_p,
                        t=parent.t,
                        seed=parent.seed + 10000 + len(all_branches[b]),
                    )
                )

    # 返回每条样本的最优分支。
    results: List[Tensor] = []
    for b in range(B):
        best = max(all_branches[b], key=_branch_sort_key)
        results.append(best.x_t)

    out_len = max(result.shape[1] for result in results)
    out = torch.full(
        (B, out_len),
        pad_token,
        dtype=torch.long,
        device=device,
    )
    for b, result in enumerate(results):
        out[b, :result.shape[1]] = result

    return out