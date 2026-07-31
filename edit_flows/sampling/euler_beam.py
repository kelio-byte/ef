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
    _event_probability,
    get_adaptive_h,
)
from edit_flows.sampling.ops import apply_ins_del_operations
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


# ---------------------------------------------------------------------------
# 分支状态
# ---------------------------------------------------------------------------


class _BranchState:
    """单条分支的轻量可变状态。"""

    __slots__ = ("x_t", "weight", "path_log_p", "log_mass", "t", "seed")

    def __init__(
        self,
        x_t: Tensor,         # (1, L) 当前 token 序列
        weight: float = 1.0, # 共识权重（多分支汇聚 = 高权重）
        path_log_p: float = 0.0,  # 编辑路径累计 log-prob（≤0，越大越好）
        log_mass: float = 0.0,  # Monte Carlo 估计的状态概率质量
        t: float = 0.0,      # 当前连续时间
        seed: int = 0,       # 随机种子
    ):
        self.x_t = x_t
        self.weight = weight
        self.path_log_p = path_log_p
        self.log_mass = log_mass
        self.t = t
        self.seed = seed

    def clone(self) -> _BranchState:
        return _BranchState(
            x_t=self.x_t.clone(),
            weight=self.weight,
            path_log_p=self.path_log_p,
            log_mass=self.log_mass,
            t=self.t,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _branch_sort_key(br: _BranchState) -> Tuple[float, float]:
    """正式排序：只按聚合状态质量；seed 提供确定性平局顺序。"""
    return (br.log_mass, -float(br.seed))


def _legacy_branch_sort_key(br: _BranchState) -> Tuple[float, float]:
    """旧实验排序：累计触发分数越负，反而排名越高。仅用于消融。"""
    return (-br.path_log_p, br.weight)


def _mix_child_seed(parent_seed: int, step: int, child_index: int) -> int:
    """稳定混合 child seed；child 0 保留父随机流以兼容 M=1。"""
    if child_index < 0:
        raise ValueError(f"child_index must be >= 0, got {child_index}")
    if child_index == 0:
        return int(parent_seed)
    mask = (1 << 64) - 1
    x = (
        (int(parent_seed) & mask)
        ^ (((step + 1) * 0x9E3779B97F4A7C15) & mask)
        ^ (((child_index + 1) * 0xD1B54A32D192ED03) & mask)
    )
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & mask
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & mask
    x ^= x >> 31
    return x & ((1 << 63) - 1)


def _logaddexp_float(a: float, b: float) -> float:
    """稳定计算两个 Python log-weight 的 logsumexp。"""
    high = max(a, b)
    low = min(a, b)
    return high + math.log1p(math.exp(low - high))


def _merge_state_candidates(
    candidates: List[Tuple[_BranchState, Tuple[int, ...]]],
    n_branches: int,
) -> List[_BranchState]:
    """按 token 状态合并 Monte Carlo 质量并保留 Top-K。"""
    merged: Dict[Tuple[int, ...], _BranchState] = {}
    for branch, key in candidates:
        if key not in merged:
            merged[key] = branch
            continue
        representative = merged[key]
        combined_mass = _logaddexp_float(
            representative.log_mass, branch.log_mass,
        )
        if branch.seed < representative.seed:
            branch.log_mass = combined_mass
            branch.weight += representative.weight
            merged[key] = branch
        else:
            representative.log_mass = combined_mass
            representative.weight += branch.weight
    return sorted(
        merged.values(), key=_branch_sort_key, reverse=True,
    )[:n_branches]


def _token_key(x_t: Tensor, pad_token: int, bos_token: int) -> Tuple[int, ...]:
    """可哈希的 token 序列标识（排除 PAD 和 BOS）。"""
    return tuple(int(t) for t in x_t[0].tolist()
                 if int(t) not in (pad_token, bos_token))


def _step_log_p_batch(
    actions: dict,
    log_rates_eff: Tensor,
    log_ins_probs: Tensor,
    log_sub_probs: Tensor,
    adapt_h: Tensor,
    score_mode: str = "full_probability",
) -> Tensor:
    """批量计算每条分支本步完整动作集合的 log-prob。"""
    rates = torch.exp(log_rates_eff)
    eps = 1e-12
    log_eps = math.log(eps)
    ins_mu = adapt_h * rates[:, :, 0]
    if score_mode == "legacy_triggered_reverse":
        event_log_p = torch.log(
            (-torch.expm1(-adapt_h.unsqueeze(-1) * rates)).clamp_min(eps)
        )
        ins_token_log_p = log_ins_probs.gather(
            2, actions["ins_tokens"].unsqueeze(-1),
        ).squeeze(-1).clamp_min(log_eps)
        sub_token_log_p = log_sub_probs.gather(
            2, actions["sub_tokens"].unsqueeze(-1),
        ).squeeze(-1).clamp_min(log_eps)
        return (
            (event_log_p[:, :, 0] + ins_token_log_p)
            .masked_fill(~actions["ins_mask"], 0.0).sum(dim=1)
            + (event_log_p[:, :, 1] + sub_token_log_p)
            .masked_fill(~actions["sub_mask"], 0.0).sum(dim=1)
            + event_log_p[:, :, 2]
            .masked_fill(~actions["del_mask"], 0.0).sum(dim=1)
        )
    if score_mode != "full_probability":
        raise ValueError(f"Unsupported score_mode: {score_mode}")
    ds_rates = rates[:, :, 1] + rates[:, :, 2]
    ds_mu = adapt_h * ds_rates
    ins_event_log_p = torch.log((-torch.expm1(-ins_mu)).clamp_min(eps))
    ds_event_log_p = torch.log((-torch.expm1(-ds_mu)).clamp_min(eps))
    ins_token_log_p = log_ins_probs.gather(
        2, actions["ins_tokens"].unsqueeze(-1),
    ).squeeze(-1).clamp_min(log_eps)
    sub_token_log_p = log_sub_probs.gather(
        2, actions["sub_tokens"].unsqueeze(-1),
    ).squeeze(-1).clamp_min(log_eps)
    ins_contrib = torch.where(
        actions["ins_mask"], ins_event_log_p + ins_token_log_p, -ins_mu,
    )
    sub_contrib = (
        ds_event_log_p
        + torch.log(
            (rates[:, :, 1] / ds_rates.clamp_min(eps)).clamp_min(eps)
        )
        + sub_token_log_p
    )
    del_contrib = (
        ds_event_log_p
        + torch.log(
            (rates[:, :, 2] / ds_rates.clamp_min(eps)).clamp_min(eps)
        )
    )
    ds_contrib = torch.where(
        actions["sub_mask"],
        sub_contrib,
        torch.where(actions["del_mask"], del_contrib, -ds_mu),
    )
    return ins_contrib.sum(dim=1) + ds_contrib.sum(dim=1)


def _step_log_p(
    actions: dict,
    log_rates_eff: Tensor,
    log_ins_probs: Tensor,
    log_sub_probs: Tensor,
    adapt_h: float,
) -> float:
    """单分支兼容包装；主采样循环使用 `_step_log_p_batch()`。"""
    h = torch.tensor(
        [[adapt_h]], dtype=log_rates_eff.dtype, device=log_rates_eff.device,
    )
    return _step_log_p_batch(
        actions, log_rates_eff, log_ins_probs, log_sub_probs, h,
    )[0].item()


def _apply_edits_batch(
    x_t: Tensor,
    actions: dict,
    max_seq_len: int,
    pad_token: int,
) -> Tensor:
    """批量应用采样编辑。"""
    x_next = x_t.clone()
    x_next[actions["sub_mask"]] = actions["sub_tokens"][actions["sub_mask"]]
    return apply_ins_del_operations(
        x_next,
        actions["ins_mask"],
        actions["del_mask"],
        actions["ins_tokens"],
        max_seq_len=max_seq_len,
        pad_token=pad_token,
    )


# ---------------------------------------------------------------------------
# 分支独立随机采样
# ---------------------------------------------------------------------------


def _stateless_uniform(
    seeds: Tensor,
    step: int,
    seq_len: int,
    stream: int,
    dtype: torch.dtype,
) -> Tensor:
    """生成由 (seed, step, position, stream) 决定的批量均匀随机数。"""
    modulus = 2_147_483_647
    positions = torch.arange(
        1, seq_len + 1, dtype=torch.int64, device=seeds.device,
    ).unsqueeze(0)
    x = torch.remainder(
        torch.remainder(seeds.to(torch.int64), modulus).unsqueeze(1) * 48_271
        + (step + 1) * 69_621
        + positions * 1_013_904_223
        + (stream + 1) * 1_664_525,
        modulus,
    )
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
    x = torch.remainder(x * 73_856_093 + 19_349_663, modulus)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 13))
    return ((x.to(torch.float64) + 0.5) / modulus).to(dtype)


def _sample_tokens_from_uniform(log_probs: Tensor, uniform: Tensor) -> Tensor:
    """用均匀随机数对最后一维 categorical 分布做 inverse-CDF 采样。"""
    cdf = torch.exp(log_probs).cumsum(dim=-1)
    return (cdf < uniform.unsqueeze(-1)).sum(dim=-1).clamp_max(
        log_probs.shape[-1] - 1,
    )


def _sample_actions_per_branch(
    branch_seeds: Tensor,
    x_t: Tensor,
    log_rates: Tensor,
    log_ins_probs: Tensor,
    log_sub_probs: Tensor,
    adapt_h: Tensor,
    pad_token: int,
    event_prob_mode: str,
    step: int,
) -> dict:
    """按 branch seed 无状态、批量地采样所有分支动作。"""
    seeds = branch_seeds.to(device=x_t.device, dtype=torch.int64)
    rates = torch.exp(log_rates)
    ins_probs = torch.exp(log_ins_probs)
    sub_probs = torch.exp(log_sub_probs)
    lambda_ins = rates[:, :, 0]
    lambda_sub = rates[:, :, 1]
    lambda_del = rates[:, :, 2]

    ins_prob = _event_probability(adapt_h * lambda_ins, event_prob_mode)
    ds_prob = _event_probability(
        adapt_h * (lambda_sub + lambda_del), event_prob_mode,
    )
    seq_len = x_t.shape[1]
    ins_mask = _stateless_uniform(
        seeds, step, seq_len, stream=0, dtype=lambda_ins.dtype,
    ) < ins_prob
    ds_mask = _stateless_uniform(
        seeds, step, seq_len, stream=1, dtype=lambda_sub.dtype,
    ) < ds_prob
    prob_del = lambda_del / (lambda_sub + lambda_del + 1e-8)
    del_mask = ds_mask & (
        _stateless_uniform(
            seeds, step, seq_len, stream=2, dtype=lambda_del.dtype,
        ) < prob_del
    )
    sub_mask = ds_mask & ~del_mask

    ins_tokens = _sample_tokens_from_uniform(
        log_ins_probs,
        _stateless_uniform(
            seeds, step, seq_len, stream=3, dtype=log_ins_probs.dtype,
        ),
    )
    sub_tokens = _sample_tokens_from_uniform(
        log_sub_probs,
        _stateless_uniform(
            seeds, step, seq_len, stream=4, dtype=log_sub_probs.dtype,
        ),
    )

    non_pad_mask = x_t != pad_token
    ins_mask &= non_pad_mask
    del_mask &= non_pad_mask
    sub_mask &= non_pad_mask
    ins_tokens = ins_tokens.masked_fill(~non_pad_mask, pad_token)
    sub_tokens = sub_tokens.masked_fill(~non_pad_mask, pad_token)
    return {
        "rates": rates,
        "ins_mask": ins_mask,
        "del_mask": del_mask,
        "sub_mask": sub_mask,
        "ins_tokens": ins_tokens,
        "sub_tokens": sub_tokens,
        "ins_probs": ins_probs,
        "sub_probs": sub_probs,
    }


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_euler_beam(
    model,
    x_0: Tensor,                     # (B, L_0) — 含 BOS, PAD 填充
    scheduler: KappaScheduler,
    n_branches: int = 5,             # 每条样本维护的并行分支数
    n_children: int = 1,             # 每个父分支每步采样的后继数
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
    sample_seeds: Optional[List[int]] = None,
    score_mode: str = "full_probability",
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
        n_children: 每个父分支生成的独立随机后继数。
        n_steps: Euler 步数 (与 sample_euler 一致)。
        base_seed: 基础随机种子。

    Returns:
        x_final: (B, L_out) 每条样本的最优分支, PAD 填充到等长。
    """
    if n_branches < 1:
        raise ValueError(f"n_branches must be >= 1, got {n_branches}")
    if n_children < 1:
        raise ValueError(f"n_children must be >= 1, got {n_children}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if x_0.shape[0] < 1:
        raise ValueError("x_0 batch must contain at least one sample")
    if use_origin_mask:
        raise NotImplementedError(
            "Euler-Beam does not yet track origin_mask across edits"
        )
    if sample_seeds is not None and len(sample_seeds) != x_0.shape[0]:
        raise ValueError(
            f"sample_seeds length must equal batch size {x_0.shape[0]}, "
            f"got {len(sample_seeds)}"
        )
    if score_mode not in ("full_probability", "legacy_triggered_reverse"):
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    sort_key = (
        _legacy_branch_sort_key
        if score_mode == "legacy_triggered_reverse"
        else _branch_sort_key
    )

    device = next(model.parameters()).device
    B = x_0.shape[0]
    default_h = 1.0 / n_steps

    # ── 初始化: 每条样本创建 n_branches 条分支 ──
    all_branches: List[List[_BranchState]] = []
    for b in range(B):
        sample_seed = (
            sample_seeds[b]
            if sample_seeds is not None
            else base_seed + b * n_branches
        )
        branches = []
        for k in range(n_branches):
            branches.append(_BranchState(
                x_t=x_0[b:b + 1].to(device),
                log_mass=(-math.log(n_branches) if n_children > 1 else 0.0),
                t=0.0,
                seed=sample_seed + k,
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

        # 批量计算父分支步长。模型前向规模始终是 K；仅模型输出和动作
        # 张量扩展为 K×M，避免重复 Transformer forward。
        adapt_h_br = get_adaptive_h(default_h, branch_t_vals, scheduler)
        parent_index_values = [
            parent_i
            for parent_i in range(N_br)
            for _ in range(n_children)
        ]
        parent_indices = torch.tensor(
            parent_index_values, dtype=torch.long, device=device,
        )
        x_candidates = x_br.index_select(0, parent_indices)
        lr_candidates = lr_br.index_select(0, parent_indices)
        lip_candidates = lip_br.index_select(0, parent_indices)
        lsp_candidates = lsp_br.index_select(0, parent_indices)
        adapt_h_candidates = adapt_h_br.index_select(0, parent_indices)
        child_seed_values = [
            _mix_child_seed(branch.seed, step, child_index)
            for _, _, branch in flat
            for child_index in range(n_children)
        ]
        child_seeds = torch.tensor(
            child_seed_values, dtype=torch.int64, device=device,
        )
        actions_batch = _sample_actions_per_branch(
            child_seeds,
            x_candidates,
            lr_candidates,
            lip_candidates,
            lsp_candidates,
            adapt_h_candidates,
            pad_token=pad_token, event_prob_mode=event_prob_mode,
            step=step,
        )

        # 6. 批量计分、应用编辑，并各自一次性传回 CPU 元数据
        step_log_ps = _step_log_p_batch(
            actions_batch,
            lr_candidates,
            lip_candidates,
            lsp_candidates,
            adapt_h_candidates,
            score_mode=score_mode,
        ).cpu().tolist()
        adapt_h_values = adapt_h_candidates.squeeze(-1).cpu().tolist()
        x_next_batch = _apply_edits_batch(
            x_candidates, actions_batch, max_seq_len, pad_token,
        )
        x_next_cpu = x_next_batch.cpu()

        new_branches: Dict[int, List[_BranchState]] = {b: [] for b in range(B)}
        new_keys: Dict[int, List[Tuple[int, ...]]] = {b: [] for b in range(B)}

        child_log_share = math.log(n_children)
        for i, parent_i in enumerate(parent_index_values):
            b, k, s = flat[parent_i]
            new_branches[b].append(_BranchState(
                x_t=x_next_batch[i:i + 1],
                weight=s.weight,
                path_log_p=s.path_log_p + step_log_ps[i],
                log_mass=(
                    s.log_mass - child_log_share
                    if n_children > 1 else s.log_mass
                ),
                t=s.t + adapt_h_values[i],
                seed=child_seed_values[i],
            ))
            new_keys[b].append(
                _token_key(x_next_cpu[i:i + 1], pad_token, bos_token)
            )

        # 7. 逐样本: 去重、排序、剪枝、分裂
        for b in range(B):
            candidates = new_branches[b]

            if n_children > 1 and score_mode == "full_probability":
                paired = list(zip(candidates, new_keys[b]))
                all_branches[b] = _merge_state_candidates(
                    paired, n_branches,
                )
                continue

            # 7a. 合并相同 token 序列的分支
            # 仅在不同分支汇聚到相同结果时合并。如果所有分支都相同（尚未发散），
            # 不合并 —— 保留所有分支的种子多样性以便后续发散。
            merged: Dict[Tuple[int, ...], _BranchState] = {}
            for br, key in zip(candidates, new_keys[b]):
                if key in merged:
                    merged[key].weight += br.weight
                    if sort_key(br) > sort_key(merged[key]):
                        merged[key].path_log_p = br.path_log_p
                        merged[key].t = br.t
                        merged[key].seed = br.seed
                else:
                    merged[key] = br

            if len(merged) == 1:
                # 全相同 → 保留原始分支（种子多样性）
                ranked = candidates
            else:
                ranked = sorted(merged.values(), key=sort_key, reverse=True)
            all_branches[b] = ranked[:n_branches]

            # M=1 兼容路径保留旧机械复制；M>1 已有真实 offspring，不复制。
            if n_children > 1:
                continue
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
        best = max(all_branches[b], key=sort_key)
        results.append(best.x_t)

    out_len = max(r.shape[1] for r in results)
    out = torch.full((B, out_len), pad_token, dtype=torch.long, device=device)
    for b, r in enumerate(results):
        out[b, :r.shape[1]] = r

    return out
