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
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from edit_flows.core.rate_scale import apply_rate_parameterization, get_rate_scale
from edit_flows.core.scheduler import KappaScheduler
from edit_flows.sampling.center_bias import (
    align_position_scores,
    renormalize_position_biased_log_rates,
)
from edit_flows.sampling.euler import (
    _compute_model_time,
    _event_probability,
    _forward_edit_model,
    _prepare_product_memory_for_sampling,
    get_adaptive_h,
)
from edit_flows.sampling.ops import (
    apply_ins_del_operations,
    edit_position_masks,
    legal_token_log_probs,
)
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


FirstEditSignature = Optional[Tuple[Tuple[str, int, int], ...]]


# ---------------------------------------------------------------------------
# 分支状态
# ---------------------------------------------------------------------------


class _BranchState:
    """单条分支的轻量可变状态。"""

    __slots__ = (
        "x_t", "weight", "path_log_p", "log_mass", "t", "seed", "state_key",
        "first_edit_signature", "first_event_pending", "first_event_record",
        "first_event_reweighted",
    )

    def __init__(
        self,
        x_t: Tensor,         # (1, L) 当前 token 序列
        weight: float = 1.0, # 共识权重（多分支汇聚 = 高权重）
        path_log_p: float = 0.0,  # 编辑路径累计 log-prob（≤0，越大越好）
        log_mass: float = 0.0,  # Monte Carlo 估计的状态概率质量
        t: float = 0.0,      # 当前连续时间
        seed: int = 0,       # 随机种子
        state_key: Optional[Tuple[int, ...]] = None,
        first_edit_signature: FirstEditSignature = None,
        first_event_pending: bool = False,
        first_event_record: Optional[dict] = None,
        first_event_reweighted: bool = False,
    ):
        self.x_t = x_t
        self.weight = weight
        self.path_log_p = path_log_p
        self.log_mass = log_mass
        self.t = t
        self.seed = seed
        self.state_key = state_key
        self.first_edit_signature = first_edit_signature
        self.first_event_pending = first_event_pending
        self.first_event_record = first_event_record
        self.first_event_reweighted = first_event_reweighted

    def clone(self) -> _BranchState:
        return _BranchState(
            x_t=self.x_t.clone(),
            weight=self.weight,
            path_log_p=self.path_log_p,
            log_mass=self.log_mass,
            t=self.t,
            seed=self.seed,
            state_key=self.state_key,
            first_edit_signature=self.first_edit_signature,
            first_event_pending=self.first_event_pending,
            first_event_record=self.first_event_record,
            first_event_reweighted=self.first_event_reweighted,
        )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _branch_sort_key(br: _BranchState) -> Tuple[float, float]:
    """正式排序：只按聚合状态质量；seed 提供确定性平局顺序。"""
    return (br.log_mass, -float(br.seed))


def _trajectory_branch_sort_key(br: _BranchState) -> Tuple[float, float]:
    """M=1 兼容排序：完整单轨迹 log-prob 越大越好。"""
    return (br.path_log_p, br.weight)


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
    origin_key: Optional[Tuple[int, ...]] = None,
    changed_state_bonus: float = 0.0,
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
        best_path_log_p = max(
            representative.path_log_p, branch.path_log_p,
        )
        if branch.seed < representative.seed:
            branch.log_mass = combined_mass
            branch.weight += representative.weight
            branch.path_log_p = best_path_log_p
            merged[key] = branch
        else:
            representative.log_mass = combined_mass
            representative.weight += branch.weight
            representative.path_log_p = best_path_log_p
    ranked_items = sorted(
        merged.items(),
        key=lambda item: (
            item[1].log_mass
            + changed_state_bonus * float(item[0] != origin_key),
            item[1].log_mass,
            -float(item[1].seed),
        ),
        reverse=True,
    )
    return [branch for _, branch in ranked_items[:n_branches]]


def _select_k1_m2_children(
    candidates: List[_BranchState],
    keys: List[Tuple[int, ...]],
    origin_key: Tuple[int, ...],
    changed_state_bonus: float,
) -> _BranchState:
    """Exact two-child specialization of merge-and-Top-1 selection."""
    if len(candidates) != 2 or len(keys) != 2:
        raise ValueError("K1M2 selection requires exactly two children")
    first, second = candidates
    first_key, second_key = keys
    if first_key == second_key:
        combined_mass = _logaddexp_float(first.log_mass, second.log_mass)
        best_path_log_p = max(first.path_log_p, second.path_log_p)
        if second.seed < first.seed:
            second.log_mass = combined_mass
            second.weight += first.weight
            second.path_log_p = best_path_log_p
            return second
        first.log_mass = combined_mass
        first.weight += second.weight
        first.path_log_p = best_path_log_p
        return first

    def rank(branch: _BranchState, key: Tuple[int, ...]):
        return (
            branch.log_mass
            + changed_state_bonus * float(key != origin_key),
            branch.log_mass,
            -float(branch.seed),
        )

    if rank(second, second_key) > rank(first, first_key):
        return second
    return first


def _first_edit_signatures_from_actions(
    actions: dict,
    row_indices: Optional[List[int]] = None,
) -> List[FirstEditSignature]:
    """Return a compact signature for the first active edit in each row.

    Only the first affected position is retained.  If multiple edit types are
    active at that position, all of them are included in a deterministic
    order.  The reduction is performed on-device and only a handful of values
    per row are copied to CPU, so enabling the diversity policy does not copy
    the full action tensors.
    """
    row_count = actions["ins_mask"].shape[0]
    if row_indices is None:
        selected = torch.arange(
            row_count, dtype=torch.long, device=actions["ins_mask"].device,
        )
    else:
        selected = torch.tensor(
            row_indices, dtype=torch.long, device=actions["ins_mask"].device,
        )
    if selected.numel() == 0:
        return []

    ins_mask = actions["ins_mask"].index_select(0, selected)
    sub_mask = actions["sub_mask"].index_select(0, selected)
    del_mask = actions["del_mask"].index_select(0, selected)
    ins_tokens = actions["ins_tokens"].index_select(0, selected)
    sub_tokens = actions["sub_tokens"].index_select(0, selected)
    active = ins_mask | sub_mask | del_mask
    has_active = active.any(dim=1)
    first_position = active.to(torch.long).argmax(dim=1)
    row = torch.arange(
        selected.shape[0], dtype=torch.long, device=active.device,
    )
    summary = torch.stack((
        has_active.to(torch.long),
        first_position,
        ins_mask[row, first_position].to(torch.long),
        sub_mask[row, first_position].to(torch.long),
        del_mask[row, first_position].to(torch.long),
        ins_tokens[row, first_position],
        sub_tokens[row, first_position],
    ), dim=1).detach().cpu().tolist()

    signatures: List[FirstEditSignature] = []
    for (
        has_edit, position, has_ins, has_sub, has_del,
        ins_token, sub_token,
    ) in summary:
        if not has_edit:
            signatures.append(None)
            continue
        events: List[Tuple[str, int, int]] = []
        if has_ins:
            events.append(("ins", int(position), int(ins_token)))
        if has_sub:
            events.append(("sub", int(position), int(sub_token)))
        if has_del:
            events.append(("del", int(position), -1))
        signatures.append(tuple(events))
    return signatures


def _merge_state_candidates_with_first_edit_diversity(
    candidates: List[Tuple[_BranchState, Tuple[int, ...]]],
    n_branches: int,
    origin_key: Optional[Tuple[int, ...]] = None,
    changed_state_bonus: float = 0.0,
) -> List[_BranchState]:
    """Merge states while reserving slots for distinct first-edit signatures.

    State mass is merged only when both the token state and the first-edit
    signature match.  This keeps different search hypotheses separate long
    enough for the diversity constraint to act.  If fewer than ``n_branches``
    signatures exist, the sampler keeps only the available unique hypotheses;
    it does not manufacture duplicate no-op branches that would cost extra
    forwards without adding coverage.
    """
    merged: Dict[
        Tuple[Tuple[int, ...], FirstEditSignature], _BranchState
    ] = {}
    for branch, key in candidates:
        merge_key = (key, branch.first_edit_signature)
        if merge_key not in merged:
            merged[merge_key] = branch
            continue
        representative = merged[merge_key]
        combined_mass = _logaddexp_float(
            representative.log_mass, branch.log_mass,
        )
        best_path_log_p = max(
            representative.path_log_p, branch.path_log_p,
        )
        if branch.seed < representative.seed:
            branch.log_mass = combined_mass
            branch.weight += representative.weight
            branch.path_log_p = best_path_log_p
            merged[merge_key] = branch
        else:
            representative.log_mass = combined_mass
            representative.weight += branch.weight
            representative.path_log_p = best_path_log_p

    def rank(branch: _BranchState, key: Tuple[int, ...]):
        return (
            branch.log_mass
            + changed_state_bonus * float(key != origin_key),
            branch.log_mass,
            -float(branch.seed),
        )

    best_by_signature: Dict[FirstEditSignature, Tuple[_BranchState, Tuple[int, ...]]] = {}
    merged_items = list(merged.items())
    for (_, _), branch in merged_items:
        key = branch.state_key
        if key is None:
            raise RuntimeError(
                "first-edit diversity requires tracked branch state keys"
            )
        signature = branch.first_edit_signature
        previous = best_by_signature.get(signature)
        if previous is None or rank(branch, key) > rank(*previous):
            best_by_signature[signature] = (branch, key)

    selected = [
        branch for branch, _ in sorted(
            best_by_signature.values(),
            key=lambda item: rank(*item),
            reverse=True,
        )[:n_branches]
    ]
    return sorted(
        selected[:n_branches],
        key=lambda branch: rank(branch, branch.state_key or ()),
        reverse=True,
    )


def _select_first_edit_diverse_run_group(
    new_branches: Dict[int, List[_BranchState]],
    new_keys: Dict[int, List[Tuple[int, ...]]],
    origin_keys: List[Tuple[int, ...]],
    group_indices: List[int],
    n_children: int,
    changed_state_bonus: float,
) -> Dict[int, _BranchState]:
    """Select one child per run while covering distinct first edits.

    The production R9K1M2 layout uses nine independent ``n_branches=1``
    runs for each augmented product.  Per-run Top-1 selection would discard
    the alternatives before they can be compared, so this small group-level
    allocator first prefers signatures supported by fewer runs, then fills
    the remaining run slots by the existing mass ranking.
    """
    slot_candidates: Dict[int, List[_BranchState]] = {}
    for sample_index in group_indices:
        candidates = _merge_state_candidates_with_first_edit_diversity(
            list(zip(
                new_branches[sample_index], new_keys[sample_index],
            )),
            n_branches=n_children,
            origin_key=origin_keys[sample_index],
            changed_state_bonus=changed_state_bonus,
        )
        if not candidates:
            raise RuntimeError(
                "first-edit diversity produced no candidate for run "
                f"{sample_index}"
            )
        slot_candidates[sample_index] = candidates

    def rank(sample_index: int, branch: _BranchState):
        key = branch.state_key or ()
        return (
            branch.log_mass
            + changed_state_bonus * float(
                key != origin_keys[sample_index]
            ),
            branch.log_mass,
            -float(branch.seed),
        )

    signature_support: Dict[FirstEditSignature, set[int]] = {}
    for sample_index, candidates in slot_candidates.items():
        for branch in candidates:
            signature_support.setdefault(
                branch.first_edit_signature, set(),
            ).add(sample_index)

    selected: Dict[int, _BranchState] = {}
    selected_signatures: set[FirstEditSignature] = set()
    while len(selected) < len(group_indices):
        available = [
            (sample_index, branch)
            for sample_index in group_indices
            if sample_index not in selected
            for branch in slot_candidates[sample_index]
            if branch.first_edit_signature not in selected_signatures
        ]
        if not available:
            break
        sample_index, branch = max(
            available,
            key=lambda item: (
                -len(signature_support[item[1].first_edit_signature]),
                *rank(*item),
            ),
        )
        selected[sample_index] = branch
        selected_signatures.add(branch.first_edit_signature)

    # There may be fewer unique signatures than runs.  Fill those slots by
    # the old local ranking; this does not create extra runs or extra forwards.
    for sample_index in group_indices:
        if sample_index in selected:
            continue
        selected[sample_index] = max(
            slot_candidates[sample_index],
            key=lambda branch: rank(sample_index, branch),
        )
    return selected


def _token_keys_batch(
    x_t: Tensor, pad_token: int, bos_token: int,
) -> List[Tuple[int, ...]]:
    """整批传回 CPU 后构造状态 key，避免逐行 Tensor 标量转换。"""
    rows = x_t.detach().cpu().tolist()
    excluded = (pad_token, bos_token)
    return [
        tuple(token for token in row if token not in excluded)
        for row in rows
    ]


def _step_log_p_batch(
    actions: dict,
    log_rates_eff: Tensor,
    log_ins_probs: Tensor,
    log_sub_probs: Tensor,
    adapt_h: Tensor,
    score_mode: str = "full_probability",
    state_tokens: Optional[Tensor] = None,
    pad_token: int = PAD_TOKEN,
) -> Tensor:
    """批量计算每条分支本步完整动作集合的 log-prob。"""
    rates = torch.exp(log_rates_eff)
    if state_tokens is not None:
        if state_tokens.shape != rates.shape[:2]:
            raise ValueError(
                "state_tokens must have shape [batch, length] matching rates"
            )
        insert_positions, sub_del_positions = edit_position_masks(
            state_tokens, pad_token=pad_token,
        )
    else:
        insert_positions = actions.get("insert_position_mask")
        sub_del_positions = actions.get("sub_del_position_mask")
        if insert_positions is None:
            insert_positions = torch.ones_like(rates[:, :, 0], dtype=torch.bool)
        if sub_del_positions is None:
            sub_del_positions = torch.ones_like(
                rates[:, :, 1], dtype=torch.bool,
            )

    # The stochastic samplers draw INS/SUB tokens from Q conditioned on the
    # valid action support.  Their returned log normalizers make beam scoring
    # use the same conditional Q instead of silently scoring a different
    # distribution.  Hand-built test actions retain legacy/raw-Q behavior.
    ins_log_normalizer = actions.get("ins_token_log_normalizer")
    sub_log_normalizer = actions.get("sub_token_log_normalizer")
    ins_rates = rates[:, :, 0] * insert_positions.to(rates.dtype)
    sub_rates = rates[:, :, 1] * sub_del_positions.to(rates.dtype)
    del_rates = rates[:, :, 2] * sub_del_positions.to(rates.dtype)
    eps = 1e-12
    log_eps = math.log(eps)
    ins_mu = adapt_h * ins_rates
    if score_mode == "legacy_triggered_reverse":
        event_log_p = torch.log(
            (-torch.expm1(
                -adapt_h.unsqueeze(-1)
                * torch.stack((ins_rates, sub_rates, del_rates), dim=-1)
            )).clamp_min(eps)
        )
        ins_token_log_p = log_ins_probs.gather(
            2, actions["ins_tokens"].unsqueeze(-1),
        ).squeeze(-1)
        sub_token_log_p = log_sub_probs.gather(
            2, actions["sub_tokens"].unsqueeze(-1),
        ).squeeze(-1)
        if ins_log_normalizer is not None:
            ins_token_log_p = ins_token_log_p - ins_log_normalizer
        if sub_log_normalizer is not None:
            sub_token_log_p = sub_token_log_p - sub_log_normalizer
        ins_token_log_p = ins_token_log_p.clamp_min(log_eps)
        sub_token_log_p = sub_token_log_p.clamp_min(log_eps)
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
    ds_rates = sub_rates + del_rates
    ds_mu = adapt_h * ds_rates
    ins_event_log_p = torch.log((-torch.expm1(-ins_mu)).clamp_min(eps))
    ds_event_log_p = torch.log((-torch.expm1(-ds_mu)).clamp_min(eps))
    ins_token_log_p = log_ins_probs.gather(
        2, actions["ins_tokens"].unsqueeze(-1),
    ).squeeze(-1)
    sub_token_log_p = log_sub_probs.gather(
        2, actions["sub_tokens"].unsqueeze(-1),
    ).squeeze(-1)
    if ins_log_normalizer is not None:
        ins_token_log_p = ins_token_log_p - ins_log_normalizer
    if sub_log_normalizer is not None:
        sub_token_log_p = sub_token_log_p - sub_log_normalizer
    ins_token_log_p = ins_token_log_p.clamp_min(log_eps)
    sub_token_log_p = sub_token_log_p.clamp_min(log_eps)
    ins_contrib = torch.where(
        actions["ins_mask"], ins_event_log_p + ins_token_log_p, -ins_mu,
    )
    sub_contrib = (
        ds_event_log_p
        + torch.log(
            (sub_rates / ds_rates.clamp_min(eps)).clamp_min(eps)
        )
        + sub_token_log_p
    )
    del_contrib = (
        ds_event_log_p
        + torch.log(
            (del_rates / ds_rates.clamp_min(eps)).clamp_min(eps)
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
    state_tokens: Optional[Tensor] = None,
    pad_token: int = PAD_TOKEN,
) -> float:
    """单分支兼容包装；主采样循环使用 `_step_log_p_batch()`。"""
    h = torch.tensor(
        [[adapt_h]], dtype=log_rates_eff.dtype, device=log_rates_eff.device,
    )
    return _step_log_p_batch(
        actions, log_rates_eff, log_ins_probs, log_sub_probs, h,
        state_tokens=state_tokens, pad_token=pad_token,
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


def _apply_q_temperature(log_probs: Tensor, temperature: float) -> Tensor:
    """Sharpen or flatten token probabilities while preserving log-normalization."""
    if temperature == 1.0:
        return log_probs
    return torch.log_softmax(log_probs / temperature, dim=-1)


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
    position_scores: Optional[Tensor] = None,
    position_bias_active: Optional[Tensor] = None,
    position_bias_max_multiplier: float = 3.0,
) -> dict:
    """按 branch seed 无状态、批量地采样所有分支动作。"""
    seeds = branch_seeds.to(device=x_t.device, dtype=torch.int64)
    legal_log_ins_probs, ins_log_normalizer = legal_token_log_probs(
        log_ins_probs,
    )
    legal_log_sub_probs, sub_log_normalizer = legal_token_log_probs(
        log_sub_probs,
        current_tokens=x_t,
    )
    insert_positions, sub_del_positions = edit_position_masks(
        x_t, pad_token=pad_token,
    )
    # Apply the same legal support to event rates before the stateless random
    # draws.  This prevents forbidden BOS SUB/DEL (and Q-empty INS/SUB) from
    # becoming silent no-ops and keeps proposal sampling aligned with scoring.
    ins_sample_positions = insert_positions & torch.isfinite(ins_log_normalizer)
    sub_sample_positions = sub_del_positions & torch.isfinite(sub_log_normalizer)
    position_bias_diagnostics = None
    if position_scores is not None:
        if position_bias_active is None:
            raise ValueError(
                "position_bias_active is required with position_scores"
            )
        legal_position_masks = torch.stack(
            (ins_sample_positions, sub_sample_positions, sub_del_positions),
            dim=-1,
        )
        log_rates, position_bias_diagnostics = (
            renormalize_position_biased_log_rates(
                log_rates,
                position_scores,
                legal_position_masks,
                position_bias_active,
                max_multiplier=position_bias_max_multiplier,
            )
        )
    rates = torch.exp(log_rates)
    lambda_ins = rates[:, :, 0] * ins_sample_positions.to(rates.dtype)
    lambda_sub = rates[:, :, 1] * sub_sample_positions.to(rates.dtype)
    lambda_del = rates[:, :, 2] * sub_del_positions.to(rates.dtype)

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
        legal_log_ins_probs,
        _stateless_uniform(
            seeds, step, seq_len, stream=3, dtype=log_ins_probs.dtype,
        ),
    )
    sub_tokens = _sample_tokens_from_uniform(
        legal_log_sub_probs,
        _stateless_uniform(
            seeds, step, seq_len, stream=4, dtype=log_sub_probs.dtype,
        ),
    )

    ins_mask &= ins_sample_positions
    del_mask &= sub_del_positions
    sub_mask &= sub_sample_positions
    ins_tokens = ins_tokens.masked_fill(~ins_mask, pad_token)
    sub_tokens = sub_tokens.masked_fill(~sub_mask, pad_token)
    return {
        "ins_mask": ins_mask,
        "del_mask": del_mask,
        "sub_mask": sub_mask,
        "ins_tokens": ins_tokens,
        "sub_tokens": sub_tokens,
        "ins_token_log_normalizer": ins_log_normalizer,
        "sub_token_log_normalizer": sub_log_normalizer,
        "insert_position_mask": insert_positions,
        "sub_del_position_mask": sub_del_positions,
        # Beam ranking must score the same (possibly center-reweighted)
        # proposal distribution from which the child was sampled.
        "effective_log_rates": log_rates,
        "position_bias_diagnostics": position_bias_diagnostics,
    }


def _first_event_actions_from_row(
    actions: dict,
    row_index: int,
    position_scores: Tensor,
) -> list[dict]:
    """Serialize one selected lineage's first sampled action set.

    This is intentionally invoked only for diagnostic runs.  The normal
    R9K1M2 full-test path keeps a compact aggregate instead of materializing
    one large JSON record per sampled trajectory.
    """
    action_rows = []
    for mode_name, mode_index, mask_name in (
        ("INS", 0, "ins_mask"),
        ("SUB", 1, "sub_mask"),
        ("DEL", 2, "del_mask"),
    ):
        positions = torch.nonzero(
            actions[mask_name][row_index], as_tuple=False,
        ).squeeze(-1).tolist()
        for position in positions:
            token_name = (
                "ins_tokens" if mode_name == "INS" else "sub_tokens"
            )
            action_rows.append(
                {
                    "mode": mode_name,
                    "position": int(position),
                    "token_id": (
                        int(actions[token_name][row_index, position].item())
                        if mode_name in {"INS", "SUB"}
                        else None
                    ),
                    "center_score": float(
                        position_scores[row_index, position, mode_index].item()
                    ),
                }
            )
    return action_rows


def _set_second_child_noop(actions: dict, n_children: int) -> None:
    """原地将每个 parent 的 child 1 设为 no-op；child 0 保持随机动作。"""
    noop_rows = torch.arange(
        1, actions["ins_mask"].shape[0], n_children,
        device=actions["ins_mask"].device,
    )
    actions["ins_mask"][noop_rows] = False
    actions["sub_mask"][noop_rows] = False
    actions["del_mask"][noop_rows] = False


def _profile_start(profile: Optional[Dict[str, object]], device) -> float:
    if profile is None:
        return 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _profile_finish(
    profile: Optional[Dict[str, object]],
    key: str,
    started_at: float,
    device,
) -> None:
    if profile is None:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    profile[key] = (
        float(profile.get(key, 0.0)) + time.perf_counter() - started_at
    )


def _record_padding_profile(
    profile: Optional[Dict[str, object]],
    x_batch: Tensor,
    pad_token: int,
) -> None:
    """Record length/padding diagnostics only for explicit profile runs."""
    if profile is None:
        return
    lengths = (x_batch != pad_token).sum(dim=1).cpu().tolist()
    padded_length = x_batch.shape[1]
    row_count = len(lengths)
    profile["actual_token_count"] = (
        int(profile.get("actual_token_count", 0)) + sum(lengths)
    )
    profile["padded_token_count"] = (
        int(profile.get("padded_token_count", 0))
        + row_count * padded_length
    )
    profile["actual_attention_cost_proxy"] = (
        int(profile.get("actual_attention_cost_proxy", 0))
        + sum(length * length for length in lengths)
    )
    profile["padded_attention_cost_proxy"] = (
        int(profile.get("padded_attention_cost_proxy", 0))
        + row_count * padded_length * padded_length
    )
    profile["max_active_length"] = max(
        int(profile.get("max_active_length", 0)),
        max(lengths),
    )
    histogram = profile.setdefault("active_length_histogram", {})
    for length in lengths:
        histogram[str(length)] = histogram.get(str(length), 0) + 1


def _record_protected_parent_profile(
    profile: Optional[Dict[str, object]],
    flat: List[Tuple[int, int, _BranchState]],
    x_batch: Tensor,
    pad_token: int,
    bos_token: int,
    sample_group_size: int,
) -> None:
    """Measure exact-state forward sharing without merging search lineages."""
    if profile is None:
        return
    keys = _token_keys_batch(x_batch, pad_token, bos_token)
    unique_by_group: Dict[
        int, set[Tuple[float, Tuple[int, ...]]]
    ] = {}
    for (sample_index, _, branch), key in zip(flat, keys):
        group_index = sample_index // sample_group_size
        unique_by_group.setdefault(group_index, set()).add((branch.t, key))

    logical_rows = len(flat)
    unique_rows = sum(len(states) for states in unique_by_group.values())
    profile["protected_sample_group_size"] = sample_group_size
    profile["protected_parent_rows"] = (
        int(profile.get("protected_parent_rows", 0)) + logical_rows
    )
    profile["protected_unique_parent_rows"] = (
        int(profile.get("protected_unique_parent_rows", 0)) + unique_rows
    )
    profile["potential_shared_parent_rows"] = (
        int(profile.get("potential_shared_parent_rows", 0))
        + logical_rows - unique_rows
    )
    profile.setdefault("protected_parent_rows_by_step", []).append(
        logical_rows
    )
    profile.setdefault("protected_unique_parent_rows_by_step", []).append(
        unique_rows
    )


def _shared_forward_row_map(
    flat: List[Tuple[int, int, _BranchState]],
    sample_group_size: int,
) -> Tuple[List[int], List[int]]:
    """Map logical lineages to one deterministic forward per exact state."""
    unique_rows: List[int] = []
    inverse_rows: List[int] = []
    seen: Dict[Tuple[int, float, Tuple[int, ...]], int] = {}
    for row, (sample_index, _, branch) in enumerate(flat):
        if branch.state_key is None:
            raise RuntimeError(
                "share_identical_forwards requires tracked branch state keys"
            )
        signature = (
            sample_index // sample_group_size,
            branch.t,
            branch.state_key,
        )
        unique_position = seen.get(signature)
        if unique_position is None:
            unique_position = len(unique_rows)
            seen[signature] = unique_position
            unique_rows.append(row)
        inverse_rows.append(unique_position)
    return unique_rows, inverse_rows


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


@torch.inference_mode()
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
    changed_state_bonus: float = 0.0,
    child_policy: str = "stochastic",
    record_trajectory: bool = False, # 未实现, 预留
    record_all_events: bool = False, # 未实现, 预留
    x_1: Optional[Tensor] = None,    # 未使用, 预留 (与 sample_euler 接口对齐)
    vocab_size: Optional[int] = None, # 未使用, 预留
    profile: Optional[Dict[str, object]] = None,
    profile_sample_group_size: int = 1,
    share_identical_forwards: bool = False,
    q_temperature: float = 1.0,
    first_edit_diversity: bool = False,
    initial_branch_seeds: Optional[List[List[int]]] = None,
    sampling_stats: Optional[Dict[str, int]] = None,
    first_event_position_scores: Optional[Tensor] = None,
    first_event_position_bias_enabled: Optional[Tensor] = None,
    first_event_bias_max_multiplier: float = 3.0,
    first_event_bias_stats: Optional[dict] = None,
    first_event_row_metadata: Optional[List[dict]] = None,
    first_event_bias_record_events: bool = False,
    first_event_record_sink: Optional[Callable[[dict], None]] = None,
    product_memory: Optional[Tensor] = None,
    product_memory_padding_mask: Optional[Tensor] = None,
) -> Tensor:
    """Euler 采样 + 分支维护。

    Args:
        model: EditFlowsTransformer 模型。
        x_0: (B, L_0) 初始序列。
        n_branches: 每条样本最多维护的并行分支数。
        n_children: 每个父分支生成的独立随机后继数。
        n_steps: Euler 步数 (与 sample_euler 一致)。
        base_seed: 基础随机种子。
        initial_branch_seeds: 可选的每样本、每初始分支 seed 布局。
        changed_state_bonus: 给非原始 token 状态的固定搜索先验；0 表示禁用。
        child_policy: `stochastic` 或 M=2 的启发式 `stochastic_noop`。
        profile_sample_group_size: 仅用于显式profile；相邻多少个sample
            属于同一product的受保护lineage组。启用首步多样性且
            n_branches=1时，该组也是跨独立run分配首步的范围。
        share_identical_forwards: 对同一product内相同时间、相同token状态
            只执行一次确定性模型前向，再映射回独立seed lineage。
        q_temperature: 对insert/substitute token posterior应用的采样温度；
            1.0保持checkpoint原始分布。
        first_edit_diversity: 在多后继模式下，首次真实状态变化时优先保留
            不同的编辑 signature；不增加模型前向或最终输出槽位。
        first_event_position_scores: 固定初始产物位置的 [B,L,3] 中心分数。
            仅在每条 lineage 的第一个真实编辑前重加权 INS/SUB/DEL 的
            位置速率；每个 mode 的合法总 hazard 保持不变。
        first_event_position_bias_enabled: [B]；允许同一批中一部分 run
            接受中心偏向，其他 run 仍按原 R9K1M2 采样。
        first_event_bias_record_events: 仅诊断用。True 时保留每条最终
            lineage 的首事件详情；False 时只保留轻量计数，避免全量 test
            写出大体积 JSON。
        first_event_record_sink: 可选的逐条首事件消费函数。仅在
            ``first_event_bias_record_events=True`` 时有效；提供后，选中
            lineage 的首事件会立即交给该函数而不累积到
            ``first_event_bias_stats['records']``。这使全量诊断可以流式
            汇总，而不在内存中保留近百万个 Python 字典。
        product_memory: 可选的、按初始输入行排列的 immutable product
            memory。每个分支会按其所属初始输入行读取同一份 memory；
            不提供时，product-memory 模型会在本函数入口对 ``x_0``
            每行编码一次。
        product_memory_padding_mask: 与 ``product_memory`` 对应的 padding
            mask。仅在提供 ``product_memory`` 时必填。

    Returns:
        x_final: (B * n_branches, L_out) 每条样本的全部排名分支，按样本优先
            排列并 PAD 到等长。
    """
    if n_branches < 1:
        raise ValueError(f"n_branches must be >= 1, got {n_branches}")
    if n_children < 1:
        raise ValueError(f"n_children must be >= 1, got {n_children}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if not math.isfinite(q_temperature) or q_temperature <= 0:
        raise ValueError(
            f"q_temperature must be finite and > 0, got {q_temperature}"
        )
    if x_0.shape[0] < 1:
        raise ValueError("x_0 batch must contain at least one sample")
    if first_event_record_sink is not None and not first_event_bias_record_events:
        raise ValueError(
            "first_event_record_sink requires "
            "first_event_bias_record_events=True"
        )
    if first_event_record_sink is not None and first_event_bias_stats is None:
        raise ValueError(
            "first_event_record_sink requires first_event_bias_stats"
        )
    if profile_sample_group_size < 1:
        raise ValueError(
            "profile_sample_group_size must be >= 1, got "
            f"{profile_sample_group_size}"
        )
    if x_0.shape[0] % profile_sample_group_size != 0:
        raise ValueError(
            "x_0 batch size must be divisible by profile_sample_group_size, "
            f"got {x_0.shape[0]} and {profile_sample_group_size}"
        )
    if use_origin_mask:
        raise NotImplementedError(
            "Euler-Beam does not yet track origin_mask across edits"
        )
    if sample_seeds is not None and len(sample_seeds) != x_0.shape[0]:
        raise ValueError(
            f"sample_seeds length must equal batch size {x_0.shape[0]}, "
            f"got {len(sample_seeds)}"
        )
    if initial_branch_seeds is not None:
        if sample_seeds is not None:
            raise ValueError(
                "initial_branch_seeds and sample_seeds are mutually exclusive"
            )
        if len(initial_branch_seeds) != x_0.shape[0]:
            raise ValueError(
                "initial_branch_seeds length must equal batch size "
                f"{x_0.shape[0]}, got {len(initial_branch_seeds)}"
            )
        invalid_lengths = [
            len(seeds) for seeds in initial_branch_seeds
            if len(seeds) != n_branches
        ]
        if invalid_lengths:
            raise ValueError(
                "each initial_branch_seeds row must contain n_branches "
                f"seeds ({n_branches}), got {invalid_lengths[0]}"
            )
    if score_mode not in ("full_probability", "legacy_triggered_reverse"):
        raise ValueError(f"Unsupported score_mode: {score_mode}")
    if changed_state_bonus < 0:
        raise ValueError(
            f"changed_state_bonus must be >= 0, got {changed_state_bonus}"
        )
    if child_policy not in ("stochastic", "stochastic_noop"):
        raise ValueError(f"Unsupported child_policy: {child_policy}")
    if child_policy == "stochastic_noop" and n_children != 2:
        raise ValueError(
            "stochastic_noop requires n_children=2"
        )
    if first_edit_diversity and n_children < 2:
        raise ValueError(
            "first_edit_diversity requires n_children >= 2"
        )
    if first_edit_diversity and n_branches == 1 and \
       profile_sample_group_size < 2:
        raise ValueError(
            "first_edit_diversity with n_branches=1 requires "
            "profile_sample_group_size >= 2"
        )
    if first_edit_diversity and score_mode != "full_probability":
        raise ValueError(
            "first_edit_diversity requires score_mode='full_probability'"
        )

    if score_mode == "legacy_triggered_reverse":
        sort_key = _legacy_branch_sort_key
    elif n_children == 1:
        sort_key = _trajectory_branch_sort_key
    else:
        sort_key = _branch_sort_key

    device = next(model.parameters()).device
    x_0 = x_0.to(device)
    B = x_0.shape[0]
    # x_t changes as branches evolve, whereas the extra product context must
    # remain tied to the original x_0.  Prepare it before creating branches,
    # then use each branch's stable sample index to gather the right cache row
    # at every forward.  This is also compatible with shared state forwards:
    # the cache gathers use the same unique-row map as x_t and t.
    product_memory, product_memory_padding_mask = (
        _prepare_product_memory_for_sampling(
            model,
            x_0,
            pad_token=pad_token,
            product_memory=product_memory,
            product_memory_padding_mask=product_memory_padding_mask,
        )
    )
    if product_memory is not None:
        if product_memory.shape[0] != B:
            raise ValueError(
                "product_memory batch size must equal x_0 batch size, got "
                f"{product_memory.shape[0]} and {B}"
            )
        if product_memory_padding_mask is None:
            raise RuntimeError("validated product memory is missing its mask")
        if product_memory_padding_mask.shape != product_memory.shape[:2]:
            raise ValueError(
                "product_memory_padding_mask must match product_memory "
                "batch and length"
            )
    default_h = 1.0 / n_steps
    origin_keys = _token_keys_batch(x_0, pad_token, bos_token)
    noop_step = min(n_steps - 1, int(0.9 * n_steps))

    if first_event_position_scores is not None:
        if (
            first_event_position_scores.ndim != 3
            or first_event_position_scores.shape[0] != B
            or first_event_position_scores.shape[1] != x_0.shape[1]
            or first_event_position_scores.shape[2] != 3
        ):
            raise ValueError(
                "first_event_position_scores must have shape "
                "[batch, initial_length, 3]"
            )
        first_event_position_scores = first_event_position_scores.to(
            device=device, dtype=torch.float32,
        )
        if not torch.isfinite(first_event_position_scores).all():
            raise ValueError("first_event_position_scores must be finite")
        if (first_event_position_scores < 0).any() or (
            first_event_position_scores > 1
        ).any():
            raise ValueError("first_event_position_scores must lie in [0, 1]")
        if first_event_position_bias_enabled is None:
            first_event_position_bias_enabled = torch.ones(
                B, dtype=torch.bool, device=device,
            )
        else:
            if first_event_position_bias_enabled.shape != (B,):
                raise ValueError(
                    "first_event_position_bias_enabled must have shape "
                    "[batch]"
                )
            first_event_position_bias_enabled = (
                first_event_position_bias_enabled.to(
                    device=device, dtype=torch.bool,
                )
            )
        if first_event_row_metadata is not None and len(
            first_event_row_metadata
        ) != B:
            raise ValueError(
                "first_event_row_metadata must have one item per batch row"
            )
        if (
            first_event_bias_max_multiplier < 1
            or not torch.isfinite(
                torch.tensor(first_event_bias_max_multiplier)
            )
        ):
            raise ValueError(
                "first_event_bias_max_multiplier must be finite and >= 1"
            )
        if (
            first_event_bias_stats is not None
            and first_event_position_scores is not None
        ):
            first_event_bias_stats.setdefault("biased_row_steps", 0)
            first_event_bias_stats.setdefault("guided_row_steps", 0)
            first_event_bias_stats.setdefault("first_event_count", 0)
            first_event_bias_stats.setdefault("no_event_count", 0)
            first_event_bias_stats.setdefault(
                "max_hazard_relative_error", 0.0,
            )
            first_event_bias_stats.setdefault("records", [])
            first_event_bias_stats.setdefault(
                "record_event_details", first_event_bias_record_events,
            )
    else:
        if first_event_position_bias_enabled is not None:
            raise ValueError(
                "first_event_position_bias_enabled requires "
                "first_event_position_scores"
            )
        if first_event_row_metadata is not None:
            raise ValueError(
                "first_event_row_metadata requires "
                "first_event_position_scores"
            )
        if first_event_bias_record_events:
            raise ValueError(
                "first_event_bias_record_events requires "
                "first_event_position_scores"
            )

    # ── 初始化: 每条样本创建 n_branches 条分支 ──
    all_branches: List[List[_BranchState]] = []
    for b in range(B):
        if initial_branch_seeds is None:
            sample_seed = (
                sample_seeds[b]
                if sample_seeds is not None
                else base_seed + b * n_branches
            )
        branches = []
        for k in range(n_branches):
            branch_seed = (
                initial_branch_seeds[b][k]
                if initial_branch_seeds is not None
                else sample_seed + k
            )
            branches.append(_BranchState(
                x_t=x_0[b:b + 1].to(device),
                log_mass=(-math.log(n_branches) if n_children > 1 else 0.0),
                t=0.0,
                seed=branch_seed,
                state_key=origin_keys[b],
                # Unlike the bias itself, this flag remains active for
                # unguided rows too so mixed designs can still count their
                # first real action fairly.
                first_event_pending=(
                    first_event_position_scores is not None
                ),
            ))
        all_branches.append(branches)

    # ── 主循环: n_steps 步 Euler 推进 ──
    for step in range(n_steps):
        section_started = _profile_start(profile, device)
        # 1. 收集所有未完成的分支 (t < 1.0)
        flat: List[Tuple[int, int, _BranchState]] = []
        for b in range(B):
            for k, br in enumerate(all_branches[b]):
                if br.t < 1.0:
                    flat.append((b, k, br))

        if not flat:
            break

        N = len(flat)
        branch_tensors = [s.x_t for _, _, s in flat]
        branch_widths = [tensor.shape[1] for tensor in branch_tensors]
        max_L = max(branch_widths)

        # 2. 构建批量 tensor (单次 GPU 传输)
        # Normal Euler-Beam states originate from one batched edit result and
        # therefore share a width.  Concatenating them avoids N small GPU slice
        # assignments.  Keep the padded fallback for external/irregular states.
        if all(width == max_L for width in branch_widths):
            x_batch = torch.cat(branch_tensors, dim=0)
            if profile is not None:
                profile["uniform_width_fast_path_steps"] = (
                    int(profile.get("uniform_width_fast_path_steps", 0)) + 1
                )
        else:
            x_batch = torch.full(
                (N, max_L), pad_token, dtype=torch.long, device=device,
            )
            for i, tensor in enumerate(branch_tensors):
                x_batch[i, :tensor.shape[1]] = tensor
        t_vals = torch.tensor(
            [s.t for _, _, s in flat], dtype=torch.float, device=device,
        ).unsqueeze(-1)
        _profile_finish(
            profile, "prepare_branches_seconds", section_started, device,
        )
        _record_padding_profile(profile, x_batch, pad_token)
        _record_protected_parent_profile(
            profile,
            flat,
            x_batch,
            pad_token,
            bos_token,
            profile_sample_group_size,
        )

        # 3. 单次模型前向。可选地只计算product内完全相同的parent状态一次，
        # 然后映射回逻辑lineage；随机动作仍按各自seed独立生成。
        section_started = _profile_start(profile, device)
        parent_sample_indices = torch.tensor(
            [b for b, _, _ in flat], dtype=torch.long, device=device,
        )
        parent_product_memory = None
        parent_product_memory_padding_mask = None
        if product_memory is not None:
            parent_product_memory = product_memory.index_select(
                0, parent_sample_indices,
            )
            parent_product_memory_padding_mask = (
                product_memory_padding_mask.index_select(
                    0, parent_sample_indices,
                )
            )

        inverse_forward_indices = None
        if share_identical_forwards:
            unique_rows, inverse_rows = _shared_forward_row_map(
                flat, profile_sample_group_size,
            )
            unique_indices = torch.tensor(
                unique_rows, dtype=torch.long, device=device,
            )
            inverse_forward_indices = torch.tensor(
                inverse_rows, dtype=torch.long, device=device,
            )
            x_model = x_batch.index_select(0, unique_indices)
            t_model_input = t_vals.index_select(0, unique_indices)
            if parent_product_memory is not None:
                product_memory_model = parent_product_memory.index_select(
                    0, unique_indices,
                )
                product_memory_padding_mask_model = (
                    parent_product_memory_padding_mask.index_select(
                        0, unique_indices,
                    )
                )
            else:
                product_memory_model = None
                product_memory_padding_mask_model = None
        else:
            x_model = x_batch
            t_model_input = t_vals
            product_memory_model = parent_product_memory
            product_memory_padding_mask_model = (
                parent_product_memory_padding_mask
            )
        physical_forward_rows = x_model.shape[0]
        for stats in (profile, sampling_stats):
            if stats is not None:
                stats["model_forward_parent_rows"] = (
                    int(stats.get("model_forward_parent_rows", 0))
                    + physical_forward_rows
                )
                stats["shared_model_parent_rows"] = (
                    int(stats.get("shared_model_parent_rows", 0))
                    + N - physical_forward_rows
                )

        x_pad_mask = x_model == pad_token
        t_model = _compute_model_time(
            t_model_input, scheduler, time_input, train_scheduler,
        )
        log_rates, log_ins_probs, log_sub_probs = _forward_edit_model(
            model,
            x_model,
            t_model,
            x_pad_mask,
            product_memory=product_memory_model,
            product_memory_padding_mask=product_memory_padding_mask_model,
        )

        # 4. 速率修正 (与 sample_euler 完全一致)
        if not use_rate_reparam and train_scheduler is not None and \
           scheduler.name != train_scheduler.name:
            k_sample = get_rate_scale(
                t_model_input, scheduler,
                clamp_kappa=clamp_kappa, clamp_max=clamp_max,
            )
            k_train = get_rate_scale(
                t_model, train_scheduler,
                clamp_kappa=clamp_kappa, clamp_max=clamp_max,
            )
            log_correction = torch.log(
                k_sample / k_train.clamp_min(1e-2)
            ).unsqueeze(1)
            log_rates = log_rates + log_correction

        log_rates_eff = apply_rate_parameterization(
            log_rates, t_model_input, scheduler,
            use_rate_reparam=use_rate_reparam,
            clamp_kappa=clamp_kappa, clamp_max=clamp_max,
        )
        if inverse_forward_indices is not None:
            log_rates_eff = log_rates_eff.index_select(
                0, inverse_forward_indices,
            )
            log_ins_probs = log_ins_probs.index_select(
                0, inverse_forward_indices,
            )
            log_sub_probs = log_sub_probs.index_select(
                0, inverse_forward_indices,
            )
        log_ins_probs = _apply_q_temperature(
            log_ins_probs, q_temperature,
        )
        log_sub_probs = _apply_q_temperature(
            log_sub_probs, q_temperature,
        )
        if profile is not None:
            profile["model_forward_calls"] = (
                int(profile.get("model_forward_calls", 0))
                + 1
            )
        _profile_finish(
            profile, "model_forward_and_rates_seconds",
            section_started, device,
        )

        # 5. 直接复用模型父 batch。模型已经把 PAD 位置屏蔽为 -1e9，
        # 无需再分配并逐行复制 x/rates/token-probs 的同形张量。
        N_br = len(flat)
        x_br = x_batch
        lr_br = log_rates_eff
        lip_br = log_ins_probs
        lsp_br = log_sub_probs
        branch_t_vals = t_vals

        # 批量计算父分支步长。模型前向规模始终是 K；仅模型输出和动作
        # 张量扩展为 K×M，避免重复 Transformer forward。
        section_started = _profile_start(profile, device)
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
        candidate_position_scores = None
        candidate_position_bias_active = None
        parent_first_event_pending = None
        parent_center_bias_enabled = None
        if first_event_position_scores is not None:
            parent_position_scores = first_event_position_scores.index_select(
                0, parent_sample_indices,
            )
            candidate_position_scores = align_position_scores(
                parent_position_scores.index_select(0, parent_indices),
                x_candidates.shape[1],
            )
            parent_first_event_pending = torch.tensor(
                [branch.first_event_pending for _, _, branch in flat],
                dtype=torch.bool,
                device=device,
            )
            parent_center_bias_enabled = (
                first_event_position_bias_enabled.index_select(
                    0, parent_sample_indices,
                )
            )
            candidate_position_bias_active = (
                parent_first_event_pending
                & parent_center_bias_enabled
            ).index_select(0, parent_indices)
        actions_batch = _sample_actions_per_branch(
            child_seeds,
            x_candidates,
            lr_candidates,
            lip_candidates,
            lsp_candidates,
            adapt_h_candidates,
            pad_token=pad_token, event_prob_mode=event_prob_mode,
            step=step,
            position_scores=candidate_position_scores,
            position_bias_active=candidate_position_bias_active,
            position_bias_max_multiplier=first_event_bias_max_multiplier,
        )
        if child_policy == "stochastic_noop" and step == noop_step:
            _set_second_child_noop(actions_batch, n_children)
        effective_lr_candidates = actions_batch.get(
            "effective_log_rates", lr_candidates,
        )
        if (
            first_event_bias_stats is not None
            and first_event_position_scores is not None
        ):
            # Count logical selected parents, not their M children.  This is
            # directly comparable with ordinary Euler's per-trajectory count.
            first_event_bias_stats["biased_row_steps"] += int(
                parent_first_event_pending.sum().item()
            )
            first_event_bias_stats["guided_row_steps"] += int(
                (
                    parent_first_event_pending
                    & parent_center_bias_enabled
                ).sum().item()
            )
            bias_diagnostics = actions_batch.get(
                "position_bias_diagnostics"
            )
            if bias_diagnostics is not None:
                changed_errors = bias_diagnostics["relative_error"][
                    bias_diagnostics["changed"]
                ]
                if changed_errors.numel():
                    first_event_bias_stats["max_hazard_relative_error"] = max(
                        first_event_bias_stats[
                            "max_hazard_relative_error"
                        ],
                        float(changed_errors.max().item()),
                    )
        _profile_finish(
            profile, "child_proposal_seconds", section_started, device,
        )

        # 6. 批量计分、应用编辑，并各自一次性传回 CPU 元数据
        section_started = _profile_start(profile, device)
        step_log_ps = _step_log_p_batch(
            actions_batch,
            effective_lr_candidates,
            lip_candidates,
            lsp_candidates,
            adapt_h_candidates,
            score_mode=score_mode,
            state_tokens=x_candidates,
            pad_token=pad_token,
        ).cpu().tolist()
        adapt_h_values = adapt_h_candidates.squeeze(-1).cpu().tolist()
        _profile_finish(
            profile, "step_scoring_seconds", section_started, device,
        )

        section_started = _profile_start(profile, device)
        x_next_batch = _apply_edits_batch(
            x_candidates, actions_batch, max_seq_len, pad_token,
        )
        next_keys = _token_keys_batch(x_next_batch, pad_token, bos_token)
        first_edit_signatures: List[FirstEditSignature] = [None] * len(
            parent_index_values
        )
        signature_row_indices: List[int] = []
        if first_edit_diversity:
            for i, parent_i in enumerate(parent_index_values):
                _, _, parent = flat[parent_i]
                if parent.first_edit_signature is None and \
                   next_keys[i] != parent.state_key:
                    signature_row_indices.append(i)
            computed_signatures = _first_edit_signatures_from_actions(
                actions_batch, signature_row_indices,
            )
            for i, signature in zip(
                signature_row_indices, computed_signatures,
            ):
                first_edit_signatures[i] = signature
            for stats in (profile, sampling_stats):
                if stats is not None:
                    stats["first_edit_signature_candidates"] = (
                        int(stats.get(
                            "first_edit_signature_candidates", 0,
                        )) + len(signature_row_indices)
                    )
                    stats["first_edit_signature_assigned"] = (
                        int(stats.get(
                            "first_edit_signature_assigned", 0,
                        )) + sum(
                            signature is not None
                            for signature in computed_signatures
                        )
                    )
        _profile_finish(
            profile, "apply_edits_seconds", section_started, device,
        )

        candidate_has_events = (
            actions_batch["ins_mask"]
            | actions_batch["sub_mask"]
            | actions_batch["del_mask"]
        ).any(dim=1).detach().cpu().tolist()
        candidate_position_reweighted = None
        bias_diagnostics = actions_batch.get("position_bias_diagnostics")
        if bias_diagnostics is not None:
            candidate_position_reweighted = (
                bias_diagnostics["changed"].any(dim=1).detach().cpu().tolist()
            )

        section_started = _profile_start(profile, device)
        new_branches: Dict[int, List[_BranchState]] = {b: [] for b in range(B)}
        new_keys: Dict[int, List[Tuple[int, ...]]] = {b: [] for b in range(B)}

        child_log_share = math.log(n_children)
        for i, parent_i in enumerate(parent_index_values):
            b, k, s = flat[parent_i]
            first_edit_signature = s.first_edit_signature
            if first_edit_signature is None:
                first_edit_signature = first_edit_signatures[i]
            first_event_record = s.first_event_record
            child_has_event = bool(candidate_has_events[i])
            first_event_reweighted = s.first_event_reweighted
            if (
                first_event_position_scores is not None
                and s.first_event_pending
                and child_has_event
                and first_event_bias_record_events
            ):
                action_rows = _first_event_actions_from_row(
                    actions_batch, i, candidate_position_scores,
                )
                first_event_record = {
                    "batch_row": int(b),
                    "first_event_step_idx": int(step),
                    "first_event_t": float(s.t),
                    "action_count": len(action_rows),
                    "actions": action_rows,
                    "position_bias_enabled": bool(
                        first_event_position_bias_enabled[b].item()
                    ),
                    "position_bias_reweighted": bool(
                        candidate_position_reweighted[i]
                        if candidate_position_reweighted is not None
                        else False
                    ),
                }
                if first_event_row_metadata is not None:
                    first_event_record["row_metadata"] = (
                        first_event_row_metadata[b]
                    )
            if (
                first_event_position_scores is not None
                and s.first_event_pending
                and child_has_event
            ):
                first_event_reweighted = bool(
                    candidate_position_reweighted[i]
                    if candidate_position_reweighted is not None
                    else False
                )
            new_branches[b].append(_BranchState(
                x_t=x_next_batch[i:i + 1],
                weight=(1.0 if n_children > 1 else s.weight),
                path_log_p=s.path_log_p + step_log_ps[i],
                log_mass=(
                    s.log_mass - child_log_share
                    if n_children > 1 else s.log_mass
                ),
                t=s.t + adapt_h_values[i],
                seed=child_seed_values[i],
                state_key=next_keys[i],
                first_edit_signature=first_edit_signature,
                first_event_pending=(
                    s.first_event_pending and not child_has_event
                ),
                first_event_record=first_event_record,
                first_event_reweighted=first_event_reweighted,
            ))
            new_keys[b].append(next_keys[i])

        # 7. 逐样本: 去重、排序、剪枝、分裂
        for b in range(B):
            if first_edit_diversity and n_branches == 1:
                # Production R9K1M2 has one branch per independent run.  Its
                # diversity decision is made across the run group below.
                continue
            candidates = new_branches[b]

            if n_children > 1 and score_mode == "full_probability":
                if profile is not None and n_branches == 1 and \
                   n_children == 2:
                    first_key, second_key = new_keys[b]
                    profile["k1_child_pairs"] = (
                        int(profile.get("k1_child_pairs", 0)) + 1
                    )
                    if first_key == second_key:
                        profile["k1_identical_child_pairs"] = (
                            int(profile.get("k1_identical_child_pairs", 0))
                            + 1
                        )
                    else:
                        changed_first = first_key != origin_keys[b]
                        changed_second = second_key != origin_keys[b]
                        if changed_first != changed_second and \
                           changed_state_bonus > 0:
                            profile["k1_bonus_decisions"] = (
                                int(profile.get("k1_bonus_decisions", 0)) + 1
                            )
                        else:
                            profile["k1_seed_tiebreak_decisions"] = (
                                int(profile.get(
                                    "k1_seed_tiebreak_decisions", 0,
                                )) + 1
                            )
                    if child_policy == "stochastic_noop" and \
                       step == noop_step:
                        profile["k1_noop_anchor_pairs"] = (
                            int(profile.get("k1_noop_anchor_pairs", 0)) + 1
                        )
                if n_branches == 1 and n_children == 2:
                    all_branches[b] = [_select_k1_m2_children(
                        candidates,
                        new_keys[b],
                        origin_keys[b],
                        changed_state_bonus,
                    )]
                elif first_edit_diversity:
                    all_branches[b] = (
                        _merge_state_candidates_with_first_edit_diversity(
                            list(zip(candidates, new_keys[b])),
                            n_branches,
                            origin_key=origin_keys[b],
                            changed_state_bonus=changed_state_bonus,
                        )
                    )
                else:
                    paired = list(zip(candidates, new_keys[b]))
                    all_branches[b] = _merge_state_candidates(
                        paired,
                        n_branches,
                        origin_key=origin_keys[b],
                        changed_state_bonus=changed_state_bonus,
                    )
                if first_edit_diversity:
                    for stats in (profile, sampling_stats):
                        if stats is not None:
                            stats["first_edit_signature_groups"] = (
                                int(stats.get(
                                    "first_edit_signature_groups", 0,
                                )) + len({
                                    branch.first_edit_signature
                                    for branch in all_branches[b]
                                })
                            )
                            stats["first_edit_diverse_slots"] = (
                                int(stats.get("first_edit_diverse_slots", 0))
                                + len(all_branches[b])
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
                    state_key=parent.state_key,
                    first_edit_signature=parent.first_edit_signature,
                    first_event_pending=parent.first_event_pending,
                    first_event_record=parent.first_event_record,
                    first_event_reweighted=parent.first_event_reweighted,
                ))
        if first_edit_diversity and n_branches == 1:
            for group_start in range(0, B, profile_sample_group_size):
                group_indices = list(range(
                    group_start,
                    group_start + profile_sample_group_size,
                ))
                selected = _select_first_edit_diverse_run_group(
                    new_branches,
                    new_keys,
                    origin_keys,
                    group_indices,
                    n_children,
                    changed_state_bonus,
                )
                selected_signatures = {
                    branch.first_edit_signature
                    for branch in selected.values()
                }
                for b in group_indices:
                    all_branches[b] = [selected[b]]
                for stats in (profile, sampling_stats):
                    if stats is not None:
                        stats["first_edit_signature_groups"] = (
                            int(stats.get(
                                "first_edit_signature_groups", 0,
                            )) + len(selected_signatures)
                        )
                        stats["first_edit_diverse_slots"] = (
                            int(stats.get("first_edit_diverse_slots", 0))
                            + len(group_indices)
                        )
        _profile_finish(
            profile, "merge_and_prune_seconds", section_started, device,
        )
        if profile is not None:
            profile["steps"] = profile.get("steps", 0) + 1
            profile["parent_branch_evaluations"] = (
                profile.get("parent_branch_evaluations", 0) + N_br
            )
            profile["child_candidate_evaluations"] = (
                profile.get("child_candidate_evaluations", 0)
                + len(parent_index_values)
            )
        if sampling_stats is not None:
            sampling_stats["steps"] = sampling_stats.get("steps", 0) + 1
            sampling_stats["parent_branch_evaluations"] = (
                sampling_stats.get("parent_branch_evaluations", 0) + N_br
            )
            sampling_stats["child_candidate_evaluations"] = (
                sampling_stats.get("child_candidate_evaluations", 0)
                + len(parent_index_values)
            )

    # ── 返回每条样本的全部最终分支（固定 K 槽位）──
    section_started = _profile_start(profile, device)
    results: List[Tensor] = []
    shortfall_samples = 0
    shortfall_outputs = 0
    if (
        first_event_bias_stats is not None
        and first_event_position_scores is not None
        and not first_event_bias_record_events
    ):
        # The normal full-test path deliberately retains only counts.  A
        # detailed record for every augmented R9 lineage would otherwise be
        # hundreds of MB and add avoidable CPU serialization work.
        first_event_bias_stats["summary_from_final_lineages"] = True
        first_event_bias_stats.setdefault(
            "first_event_trajectory_role_counts", {},
        )
        first_event_bias_stats.setdefault(
            "reweighted_first_event_trajectory_role_counts", {},
        )
        first_event_bias_stats.setdefault(
            "no_event_trajectory_role_counts", {},
        )
    for b in range(B):
        if n_children > 1 and score_mode == "full_probability":
            ranked = all_branches[b]
        else:
            ranked = sorted(
                all_branches[b], key=sort_key, reverse=True,
            )
        selected = ranked[:n_branches]
        missing = n_branches - len(selected)
        if missing:
            shortfall_samples += 1
            shortfall_outputs += missing
            selected.extend([ranked[0]] * missing)
        results.extend(branch.x_t for branch in selected)
        if (
            first_event_bias_stats is not None
            and first_event_position_scores is not None
        ):
            if first_event_row_metadata is None:
                role = "unspecified"
            else:
                role = str(
                    first_event_row_metadata[b].get(
                        "trajectory_role", "unspecified",
                    )
                )
            for branch_rank, branch in enumerate(selected):
                if branch.first_event_pending:
                    first_event_bias_stats["no_event_count"] += 1
                    no_event_counts = first_event_bias_stats.setdefault(
                        "no_event_trajectory_role_counts", {},
                    )
                    no_event_counts[role] = int(
                        no_event_counts.get(role, 0)
                    ) + 1
                    continue
                first_event_bias_stats["first_event_count"] += 1
                if first_event_bias_record_events:
                    if branch.first_event_record is None:
                        raise RuntimeError(
                            "first-event diagnostics lost a selected "
                            "lineage record"
                        )
                    record = dict(branch.first_event_record)
                    record["beam_branch_rank"] = int(branch_rank)
                    if first_event_record_sink is None:
                        first_event_bias_stats["records"].append(record)
                    else:
                        first_event_record_sink(record)
                    continue
                first_event_counts = first_event_bias_stats[
                    "first_event_trajectory_role_counts"
                ]
                first_event_counts[role] = int(
                    first_event_counts.get(role, 0)
                ) + 1
                if branch.first_event_reweighted:
                    reweighted_counts = first_event_bias_stats[
                        "reweighted_first_event_trajectory_role_counts"
                    ]
                    reweighted_counts[role] = int(
                        reweighted_counts.get(role, 0)
                    ) + 1

    if sampling_stats is not None:
        sampling_stats["final_branch_shortfall_samples"] = (
            sampling_stats.get("final_branch_shortfall_samples", 0)
            + shortfall_samples
        )
        sampling_stats["final_branch_shortfall_outputs"] = (
            sampling_stats.get("final_branch_shortfall_outputs", 0)
            + shortfall_outputs
        )

    out_len = max(r.shape[1] for r in results)
    out = torch.full(
        (len(results), out_len), pad_token, dtype=torch.long, device=device,
    )
    for row_index, result in enumerate(results):
        out[row_index, :result.shape[1]] = result

    _profile_finish(
        profile, "finalize_output_seconds", section_started, device,
    )

    return out
