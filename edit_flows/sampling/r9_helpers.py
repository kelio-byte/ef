"""用途：提供 K1M2 分支采样所需的随机数、动作和候选筛选辅助函数。
输入：模型分布、编辑状态、分支信息和随机种子。
输出：采样动作、子候选及概率信息；不负责展开 R 次独立运行。
"""

from __future__ import annotations
from typing import List, Tuple, Optional, TYPE_CHECKING
import math
import torch
from torch import Tensor
from edit_flows.sampling.ops import (
    apply_ins_del_operations,
    edit_position_masks,
    legal_token_log_probs,
)
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN
from edit_flows.sampling.common import _event_probability

if TYPE_CHECKING:
    from .r9 import Branch


def _mix_child_seed(parent_seed: int, step: int, child_index: int) -> int:
    """作用：为子分支生成稳定随机种子。输入：父种子、步数和子分支序号。输出：子分支种子整数。"""
    if child_index < 0:
        raise ValueError(f"child_index must be >= 0, got {child_index}")
    if child_index == 0:
        return int(parent_seed)
    mask = (1 << 64) - 1
    x = (
        int(parent_seed) & mask
        ^ (step + 1) * 11400714819323198485 & mask
        ^ (child_index + 1) * 15111065706836454659 & mask
    )
    x = (x ^ x >> 30) * 13787848793156543929 & mask
    x = (x ^ x >> 27) * 10723151780598845931 & mask
    x ^= x >> 31
    return x & (1 << 63) - 1


def _logaddexp_float(a: float, b: float) -> float:
    """作用：稳定合并两个对数权重。输入：两个 log 权重。输出：logsumexp 标量。"""
    high = max(a, b)
    low = min(a, b)
    return high + math.log1p(math.exp(low - high))


def _select_k1_m2_children(
    candidates: List[Branch],
    keys: List[Tuple[int, ...]],
    origin_key: Tuple[int, ...],
    changed_state_bonus: float,
) -> Branch:
    """作用：合并重复状态并按 K1M2 规则保留一个子分支。输入：两个候选、状态键和变化奖励。输出：保留的分支。"""
    if len(candidates) != 2 or len(keys) != 2:
        raise ValueError("K1M2 selection requires exactly two children")
    (first, second) = candidates
    (first_key, second_key) = keys
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

    def rank(branch: Branch, key: Tuple[int, ...]):
        """作用：计算候选分支排序键。输入：分支和状态键。输出：按权重、变化奖励及种子组成的排序元组。"""
        return (
            branch.log_mass + changed_state_bonus * float(key != origin_key),
            branch.log_mass,
            -float(branch.seed),
        )

    if rank(second, second_key) > rank(first, first_key):
        return second
    return first


def _token_keys_batch(
    x_t: Tensor, pad_token: int, bos_token: int
) -> List[Tuple[int, ...]]:
    """作用：将一批状态编码为可比较键。输入：token 张量和 PAD/BOS 编号。输出：逐行 token 元组列表。"""
    rows = x_t.detach().cpu().tolist()
    excluded = (pad_token, bos_token)
    return [tuple((token for token in row if token not in excluded)) for row in rows]


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
    """作用：计算每条分支本步动作的对数概率。输入：动作、模型分布、步长和状态。输出：每条分支的 log 概率。"""
    rates = torch.exp(log_rates_eff)
    if state_tokens is not None:
        if state_tokens.shape != rates.shape[:2]:
            raise ValueError(
                "state_tokens must have shape [batch, length] matching rates"
            )
        (insert_positions, sub_del_positions) = edit_position_masks(
            state_tokens, pad_token=pad_token
        )
    else:
        insert_positions = actions.get("insert_position_mask")
        sub_del_positions = actions.get("sub_del_position_mask")
        if insert_positions is None:
            insert_positions = torch.ones_like(rates[:, :, 0], dtype=torch.bool)
        if sub_del_positions is None:
            sub_del_positions = torch.ones_like(rates[:, :, 1], dtype=torch.bool)
    ins_log_normalizer = actions.get("ins_token_log_normalizer")
    sub_log_normalizer = actions.get("sub_token_log_normalizer")
    ins_rates = rates[:, :, 0] * insert_positions.to(rates.dtype)
    sub_rates = rates[:, :, 1] * sub_del_positions.to(rates.dtype)
    del_rates = rates[:, :, 2] * sub_del_positions.to(rates.dtype)
    eps = 1e-12
    log_eps = math.log(eps)
    ins_mu = adapt_h * ins_rates
    ds_rates = sub_rates + del_rates
    ds_mu = adapt_h * ds_rates
    ins_event_log_p = torch.log((-torch.expm1(-ins_mu)).clamp_min(eps))
    ds_event_log_p = torch.log((-torch.expm1(-ds_mu)).clamp_min(eps))
    ins_token_log_p = log_ins_probs.gather(
        2, actions["ins_tokens"].unsqueeze(-1)
    ).squeeze(-1)
    sub_token_log_p = log_sub_probs.gather(
        2, actions["sub_tokens"].unsqueeze(-1)
    ).squeeze(-1)
    if ins_log_normalizer is not None:
        ins_token_log_p = ins_token_log_p - ins_log_normalizer
    if sub_log_normalizer is not None:
        sub_token_log_p = sub_token_log_p - sub_log_normalizer
    ins_token_log_p = ins_token_log_p.clamp_min(log_eps)
    sub_token_log_p = sub_token_log_p.clamp_min(log_eps)
    ins_contrib = torch.where(
        actions["ins_mask"], ins_event_log_p + ins_token_log_p, -ins_mu
    )
    sub_contrib = (
        ds_event_log_p
        + torch.log((sub_rates / ds_rates.clamp_min(eps)).clamp_min(eps))
        + sub_token_log_p
    )
    del_contrib = ds_event_log_p + torch.log(
        (del_rates / ds_rates.clamp_min(eps)).clamp_min(eps)
    )
    ds_contrib = torch.where(
        actions["sub_mask"],
        sub_contrib,
        torch.where(actions["del_mask"], del_contrib, -ds_mu),
    )
    return ins_contrib.sum(dim=1) + ds_contrib.sum(dim=1)


def _apply_edits_batch(
    x_t: Tensor, actions: dict, max_seq_len: int, pad_token: int
) -> Tensor:
    """作用：批量执行已采样编辑。输入：状态、动作和序列长度限制。输出：编辑后的状态张量。"""
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


def _stateless_uniform(
    seeds: Tensor, step: int, seq_len: int, stream: int, dtype: torch.dtype
) -> Tensor:
    """作用：生成可复现的无状态均匀随机数。输入：种子、步数、位置、随机流编号和 dtype。输出：均匀随机张量。"""
    modulus = 2147483647
    positions = torch.arange(
        1, seq_len + 1, dtype=torch.int64, device=seeds.device
    ).unsqueeze(0)
    x = torch.remainder(
        torch.remainder(seeds.to(torch.int64), modulus).unsqueeze(1) * 48271
        + (step + 1) * 69621
        + positions * 1013904223
        + (stream + 1) * 1664525,
        modulus,
    )
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
    x = torch.remainder(x * 73856093 + 19349663, modulus)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 13))
    return ((x.to(torch.float64) + 0.5) / modulus).to(dtype)


def _sample_tokens_from_uniform(log_probs: Tensor, uniform: Tensor) -> Tensor:
    """作用：从 token 类别分布中采样。输入：对数概率和均匀随机数。输出：采样 token 编号张量。"""
    cdf = torch.exp(log_probs).cumsum(dim=-1)
    return (cdf < uniform.unsqueeze(-1)).sum(dim=-1).clamp_max(log_probs.shape[-1] - 1)


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
    """作用：批量采样各分支的插入、删除和替换动作。输入：分支种子、状态、速率及 token 分布。输出：包含动作掩码和 token 的字典。"""
    seeds = branch_seeds.to(device=x_t.device, dtype=torch.int64)
    (legal_log_ins_probs, ins_log_normalizer) = legal_token_log_probs(log_ins_probs)
    (legal_log_sub_probs, sub_log_normalizer) = legal_token_log_probs(
        log_sub_probs, current_tokens=x_t
    )
    (insert_positions, sub_del_positions) = edit_position_masks(
        x_t, pad_token=pad_token
    )
    ins_sample_positions = insert_positions & torch.isfinite(ins_log_normalizer)
    sub_sample_positions = sub_del_positions & torch.isfinite(sub_log_normalizer)
    rates = torch.exp(log_rates)
    lambda_ins = rates[:, :, 0] * ins_sample_positions.to(rates.dtype)
    lambda_sub = rates[:, :, 1] * sub_sample_positions.to(rates.dtype)
    lambda_del = rates[:, :, 2] * sub_del_positions.to(rates.dtype)
    ins_prob = _event_probability(adapt_h * lambda_ins, event_prob_mode)
    ds_prob = _event_probability(adapt_h * (lambda_sub + lambda_del), event_prob_mode)
    seq_len = x_t.shape[1]
    ins_mask = (
        _stateless_uniform(seeds, step, seq_len, stream=0, dtype=lambda_ins.dtype)
        < ins_prob
    )
    ds_mask = (
        _stateless_uniform(seeds, step, seq_len, stream=1, dtype=lambda_sub.dtype)
        < ds_prob
    )
    prob_del = lambda_del / (lambda_sub + lambda_del + 1e-08)
    del_mask = ds_mask & (
        _stateless_uniform(seeds, step, seq_len, stream=2, dtype=lambda_del.dtype)
        < prob_del
    )
    sub_mask = ds_mask & ~del_mask
    ins_tokens = _sample_tokens_from_uniform(
        legal_log_ins_probs,
        _stateless_uniform(seeds, step, seq_len, stream=3, dtype=log_ins_probs.dtype),
    )
    sub_tokens = _sample_tokens_from_uniform(
        legal_log_sub_probs,
        _stateless_uniform(seeds, step, seq_len, stream=4, dtype=log_sub_probs.dtype),
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
        "effective_log_rates": log_rates,
    }


def _set_second_child_noop(actions: dict, n_children: int) -> None:
    """作用：将每个父分支的第二个子分支设为 no-op。输入：动作字典和子分支数。输出：原地修改动作字典。"""
    noop_rows = torch.arange(
        1, actions["ins_mask"].shape[0], n_children, device=actions["ins_mask"].device
    )
    actions["ins_mask"][noop_rows] = False
    actions["sub_mask"][noop_rows] = False
    actions["del_mask"][noop_rows] = False
