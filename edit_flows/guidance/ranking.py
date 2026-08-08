"""Shared-anchor pairwise ranking utilities for action-level guidance.

The multi-terminal guidance records for one product have different sampled
states and times.  Comparing their scores directly would therefore mix up
terminal quality with state difficulty.  This module evaluates several
terminal action sets under one shared ``(product, state, time)`` anchor and
provides a small differentiable ranking loss.  It remains an action-level
approximation; it is not an exact variable-length Z-space DGM transition.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from edit_flows.guidance.targets import build_action_target_masks


def _validate_guidance_tensors(
    guidance: tuple[Tensor, Tensor, Tensor],
    *,
    state_length: int | None = None,
) -> None:
    if len(guidance) != 3:
        raise ValueError("guidance must contain insert, substitute and delete tensors")
    insert, substitute, delete = guidance
    if insert.ndim != 3 or substitute.shape != insert.shape:
        raise ValueError("insert and substitute guidance must have equal rank-3 shapes")
    if delete.ndim != 3 or delete.shape[:2] != insert.shape[:2] or delete.shape[-1] != 1:
        raise ValueError("delete guidance must have shape [batch, length, 1]")
    if not all(torch.is_floating_point(value) for value in guidance):
        raise TypeError("guidance tensors must be floating point")
    if not all(torch.isfinite(value).all() for value in guidance):
        raise ValueError("guidance tensors must contain only finite values")
    if (insert <= 0).any() or (substitute <= 0).any() or (delete <= 0).any():
        raise ValueError("guidance tensors must be strictly positive")
    if state_length is not None and insert.shape[1] != state_length:
        raise ValueError(
            f"guidance length {insert.shape[1]} does not match state length {state_length}"
        )


def score_masked_action_sets(
    guidance: tuple[Tensor, Tensor, Tensor],
    insert_mask: Tensor,
    substitute_mask: Tensor,
    delete_mask: Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Return mean-log-guidance scores and selected-action counts.

    ``guidance`` and masks describe one state-terminal pair per batch row.
    The score is the mean of ``log(H)`` over all selected insert, substitute
    and delete actions.  A zero selected-action count is represented by a
    zero score and a zero count; callers must exclude that row from ranking.
    """
    if eps <= 0 or not torch.isfinite(torch.tensor(float(eps))):
        raise ValueError("eps must be finite and positive")
    insert, substitute, delete = guidance
    _validate_guidance_tensors(guidance)
    if insert_mask.dtype != torch.bool or substitute_mask.dtype != torch.bool or delete_mask.dtype != torch.bool:
        raise TypeError("action masks must be boolean tensors")
    if insert_mask.shape != insert.shape or substitute_mask.shape != substitute.shape:
        raise ValueError("insert/substitute masks must match guidance shapes")
    if delete_mask.shape != delete.shape:
        raise ValueError("delete mask must match delete guidance shape")
    if not (insert_mask.device == insert.device == substitute_mask.device == delete_mask.device):
        raise ValueError("guidance and masks must be on the same device")

    values = (
        torch.log(insert.clamp_min(eps)),
        torch.log(substitute.clamp_min(eps)),
        torch.log(delete.clamp_min(eps)),
    )
    masks = (insert_mask, substitute_mask, delete_mask)
    score_sum = torch.zeros(insert.shape[0], dtype=insert.dtype, device=insert.device)
    count = torch.zeros(insert.shape[0], dtype=insert.dtype, device=insert.device)
    for value, mask in zip(values, masks):
        score_sum = score_sum + (value * mask.to(dtype=value.dtype)).sum(dim=tuple(range(1, value.ndim)))
        count = count + mask.sum(dim=tuple(range(1, mask.ndim))).to(dtype=count.dtype)
    scores = torch.where(count > 0, score_sum / count.clamp_min(1.0), torch.zeros_like(score_sum))
    return scores, count


def score_terminal_action_sets(
    guidance: tuple[Tensor, Tensor, Tensor],
    state_tokens: Tensor,
    terminal_tokens: Tensor,
    *,
    vocab_size: int,
    pad_token: int = 0,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Build alignment masks and score one terminal per batch row.

    The alignment is intentionally detached and performed on CPU, matching
    the existing action-target training path.  No gradient is required
    through the discrete alignment or terminal tokens.
    """
    if state_tokens.ndim != 2 or terminal_tokens.ndim != 2:
        raise ValueError("state_tokens and terminal_tokens must be rank-2")
    if state_tokens.shape[0] != terminal_tokens.shape[0]:
        raise ValueError("state and terminal batches must have equal size")
    _validate_guidance_tensors(guidance, state_length=state_tokens.shape[1])
    insert_mask, substitute_mask, delete_mask = build_action_target_masks(
        state_tokens.detach().cpu(),
        terminal_tokens.detach().cpu(),
        vocab_size=vocab_size,
        pad_token=pad_token,
    )
    device = guidance[0].device
    return score_masked_action_sets(
        guidance,
        insert_mask.to(device=device),
        substitute_mask.to(device=device),
        delete_mask.to(device=device),
        eps=eps,
    )


def _group_rows(source_index: Tensor, group_size: int) -> list[list[int]]:
    if source_index.ndim != 1:
        raise ValueError("source_index must have shape [batch]")
    if group_size < 1:
        raise ValueError("group_size must be positive")
    groups: dict[int, list[int]] = {}
    for row, source in enumerate(source_index.detach().cpu().tolist()):
        groups.setdefault(int(source), []).append(row)
    invalid = {
        source: len(rows)
        for source, rows in groups.items()
        if len(rows) != group_size
    }
    if invalid:
        preview = list(sorted(invalid.items()))[:5]
        raise ValueError(
            "shared-anchor ranking requires complete product groups; "
            f"invalid groups (first five)={preview}"
        )
    return [groups[source] for source in sorted(groups)]


def _safe_pearson(left: Tensor, right: Tensor) -> Tensor:
    if left.numel() < 2:
        return left.new_zeros(())
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = torch.sqrt(left_centered.square().sum() * right_centered.square().sum())
    numerator = (left_centered * right_centered).sum()
    return torch.where(
        denominator > 1e-12,
        numerator / denominator,
        left.new_zeros(()),
    )


def _within_group_pearson(
    left: Tensor,
    right: Tensor,
    group_ids: Tensor,
) -> Tensor:
    """Pearson correlation after removing each group's score offset."""
    if left.shape != right.shape or left.ndim != 1 or group_ids.shape != left.shape:
        raise ValueError("within-group correlation inputs must be equal-shaped rank-1 tensors")
    if left.numel() < 2:
        return left.new_zeros(())
    left_residual = left.clone()
    right_residual = right.clone()
    for group_id in torch.unique(group_ids):
        mask = group_ids == group_id
        if int(mask.sum().item()) < 2:
            continue
        left_residual[mask] -= left[mask].mean()
        right_residual[mask] -= right[mask].mean()
    return _safe_pearson(left_residual, right_residual)


def shared_anchor_pairwise_loss(
    guidance: tuple[Tensor, Tensor, Tensor],
    state_tokens: Tensor,
    terminal_tokens: Tensor,
    source_index: Tensor,
    reward: Tensor,
    *,
    vocab_size: int,
    group_size: int = 4,
    anchor_rotation: int = 0,
    all_anchors: bool = False,
    temperature: float = 1.0,
    equal_tolerance: float = 1e-6,
    pad_token: int = 0,
    eps: float = 1e-8,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute pairwise ranking under shared state/time anchors.

    ``guidance`` is the model output for the original batch.  During
    training, one anchor row per product is selected (with a deterministic
    rotation); during validation ``all_anchors=True`` evaluates every row as
    an anchor.  Every selected anchor reuses one guidance output while its
    action masks are built against all terminals in the same product group.
    """
    if not isinstance(anchor_rotation, int) or anchor_rotation < 0:
        raise ValueError("anchor_rotation must be a non-negative integer")
    if temperature <= 0 or not torch.isfinite(torch.tensor(float(temperature))):
        raise ValueError("temperature must be finite and positive")
    if equal_tolerance < 0 or not torch.isfinite(torch.tensor(float(equal_tolerance))):
        raise ValueError("equal_tolerance must be finite and non-negative")
    if state_tokens.ndim != 2 or terminal_tokens.ndim != 2:
        raise ValueError("state_tokens and terminal_tokens must be rank-2")
    if state_tokens.shape[0] != terminal_tokens.shape[0]:
        raise ValueError("state and terminal batches must have equal size")
    if source_index.shape != reward.shape or source_index.ndim != 1:
        raise ValueError("source_index and reward must have shape [batch]")
    if reward.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        reward = reward.float()
    reward = reward.to(device=state_tokens.device, dtype=torch.float32)
    if not torch.isfinite(reward).all() or (reward < 0).any():
        raise ValueError("reward must contain finite non-negative values")
    _validate_guidance_tensors(guidance, state_length=state_tokens.shape[1])
    groups = _group_rows(source_index, group_size)
    group_count = len(groups)

    pair_specs: list[tuple[int, int, int]] = []
    considered_pairs = 0
    selected_candidate_keys: set[tuple[int, int]] = set()
    zero_anchor_candidates: list[tuple[int, int]] = []
    for group in groups:
        anchors = group if all_anchors else [
            group[(anchor_rotation + int(source_index[group[0]].item()) % group_size) % group_size]
        ]
        for anchor in anchors:
            for left_offset in range(group_size):
                left = group[left_offset]
                for right_offset in range(left_offset + 1, group_size):
                    right = group[right_offset]
                    left_reward = float(reward[left].item())
                    right_reward = float(reward[right].item())
                    if abs(left_reward - right_reward) <= equal_tolerance:
                        continue
                    considered_pairs += 1
                    high, low = (left, right) if left_reward > right_reward else (right, left)
                    pair_specs.append((anchor, high, low))
                    selected_candidate_keys.add((anchor, high))
                    selected_candidate_keys.add((anchor, low))

    if not pair_specs:
        zero = sum((value.sum() for value in guidance)) * 0.0
        metrics = {
            "pair_count": zero.detach(),
            "candidate_pair_count": zero.detach(),
            "pair_accuracy_strict": zero.detach(),
            "pair_accuracy_tie_half": zero.detach(),
            "pair_tie_fraction": zero.detach(),
            "pair_margin_mean": zero.detach(),
            "valid_pair_group_fraction": zero.detach(),
            "no_action_candidate_fraction": zero.detach(),
            "reward_score_pearson": zero.detach(),
            "reward_score_pearson_within_group": zero.detach(),
        }
        return zero, metrics

    unique_keys = sorted(selected_candidate_keys)
    anchor_rows = torch.tensor([key[0] for key in unique_keys], dtype=torch.long)
    terminal_rows = torch.tensor([key[1] for key in unique_keys], dtype=torch.long)
    device = guidance[0].device
    anchor_rows_device = anchor_rows.to(device=device)
    terminal_rows_device = terminal_rows.to(device=device)
    anchor_guidance = tuple(value.index_select(0, anchor_rows_device) for value in guidance)
    anchor_states = state_tokens.index_select(0, anchor_rows_device)
    candidate_terminals = terminal_tokens.index_select(0, terminal_rows_device)
    scores, counts = score_terminal_action_sets(
        anchor_guidance,
        anchor_states,
        candidate_terminals,
        vocab_size=vocab_size,
        pad_token=pad_token,
        eps=eps,
    )
    score_by_key = {
        key: (scores[index], counts[index])
        for index, key in enumerate(unique_keys)
    }
    pair_values: list[Tensor] = []
    margins: list[Tensor] = []
    valid_group_sources: set[int] = set()
    for anchor, high, low in pair_specs:
        high_score, high_count = score_by_key[(anchor, high)]
        low_score, low_count = score_by_key[(anchor, low)]
        if high_count.item() <= 0 or low_count.item() <= 0:
            continue
        margin = high_score - low_score
        pair_values.append(F.softplus(-margin / float(temperature)))
        margins.append(margin.detach())
        valid_group_sources.add(int(source_index[anchor].item()))

    if pair_values:
        pair_loss = torch.stack(pair_values).mean()
        margin_tensor = torch.stack(margins)
        valid_pair_count = len(pair_values)
        strict_accuracy = torch.stack([
            margin > 0 for margin in margins
        ]).float().mean()
        tie_fraction = torch.stack([
            margin == 0 for margin in margins
        ]).float().mean()
        tie_half_accuracy = (
            (torch.stack([margin > 0 for margin in margins]).float()
             + 0.5 * torch.stack([margin == 0 for margin in margins]).float())
            .mean()
        )
        pair_margin_mean = margin_tensor.mean()
    else:
        pair_loss = sum((value.sum() for value in guidance)) * 0.0
        valid_pair_count = 0
        strict_accuracy = pair_loss.detach()
        tie_fraction = pair_loss.detach()
        tie_half_accuracy = pair_loss.detach()
        pair_margin_mean = pair_loss.detach()

    candidate_counts = counts
    no_action_fraction = (candidate_counts <= 0).float().mean()
    valid_scores = candidate_counts > 0
    candidate_rewards = torch.stack([
        reward[key[1]] for key in unique_keys
    ]).to(device=device)
    candidate_groups = torch.tensor(
        [int(source_index[key[0]].item()) for key in unique_keys],
        dtype=torch.long, device=device,
    )
    reward_score_pearson = _safe_pearson(
        candidate_rewards[valid_scores], scores[valid_scores],
    ).detach()
    reward_score_pearson_within_group = _within_group_pearson(
        scores[valid_scores],
        candidate_rewards[valid_scores],
        candidate_groups[valid_scores],
    ).detach()
    valid_pair_group_fraction = scores.new_tensor(
        len(valid_group_sources) / max(group_count, 1),
    )
    metrics = {
        "pair_count": scores.new_tensor(float(valid_pair_count)),
        "candidate_pair_count": scores.new_tensor(float(considered_pairs)),
        "pair_accuracy_strict": strict_accuracy.detach(),
        "pair_accuracy_tie_half": tie_half_accuracy.detach(),
        "pair_tie_fraction": tie_fraction.detach(),
        "pair_margin_mean": pair_margin_mean.detach(),
        "valid_pair_group_fraction": valid_pair_group_fraction,
        "no_action_candidate_fraction": no_action_fraction.detach(),
        "reward_score_pearson": reward_score_pearson,
        "reward_score_pearson_within_group": reward_score_pearson_within_group,
    }
    return pair_loss, metrics
