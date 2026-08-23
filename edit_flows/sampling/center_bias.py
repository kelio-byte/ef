"""First-event reaction-center position bias for Euler sampling."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def align_position_scores(scores: Tensor, sequence_length: int) -> Tensor:
    """Pad or truncate frozen initial-state scores to the current batch width."""
    if scores.ndim != 3 or scores.shape[-1] != 3:
        raise ValueError("position scores must have shape [batch, length, 3]")
    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    if scores.shape[1] == sequence_length:
        return scores
    if scores.shape[1] > sequence_length:
        return scores[:, :sequence_length]
    padding = torch.zeros(
        scores.shape[0],
        sequence_length - scores.shape[1],
        3,
        dtype=scores.dtype,
        device=scores.device,
    )
    return torch.cat((scores, padding), dim=1)


def renormalize_position_biased_log_rates(
    log_rates: Tensor,
    position_scores: Tensor,
    legal_position_masks: Tensor,
    active_rows: Tensor,
    *,
    max_multiplier: float = 3.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Bias positions while preserving each row and mode's legal hazard.

    Position scores are normally 0, 0.5, or 1. A score of 1 receives
    max_multiplier relative weight before normalization. INS, SUB, and DEL
    are normalized independently. Rows with constant legal scores are
    returned bit-for-bit unchanged.
    """
    if log_rates.ndim != 3 or log_rates.shape[-1] != 3:
        raise ValueError("log_rates must have shape [batch, length, 3]")
    if position_scores.shape != log_rates.shape:
        raise ValueError("position_scores must match log_rates")
    if legal_position_masks.shape != log_rates.shape:
        raise ValueError("legal_position_masks must match log_rates")
    if active_rows.shape != (log_rates.shape[0],):
        raise ValueError("active_rows must have shape [batch]")
    if max_multiplier < 1 or not math.isfinite(max_multiplier):
        raise ValueError("max_multiplier must be finite and >= 1")
    if not torch.isfinite(position_scores).all():
        raise ValueError("position_scores must be finite")
    if (position_scores < 0).any() or (position_scores > 1).any():
        raise ValueError("position_scores must lie in [0, 1]")

    negative_infinity = torch.tensor(
        float("-inf"), dtype=log_rates.dtype, device=log_rates.device
    )
    base_legal = log_rates.masked_fill(
        ~legal_position_masks, negative_infinity
    )
    before = torch.exp(log_rates).masked_fill(
        ~legal_position_masks, 0.0
    ).sum(dim=1)
    if max_multiplier == 1.0:
        zeros = torch.zeros_like(before)
        return log_rates, {
            "before_hazard": before,
            "after_hazard": before,
            "absolute_error": zeros,
            "relative_error": zeros,
            "changed": torch.zeros_like(before, dtype=torch.bool),
        }

    log_multiplier = math.log(max_multiplier)
    weighted_legal = (
        log_rates + log_multiplier * position_scores
    ).masked_fill(~legal_position_masks, negative_infinity)
    base_log_total = torch.logsumexp(base_legal, dim=1)
    weighted_log_total = torch.logsumexp(weighted_legal, dim=1)
    has_legal = legal_position_masks.any(dim=1)

    positive_infinity = torch.tensor(
        float("inf"), dtype=position_scores.dtype, device=position_scores.device
    )
    legal_min = position_scores.masked_fill(
        ~legal_position_masks, positive_infinity
    ).amin(dim=1)
    legal_max = position_scores.masked_fill(
        ~legal_position_masks, negative_infinity
    ).amax(dim=1)
    changed = (
        active_rows.unsqueeze(1)
        & has_legal
        & (legal_max != legal_min)
    )
    correction = torch.where(
        changed,
        base_log_total - weighted_log_total,
        torch.zeros_like(base_log_total),
    )
    apply_mask = legal_position_masks & changed.unsqueeze(1)
    biased = (
        log_rates
        + log_multiplier * position_scores
        + correction.unsqueeze(1)
    )
    output = torch.where(apply_mask, biased, log_rates)
    after = torch.exp(output).masked_fill(
        ~legal_position_masks, 0.0
    ).sum(dim=1)
    absolute_error = (after - before).abs()
    relative_error = absolute_error / before.clamp_min(1e-12)
    return output, {
        "before_hazard": before,
        "after_hazard": after,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "changed": changed,
    }
