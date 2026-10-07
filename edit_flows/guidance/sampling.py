"""Action-level guidance transforms for the ordinary Euler sampler.

The Edit Flows network emits per-position edit rates and conditional token
posteriors.  This module applies a positive guidance weight to the resulting
action rates, then preserves the original total edit intensity at each
position.  It is an explicit approximate adapter, not the fixed-coordinate
Z-space DGM construction.
"""

from __future__ import annotations

import torch
from torch import Tensor


def apply_action_guidance(
    log_rates: Tensor,
    log_insert_probs: Tensor,
    log_substitute_probs: Tensor,
    guidance_insert: Tensor,
    guidance_substitute: Tensor,
    guidance_delete: Tensor,
    *,
    beta: float = 1.0,
    eps: float = 1e-12,
    rate_normalization: str = "per_position",
    position_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reweight one Euler action distribution with positive guidance.

    ``log_rates`` has final dimension ``(insert, substitute, delete)``;
    insert/substitute guidance have the same ``[B, L, V]`` shape as their
    token posteriors and delete guidance has shape ``[B, L, 1]``.  The
    ``rate_normalization='per_position'`` preserves total rate at every
    position.  ``'per_sample'`` preserves the sum across editable positions,
    allowing guidance to move intensity between positions.  Therefore a
    constant guidance tensor changes neither rates nor token posteriors in
    either mode.  ``beta=0`` is an exact identity path.
    """
    if log_rates.ndim != 3 or log_rates.shape[-1] != 3:
        raise ValueError("log_rates must have shape [B, L, 3]")
    if log_insert_probs.ndim != 3 or log_substitute_probs.shape != log_insert_probs.shape:
        raise ValueError("insert/substitute log probabilities must be [B, L, V]")
    if log_rates.shape[:2] != log_insert_probs.shape[:2]:
        raise ValueError("rate and token posterior batch/length dimensions differ")
    if guidance_insert.shape != log_insert_probs.shape:
        raise ValueError("insert guidance shape does not match insert posterior")
    if guidance_substitute.shape != log_substitute_probs.shape:
        raise ValueError("substitute guidance shape does not match substitute posterior")
    if guidance_delete.shape != log_rates.shape[:2] + (1,):
        raise ValueError("delete guidance must have shape [B, L, 1]")
    if beta < 0 or not torch.isfinite(torch.tensor(beta)):
        raise ValueError("beta must be finite and non-negative")
    if eps <= 0 or not torch.isfinite(torch.tensor(eps)):
        raise ValueError("eps must be finite and positive")
    if rate_normalization not in {"per_position", "per_sample"}:
        raise ValueError(
            "rate_normalization must be 'per_position' or 'per_sample'"
        )
    if position_mask is not None:
        if position_mask.shape != log_rates.shape[:2]:
            raise ValueError("position_mask must match [batch, length]")
        if position_mask.dtype != torch.bool:
            raise TypeError("position_mask must be boolean")
    if beta == 0:
        return log_rates, log_insert_probs, log_substitute_probs
    for name, tensor in (
        ("log_rates", log_rates),
        ("log_insert_probs", log_insert_probs),
        ("log_substitute_probs", log_substitute_probs),
        ("guidance_insert", guidance_insert),
        ("guidance_substitute", guidance_substitute),
        ("guidance_delete", guidance_delete),
    ):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain finite values")
    if (
        (guidance_insert <= 0).any()
        or (guidance_substitute <= 0).any()
        or (guidance_delete <= 0).any()
    ):
        raise ValueError("guidance tensors must be strictly positive")

    rates = log_rates.exp()
    insert_probs = log_insert_probs.exp()
    substitute_probs = log_substitute_probs.exp()
    base_insert = rates[:, :, 0:1] * insert_probs
    base_substitute = rates[:, :, 1:2] * substitute_probs
    base_delete = rates[:, :, 2:3]
    # H^beta is the density-ratio contribution.  Clamp only the log argument
    # for numerical safety; the guidance model itself is required to be > 0.
    h_insert = (guidance_insert.clamp_min(eps).log() * beta).exp()
    h_substitute = (guidance_substitute.clamp_min(eps).log() * beta).exp()
    h_delete = (guidance_delete.clamp_min(eps).log() * beta).exp()

    weighted_insert = base_insert * h_insert
    weighted_substitute = base_substitute * h_substitute
    weighted_delete = base_delete * h_delete
    weighted_total = (
        weighted_insert.sum(dim=-1)
        + weighted_substitute.sum(dim=-1)
        + weighted_delete.squeeze(-1)
    )
    base_total = rates.sum(dim=-1)
    if rate_normalization == "per_position":
        scale = torch.where(
            base_total > 0,
            base_total / weighted_total.clamp_min(eps),
            torch.ones_like(base_total),
        ).unsqueeze(-1)
        weighted_insert = weighted_insert * scale
        weighted_substitute = weighted_substitute * scale
        weighted_delete = weighted_delete * scale
    else:
        active = (
            position_mask
            if position_mask is not None
            else torch.ones_like(base_total, dtype=torch.bool)
        )
        active_float = active.to(dtype=base_total.dtype)
        base_sample_total = (base_total * active_float).sum(dim=-1)
        weighted_sample_total = (weighted_total * active_float).sum(dim=-1)
        scale = torch.where(
            base_sample_total > 0,
            base_sample_total / weighted_sample_total.clamp_min(eps),
            torch.ones_like(base_sample_total),
        ).reshape(-1, 1, 1)
        weighted_insert = weighted_insert * scale
        weighted_substitute = weighted_substitute * scale
        weighted_delete = weighted_delete * scale
        if position_mask is not None:
            active_3d = active.unsqueeze(-1)
            weighted_insert = torch.where(
                active_3d, weighted_insert, base_insert,
            )
            weighted_substitute = torch.where(
                active_3d, weighted_substitute, base_substitute,
            )
            weighted_delete = torch.where(
                active_3d, weighted_delete, base_delete,
            )

    guided_rates = torch.stack(
        [
            weighted_insert.sum(dim=-1),
            weighted_substitute.sum(dim=-1),
            weighted_delete.squeeze(-1),
        ],
        dim=-1,
    )
    guided_insert_probs = weighted_insert / guided_rates[:, :, 0:1].clamp_min(eps)
    guided_substitute_probs = weighted_substitute / guided_rates[:, :, 1:2].clamp_min(eps)
    # If a base type has zero rate, no event can be sampled.  A finite
    # normalized posterior is nevertheless needed by torch.multinomial.
    guided_insert_probs = torch.where(
        guided_rates[:, :, 0:1] > eps,
        guided_insert_probs,
        insert_probs,
    )
    guided_substitute_probs = torch.where(
        guided_rates[:, :, 1:2] > eps,
        guided_substitute_probs,
        substitute_probs,
    )
    return (
        guided_rates.clamp_min(eps).log(),
        guided_insert_probs.clamp_min(eps).log(),
        guided_substitute_probs.clamp_min(eps).log(),
    )
