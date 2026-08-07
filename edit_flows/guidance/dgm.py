"""Small, model-agnostic utilities for discrete guidance matching.

This module intentionally contains only the algebra shared by synthetic tests
and future guidance samplers.  It does not modify the Edit Flows checkpoint or
claim that action-level reweighting is the exact variable-length Z-space DGM
construction.  The latter still requires a separate transition adapter.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F


def positive_guidance(raw_guidance: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Map unconstrained network outputs to finite strictly-positive weights."""
    if not torch.is_floating_point(raw_guidance):
        raise TypeError("raw_guidance must be a floating-point tensor")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if not torch.isfinite(raw_guidance).all():
        raise ValueError("raw_guidance must contain only finite values")
    return F.softplus(raw_guidance) + eps


def guided_log_probs(
    base_log_probs: Tensor,
    guidance: Tensor,
    *,
    dim: int = -1,
) -> Tensor:
    """Reweight a base categorical distribution by positive guidance.

    The returned tensor is normalized in ``dim`` and represents

    ``q(s | x) ∝ p(s | x) * H(s, x)``.

    ``base_log_probs`` may contain unnormalized log weights, which is useful
    when the base model already emitted log rates plus token log-probabilities.
    ``guidance`` must be strictly positive and have the same shape.
    """
    if base_log_probs.shape != guidance.shape:
        raise ValueError(
            "base_log_probs and guidance must have the same shape, got "
            f"{tuple(base_log_probs.shape)} and {tuple(guidance.shape)}"
        )
    if not torch.is_floating_point(base_log_probs):
        raise TypeError("base_log_probs must be a floating-point tensor")
    if not torch.is_floating_point(guidance):
        raise TypeError("guidance must be a floating-point tensor")
    if not torch.isfinite(base_log_probs).all():
        raise ValueError("base_log_probs must contain only finite values")
    if not torch.isfinite(guidance).all() or (guidance <= 0).any():
        raise ValueError("guidance must contain finite strictly-positive values")
    if guidance.shape[dim] < 1:
        raise ValueError("the categorical dimension must be non-empty")
    return F.log_softmax(base_log_probs + guidance.log(), dim=dim)


def positive_guidance_bregman_loss(
    guidance: Tensor,
    reward: Tensor,
    *,
    mask: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    """Compute the positive guidance Bregman objective ``H - r log H``.

    ``reward`` is broadcast over all non-batch dimensions.  The function
    expects an already-positive guidance prediction; a future model wrapper
    should call :func:`positive_guidance` before passing its output here.
    ``mask`` can select valid coordinates and is broadcast like ``reward``.
    """
    if guidance.ndim < 1:
        raise ValueError("guidance must have at least a batch dimension")
    if not torch.is_floating_point(guidance):
        raise TypeError("guidance must be a floating-point tensor")
    reward = reward.to(device=guidance.device, dtype=guidance.dtype)
    if reward.ndim > guidance.ndim:
        raise ValueError(
            f"reward shape {tuple(reward.shape)} has more dimensions than "
            f"guidance shape {tuple(guidance.shape)}"
        )
    # A terminal reward is normally one scalar per batch item.  Append
    # singleton coordinate dimensions so [B] broadcasts over [B, L, V].
    if reward.ndim < guidance.ndim:
        if reward.ndim == 0 or reward.shape[0] == guidance.shape[0]:
            reward = reward.reshape(
                *reward.shape,
                *([1] * (guidance.ndim - reward.ndim)),
            )
        else:
            raise ValueError(
                f"reward shape {tuple(reward.shape)} cannot broadcast to "
                f"guidance shape {tuple(guidance.shape)}"
            )
    try:
        reward = torch.broadcast_to(reward, guidance.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"reward shape {tuple(reward.shape)} cannot broadcast to "
            f"guidance shape {tuple(guidance.shape)}"
        ) from exc
    if not torch.isfinite(guidance).all() or (guidance <= 0).any():
        raise ValueError("guidance must contain finite strictly-positive values")
    if not torch.isfinite(reward).all() or (reward < 0).any():
        raise ValueError("reward must contain finite non-negative values")

    loss = guidance - reward * guidance.log()
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor")
        mask = mask.to(device=guidance.device)
        try:
            mask = torch.broadcast_to(mask, loss.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} cannot broadcast to "
                f"guidance shape {tuple(loss.shape)}"
            ) from exc
        loss = loss.masked_select(mask)
        if loss.numel() == 0:
            raise ValueError("mask selects no guidance entries")

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"unsupported reduction: {reduction}")
