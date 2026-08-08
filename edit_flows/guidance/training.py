"""Training helpers for the first action-level guidance adapter."""

from __future__ import annotations

import torch
from torch import Tensor

from edit_flows.guidance.dgm import positive_guidance_bregman_loss
from edit_flows.guidance.targets import (
    build_action_target_masks,
    make_action_reward_targets,
)


def _safe_pearson_correlation(left: Tensor, right: Tensor) -> Tensor:
    """Return a finite batch Pearson correlation, or zero if undefined."""
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("correlation inputs must be equal-shaped rank-1 tensors")
    if left.numel() < 2:
        return left.new_zeros(())
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = torch.sqrt(
        left_centered.square().sum() * right_centered.square().sum(),
    )
    numerator = (left_centered * right_centered).sum()
    return torch.where(
        denominator > 1e-12,
        numerator / denominator,
        left.new_zeros(()),
    )


def guidance_action_loss(
    model,
    batch: dict[str, Tensor],
    *,
    background: float = 1e-4,
    background_loss_weight: float = 0.01,
) -> tuple[Tensor, dict[str, float]]:
    """Compute action-specific positive-guidance loss for one padded batch."""
    if (
        background_loss_weight < 0
        or not torch.isfinite(torch.tensor(background_loss_weight))
    ):
        raise ValueError("background_loss_weight must be finite and non-negative")
    required = {
        "product_tokens", "state_tokens", "terminal_tokens", "time", "reward",
    }
    missing = sorted(required.difference(batch))
    if missing:
        raise KeyError(f"guidance batch is missing fields: {missing}")
    device = next(model.parameters()).device
    product_tokens = batch["product_tokens"].to(device=device)
    state_tokens = batch["state_tokens"].to(device=device)
    terminal_tokens = batch["terminal_tokens"]
    time_step = batch["time"].to(device=device)
    reward = batch["reward"].to(device=device)
    state_padding = state_tokens == model.pad_token
    product_padding = product_tokens == model.pad_token
    insert_mask, substitute_mask, delete_mask = build_action_target_masks(
        batch["state_tokens"].detach().cpu(),
        terminal_tokens.detach().cpu(),
        vocab_size=model.vocab_size,
        pad_token=model.pad_token,
    )
    insert_mask = insert_mask.to(device=device)
    substitute_mask = substitute_mask.to(device=device)
    delete_mask = delete_mask.to(device=device)
    target_insert, target_substitute, target_delete = make_action_reward_targets(
        insert_mask,
        substitute_mask,
        delete_mask,
        reward,
        background=background,
    )
    guidance_insert, guidance_substitute, guidance_delete = model(
        product_tokens,
        state_tokens,
        time_step,
        product_padding,
        state_padding,
    )
    state_action_mask = (~state_padding).unsqueeze(-1)

    def balanced_loss(
        guidance: Tensor,
        target: Tensor,
        selected_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        valid_mask = state_action_mask.expand_as(guidance)
        selected_mask = selected_mask & valid_mask
        background_mask = valid_mask & ~selected_mask
        pointwise = positive_guidance_bregman_loss(
            guidance, target, reduction="none",
        )
        selected_values = pointwise.masked_select(selected_mask)
        background_values = pointwise.masked_select(background_mask)
        selected_loss = (
            selected_values.mean()
            if selected_values.numel() else pointwise.new_zeros(())
        )
        background_loss = (
            background_values.mean()
            if background_values.numel() else pointwise.new_zeros(())
        )
        total = selected_loss + background_loss_weight * background_loss
        if selected_values.numel() == 0:
            total = background_loss
        return total, selected_loss, background_loss

    loss_insert, selected_insert_loss, background_insert_loss = balanced_loss(
        guidance_insert, target_insert, insert_mask,
    )
    loss_substitute, selected_substitute_loss, background_substitute_loss = balanced_loss(
        guidance_substitute, target_substitute, substitute_mask,
    )
    loss_delete, selected_delete_loss, background_delete_loss = balanced_loss(
        guidance_delete, target_delete, delete_mask,
    )
    loss = (loss_insert + loss_substitute + loss_delete) / 3.0
    with torch.no_grad():
        selected = (
            insert_mask.any(dim=-1).float().mean()
            + substitute_mask.any(dim=-1).float().mean()
            + delete_mask.any(dim=-1).float().mean()
        ) / 3.0
        total_selected = torch.stack([
            mask.sum(dim=tuple(range(1, mask.ndim)))
            for mask in (insert_mask, substitute_mask, delete_mask)
        ]).sum(dim=0)
        selected_guidance = torch.stack([
            (guidance * mask.to(dtype=guidance.dtype)).sum(
                dim=tuple(range(1, guidance.ndim)),
            )
            for guidance, mask in (
                (guidance_insert, insert_mask),
                (guidance_substitute, substitute_mask),
                (guidance_delete, delete_mask),
            )
        ]).sum(dim=0) / total_selected.clamp_min(1).to(
            dtype=guidance_insert.dtype,
        )
        selected_rows = total_selected > 0
        corr = _safe_pearson_correlation(
            reward[selected_rows],
            selected_guidance[selected_rows],
        )
        metrics = {
            "loss": float(loss.item()),
            "loss_insert": float(loss_insert.item()),
            "loss_substitute": float(loss_substitute.item()),
            "loss_delete": float(loss_delete.item()),
            "loss_insert_selected": float(selected_insert_loss.item()),
            "loss_substitute_selected": float(selected_substitute_loss.item()),
            "loss_delete_selected": float(selected_delete_loss.item()),
            "loss_insert_background": float(background_insert_loss.item()),
            "loss_substitute_background": float(background_substitute_loss.item()),
            "loss_delete_background": float(background_delete_loss.item()),
            "background_loss_weight": float(background_loss_weight),
            "reward_mean": float(reward.float().mean().item()),
            "selected_action_fraction": float(selected.item()),
            "selected_row_fraction": float(selected_rows.float().mean().item()),
            "selected_guidance_mean": float(
                selected_guidance[selected_rows].mean().item()
                if selected_rows.any() else 0.0
            ),
            "reward_selected_guidance_corr": float(corr.item()),
            "guidance_insert_mean": float(guidance_insert.mean().item()),
            "guidance_substitute_mean": float(guidance_substitute.mean().item()),
            "guidance_delete_mean": float(guidance_delete.mean().item()),
        }
    return loss, metrics


def train_guidance_step(
    model,
    batch: dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    background: float = 1e-4,
    background_loss_weight: float = 0.01,
    max_grad_norm: float | None = 1.0,
) -> dict[str, float]:
    """Run one optimizer step and return scalar diagnostics."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = guidance_action_loss(
        model, batch, background=background,
        background_loss_weight=background_loss_weight,
    )
    loss.backward()
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm,
        )
        metrics["grad_norm"] = float(grad_norm.item())
    optimizer.step()
    return metrics


@torch.no_grad()
def evaluate_guidance_step(
    model,
    batch: dict[str, Tensor],
    *,
    background: float = 1e-4,
    background_loss_weight: float = 0.01,
) -> dict[str, float]:
    """Evaluate guidance loss without changing parameters or RNG state."""
    was_training = model.training
    model.eval()
    _, metrics = guidance_action_loss(
        model, batch, background=background,
        background_loss_weight=background_loss_weight,
    )
    if was_training:
        model.train()
    return metrics
