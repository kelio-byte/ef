"""Training helpers for the first action-level guidance adapter."""

from __future__ import annotations

import torch
from torch import Tensor

from edit_flows.guidance.dgm import positive_guidance_bregman_loss
from edit_flows.guidance.targets import (
    build_action_target_masks,
    make_action_reward_targets,
)


def guidance_action_loss(
    model,
    batch: dict[str, Tensor],
    *,
    background: float = 1e-4,
) -> tuple[Tensor, dict[str, float]]:
    """Compute action-specific positive-guidance loss for one padded batch."""
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
    loss_insert = positive_guidance_bregman_loss(
        guidance_insert, target_insert, mask=state_action_mask,
    )
    loss_substitute = positive_guidance_bregman_loss(
        guidance_substitute, target_substitute, mask=state_action_mask,
    )
    loss_delete = positive_guidance_bregman_loss(
        guidance_delete, target_delete, mask=state_action_mask,
    )
    loss = (loss_insert + loss_substitute + loss_delete) / 3.0
    with torch.no_grad():
        selected = (
            insert_mask.any(dim=-1).float().mean()
            + substitute_mask.any(dim=-1).float().mean()
            + delete_mask.any(dim=-1).float().mean()
        ) / 3.0
        metrics = {
            "loss": float(loss.item()),
            "loss_insert": float(loss_insert.item()),
            "loss_substitute": float(loss_substitute.item()),
            "loss_delete": float(loss_delete.item()),
            "reward_mean": float(reward.float().mean().item()),
            "selected_action_fraction": float(selected.item()),
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
    max_grad_norm: float | None = 1.0,
) -> dict[str, float]:
    """Run one optimizer step and return scalar diagnostics."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = guidance_action_loss(
        model, batch, background=background,
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
) -> dict[str, float]:
    """Evaluate guidance loss without changing parameters or RNG state."""
    was_training = model.training
    model.eval()
    _, metrics = guidance_action_loss(
        model, batch, background=background,
    )
    if was_training:
        model.train()
    return metrics
