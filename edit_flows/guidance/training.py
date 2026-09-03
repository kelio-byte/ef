"""Training helpers for the first action-level guidance adapter."""

from __future__ import annotations

import torch
from torch import Tensor

from edit_flows.guidance.dgm import positive_guidance_bregman_loss
from edit_flows.guidance.targets import (
    build_action_target_masks,
    make_action_reward_targets,
)
from edit_flows.guidance.ranking import (
    score_calibration_loss,
    shared_anchor_pairwise_loss,
)


_ACTION_TARGET_FIELDS = {
    "terminal": "terminal_tokens",
    "transition": "transition_tokens",
}


def _action_target_field(action_target_source: str) -> str:
    """Resolve the serialized target field for one guidance objective.

    ``terminal`` preserves the original endpoint-alignment objective.  The
    optional ``transition`` mode instead labels only the sampled first Euler
    transition from the shared current state.
    """
    try:
        return _ACTION_TARGET_FIELDS[action_target_source]
    except KeyError as exc:
        choices = ", ".join(sorted(_ACTION_TARGET_FIELDS))
        raise ValueError(
            f"action_target_source must be one of {{{choices}}}, got "
            f"{action_target_source!r}"
        ) from exc


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
    pairwise_loss_weight: float = 0.0,
    pairwise_temperature: float = 1.0,
    pairwise_equal_tolerance: float = 1e-6,
    pairwise_group_size: int = 4,
    pairwise_anchor_rotation: int = 0,
    pairwise_all_anchors: bool = False,
    score_calibration_weight: float = 0.0,
    action_target_source: str = "terminal",
) -> tuple[Tensor, dict[str, float]]:
    """Compute action-specific positive-guidance loss for one padded batch."""
    if (
        background_loss_weight < 0
        or not torch.isfinite(torch.tensor(background_loss_weight))
    ):
        raise ValueError("background_loss_weight must be finite and non-negative")
    if (
        pairwise_loss_weight < 0
        or not torch.isfinite(torch.tensor(pairwise_loss_weight))
    ):
        raise ValueError("pairwise_loss_weight must be finite and non-negative")
    if (
        score_calibration_weight < 0
        or not torch.isfinite(torch.tensor(score_calibration_weight))
    ):
        raise ValueError("score_calibration_weight must be finite and non-negative")
    action_target_field = _action_target_field(action_target_source)
    required = {
        "product_tokens", "state_tokens", "time", "reward", action_target_field,
    }
    missing = sorted(required.difference(batch))
    if missing:
        raise KeyError(f"guidance batch is missing fields: {missing}")
    pairwise_requested = pairwise_loss_weight > 0 or pairwise_all_anchors
    calibration_requested = score_calibration_weight > 0
    if (pairwise_requested or calibration_requested) and "source_index" not in batch:
        raise KeyError("pairwise guidance requires source_index in the batch")
    device = next(model.parameters()).device
    product_tokens = batch["product_tokens"].to(device=device)
    state_tokens = batch["state_tokens"].to(device=device)
    action_target_tokens = batch[action_target_field]
    time_step = batch["time"].to(device=device)
    reward = batch["reward"].to(device=device)
    state_padding = state_tokens == model.pad_token
    product_padding = product_tokens == model.pad_token
    insert_mask, substitute_mask, delete_mask = build_action_target_masks(
        batch["state_tokens"].detach().cpu(),
        action_target_tokens.detach().cpu(),
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
    bregman_loss = (loss_insert + loss_substitute + loss_delete) / 3.0
    pairwise_loss = bregman_loss * 0.0
    pairwise_metrics: dict[str, Tensor] = {}
    if pairwise_requested:
        pairwise_loss, pairwise_metrics = shared_anchor_pairwise_loss(
            (guidance_insert, guidance_substitute, guidance_delete),
            state_tokens,
            action_target_tokens,
            batch["source_index"],
            reward,
            vocab_size=model.vocab_size,
            group_size=pairwise_group_size,
            anchor_rotation=pairwise_anchor_rotation,
            all_anchors=pairwise_all_anchors,
            temperature=pairwise_temperature,
            equal_tolerance=pairwise_equal_tolerance,
            pad_token=model.pad_token,
        )
    calibration_loss = bregman_loss * 0.0
    calibration_metrics: dict[str, Tensor] = {}
    if calibration_requested:
        calibration_loss, calibration_metrics = score_calibration_loss(
            (guidance_insert, guidance_substitute, guidance_delete),
            state_tokens,
            action_target_tokens,
            batch["source_index"],
            reward,
            vocab_size=model.vocab_size,
            group_size=pairwise_group_size,
            anchor_rotation=pairwise_anchor_rotation,
            all_anchors=pairwise_all_anchors,
            equal_tolerance=pairwise_equal_tolerance,
            pad_token=model.pad_token,
        )
    total_loss = (
        bregman_loss
        + pairwise_loss_weight * pairwise_loss
        + score_calibration_weight * calibration_loss
    )
    pairwise_metrics.update(calibration_metrics)
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
            "loss": float(total_loss.item()),
            "loss_total": float(total_loss.item()),
            "loss_bregman": float(bregman_loss.item()),
            "loss_pairwise": float(pairwise_loss.item()),
            "loss_score_calibration": float(calibration_loss.item()),
            "pairwise_loss_weight": float(pairwise_loss_weight),
            "score_calibration_weight": float(score_calibration_weight),
            "pairwise_temperature": float(pairwise_temperature),
            "pairwise_equal_tolerance": float(pairwise_equal_tolerance),
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
        for name, value in pairwise_metrics.items():
            metrics[name] = float(value.item())
    return total_loss, metrics


def train_guidance_step(
    model,
    batch: dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    background: float = 1e-4,
    background_loss_weight: float = 0.01,
    max_grad_norm: float | None = 1.0,
    pairwise_loss_weight: float = 0.0,
    pairwise_temperature: float = 1.0,
    pairwise_equal_tolerance: float = 1e-6,
    pairwise_group_size: int = 4,
    pairwise_anchor_rotation: int = 0,
    pairwise_all_anchors: bool = False,
    score_calibration_weight: float = 0.0,
    action_target_source: str = "terminal",
) -> dict[str, float]:
    """Run one optimizer step and return scalar diagnostics."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = guidance_action_loss(
        model, batch, background=background,
        background_loss_weight=background_loss_weight,
        pairwise_loss_weight=pairwise_loss_weight,
        pairwise_temperature=pairwise_temperature,
        pairwise_equal_tolerance=pairwise_equal_tolerance,
        pairwise_group_size=pairwise_group_size,
        pairwise_anchor_rotation=pairwise_anchor_rotation,
        pairwise_all_anchors=pairwise_all_anchors,
        score_calibration_weight=score_calibration_weight,
        action_target_source=action_target_source,
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
    pairwise_loss_weight: float = 0.0,
    pairwise_temperature: float = 1.0,
    pairwise_equal_tolerance: float = 1e-6,
    pairwise_group_size: int = 4,
    pairwise_anchor_rotation: int = 0,
    pairwise_all_anchors: bool = False,
    score_calibration_weight: float = 0.0,
    action_target_source: str = "terminal",
) -> dict[str, float]:
    """Evaluate guidance loss without changing parameters or RNG state."""
    was_training = model.training
    model.eval()
    _, metrics = guidance_action_loss(
        model, batch, background=background,
        background_loss_weight=background_loss_weight,
        pairwise_loss_weight=pairwise_loss_weight,
        pairwise_temperature=pairwise_temperature,
        pairwise_equal_tolerance=pairwise_equal_tolerance,
        pairwise_group_size=pairwise_group_size,
        pairwise_anchor_rotation=pairwise_anchor_rotation,
        pairwise_all_anchors=pairwise_all_anchors,
        score_calibration_weight=score_calibration_weight,
        action_target_source=action_target_source,
    )
    if was_training:
        model.train()
    return metrics
