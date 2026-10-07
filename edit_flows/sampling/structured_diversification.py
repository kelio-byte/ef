"""Structured first-edit diversification for retrosynthesis sampling.

This module deliberately does not modify ordinary Euler or Euler-Beam.  It
creates a fixed number of distinct atomic first edits from one model forward,
then continues every resulting state independently with ordinary Euler.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch import Tensor

from edit_flows.core.rate_scale import apply_rate_parameterization, get_rate_scale
from edit_flows.core.scheduler import KappaScheduler
from edit_flows.sampling.euler import (
    _compute_model_time,
    get_adaptive_h,
    sample_euler,
)
from edit_flows.sampling.ops import apply_ins_del_operations
from edit_flows.utils.tokens import BOS_TOKEN, GAP_TOKEN, PAD_TOKEN, UNK_TOKEN


_OPERATION_NAMES = ("INS", "SUB", "DEL")


def _valid_token_mask(
    vocab_size: int,
    *,
    device: torch.device,
    forbidden_token_ids: tuple[int, ...],
) -> Tensor:
    mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
    for token_id in forbidden_token_ids:
        if 0 <= int(token_id) < vocab_size:
            mask[int(token_id)] = False
    return mask


def _select_q_token(
    log_probs: Tensor,
    valid_mask: Tensor,
    *,
    token_selection: str,
) -> tuple[int, float]:
    """Select one legal token and return ``(token_id, log_q)``."""
    masked = log_probs.masked_fill(~valid_mask, float("-inf"))
    if not torch.isfinite(masked).any():
        raise ValueError("structured proposal has no legal token")
    if token_selection == "argmax":
        token = int(torch.argmax(masked).item())
    elif token_selection == "sample":
        token = int(torch.multinomial(torch.softmax(masked, dim=-1), 1).item())
    else:
        raise ValueError(
            "token_selection must be 'argmax' or 'sample', "
            f"got {token_selection!r}"
        )
    return token, float(masked[token].item())


def _token_key(row: Tensor, pad_token: int, bos_token: int) -> tuple[int, ...]:
    return tuple(
        int(token)
        for token in row.detach().cpu().tolist()
        if token not in (pad_token, bos_token)
    )


def _prepare_structured_model_output(
    model,
    x_0: Tensor,
    scheduler: KappaScheduler,
    *,
    pad_token: int,
    origin_mask: Optional[Tensor],
    use_rate_reparam: bool,
    clamp_kappa: bool,
    clamp_max: float,
    time_input: str,
    train_scheduler: Optional[KappaScheduler],
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the same model/time/rate path used by ordinary Euler at ``t=0``."""
    device = next(model.parameters()).device
    batch_size = x_0.shape[0]
    t = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
    x_current = x_0.to(device=device)
    padding_mask = x_current == pad_token
    t_model = _compute_model_time(
        t, scheduler, time_input, train_scheduler,
    )
    log_rates, log_ins_probs, log_sub_probs = model(
        x_current, t_model, padding_mask, origin_mask=origin_mask,
    )
    if not use_rate_reparam and train_scheduler is not None \
            and scheduler.name != train_scheduler.name:
        k_sample = get_rate_scale(
            t, scheduler, clamp_kappa=clamp_kappa, clamp_max=clamp_max,
        )
        k_train = get_rate_scale(
            t_model, train_scheduler,
            clamp_kappa=clamp_kappa, clamp_max=clamp_max,
        )
        log_rates = log_rates + torch.log(
            k_sample / k_train.clamp_min(1e-2),
        ).unsqueeze(1)
    log_rates = apply_rate_parameterization(
        log_rates,
        t,
        scheduler,
        use_rate_reparam=use_rate_reparam,
        clamp_kappa=clamp_kappa,
        clamp_max=clamp_max,
    )
    return log_rates, log_ins_probs, log_sub_probs


def _rank_direction_candidates(
    x_row: Tensor,
    log_rates_row: Tensor,
    *,
    max_seq_len: int,
    pad_token: int,
    bos_token: int,
) -> list[tuple[float, int, int]]:
    """Return legal ``(log_rate, position, operation_index)`` candidates."""
    non_pad = x_row != pad_token
    sequence_length = int(non_pad.sum().item())
    direction_candidates: list[tuple[float, int, int]] = []
    for position in range(x_row.shape[0]):
        if not bool(non_pad[position].item()):
            continue
        for operation in range(3):
            # Position 0 is BOS: it is a legal anchor for insertion after
            # BOS, but never a molecule token that may be substituted/deleted.
            if position == 0 and operation != 0:
                continue
            if operation == 0 and sequence_length >= max_seq_len:
                continue
            score = float(log_rates_row[position, operation].item())
            if not math.isfinite(score) or score <= -1e8:
                continue
            direction_candidates.append((score, position, operation))
    # Stable tie breaks are intentional: they make argmax structured sampling
    # independent of Python hash ordering and easy to reproduce.
    direction_candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return direction_candidates


def _rank_concrete_fallbacks(
    x_row: Tensor,
    log_rates_row: Tensor,
    log_ins_row: Tensor,
    log_sub_row: Tensor,
    selected_actions: set[tuple[int, int, int]],
    *,
    max_seq_len: int,
    pad_token: int,
    forbidden_token_ids: tuple[int, ...],
) -> list[tuple[float, int, int, int, float]]:
    """Build extra concrete actions only when unique direction pairs are scarce."""
    non_pad = x_row != pad_token
    sequence_length = int(non_pad.sum().item())
    vocab_size = log_ins_row.shape[-1]
    valid_tokens = _valid_token_mask(
        vocab_size,
        device=log_ins_row.device,
        forbidden_token_ids=forbidden_token_ids,
    )
    candidates: list[tuple[float, int, int, int, float]] = []
    for position in range(x_row.shape[0]):
        if not bool(non_pad[position].item()):
            continue
        for operation in (0, 1):
            if position == 0 and operation != 0:
                continue
            if operation == 0:
                if sequence_length >= max_seq_len:
                    continue
                log_q = log_ins_row[position].masked_fill(
                    ~valid_tokens, float("-inf"),
                )
            else:
                sub_valid = valid_tokens.clone()
                current_token = int(x_row[position].item())
                if 0 <= current_token < vocab_size:
                    sub_valid[current_token] = False
                log_q = log_sub_row[position].masked_fill(
                    ~sub_valid, float("-inf"),
                )
            if not torch.isfinite(log_q).any():
                continue
            top_count = min(16, int(torch.isfinite(log_q).sum().item()))
            top_values, top_tokens = torch.topk(log_q, top_count)
            for token_score, token_tensor in zip(top_values, top_tokens):
                token = int(token_tensor.item())
                action_key = (position, operation, token)
                if action_key in selected_actions:
                    continue
                score = float(log_rates_row[position, operation].item()) \
                    + float(token_score.item())
                candidates.append(
                    (score, position, operation, token,
                     float(token_score.item()))
                )
        if position != 0:
            operation = 2
            score = float(log_rates_row[position, operation].item())
            if math.isfinite(score) and score > -1e8:
                action_key = (position, operation, -1)
                if action_key not in selected_actions:
                    candidates.append((score, position, operation, -1, 0.0))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    return candidates


@torch.inference_mode()
def sample_structured_diversification(
    model,
    x_0: Tensor,
    scheduler: KappaScheduler,
    *,
    n_trajectories: int = 9,
    n_steps: int = 100,
    max_seq_len: int = 512,
    pad_token: int = PAD_TOKEN,
    bos_token: int = BOS_TOKEN,
    use_rate_reparam: bool = False,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
    time_input: str = "t",
    train_scheduler: Optional[KappaScheduler] = None,
    event_prob_mode: str = "poisson",
    use_origin_mask: bool = False,
    token_selection: str = "argmax",
    product_indices: Optional[list[int]] = None,
    action_records: Optional[list[dict[str, Any]]] = None,
    sampling_stats: Optional[dict[str, Any]] = None,
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Create distinct first-edit trajectories, then continue with Euler M=1.

    The first structured edit is selected from the current model's rate heads:
    one candidate per ``(position, operation)`` is selected in descending
    ``log(lambda)`` order.  INS/SUB use the highest-probability legal token
    from their corresponding Q head by default.  If a very short sequence has
    fewer than ``n_trajectories`` direction pairs, distinct concrete tokens are
    used as a documented fallback to preserve the final output budget.

    Returns ``(final_states, records)``.  ``final_states`` is product-major
    with ``n_trajectories`` rows per input product.
    """
    if x_0.ndim != 2 or x_0.shape[0] < 1:
        raise ValueError("x_0 must have shape [batch, length] with batch > 0")
    if n_trajectories < 1:
        raise ValueError("n_trajectories must be >= 1")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if token_selection not in {"argmax", "sample"}:
        raise ValueError(
            "token_selection must be 'argmax' or 'sample', "
            f"got {token_selection!r}"
        )
    if product_indices is not None and len(product_indices) != x_0.shape[0]:
        raise ValueError("product_indices must have one value per input product")

    device = next(model.parameters()).device
    x_products = x_0.to(device=device).clone()
    initial_origin_mask = (
        torch.ones_like(x_products, dtype=torch.bool, device=device)
        if use_origin_mask else None
    )
    log_rates, log_ins_probs, log_sub_probs = (
        _prepare_structured_model_output(
            model,
            x_products,
            scheduler,
            pad_token=pad_token,
            origin_mask=initial_origin_mask,
            use_rate_reparam=use_rate_reparam,
            clamp_kappa=clamp_kappa,
            clamp_max=clamp_max,
            time_input=time_input,
            train_scheduler=train_scheduler,
        )
    )

    forbidden_token_ids = (pad_token, bos_token, GAP_TOKEN, UNK_TOKEN)
    batch_size = x_products.shape[0]
    selected_per_product: list[list[dict[str, Any]]] = []
    first_positions: list[int] = []
    first_operations: list[int] = []
    first_tokens: list[int] = []

    for batch_index in range(batch_size):
        x_row = x_products[batch_index]
        direction_candidates = _rank_direction_candidates(
            x_row,
            log_rates[batch_index],
            max_seq_len=max_seq_len,
            pad_token=pad_token,
            bos_token=bos_token,
        )
        selected_pairs: set[tuple[int, int]] = set()
        selected_actions: set[tuple[int, int, int]] = set()
        selected: list[dict[str, Any]] = []

        for log_rate, position, operation in direction_candidates:
            if len(selected) >= n_trajectories:
                break
            direction_key = (position, operation)
            if direction_key in selected_pairs:
                continue
            if operation == 2:
                token = -1
                log_q = 0.0
            else:
                valid_tokens = _valid_token_mask(
                    log_ins_probs.shape[-1],
                    device=device,
                    forbidden_token_ids=forbidden_token_ids,
                )
                if operation == 1:
                    current_token = int(x_row[position].item())
                    if 0 <= current_token < valid_tokens.shape[0]:
                        valid_tokens[current_token] = False
                source = (
                    log_ins_probs[batch_index, position]
                    if operation == 0
                    else log_sub_probs[batch_index, position]
                )
                try:
                    token, log_q = _select_q_token(
                        source,
                        valid_tokens,
                        token_selection=token_selection,
                    )
                except ValueError:
                    continue
            action_key = (position, operation, token)
            selected_pairs.add(direction_key)
            selected_actions.add(action_key)
            selected.append({
                "trajectory": len(selected) + 1,
                "position": position,
                "operation": _OPERATION_NAMES[operation],
                "token": token,
                "direction_log_rate": log_rate,
                "token_log_probability": log_q,
                "selection_mode": "unique_direction",
            })

        if len(selected) < n_trajectories:
            fallbacks = _rank_concrete_fallbacks(
                x_row,
                log_rates[batch_index],
                log_ins_probs[batch_index],
                log_sub_probs[batch_index],
                selected_actions,
                max_seq_len=max_seq_len,
                pad_token=pad_token,
                forbidden_token_ids=forbidden_token_ids,
            )
            for score, position, operation, token, log_q in fallbacks:
                if len(selected) >= n_trajectories:
                    break
                action_key = (position, operation, token)
                if action_key in selected_actions:
                    continue
                selected_actions.add(action_key)
                selected.append({
                    "trajectory": len(selected) + 1,
                    "position": position,
                    "operation": _OPERATION_NAMES[operation],
                    "token": token,
                    "direction_log_rate": float(
                        log_rates[batch_index, position, operation].item()
                    ),
                    "token_log_probability": log_q,
                    "concrete_action_log_score": score,
                    "selection_mode": "concrete_fallback",
                })

        if len(selected) < n_trajectories:
            raise RuntimeError(
                "structured proposal could not produce the requested output "
                f"budget: product batch index {batch_index}, "
                f"requested {n_trajectories}, got {len(selected)}"
            )

        for item in selected:
            operation = _OPERATION_NAMES.index(item["operation"])
            first_positions.append(int(item["position"]))
            first_operations.append(operation)
            first_tokens.append(int(item["token"]))
        selected_per_product.append(selected)

    n_rows = batch_size * n_trajectories
    x_first = x_products.repeat_interleave(n_trajectories, dim=0)
    first_position_tensor = torch.tensor(
        first_positions, dtype=torch.long, device=device,
    )
    first_operation_tensor = torch.tensor(
        first_operations, dtype=torch.long, device=device,
    )
    first_token_tensor = torch.tensor(
        first_tokens, dtype=torch.long, device=device,
    )
    ins_mask = torch.zeros_like(x_first, dtype=torch.bool)
    sub_mask = torch.zeros_like(x_first, dtype=torch.bool)
    del_mask = torch.zeros_like(x_first, dtype=torch.bool)
    ins_tokens = torch.full_like(x_first, pad_token)
    sub_tokens = torch.full_like(x_first, pad_token)
    ins_rows = first_operation_tensor == 0
    sub_rows = first_operation_tensor == 1
    del_rows = first_operation_tensor == 2
    ins_mask[ins_rows, first_position_tensor[ins_rows]] = True
    sub_mask[sub_rows, first_position_tensor[sub_rows]] = True
    del_mask[del_rows, first_position_tensor[del_rows]] = True
    ins_tokens[ins_rows, first_position_tensor[ins_rows]] = (
        first_token_tensor[ins_rows]
    )
    sub_tokens[sub_rows, first_position_tensor[sub_rows]] = (
        first_token_tensor[sub_rows]
    )
    x_first[sub_mask] = sub_tokens[sub_mask]
    x_first = apply_ins_del_operations(
        x_first,
        ins_mask,
        del_mask,
        ins_tokens,
        max_seq_len=max_seq_len,
        pad_token=pad_token,
    )
    continuation_origin_mask = None
    if use_origin_mask:
        origin_markers = torch.where(
            x_products.repeat_interleave(n_trajectories, dim=0) == pad_token,
            torch.full_like(
                x_products.repeat_interleave(n_trajectories, dim=0), 2,
            ),
            torch.ones_like(
                x_products.repeat_interleave(n_trajectories, dim=0),
            ),
        )
        origin_markers[sub_mask] = 0
        origin_ins_tokens = torch.zeros_like(ins_tokens)
        continuation_origin_markers = apply_ins_del_operations(
            origin_markers,
            ins_mask,
            del_mask,
            origin_ins_tokens,
            max_seq_len=max_seq_len,
            pad_token=2,
        )
        continuation_origin_mask = continuation_origin_markers == 1
    first_step = get_adaptive_h(
        1.0 / n_steps,
        torch.zeros(n_rows, 1, dtype=torch.float32, device=device),
        scheduler,
    )
    start_time = first_step.clamp_max(1.0)

    # All cross-trajectory competition ends here.  The continuation is one
    # ordinary Euler M=1 rollout over the complete batch.
    final_states, _ = sample_euler(
        model,
        x_first,
        scheduler,
        n_steps=n_steps,
        max_seq_len=max_seq_len,
        pad_token=pad_token,
        bos_token=bos_token,
        use_rate_reparam=use_rate_reparam,
        clamp_kappa=clamp_kappa,
        clamp_max=clamp_max,
        time_input=time_input,
        train_scheduler=train_scheduler,
        event_prob_mode=event_prob_mode,
        use_origin_mask=use_origin_mask,
        initial_origin_mask=continuation_origin_mask,
        start_time=start_time,
    )

    records: list[dict[str, Any]] = []
    for batch_index, selected in enumerate(selected_per_product):
        final_start = batch_index * n_trajectories
        final_rows = final_states[final_start:final_start + n_trajectories]
        final_keys = [
            _token_key(row, pad_token, bos_token) for row in final_rows
        ]
        direction_keys = [
            (int(item["position"]), str(item["operation"]))
            for item in selected
        ]
        action_keys = [
            (
                int(item["position"]),
                str(item["operation"]),
                int(item["token"]),
            )
            for item in selected
        ]
        unique_direction_count = len(set(direction_keys))
        unique_action_count = len(set(action_keys))
        final_unique_count = len(set(final_keys))
        record = {
            "product_index": (
                int(product_indices[batch_index])
                if product_indices is not None else batch_index
            ),
            "n_trajectories": n_trajectories,
            "selected_actions": selected,
            "unique_direction_count": unique_direction_count,
            "direction_duplicate_rate": 1.0 - (
                unique_direction_count / n_trajectories
            ),
            "unique_action_count": unique_action_count,
            "action_duplicate_rate": 1.0 - (
                unique_action_count / n_trajectories
            ),
            "final_unique_count": final_unique_count,
            "final_duplicate_rate": 1.0 - (
                final_unique_count / n_trajectories
            ),
        }
        records.append(record)

    if action_records is not None:
        action_records.extend(records)
    if sampling_stats is not None:
        sampling_stats["products"] = (
            int(sampling_stats.get("products", 0)) + batch_size
        )
        sampling_stats["trajectory_count"] = (
            int(sampling_stats.get("trajectory_count", 0)) + n_rows
        )
        sampling_stats["selected_direction_pairs"] = (
            int(sampling_stats.get("selected_direction_pairs", 0))
            + sum(len(set((a["position"], a["operation"]) for a in r["selected_actions"]))
                  for r in records)
        )
        sampling_stats["selected_direction_duplicate_slots"] = (
            int(sampling_stats.get("selected_direction_duplicate_slots", 0))
            + sum(
                n_trajectories - r["unique_direction_count"] for r in records
            )
        )
        sampling_stats["selected_action_duplicate_slots"] = (
            int(sampling_stats.get("selected_action_duplicate_slots", 0))
            + sum(n_trajectories - r["unique_action_count"] for r in records)
        )
        sampling_stats["final_unique_candidates"] = (
            int(sampling_stats.get("final_unique_candidates", 0))
            + sum(r["final_unique_count"] for r in records)
        )
        sampling_stats["final_duplicate_slots"] = (
            int(sampling_stats.get("final_duplicate_slots", 0))
            + sum(n_trajectories - r["final_unique_count"] for r in records)
        )
        sampling_stats["final_slots"] = (
            int(sampling_stats.get("final_slots", 0)) + n_rows
        )
        sampling_stats["mean_final_unique_per_product"] = (
            sampling_stats["final_unique_candidates"]
            / sampling_stats["products"]
        )
        sampling_stats["final_duplicate_rate"] = (
            sampling_stats["final_duplicate_slots"]
            / sampling_stats["final_slots"]
        )

    return final_states, records
