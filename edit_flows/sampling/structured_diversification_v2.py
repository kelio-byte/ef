"""Delayed, quality-gated structured diversification.

The original structured sampler forces all first edits at ``t=0`` and spends
the whole output budget on distinct ``(position, operation)`` pairs.  This
module keeps that implementation untouched and adds a smaller intervention:

* run ordinary Euler until the first sampled edit event;
* at that event, choose one highest-probability mode and two modes sampled
  without replacement from a small high-probability pool;
* create three Q-token completions for each INS/SUB mode (DEL gets three
  independent continuation streams);
* continue all nine rows independently with Euler M=1.

The continuation follows the ordinary Euler update path.  Its random draws
are stateless and row-seeded so that repeated DEL actions still have distinct,
reproducible continuation streams.
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
)
from edit_flows.sampling.euler_beam import (
    _mix_child_seed,
    _sample_actions_per_branch,
)
from edit_flows.sampling.ops import apply_ins_del_operations
from edit_flows.sampling.structured_diversification import (
    _OPERATION_NAMES,
    _token_key,
    _valid_token_mask,
)
from edit_flows.utils.tokens import BOS_TOKEN, GAP_TOKEN, PAD_TOKEN, UNK_TOKEN


def _pad_rows(rows: list[Tensor], *, pad_token: int) -> Tensor:
    """Pad a list of ``[1, length]`` rows into one model batch."""
    if not rows:
        raise ValueError("cannot pad an empty row list")
    max_length = max(int(row.shape[1]) for row in rows)
    result = torch.full(
        (len(rows), max_length),
        pad_token,
        dtype=rows[0].dtype,
        device=rows[0].device,
    )
    for row_index, row in enumerate(rows):
        result[row_index, :row.shape[1]] = row[0]
    return result


def _model_output_at_time(
    model,
    x_t: Tensor,
    t: Tensor,
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
    """Match the model/rate path in :func:`sample_euler` for arbitrary t."""
    x_pad_mask = x_t == pad_token
    t_model = _compute_model_time(
        t, scheduler, time_input, train_scheduler,
    )
    log_rates, log_ins_probs, log_sub_probs = model(
        x_t, t_model, x_pad_mask, origin_mask=origin_mask,
    )
    if (
        not use_rate_reparam
        and train_scheduler is not None
        and scheduler.name != train_scheduler.name
    ):
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


def _apply_actions_and_origin(
    x_t: Tensor,
    actions: dict[str, Tensor],
    *,
    max_seq_len: int,
    pad_token: int,
    origin_mask: Optional[Tensor],
) -> tuple[Tensor, Optional[Tensor]]:
    """Apply one Euler action batch and carry the optional origin mask."""
    x_pad_mask = x_t == pad_token
    x_next = x_t.clone()
    x_next[actions["sub_mask"]] = actions["sub_tokens"][actions["sub_mask"]]
    next_origin = None
    if origin_mask is not None:
        origin_markers = torch.where(
            x_pad_mask,
            torch.full_like(x_t, 2),
            origin_mask.long(),
        )
        origin_markers[actions["sub_mask"]] = 0
        origin_markers = apply_ins_del_operations(
            origin_markers,
            actions["ins_mask"],
            actions["del_mask"],
            torch.zeros_like(actions["ins_tokens"]),
            max_seq_len=max_seq_len,
            pad_token=2,
        )
        next_origin = origin_markers == 1
    x_next = apply_ins_del_operations(
        x_next,
        actions["ins_mask"],
        actions["del_mask"],
        actions["ins_tokens"],
        max_seq_len=max_seq_len,
        pad_token=pad_token,
    )
    return x_next, next_origin


def _legal_mode_candidates(
    x_row: Tensor,
    log_rates_row: Tensor,
    step_size: float,
    *,
    max_seq_len: int,
    pad_token: int,
) -> list[dict[str, Any]]:
    """Rank legal modes by their one-step event probability."""
    non_pad = x_row != pad_token
    sequence_length = int(non_pad.sum().item())
    candidates: list[dict[str, Any]] = []
    for position in range(1, x_row.shape[0]):
        if not bool(non_pad[position].item()):
            continue
        for operation in range(3):
            if operation == 0 and sequence_length >= max_seq_len:
                continue
            log_rate = float(log_rates_row[position, operation].item())
            if not math.isfinite(log_rate) or log_rate <= -1e8:
                continue
            rate = max(math.exp(min(log_rate, 50.0)), 0.0)
            if operation == 0:
                log_probability = math.log(
                    max(1.0 - math.exp(-step_size * rate), 1e-30),
                )
            else:
                other_rate = max(
                    math.exp(min(float(log_rates_row[position, 1].item()), 50.0)),
                    0.0,
                ) + max(
                    math.exp(min(float(log_rates_row[position, 2].item()), 50.0)),
                    0.0,
                )
                event_probability = max(
                    1.0 - math.exp(-step_size * other_rate), 1e-30,
                )
                log_probability = math.log(event_probability)
                if other_rate <= 0.0:
                    log_probability = -float("inf")
                else:
                    operation_rate = rate
                    log_probability += math.log(
                        max(operation_rate / other_rate, 1e-30),
                    )
            if not math.isfinite(log_probability):
                continue
            candidates.append({
                "position": position,
                "operation_index": operation,
                "operation": _OPERATION_NAMES[operation],
                "mode_log_rate": log_rate,
                "mode_log_probability": log_probability,
            })
    candidates.sort(
        key=lambda item: (
            -float(item["mode_log_probability"]),
            -float(item["mode_log_rate"]),
            int(item["position"]),
            int(item["operation_index"]),
        ),
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["mode_rank"] = rank
    return candidates


def _select_modes(
    candidates: list[dict[str, Any]],
    *,
    k_mode: int,
    mode_pool_size: int,
    selection_seed: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Select top-1 anchor and weighted high-probability alternatives."""
    if len(candidates) < k_mode:
        raise RuntimeError(
            "delayed structured sampler found fewer legal modes than requested: "
            f"requested {k_mode}, available {len(candidates)}"
        )
    pool_size = min(len(candidates), max(k_mode, mode_pool_size))
    pool = candidates[:pool_size]
    selected = [dict(pool[0])]
    if k_mode == 1:
        selected[0]["mode_selection"] = "anchor_top1"
        return selected
    if selection_seed is not None:
        selection_seed = int(selection_seed) & ((1 << 63) - 1)
    logits = torch.tensor(
        [float(item["mode_log_probability"]) for item in pool[1:]],
        dtype=torch.float64,
    )
    probabilities = torch.softmax(logits - logits.max(), dim=0)
    count = min(k_mode - 1, len(pool) - 1)
    generator = None
    if selection_seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(selection_seed)
    sampled = torch.multinomial(
        probabilities,
        count,
        replacement=False,
        generator=generator,
    )
    for index in sampled.tolist():
        selected.append(dict(pool[1 + int(index)]))
    selected[0]["mode_selection"] = "anchor_top1"
    for item in selected[1:]:
        item["mode_selection"] = "weighted_top_pool_without_replacement"
    return selected


def _completion_tokens(
    log_probs_row: Tensor,
    x_row: Tensor,
    operation: int,
    *,
    k_completion: int,
    pad_token: int,
    bos_token: int,
) -> list[tuple[int, float]]:
    """Return top-Q legal token completions for one selected mode."""
    if operation == 2:
        return [(-1, 0.0) for _ in range(k_completion)]
    valid_tokens = _valid_token_mask(
        log_probs_row.shape[0],
        device=log_probs_row.device,
        forbidden_token_ids=(pad_token, bos_token, GAP_TOKEN, UNK_TOKEN),
    )
    if operation == 1:
        current_token = int(x_row.item())
        if 0 <= current_token < valid_tokens.shape[0]:
            valid_tokens[current_token] = False
    masked = log_probs_row.masked_fill(~valid_tokens, float("-inf"))
    finite_count = int(torch.isfinite(masked).sum().item())
    if finite_count < k_completion:
        raise RuntimeError(
            "delayed structured sampler found fewer legal token completions "
            f"than requested: requested {k_completion}, available {finite_count}"
        )
    values, tokens = torch.topk(masked, k_completion)
    return [
        (int(token.item()), float(value.item()))
        for value, token in zip(values, tokens)
    ]


def _build_branch_actions(
    x_batch: Tensor,
    selected_modes: list[list[dict[str, Any]]],
    log_ins_probs: Tensor,
    log_sub_probs: Tensor,
    *,
    k_completion: int,
    pad_token: int,
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Expand one trigger batch into product-major structured actions."""
    batch_size = x_batch.shape[0]
    n_trajectories = len(selected_modes[0]) * k_completion
    if any(len(modes) != len(selected_modes[0]) for modes in selected_modes):
        raise ValueError("all products must have the same selected mode count")
    x_rows = x_batch.repeat_interleave(n_trajectories, dim=0)
    operation_rows: list[int] = []
    position_rows: list[int] = []
    token_rows: list[int] = []
    records: list[dict[str, Any]] = []
    for batch_index, modes in enumerate(selected_modes):
        for mode in modes:
            operation = int(mode["operation_index"])
            position = int(mode["position"])
            source = (
                log_ins_probs[batch_index, position]
                if operation == 0
                else log_sub_probs[batch_index, position]
            )
            completions = _completion_tokens(
                source,
                x_batch[batch_index, position],
                operation,
                k_completion=k_completion,
                pad_token=pad_token,
                bos_token=BOS_TOKEN,
            )
            for completion_rank, (token, token_log_probability) in enumerate(
                completions, start=1,
            ):
                operation_rows.append(operation)
                position_rows.append(position)
                token_rows.append(token)
                records.append({
                    **mode,
                    "completion_rank": completion_rank,
                    "token": token,
                    "token_log_probability": token_log_probability,
                    "selection": (
                        "top_q_completion"
                        if operation in (0, 1)
                        else "independent_continuation_stream"
                    ),
                })
    if len(records) != x_rows.shape[0]:
        raise RuntimeError(
            "structured branch action count does not match output budget: "
            f"records={len(records)}, rows={x_rows.shape[0]}"
        )
    device = x_batch.device
    position_tensor = torch.tensor(position_rows, dtype=torch.long, device=device)
    operation_tensor = torch.tensor(operation_rows, dtype=torch.long, device=device)
    token_tensor = torch.tensor(token_rows, dtype=torch.long, device=device)
    ins_mask = torch.zeros_like(x_rows, dtype=torch.bool)
    sub_mask = torch.zeros_like(x_rows, dtype=torch.bool)
    del_mask = torch.zeros_like(x_rows, dtype=torch.bool)
    ins_tokens = torch.full_like(x_rows, pad_token)
    sub_tokens = torch.full_like(x_rows, pad_token)
    ins_rows = operation_tensor == 0
    sub_rows = operation_tensor == 1
    del_rows = operation_tensor == 2
    ins_mask[ins_rows, position_tensor[ins_rows]] = True
    sub_mask[sub_rows, position_tensor[sub_rows]] = True
    del_mask[del_rows, position_tensor[del_rows]] = True
    ins_tokens[ins_rows, position_tensor[ins_rows]] = token_tensor[ins_rows]
    sub_tokens[sub_rows, position_tensor[sub_rows]] = token_tensor[sub_rows]
    x_rows[sub_mask] = sub_tokens[sub_mask]
    return x_rows, {
        "ins_mask": ins_mask,
        "sub_mask": sub_mask,
        "del_mask": del_mask,
        "ins_tokens": ins_tokens,
        "sub_tokens": sub_tokens,
        "records": records,
    }


@torch.inference_mode()
def _sample_seeded_euler_m1(
    model,
    x_t: Tensor,
    start_time: Tensor,
    branch_seeds: Tensor,
    scheduler: KappaScheduler,
    *,
    n_steps: int,
    max_seq_len: int,
    pad_token: int,
    bos_token: int,
    use_rate_reparam: bool,
    clamp_kappa: bool,
    clamp_max: float,
    time_input: str,
    train_scheduler: Optional[KappaScheduler],
    event_prob_mode: str,
    initial_origin_mask: Optional[Tensor],
) -> Tensor:
    """Ordinary Euler M=1 continuation with independent row RNG streams."""
    device = next(model.parameters()).device
    x_current = x_t.to(device=device).clone()
    t = start_time.to(device=device, dtype=torch.float32).reshape(-1, 1)
    seeds = branch_seeds.to(device=device, dtype=torch.int64)
    if x_current.shape[0] != t.shape[0] or seeds.shape[0] != t.shape[0]:
        raise ValueError("continuation inputs must have matching batch sizes")
    if initial_origin_mask is not None:
        origin_mask = initial_origin_mask.to(device=device, dtype=torch.bool)
        if origin_mask.shape != x_current.shape:
            raise ValueError("initial_origin_mask must match continuation state")
    else:
        origin_mask = None
    default_h = 1.0 / n_steps
    step = 0
    max_iterations = max(10 * n_steps, 100)
    while bool((t < 1.0).any().item()):
        log_rates, log_ins_probs, log_sub_probs = _model_output_at_time(
            model,
            x_current,
            t,
            scheduler,
            pad_token=pad_token,
            origin_mask=origin_mask,
            use_rate_reparam=use_rate_reparam,
            clamp_kappa=clamp_kappa,
            clamp_max=clamp_max,
            time_input=time_input,
            train_scheduler=train_scheduler,
        )
        adapt_h = get_adaptive_h(default_h, t, scheduler)
        actions = _sample_actions_per_branch(
            seeds,
            x_current,
            log_rates,
            log_ins_probs,
            log_sub_probs,
            adapt_h,
            pad_token=pad_token,
            event_prob_mode=event_prob_mode,
            step=step,
        )
        done = (t >= 1.0).squeeze(-1)
        if bool(done.any().item()):
            actions["ins_mask"][done] = False
            actions["sub_mask"][done] = False
            actions["del_mask"][done] = False
        x_current, origin_mask = _apply_actions_and_origin(
            x_current,
            actions,
            max_seq_len=max_seq_len,
            pad_token=pad_token,
            origin_mask=origin_mask,
        )
        t = t + adapt_h
        step += 1
        if step > max_iterations:
            raise RuntimeError(
                "seeded Euler continuation did not reach t=1 within guard"
            )
    return x_current


def _record_final_duplicate_stats(
    final_states: Tensor,
    records: list[dict[str, Any]],
    *,
    n_trajectories: int,
    pad_token: int,
    bos_token: int,
    sampling_stats: Optional[dict[str, Any]],
) -> None:
    for product_index, record in enumerate(records):
        start = product_index * n_trajectories
        rows = final_states[start:start + n_trajectories]
        keys = [_token_key(row, pad_token, bos_token) for row in rows]
        unique_count = len(set(keys))
        record["final_unique_count"] = unique_count
        record["final_duplicate_rate"] = 1.0 - unique_count / n_trajectories
    total_slots = len(records) * n_trajectories
    total_unique = sum(int(record["final_unique_count"]) for record in records)
    if sampling_stats is not None:
        previous_slots = int(sampling_stats.get("final_slots", 0))
        previous_unique = int(
            sampling_stats.get("final_unique_candidates", 0),
        )
        previous_products = int(sampling_stats.get("products", 0))
        sampling_stats["final_slots"] = previous_slots + total_slots
        sampling_stats["final_unique_candidates"] = (
            previous_unique + total_unique
        )
        sampling_stats["final_duplicate_slots"] = (
            sampling_stats["final_slots"]
            - sampling_stats["final_unique_candidates"]
        )
        sampling_stats["mean_final_unique_per_product"] = (
            sampling_stats["final_unique_candidates"]
            / max(previous_products + len(records), 1)
        )
        sampling_stats["final_duplicate_rate"] = (
            sampling_stats["final_duplicate_slots"]
            / sampling_stats["final_slots"]
            if sampling_stats["final_slots"] else 0.0
        )


@torch.inference_mode()
def sample_delayed_structured_diversification(
    model,
    x_0: Tensor,
    scheduler: KappaScheduler,
    *,
    k_mode: int = 3,
    k_completion: int = 3,
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
    mode_pool_size: Optional[int] = None,
    base_seed: int = 42,
    product_indices: Optional[list[int]] = None,
    action_records: Optional[list[dict[str, Any]]] = None,
    sampling_stats: Optional[dict[str, Any]] = None,
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Run ordinary Euler to the first edit, then do one 3x3 branch.

    The output order is product-major and mode-major, with completion rank as
    the innermost dimension.  For example, rows 0--2 are mode 1's three
    completions and rows 3--5 are mode 2's completions.
    """
    if x_0.ndim != 2 or x_0.shape[0] < 1:
        raise ValueError("x_0 must have shape [batch, length] with batch > 0")
    if k_mode < 1 or k_completion < 1:
        raise ValueError("k_mode and k_completion must be >= 1")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if product_indices is not None and len(product_indices) != x_0.shape[0]:
        raise ValueError("product_indices must have one value per input product")
    if mode_pool_size is None:
        mode_pool_size = 2 * k_mode
    if mode_pool_size < k_mode:
        raise ValueError("mode_pool_size must be >= k_mode")

    device = next(model.parameters()).device
    x_products = x_0.to(device=device).clone()
    batch_size = x_products.shape[0]
    pending_states = [x_products[i:i + 1].clone() for i in range(batch_size)]
    pending_origins = [
        torch.ones_like(state, dtype=torch.bool) if use_origin_mask else None
        for state in pending_states
    ]
    pending_times = [0.0 for _ in range(batch_size)]
    pending_steps = [0 for _ in range(batch_size)]
    pending_indices = list(range(batch_size))
    trigger_payloads: list[Optional[dict[str, Any]]] = [None] * batch_size
    fallback_trigger_count = 0

    while pending_indices:
        x_batch = _pad_rows(
            [pending_states[index] for index in pending_indices],
            pad_token=pad_token,
        )
        if use_origin_mask:
            origin_batch = _pad_rows(
                [pending_origins[index] for index in pending_indices],
                pad_token=0,
            ).bool()
        else:
            origin_batch = None
        t_batch = torch.tensor(
            [pending_times[index] for index in pending_indices],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        log_rates, log_ins_probs, log_sub_probs = _model_output_at_time(
            model,
            x_batch,
            t_batch,
            scheduler,
            pad_token=pad_token,
            origin_mask=origin_batch,
            use_rate_reparam=use_rate_reparam,
            clamp_kappa=clamp_kappa,
            clamp_max=clamp_max,
            time_input=time_input,
            train_scheduler=train_scheduler,
        )
        adapt_h = get_adaptive_h(1.0 / n_steps, t_batch, scheduler)
        trigger_seeds = torch.tensor(
            [
                _mix_child_seed(
                    int(base_seed),
                    int(product_indices[index])
                    if product_indices is not None else index,
                    pending_steps[index] + 1,
                )
                for index in pending_indices
            ],
            dtype=torch.int64,
            device=device,
        )
        normal_actions = _sample_actions_per_branch(
            trigger_seeds,
            x_batch,
            log_rates,
            log_ins_probs,
            log_sub_probs,
            adapt_h,
            pad_token=pad_token,
            event_prob_mode=event_prob_mode,
            step=0,
        )
        event_mask = (
            normal_actions["ins_mask"]
            | normal_actions["sub_mask"]
            | normal_actions["del_mask"]
        )
        next_time = t_batch + adapt_h
        # A no-event path is allowed to reach the end, but it still needs the
        # fixed 9-row output budget.  The final numerical step is the only
        # fallback trigger; normal sampled events trigger earlier.
        trigger_mask = event_mask.any(dim=1) | (next_time.squeeze(1) >= 1.0)
        # Rows that have not triggered still advance through the exact
        # ordinary Euler no-event step, even when another row in this batch
        # triggers.  Otherwise their time would be silently frozen until the
        # next compacted batch.
        if bool((~trigger_mask).any().item()):
            next_x, next_origin = _apply_actions_and_origin(
                x_batch,
                normal_actions,
                max_seq_len=max_seq_len,
                pad_token=pad_token,
                origin_mask=origin_batch,
            )
            for local_index, product_index in enumerate(pending_indices):
                if bool(trigger_mask[local_index].item()):
                    continue
                pending_states[product_index] = (
                    next_x[local_index:local_index + 1].clone()
                )
                if use_origin_mask:
                    pending_origins[product_index] = (
                        next_origin[local_index:local_index + 1].clone()
                    )
                pending_times[product_index] = float(
                    next_time[local_index].item()
                )
                pending_steps[product_index] += 1

        remaining_indices: list[int] = []
        for local_index, product_index in enumerate(pending_indices):
            if not bool(trigger_mask[local_index].item()):
                remaining_indices.append(product_index)
                continue
            if not bool(event_mask[local_index].any().item()):
                fallback_trigger_count += 1
            trigger_payloads[product_index] = {
                "x": x_batch[local_index:local_index + 1].clone(),
                "origin": (
                    origin_batch[local_index:local_index + 1].clone()
                    if origin_batch is not None else None
                ),
                "t": float(t_batch[local_index].item()),
                "h": float(adapt_h[local_index].item()),
                "next_t": float(next_time[local_index].item()),
                "log_rates": log_rates[local_index].clone(),
                "log_ins_probs": log_ins_probs[local_index].clone(),
                "log_sub_probs": log_sub_probs[local_index].clone(),
                "trigger_event_count": int(event_mask[local_index].sum().item()),
                "trigger_event_positions": torch.nonzero(
                    event_mask[local_index], as_tuple=False,
                ).squeeze(-1).tolist(),
                "trigger_step": pending_steps[product_index],
            }
        pending_indices = remaining_indices

    payloads = [payload for payload in trigger_payloads if payload is not None]
    if len(payloads) != batch_size:
        raise RuntimeError("not every product reached a structured trigger")

    selected_modes: list[list[dict[str, Any]]] = []
    branch_input_rows: list[Tensor] = []
    branch_origin_rows: list[Optional[Tensor]] = []
    branch_start_times: list[float] = []
    branch_seeds: list[int] = []
    records: list[dict[str, Any]] = []
    for product_index, payload in enumerate(payloads):
        x_trigger = payload["x"]
        step_size = float(payload["h"])
        candidates = _legal_mode_candidates(
            x_trigger[0],
            payload["log_rates"],
            step_size,
            max_seq_len=max_seq_len,
            pad_token=pad_token,
        )
        modes = _select_modes(
            candidates,
            k_mode=k_mode,
            mode_pool_size=mode_pool_size,
            selection_seed=_mix_child_seed(
                int(base_seed),
                int(product_indices[product_index])
                if product_indices is not None else product_index,
                10,
            ),
        )
        selected_modes.append(modes)
        branch_x, branch_actions = _build_branch_actions(
            x_trigger,
            [modes],
            payload["log_ins_probs"].unsqueeze(0),
            payload["log_sub_probs"].unsqueeze(0),
            k_completion=k_completion,
            pad_token=pad_token,
        )
        branch_x, branch_origin = _apply_actions_and_origin(
            branch_x,
            branch_actions,
            max_seq_len=max_seq_len,
            pad_token=pad_token,
            origin_mask=(
                payload["origin"].repeat_interleave(k_mode * k_completion, dim=0)
                if payload["origin"] is not None else None
            ),
        )
        branch_input_rows.extend(
            [branch_x[row:row + 1].clone() for row in range(branch_x.shape[0])]
        )
        if branch_origin is not None:
            branch_origin_rows.extend(
                [branch_origin[row:row + 1].clone() for row in range(branch_origin.shape[0])]
            )
        branch_start_times.extend(
            [payload["next_t"] for _ in range(k_mode * k_completion)]
        )
        product_id = (
            int(product_indices[product_index])
            if product_indices is not None else product_index
        )
        product_records = []
        for branch_index, item in enumerate(branch_actions["records"]):
            seed = _mix_child_seed(
                int(base_seed),
                product_id,
                branch_index + 1,
            )
            branch_seeds.append(seed)
            product_records.append({
                "trajectory": branch_index + 1,
                "continuation_seed": seed,
                **item,
            })
        records.append({
            "product_index": product_id,
            "n_trajectories": k_mode * k_completion,
            "trigger_t": payload["t"],
            "trigger_next_t": payload["next_t"],
            "trigger_step": payload["trigger_step"],
            "trigger_event_count": payload["trigger_event_count"],
            "trigger_event_positions": payload["trigger_event_positions"],
            "mode_candidate_count": len(candidates),
            "mode_pool_size": min(len(candidates), mode_pool_size),
            "mode_candidates": candidates[:mode_pool_size],
            "selected_actions": product_records,
            "mode_selection_seed": _mix_child_seed(
                int(base_seed),
                product_id,
                10,
            ),
            "unique_mode_count": len({
                (item["position"], item["operation"])
                for item in product_records
            }),
            "unique_first_action_count": len({
                (item["position"], item["operation"], item["token"])
                for item in product_records
            }),
            "first_duplicate_rate": 1.0 - len({
                (item["position"], item["operation"], item["token"])
                for item in product_records
            }) / (k_mode * k_completion),
            "fallback_trigger": payload["trigger_event_count"] == 0,
        })

    branch_batch = _pad_rows(branch_input_rows, pad_token=pad_token)
    branch_origin_batch = (
        _pad_rows([row for row in branch_origin_rows if row is not None], pad_token=0).bool()
        if branch_origin_rows else None
    )
    final_states = _sample_seeded_euler_m1(
        model,
        branch_batch,
        torch.tensor(branch_start_times, dtype=torch.float32, device=device),
        torch.tensor(branch_seeds, dtype=torch.int64, device=device),
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
        initial_origin_mask=branch_origin_batch,
    )
    _record_final_duplicate_stats(
        final_states,
        records,
        n_trajectories=k_mode * k_completion,
        pad_token=pad_token,
        bos_token=bos_token,
        sampling_stats=sampling_stats,
    )
    if action_records is not None:
        action_records.extend(records)
    if sampling_stats is not None:
        previous_products = int(sampling_stats.get("products", 0))
        previous_triggered = int(sampling_stats.get("triggered_products", 0))
        previous_fallbacks = int(
            sampling_stats.get("fallback_trigger_count", 0),
        )
        previous_rank_sum = float(
            sampling_stats.get("selected_mode_rank_sum", 0.0),
        )
        previous_rank_count = int(
            sampling_stats.get("selected_mode_rank_count", 0),
        )
        previous_trigger_t_sum = float(
            sampling_stats.get("trigger_t_sum", 0.0),
        )
        previous_first_dup_sum = float(
            sampling_stats.get("first_duplicate_rate_sum", 0.0),
        )
        previous_histogram = {
            str(key): int(value)
            for key, value in sampling_stats.get(
                "selected_mode_rank_histogram", {},
            ).items()
        }
        all_mode_ranks = [
            int(item["mode_rank"])
            for record in records
            for item in record["selected_actions"][::k_completion]
        ]
        local_triggered = batch_size - fallback_trigger_count
        local_rank_sum = float(sum(all_mode_ranks))
        local_rank_count = len(all_mode_ranks)
        local_trigger_t_sum = sum(record["trigger_t"] for record in records)
        local_first_dup_sum = sum(
            record["first_duplicate_rate"] for record in records
        )
        local_histogram = {
            str(rank): all_mode_ranks.count(rank)
            for rank in sorted(set(all_mode_ranks))
        }
        for rank, count in local_histogram.items():
            previous_histogram[rank] = previous_histogram.get(rank, 0) + count
        total_products = previous_products + batch_size
        total_rank_sum = previous_rank_sum + local_rank_sum
        total_rank_count = previous_rank_count + local_rank_count
        total_trigger_t_sum = previous_trigger_t_sum + local_trigger_t_sum
        total_first_dup_sum = previous_first_dup_sum + local_first_dup_sum
        sampling_stats.update({
            "sampler_version": "delayed_v2",
            "products": total_products,
            "trajectory_count": total_products * k_mode * k_completion,
            "k_mode": k_mode,
            "k_completion": k_completion,
            "mode_pool_size": mode_pool_size,
            "triggered_products": previous_triggered + local_triggered,
            "fallback_trigger_count": previous_fallbacks + fallback_trigger_count,
            "fallback_trigger_rate": (
                (previous_fallbacks + fallback_trigger_count) / total_products
            ),
            "trigger_t_sum": total_trigger_t_sum,
            "mean_trigger_t": total_trigger_t_sum / total_products,
            "min_trigger_t": min(
                float(sampling_stats.get("min_trigger_t", float("inf"))),
                min(record["trigger_t"] for record in records),
            ),
            "max_trigger_t": max(
                float(sampling_stats.get("max_trigger_t", -float("inf"))),
                max(record["trigger_t"] for record in records),
            ),
            "selected_mode_rank_sum": total_rank_sum,
            "selected_mode_rank_count": total_rank_count,
            "mean_selected_mode_rank": total_rank_sum / total_rank_count,
            "max_selected_mode_rank": max(
                int(sampling_stats.get("max_selected_mode_rank", 0)),
                max(all_mode_ranks),
            ),
            "selected_mode_rank_histogram": previous_histogram,
            "first_duplicate_rate_sum": total_first_dup_sum,
            "mean_first_duplicate_rate": total_first_dup_sum / total_products,
            "cross_trajectory_competition": False,
            "continuation": "ordinary_euler_m1_stateless_seeded",
        })
    return final_states, records
