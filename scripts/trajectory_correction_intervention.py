#!/usr/bin/env python
"""Run controlled first-completion interventions for the correction study."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from edit_flows.analysis.first_step import (
    build_model_batch,
    load_parallel_texts,
    tokenize_smiles,
)
from edit_flows.analysis.trajectory_correction import (
    aggregate_trace_summaries,
    summarize_trace,
    token_edit_distance,
)
from edit_flows.sampling.euler import (
    _oracle_token_support,
    _override_with_anchor_event,
    sample_euler,
)
from edit_flows.sampling.ops import apply_ins_del_operations
from edit_flows.utils.tokens import PAD_TOKEN

from scripts.trajectory_correction_analysis import (
    _first_augmentation,
    _load_checkpoint,
    _load_model,
    _scheduler,
    _sampling_kwargs,
)


def _token_from_model(
    probabilities: torch.Tensor,
    excluded: set[int],
) -> int | None:
    scores = probabilities.detach().clone()
    if excluded:
        scores[list(excluded)] = float("-inf")
    scores = scores.masked_fill(scores <= 0, float("-inf"))
    if not bool(torch.isfinite(scores).any().item()):
        return None
    return int(torch.argmax(scores).item())


def _completion_anchor(
    mode: str,
    sample_idx: int,
    actions: dict,
    oracle: dict,
    oracle_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]],
    oracle_sample_idx: int,
) -> tuple[dict | None, dict]:
    """Select an oracle completion anchor and, for wrong mode, a model token."""
    positions = torch.nonzero(
        oracle["pos_mask"][oracle_sample_idx], as_tuple=False,
    ).squeeze(-1).tolist()
    for position in positions:
        if bool(oracle["ins_mask"][oracle_sample_idx, position].item()):
            action_type = "ins"
            oracle_log_probs = oracle_out[1][oracle_sample_idx, position]
            model_probs = actions["ins_probs"][sample_idx, position]
        elif bool(oracle["sub_mask"][oracle_sample_idx, position].item()):
            action_type = "sub"
            oracle_log_probs = oracle_out[2][oracle_sample_idx, position]
            model_probs = actions["sub_probs"][sample_idx, position]
        else:
            continue

        support = _oracle_token_support(oracle_log_probs)
        if not support:
            continue
        correct_token = int(min(support))
        if mode == "force_correct_completion_first":
            forced_token = correct_token
        elif mode == "force_wrong_completion_first":
            forced_token = _token_from_model(model_probs, set(support))
            if forced_token is None:
                continue
        else:
            raise ValueError(f"unsupported completion intervention mode: {mode}")

        anchor = {
            "pos": int(position),
            "type": action_type,
            "token": int(forced_token),
        }
        info = {
            "mode": mode,
            "applied": True,
            "position": int(position),
            "type": action_type,
            "correct_token": correct_token,
            "forced_token": int(forced_token),
            "is_wrong_completion": int(forced_token) != correct_token,
        }
        return anchor, info

    return None, {
        "mode": mode,
        "applied": False,
        "reason": "no oracle INS/SUB completion with an available wrong token",
    }


def _make_callback(mode: str):
    def callback(
        sample_idx: int,
        actions: dict,
        x_t: torch.Tensor,
        oracle: dict,
        oracle_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]],
        oracle_sample_idx: int,
        target_x: torch.Tensor | None = None,
    ) -> dict:
        del x_t, target_x
        anchor, info = _completion_anchor(
            mode,
            sample_idx,
            actions,
            oracle,
            oracle_out,
            oracle_sample_idx,
        )
        if anchor is not None:
            _override_with_anchor_event(sample_idx, actions, anchor)
        return info

    return callback


def _event_action_specs(actions: dict, sample_idx: int) -> list[dict]:
    """Return the sampled first event as single-action descriptors.

    Replacement is treated as one effective operation.  The new intervention
    protocol only perturbs INS/SUB completion tokens; a replacement-only event
    is therefore kept in the event count but cannot be selected as an anchor.
    """
    ins_mask = actions["ins_mask"][sample_idx]
    del_mask = actions["del_mask"][sample_idx]
    sub_mask = actions["sub_mask"][sample_idx]
    specs = []
    positions = torch.nonzero(
        ins_mask | del_mask | sub_mask, as_tuple=False,
    ).squeeze(-1).tolist()
    for position in positions:
        if bool(sub_mask[position].item()):
            specs.append({
                "position": int(position),
                "type": "sub",
                "token": int(actions["sub_tokens"][sample_idx, position].item()),
            })
        elif bool(ins_mask[position].item()) and bool(del_mask[position].item()):
            specs.append({
                "position": int(position),
                "type": "replace",
                "token": int(actions["ins_tokens"][sample_idx, position].item()),
            })
        elif bool(ins_mask[position].item()):
            specs.append({
                "position": int(position),
                "type": "ins",
                "token": int(actions["ins_tokens"][sample_idx, position].item()),
            })
        else:
            specs.append({
                "position": int(position),
                "type": "del",
                "token": None,
            })
    return specs


def _apply_sample_event(
    x_t: torch.Tensor,
    actions: dict,
    sample_idx: int,
    *,
    position: int | None = None,
    action_type: str | None = None,
    token: int | None = None,
    max_seq_len: int,
) -> torch.Tensor:
    """Apply the sampled event, optionally replacing one completion token."""
    ins_mask = actions["ins_mask"][sample_idx:sample_idx + 1].clone()
    del_mask = actions["del_mask"][sample_idx:sample_idx + 1].clone()
    ins_tokens = actions["ins_tokens"][sample_idx:sample_idx + 1].clone()
    sub_mask = actions["sub_mask"][sample_idx:sample_idx + 1].clone()
    sub_tokens = actions["sub_tokens"][sample_idx:sample_idx + 1].clone()
    if position is not None and token is not None:
        if action_type == "ins":
            ins_tokens[0, position] = int(token)
        elif action_type == "sub":
            sub_tokens[0, position] = int(token)
        else:
            raise ValueError(f"cannot perturb action type {action_type}")
    # Substitution is applied before INS/DEL.  ``apply_ins_del_operations``
    # intentionally handles only insertion, deletion, and replacement (the
    # latter is encoded as INS+DEL); omitting this line would make a SUB event
    # look neutral in the distance check and would invalidate the matched
    # intervention protocol.
    x_next = x_t[sample_idx:sample_idx + 1].clone()
    x_next[sub_mask] = sub_tokens[sub_mask]
    return apply_ins_del_operations(
        x_next,
        ins_mask,
        del_mask,
        ins_tokens,
        max_seq_len=max_seq_len,
    )[0]


def _compact_ids(values: torch.Tensor) -> list[int]:
    """Move one padded state to the compact integer representation."""
    return [
        int(value)
        for value in values.detach().cpu().tolist()
        if int(value) != PAD_TOKEN
    ]


def _apply_event_ids(
    source_ids: list[int],
    specs: list[dict],
    *,
    max_seq_len: int,
    override: tuple[int, str, int] | None = None,
) -> list[int]:
    """Apply one sampled event without a GPU round-trip per candidate.

    ``apply_ins_del_operations`` inserts after the position, deletes the
    position, and represents replacement as INS+DEL at the same position.
    This compact implementation mirrors that ordering for the first-event
    distance check.  ``source_ids`` includes BOS and contains no PAD tokens.
    """
    by_position = {int(spec["position"]): spec for spec in specs}
    if override is not None:
        position, action_type, token = override
        updated = dict(by_position[int(position)])
        updated["token"] = int(token)
        by_position[int(position)] = updated

    output: list[int] = []
    for position, source_token in enumerate(source_ids):
        spec = by_position.get(position)
        if spec is None:
            output.append(source_token)
            continue
        action_type = str(spec["type"])
        if action_type in {"sub", "replace"}:
            output.append(int(spec["token"]))
        elif action_type != "del":
            output.append(source_token)
        if action_type == "ins":
            output.append(int(spec["token"]))
    return output[:max_seq_len]


def _distance_numpy(left: list[int], right: list[int]) -> int:
    """Levenshtein distance for one compact CPU sequence."""
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_token != right_token),
            ))
        previous = current
    return int(previous[-1])


def _distance_tables(
    left: list[int],
    right: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return prefix and suffix Levenshtein tables for one source state."""
    m, n = len(left), len(right)
    prefix = np.zeros((m + 1, n + 1), dtype=np.int64)
    prefix[0, :] = np.arange(n + 1, dtype=np.int64)
    prefix[:, 0] = np.arange(m + 1, dtype=np.int64)
    for i, left_token in enumerate(left, start=1):
        for j, right_token in enumerate(right, start=1):
            prefix[i, j] = min(
                prefix[i, j - 1] + 1,
                prefix[i - 1, j] + 1,
                prefix[i - 1, j - 1] + (left_token != right_token),
            )

    suffix = np.zeros((m + 1, n + 1), dtype=np.int64)
    suffix[m, :] = np.arange(n, -1, -1, dtype=np.int64)
    suffix[:, n] = np.arange(m, -1, -1, dtype=np.int64)
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            suffix[i, j] = min(
                suffix[i, j + 1] + 1,
                suffix[i + 1, j] + 1,
                suffix[i + 1, j + 1] + (left[i] != right[j]),
            )
    return prefix, suffix


def _distance_after_token_change(
    prefix: np.ndarray,
    suffix: np.ndarray,
    target: list[int],
    position: int,
    token: int,
) -> int:
    """Distance after replacing one source token at ``position``."""
    n = len(target)
    # The changed source token can be deleted, or aligned to any target token;
    # all insertions before/after it are represented by the prefix/suffix
    # tables.  This is exact and costs O(len(target)) per candidate.
    delete_case = min(
        int(prefix[position, j] + 1 + suffix[position + 1, j])
        for j in range(n + 1)
    )
    align_case = min(
        (
            int(prefix[position, j])
            + int(token != target[j])
            + int(suffix[position + 1, j + 1])
        )
        for j in range(n)
    ) if n else delete_case
    return min(delete_case, align_case)


def _event_output_positions(
    source_ids: list[int],
    specs: list[dict],
    *,
    max_seq_len: int,
) -> dict[tuple[int, str], int]:
    """Map each INS/SUB action to its output token position."""
    by_position = {int(spec["position"]): spec for spec in specs}
    output_positions: dict[tuple[int, str], int] = {}
    output_length = 0
    for position, source_token in enumerate(source_ids):
        spec = by_position.get(position)
        if spec is None:
            output_length += 1
            continue
        action_type = str(spec["type"])
        if action_type in {"sub", "replace"}:
            output_positions[(position, action_type)] = output_length
            output_length += 1
        elif action_type == "del":
            continue
        else:
            output_length += 1
            output_positions[(position, action_type)] = output_length
            output_length += 1
    return {
        key: position
        for key, position in output_positions.items()
        if position < max_seq_len
    }


def _distances_numpy(
    sequences: list[list[int]],
    target: list[int],
) -> list[int]:
    """Compute distances for many candidate states in one vectorized DP.

    The dynamic-programming recurrence still advances over sequence and target
    positions, but the candidate dimension is handled by NumPy.  This avoids
    thousands of synchronous GPU calls and Python list DPs in the harmful
    intervention search.
    """
    if not sequences:
        return []
    max_left = max(len(sequence) for sequence in sequences)
    max_right = len(target)
    left = np.zeros((len(sequences), max_left), dtype=np.int64)
    lengths = np.zeros(len(sequences), dtype=np.int64)
    for row, sequence in enumerate(sequences):
        lengths[row] = len(sequence)
        if sequence:
            left[row, :len(sequence)] = sequence
    previous = np.broadcast_to(
        np.arange(max_right + 1, dtype=np.int64),
        (len(sequences), max_right + 1),
    ).copy()
    right = np.asarray(target, dtype=np.int64)
    for position in range(max_left):
        current = np.empty_like(previous)
        current[:, 0] = position + 1
        token = left[:, position]
        for target_position in range(max_right):
            substitution = previous[:, target_position] + (
                token != right[target_position]
            )
            current[:, target_position + 1] = np.minimum(
                current[:, target_position] + 1,
                np.minimum(previous[:, target_position + 1] + 1, substitution),
            )
        active = lengths > position
        previous = np.where(active[:, None], current, previous)
    return [int(value) for value in previous[:, -1].tolist()]


def _candidate_tokens(
    probabilities: torch.Tensor,
    original_token: int,
    top_k: int,
) -> list[tuple[int, float]]:
    k = min(int(top_k), int(probabilities.numel()))
    values, indices = torch.topk(probabilities.detach(), k=k)
    candidates = []
    for value, index in zip(values.tolist(), indices.tolist()):
        if value > 0.0 and int(index) != int(original_token):
            candidates.append((int(index), float(value)))
    return candidates


def _make_order_invariant_callback(
    mode: str,
    *,
    max_seq_len: int,
    wrong_top_k: int,
    required_damage: int = 1,
):
    """Create a matched perturbation based on actual distance progress.

    The control event must reduce edit distance by exactly its number of
    effective actions.  The harmful branch changes one INS/SUB token while
    keeping every other action, position and type fixed, and accepts only a
    candidate whose post-event distance is exactly ``required_damage`` worse
    than the control event.
    """
    if mode not in {
        "progress_compatible_first",
        "force_harmful_completion_first",
    }:
        raise ValueError(f"unsupported order-invariant mode: {mode}")

    def callback(
        sample_idx: int,
        actions: dict,
        x_t: torch.Tensor,
        oracle: dict,
        oracle_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]],
        oracle_sample_idx: int,
        target_x: torch.Tensor | None = None,
    ) -> dict:
        del oracle, oracle_out, oracle_sample_idx
        base_info = {
            "mode": mode,
            "applied": False,
            "classification": "order_invariant_distance_progress",
        }
        if target_x is None:
            base_info["reason"] = "target unavailable"
            return base_info

        specs = _event_action_specs(actions, sample_idx)
        n_actions = len(specs)
        if n_actions == 0:
            base_info["reason"] = "no event"
            return base_info
        source_ids = _compact_ids(x_t[sample_idx])
        target_ids = _compact_ids(target_x)
        control_ids = _apply_event_ids(
            source_ids, specs, max_seq_len=max_seq_len,
        )
        source_prefix, source_suffix = _distance_tables(source_ids, target_ids)
        before = int(source_prefix[-1, -1])
        control_prefix, control_suffix = _distance_tables(control_ids, target_ids)
        control_after = int(control_prefix[-1, -1])
        control_delta = before - control_after
        if control_delta != n_actions:
            base_info.update({
                "reason": "sampled event is not full progress",
                "n_actions": n_actions,
                "edit_distance_before": before,
                "control_distance_after": control_after,
                "control_distance_delta": control_delta,
            })
            return base_info

        base_info.update({
            "applied": True,
            "n_actions": n_actions,
            "edit_distance_before": before,
            "control_distance_after": control_after,
            "control_distance_delta": control_delta,
        })
        if mode == "progress_compatible_first":
            return base_info

        output_positions = _event_output_positions(
            source_ids, specs, max_seq_len=max_seq_len,
        )
        candidate_rows = []
        for spec in specs:
            if spec["type"] not in {"ins", "sub"}:
                continue
            output_position = output_positions.get(
                (spec["position"], spec["type"]),
            )
            if output_position is None:
                continue
            probabilities = actions[
                "ins_probs" if spec["type"] == "ins" else "sub_probs"
            ][sample_idx, spec["position"]]
            for candidate, probability in _candidate_tokens(
                probabilities, spec["token"], wrong_top_k,
            ):
                candidate_rows.append({
                    "position": spec["position"],
                    "type": spec["type"],
                    "forced_token": candidate,
                    "probability": probability,
                    "output_position": output_position,
                    "base_token": spec["token"],
                })
        best = None
        for candidate_row in candidate_rows:
            candidate_after = _distance_after_token_change(
                control_prefix,
                control_suffix,
                target_ids,
                candidate_row["output_position"],
                candidate_row["forced_token"],
            )
            damage = candidate_after - control_after
            if damage != required_damage:
                continue
            if best is None or candidate_row["probability"] > best["probability"]:
                best = {
                    "position": candidate_row["position"],
                    "type": candidate_row["type"],
                    "base_token": candidate_row["base_token"],
                    "forced_token": candidate_row["forced_token"],
                    "probability": candidate_row["probability"],
                    "perturbed_distance_after": candidate_after,
                    "damage": damage,
                }
        if best is None:
            base_info["applied"] = False
            base_info["reason"] = "no matched harmful token"
            return base_info

        position = best["position"]
        if best["type"] == "ins":
            actions["ins_tokens"][sample_idx, position] = best["forced_token"]
        else:
            actions["sub_tokens"][sample_idx, position] = best["forced_token"]
        base_info.update(best)
        return base_info

    return callback


def _mode_summary(rows: list[dict], mode: str) -> dict:
    first_rows = [
        row for row in rows
        if row.get("first_event_found")
    ]
    intervention_rows = []
    for row in first_rows:
        trace = row.get("trace", [])
        if trace and trace[0].get("intervention") is not None:
            intervention_rows.append(trace[0]["intervention"])
    applied_rows = [
        row for row, info in zip(
            [row for row in first_rows
             if row.get("trace")
             and row["trace"][0].get("intervention") is not None],
            intervention_rows,
        )
        if info.get("applied")
    ]
    return {
        "mode": mode,
        "n_trajectories": len(rows),
        "n_first_event": len(first_rows),
        "n_intervention_records": len(intervention_rows),
        "n_intervention_applied": len(applied_rows),
        "intervention_coverage_among_first_events": (
            len(applied_rows) / len(first_rows) if first_rows else 0.0
        ),
        "final_hit_rate_all": (
            sum(bool(row.get("final_hit")) for row in rows) / len(rows)
            if rows else 0.0
        ),
        "final_valid_rate_all": (
            sum(bool(row.get("final_valid")) for row in rows) / len(rows)
            if rows else 0.0
        ),
        "forced_final_hit_rate": (
            sum(bool(row.get("final_hit")) for row in applied_rows)
            / len(applied_rows)
            if applied_rows else 0.0
        ),
        "forced_final_valid_rate": (
            sum(bool(row.get("final_valid")) for row in applied_rows)
            / len(applied_rows)
            if applied_rows else 0.0
        ),
    }


def run_analysis(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    checkpoint = _load_checkpoint(args.checkpoint, device)
    model, id2token, cfg = _load_model(
        checkpoint, device, args.vocab_file,
    )
    products, targets = load_parallel_texts(
        args.products_file, args.targets_file,
    )
    products, targets, reaction_indices = _first_augmentation(
        products,
        targets,
        augmentation=args.augmentation,
        max_reactions=args.max_reactions,
    )
    token2id = {token: index for index, token in id2token.items()}
    product_ids = [tokenize_smiles(value, token2id) for value in products]
    target_ids = [tokenize_smiles(value, token2id) for value in targets]

    scheduler_name = args.scheduler or cfg.get(
        "sample_scheduler", cfg.get("scheduler", "cubic")
    )
    scheduler = _scheduler(scheduler_name)
    train_scheduler = _scheduler(cfg.get("scheduler", "cubic"))
    if args.mode in {
        "progress_compatible_first",
        "force_harmful_completion_first",
    }:
        callback = _make_order_invariant_callback(
            args.mode,
            max_seq_len=int(cfg.get("max_seq_len", 512)),
            wrong_top_k=args.wrong_top_k,
        )
    elif args.mode != "normal":
        callback = _make_callback(args.mode)
    else:
        callback = None
    rows: list[dict] = []

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    for start in range(0, len(products), args.batch_size):
        end = min(start + args.batch_size, len(products))
        x_0, x_1 = build_model_batch(
            product_ids[start:end], target_ids[start:end],
        )
        batch_x0 = x_0.repeat_interleave(args.n_samples, dim=0).to(device)
        batch_x1 = x_1.repeat_interleave(args.n_samples, dim=0).to(device)
        kwargs = _sampling_kwargs(model, cfg, scheduler, train_scheduler, args)
        kwargs["x_1"] = batch_x1
        final, _, traces = sample_euler(
            model,
            batch_x0,
            record_compact_events=True,
            first_event_intervention=callback,
            **kwargs,
        )
        final_cpu = final.detach().cpu()
        batch_size = x_0.shape[0]
        for local_index in range(batch_size):
            for path_index in range(args.n_samples):
                row_index = local_index * args.n_samples + path_index
                trace = traces[row_index]
                row = summarize_trace(
                    trace,
                    final_cpu[row_index].tolist(),
                    target_ids[start + local_index],
                    id2token,
                )
                rows.append({
                    "reaction_index": reaction_indices[start + local_index],
                    "path_index": path_index,
                    "product": products[start + local_index],
                    "target": targets[start + local_index],
                    "final_ids": final_cpu[row_index].tolist(),
                    "trace": trace,
                    **row,
                })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _mode_summary(rows, args.mode)
    summary.update({
        "checkpoint": os.path.abspath(args.checkpoint),
        "products_file": os.path.abspath(args.products_file),
        "targets_file": os.path.abspath(args.targets_file),
        "mode": args.mode,
        "scheduler": scheduler_name,
        "n_steps": args.n_steps,
        "n_samples": args.n_samples,
        "augmentation": args.augmentation,
        "n_reactions": len(products),
        "seed": args.seed,
    })
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (output_dir / "per_trajectory.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--products_file", required=True)
    parser.add_argument("--targets_file", required=True)
    parser.add_argument("--vocab_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "normal",
            "force_correct_completion_first",
            "force_wrong_completion_first",
            "progress_compatible_first",
            "force_harmful_completion_first",
        ],
        default="normal",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scheduler", choices=["cubic", "linear"], default=None)
    parser.add_argument("--event_prob_mode", choices=["poisson", "linear"], default="poisson")
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--max_reactions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wrong_top_k",
        type=int,
        default=32,
        help="top model tokens searched for a matched harmful perturbation",
    )
    args = parser.parse_args()
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
