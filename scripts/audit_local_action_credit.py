#!/usr/bin/env python
"""Audit whether recorded first Euler transitions provide usable local credit.

This read-only diagnostic consumes shared-anchor guidance records that contain
``transition_tokens``.  It compares the action masks from the common current
state to (a) the sampled first successor and (b) the old terminal target.  No
model, reward model, dataset target, or guidance parameter is loaded.

The key group-level metric is deliberately stricter than merely counting two
non-noop children: a group is locally discriminative only when it has at least
two *different* nonempty first-step action masks whose mean terminal rewards
differ.  Multiple copies of the same first edit with stochastic future returns
are reported as noise rather than treated as separate local actions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from edit_flows.guidance.data import (
    collate_guidance_records,
    load_guidance_dataset,
)
from edit_flows.guidance.targets import build_action_target_masks
from edit_flows.utils.tokens import PAD_TOKEN

try:  # Works both as ``python scripts/foo.py`` and as a test import.
    from scripts.audit_guidance_anchors import summarize_anchor_sharing
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI execution
    from audit_guidance_anchors import summarize_anchor_sharing


def _mask_signature(
    insert_mask: torch.Tensor,
    substitute_mask: torch.Tensor,
    delete_mask: torch.Tensor,
    row: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return a hashable exact description of one row's action mask."""

    return tuple(
        tuple(tuple(int(value) for value in index) for index in torch.nonzero(
            mask[row], as_tuple=False,
        ).tolist())
        for mask in (insert_mask, substitute_mask, delete_mask)
    )


def _action_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    vocab_size: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Build compact per-record action diagnostics without retaining masks."""

    missing = sum("transition_tokens" not in record for record in records)
    if missing:
        raise ValueError(
            "local action-credit audit requires transition_tokens in every "
            f"record; missing {missing}/{len(records)}"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start:start + batch_size]
        batch = collate_guidance_records(batch_records)
        transition_masks = build_action_target_masks(
            batch["state_tokens"],
            batch["transition_tokens"],
            vocab_size=vocab_size,
            pad_token=PAD_TOKEN,
        )
        terminal_masks = build_action_target_masks(
            batch["state_tokens"],
            batch["terminal_tokens"],
            vocab_size=vocab_size,
            pad_token=PAD_TOKEN,
        )
        for local_index, record in enumerate(batch_records):
            transition_counts = [
                int(mask[local_index].sum().item())
                for mask in transition_masks
            ]
            terminal_counts = [
                int(mask[local_index].sum().item())
                for mask in terminal_masks
            ]
            reward = float(record["reward"])
            if not math.isfinite(reward) or reward < 0:
                raise ValueError(
                    f"record {start + local_index} has invalid reward {reward}"
                )
            rows.append({
                "source_index": int(record["source_index"]),
                "time_index": int(record.get("time_index", -1)),
                "time": float(record["time"]),
                "state": tuple(int(token) for token in record["state_tokens"]),
                "reward": reward,
                "transition_counts": transition_counts,
                "transition_total": sum(transition_counts),
                "terminal_counts": terminal_counts,
                "terminal_total": sum(terminal_counts),
                "transition_signature": _mask_signature(
                    *transition_masks, local_index,
                ),
            })
    return rows


def _action_summary(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    """Summarize total, per-type, and no-op counts for a target representation."""

    counts_key = f"{prefix}_counts"
    total_key = f"{prefix}_total"
    row_count = len(rows)
    type_totals = [
        sum(int(row[counts_key][kind]) for row in rows)
        for kind in range(3)
    ]
    total_actions = sum(type_totals)
    nonempty = [row for row in rows if int(row[total_key]) > 0]
    return {
        "record_count": row_count,
        "nonempty_row_count": len(nonempty),
        "nonempty_row_fraction": len(nonempty) / row_count if row_count else None,
        "noop_row_count": row_count - len(nonempty),
        "noop_row_fraction": (
            (row_count - len(nonempty)) / row_count if row_count else None
        ),
        "mean_actions_per_record": total_actions / row_count if row_count else None,
        "mean_actions_per_nonempty_record": (
            total_actions / len(nonempty) if nonempty else None
        ),
        "action_type_counts": {
            "insert": type_totals[0],
            "substitute": type_totals[1],
            "delete": type_totals[2],
        },
        "action_type_fractions": {
            "insert": type_totals[0] / total_actions if total_actions else None,
            "substitute": type_totals[1] / total_actions if total_actions else None,
            "delete": type_totals[2] / total_actions if total_actions else None,
        },
    }


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["source_index"])].append(row)
    return dict(groups)


def _group_summary(
    groups: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    expected_group_size: int,
    reward_tolerance: float,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Assess group integrity and distinct first-action reward variation."""

    if expected_group_size < 1:
        raise ValueError("expected_group_size must be positive")
    if reward_tolerance < 0 or not math.isfinite(reward_tolerance):
        raise ValueError("reward_tolerance must be finite and non-negative")

    summaries: dict[int, dict[str, Any]] = {}
    for source_index, rows in groups.items():
        states = {row["state"] for row in rows}
        times = {round(float(row["time"]), 10) for row in rows}
        complete = len(rows) == expected_group_size
        shared = len(states) == 1 and len(times) == 1
        nonempty_rows = [
            row for row in rows if int(row["transition_total"]) > 0
        ]
        rewards = [float(row["reward"]) for row in nonempty_rows]
        by_signature: dict[Any, list[float]] = defaultdict(list)
        for row in nonempty_rows:
            by_signature[row["transition_signature"]].append(float(row["reward"]))
        action_mean_rewards = [
            sum(values) / len(values) for values in by_signature.values()
        ]
        nonempty_reward_varies = (
            len(rewards) >= 2
            and max(rewards) - min(rewards) > reward_tolerance
        )
        distinct_action_reward_varies = (
            len(action_mean_rewards) >= 2
            and max(action_mean_rewards) - min(action_mean_rewards) > reward_tolerance
        )
        locally_discriminative = (
            complete
            and shared
            and distinct_action_reward_varies
        )
        summaries[source_index] = {
            "complete": complete,
            "shared_state_and_time": shared,
            "structurally_valid": complete and shared,
            "nonempty_child_count": len(nonempty_rows),
            "distinct_nonempty_action_count": len(by_signature),
            "nonempty_reward_varies": nonempty_reward_varies,
            "distinct_action_reward_varies": distinct_action_reward_varies,
            "locally_discriminative": locally_discriminative,
            "time_index": int(rows[0]["time_index"]) if len(times) == 1 else -1,
        }

    values = list(summaries.values())
    structurally_valid = [value for value in values if value["structurally_valid"]]
    locally_discriminative = [
        value for value in values if value["locally_discriminative"]
    ]
    summary = {
        "group_count": len(values),
        "expected_group_size": expected_group_size,
        "complete_group_count": sum(value["complete"] for value in values),
        "shared_state_time_group_count": sum(
            value["shared_state_and_time"] for value in values
        ),
        "structurally_valid_group_count": len(structurally_valid),
        "groups_with_two_nonempty_children_and_reward_variation": sum(
            value["nonempty_child_count"] >= 2
            and value["nonempty_reward_varies"]
            and value["structurally_valid"]
            for value in values
        ),
        "groups_with_two_distinct_actions_and_reward_variation": len(
            locally_discriminative
        ),
        "locally_discriminative_group_fraction": (
            len(locally_discriminative) / len(structurally_valid)
            if structurally_valid else None
        ),
        "distinct_action_count_distribution": {
            str(count): frequency
            for count, frequency in sorted(Counter(
                value["distinct_nonempty_action_count"] for value in values
            ).items())
        },
    }
    return summary, summaries


def _by_time_summary(
    rows: Sequence[Mapping[str, Any]],
    group_summaries: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    row_buckets: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        row_buckets[int(row["time_index"])].append(row)
    group_buckets: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for group in group_summaries.values():
        group_buckets[int(group["time_index"])].append(group)
    output: dict[str, Any] = {}
    for time_index in sorted(row_buckets):
        time_rows = row_buckets[time_index]
        time_groups = group_buckets.get(time_index, [])
        valid_groups = [
            group for group in time_groups if group["structurally_valid"]
        ]
        discriminative = [
            group for group in valid_groups if group["locally_discriminative"]
        ]
        output[str(time_index)] = {
            "transition_actions": _action_summary(time_rows, "transition"),
            "terminal_alignment_actions": _action_summary(time_rows, "terminal"),
            "group_count": len(time_groups),
            "structurally_valid_group_count": len(valid_groups),
            "locally_discriminative_group_count": len(discriminative),
            "locally_discriminative_group_fraction": (
                len(discriminative) / len(valid_groups)
                if valid_groups else None
            ),
        }
    return output


def summarize_local_action_credit(
    records: Sequence[Mapping[str, Any]],
    *,
    vocab_size: int,
    expected_group_size: int = 4,
    reward_tolerance: float = 1e-6,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Return local-vs-terminal action-mask diagnostics for guidance records."""

    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    rows = _action_rows(records, vocab_size=vocab_size, batch_size=batch_size)
    groups = _group_rows(rows)
    group_summary, group_summaries = _group_summary(
        groups,
        expected_group_size=expected_group_size,
        reward_tolerance=reward_tolerance,
    )
    return {
        "record_count": len(records),
        "anchor_sharing": summarize_anchor_sharing(records),
        "transition_actions": _action_summary(rows, "transition"),
        "terminal_alignment_actions": _action_summary(rows, "terminal"),
        "local_credit_groups": group_summary,
        "by_time_index": _by_time_summary(rows, group_summaries),
    }


def audit(
    path: str | Path,
    *,
    vocab_size: int | None = None,
    expected_group_size: int = 4,
    reward_tolerance: float = 1e-6,
    batch_size: int = 256,
    min_usable_group_fraction: float = 0.20,
) -> dict[str, Any]:
    """Load one data artifact and add provenance plus the predeclared gate."""

    records, metadata = load_guidance_dataset(path)
    resolved_vocab_size = vocab_size or metadata.get("model_vocab")
    if resolved_vocab_size is None:
        raise ValueError("vocab_size is absent; pass --vocab_size")
    summary = summarize_local_action_credit(
        records,
        vocab_size=int(resolved_vocab_size),
        expected_group_size=expected_group_size,
        reward_tolerance=reward_tolerance,
        batch_size=batch_size,
    )
    fraction = summary["local_credit_groups"][
        "locally_discriminative_group_fraction"
    ]
    summary.update({
        "data_path": str(Path(path).resolve()),
        "guidance_metadata": metadata,
        "vocab_size": int(resolved_vocab_size),
        "reward_tolerance": reward_tolerance,
        "predeclared_min_usable_group_fraction": min_usable_group_fraction,
        "predeclared_gate_pass": (
            fraction is not None and fraction > min_usable_group_fraction
        ),
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--expected_group_size", type=int, default=4)
    parser.add_argument("--reward_tolerance", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--min_usable_group_fraction", type=float, default=0.20)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()
    summary = audit(
        args.data,
        vocab_size=args.vocab_size,
        expected_group_size=args.expected_group_size,
        reward_tolerance=args.reward_tolerance,
        batch_size=args.batch_size,
        min_usable_group_fraction=args.min_usable_group_fraction,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n")


if __name__ == "__main__":
    main()
