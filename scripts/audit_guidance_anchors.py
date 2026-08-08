#!/usr/bin/env python
"""Audit whether guidance records really share product-conditioned anchors.

The pairwise guidance objective assumes that records in one ``source_index``
group use the same intermediate state and time.  This read-only utility checks
that assumption without loading a model or touching the original dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from edit_flows.guidance.data import load_guidance_dataset


def _group_records(records: Sequence[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[int(record["source_index"])].append(record)
    return dict(groups)


def summarize_anchor_sharing(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return JSON-friendly group/time/state sharing diagnostics."""
    groups = _group_records(records)
    group_sizes = Counter(len(rows) for rows in groups.values())
    state_unique_counts: list[int] = []
    time_unique_counts: list[int] = []
    same_time_pairs = 0
    same_time_state_equal_pairs = 0

    for rows in groups.values():
        states = [tuple(int(token) for token in row["state_tokens"]) for row in rows]
        times = [round(float(row["time"]), 10) for row in rows]
        state_unique_counts.append(len(set(states)))
        time_unique_counts.append(len(set(times)))
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                if times[left] != times[right]:
                    continue
                same_time_pairs += 1
                if states[left] == states[right]:
                    same_time_state_equal_pairs += 1

    group_count = len(groups)
    same_time_fraction = (
        same_time_state_equal_pairs / same_time_pairs
        if same_time_pairs else None
    )
    return {
        "record_count": len(records),
        "group_count": group_count,
        "group_size_distribution": {
            str(size): count for size, count in sorted(group_sizes.items())
        },
        "state_unique_count_mean": (
            sum(state_unique_counts) / group_count if group_count else 0.0
        ),
        "time_unique_count_mean": (
            sum(time_unique_counts) / group_count if group_count else 0.0
        ),
        "groups_all_states_equal": sum(
            count == 1 for count in state_unique_counts
        ),
        "groups_all_times_equal": sum(
            count == 1 for count in time_unique_counts
        ),
        "same_time_pair_count": same_time_pairs,
        "same_time_state_equal_pair_count": same_time_state_equal_pairs,
        "same_time_state_equal_fraction": same_time_fraction,
    }


def audit(path: str | Path) -> dict[str, Any]:
    records, metadata = load_guidance_dataset(path)
    summary = summarize_anchor_sharing(records)
    summary["path"] = str(Path(path).resolve())
    summary["schema_version"] = metadata.get("schema_version")
    summary["metadata"] = metadata
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()
    summary = audit(args.data)
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n")


if __name__ == "__main__":
    main()
