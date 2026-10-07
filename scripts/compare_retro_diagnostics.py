#!/usr/bin/env python
"""Compare two reaction-level retrosynthesis diagnostic reports.

The scorer already aggregates all SMILES augmentations belonging to one
reaction.  This utility preserves that statistical unit: it aligns the two
reports by reaction index, computes paired Top-k/Oracle differences, and uses
paired bootstrap resampling over reactions only.  It never treats augmentation
rows as independent observations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Iterable, Sequence


DEFAULT_METRICS = ("top_1", "top_2", "top_3", "top_5", "top_10", "oracle")


def _load_per_reaction(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text())
    rows = payload.get("per_reaction")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} does not contain nonempty per_reaction diagnostics")
    indexed: dict[int, dict] = {}
    for row in rows:
        try:
            index = int(row["reaction_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path} contains a row without reaction_index") from exc
        if index in indexed:
            raise ValueError(f"{path} repeats reaction_index {index}")
        indexed[index] = row
    return indexed


def _hit(row: dict, metric: str) -> int:
    if metric == "oracle":
        return int(bool(row.get("oracle_any", False)))
    if not metric.startswith("top_"):
        raise ValueError(f"unsupported metric {metric!r}")
    try:
        limit = int(metric.split("_", 1)[1])
    except ValueError as exc:  # pragma: no cover - protected by callers
        raise ValueError(f"invalid Top-k metric {metric!r}") from exc
    rank = row.get("target_final_rank")
    return int(rank is not None and int(rank) <= limit)


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of no values")
    position = (len(sorted_values) - 1) * quantile
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(sorted_values[low])
    fraction = position - low
    return float(sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction)


def _paired_bootstrap_intervals(
    differences: dict[str, list[int]],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    metric_names = list(differences)
    reaction_count = len(next(iter(differences.values())))
    if reaction_count < 1:
        raise ValueError("cannot bootstrap an empty comparison")
    if any(len(values) != reaction_count for values in differences.values()):
        raise ValueError("metric difference vectors have inconsistent lengths")

    rng = random.Random(seed)
    samples_by_metric = {name: [] for name in metric_names}
    # Draw the same reaction indices for every metric in each replicate.  This
    # preserves their paired relationship and avoids treating augmentation rows
    # or metrics as separate observations.
    for _ in range(samples):
        totals = [0] * len(metric_names)
        for _ in range(reaction_count):
            reaction = rng.randrange(reaction_count)
            for metric_index, name in enumerate(metric_names):
                totals[metric_index] += differences[name][reaction]
        for metric_index, name in enumerate(metric_names):
            samples_by_metric[name].append(
                100.0 * totals[metric_index] / reaction_count
            )
    return {
        name: [
            _percentile(sorted(values), 0.025),
            _percentile(sorted(values), 0.975),
        ]
        for name, values in samples_by_metric.items()
    }


def compare_diagnostics(
    baseline_path: Path,
    candidate_path: Path,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    bootstrap_samples: int = 5000,
    seed: int = 20260811,
) -> dict:
    """Return paired reaction-level metric changes and bootstrap intervals."""
    baseline = _load_per_reaction(baseline_path)
    candidate = _load_per_reaction(candidate_path)
    baseline_indices = set(baseline)
    candidate_indices = set(candidate)
    if baseline_indices != candidate_indices:
        only_baseline = sorted(baseline_indices - candidate_indices)
        only_candidate = sorted(candidate_indices - baseline_indices)
        raise ValueError(
            "diagnostic reports cover different reactions: "
            f"only_baseline={only_baseline[:5]}, only_candidate={only_candidate[:5]}"
        )
    reaction_indices = sorted(baseline_indices)
    if not reaction_indices:
        raise ValueError("diagnostic reports contain no reactions")

    differences: dict[str, list[int]] = {}
    result_metrics: dict[str, dict] = {}
    for metric in metrics:
        baseline_hits = [_hit(baseline[index], metric) for index in reaction_indices]
        candidate_hits = [_hit(candidate[index], metric) for index in reaction_indices]
        delta = [candidate_value - baseline_value for candidate_value, baseline_value in zip(candidate_hits, baseline_hits)]
        differences[metric] = delta
        result_metrics[metric] = {
            "baseline_percent": 100.0 * sum(baseline_hits) / len(reaction_indices),
            "candidate_percent": 100.0 * sum(candidate_hits) / len(reaction_indices),
            "delta_percentage_points": 100.0 * sum(delta) / len(reaction_indices),
            "candidate_only_count": sum(value == 1 for value in delta),
            "baseline_only_count": sum(value == -1 for value in delta),
            "both_or_neither_count": sum(value == 0 for value in delta),
        }
    intervals = _paired_bootstrap_intervals(
        differences,
        samples=bootstrap_samples,
        seed=seed,
    )
    for metric, interval in intervals.items():
        result_metrics[metric]["paired_bootstrap_95ci_delta_percentage_points"] = interval

    return {
        "schema_version": 1,
        "statistical_unit": "one original reaction after aggregation over all augmentation rows",
        "baseline_diagnostics": str(baseline_path.resolve()),
        "candidate_diagnostics": str(candidate_path.resolve()),
        "reaction_count": len(reaction_indices),
        "metrics": result_metrics,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_diagnostics", required=True, type=Path)
    parser.add_argument("--candidate_diagnostics", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args(argv)
    result = compare_diagnostics(
        args.baseline_diagnostics,
        args.candidate_diagnostics,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
