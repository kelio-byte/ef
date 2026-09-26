"""Paired reaction bootstrap intervals for two saved score summaries."""

import argparse
import json
from pathlib import Path

import numpy as np


def paired_bootstrap(baseline_path, candidate_path, output_path, samples, seed):
    baseline = json.loads(Path(baseline_path).read_text())
    candidate = json.loads(Path(candidate_path).read_text())
    if baseline["reaction_count"] != candidate["reaction_count"]:
        raise ValueError("Score summaries must contain the same reaction count")
    if baseline["targets_sha256"] != candidate["targets_sha256"]:
        raise ValueError("Score summaries must use identical ordered targets")

    n_reactions = baseline["reaction_count"]
    baseline_ranks = baseline["target_ranks"]
    candidate_ranks = candidate["target_ranks"]
    if len(baseline_ranks) != n_reactions or len(candidate_ranks) != n_reactions:
        raise ValueError("Per-reaction target ranks are missing or misaligned")
    metrics = {}
    for cutoff in (1, 3, 10):
        baseline_hits = np.array(
            [rank is not None and rank <= cutoff for rank in baseline_ranks],
            dtype=float,
        )
        candidate_hits = np.array(
            [rank is not None and rank <= cutoff for rank in candidate_ranks],
            dtype=float,
        )
        metrics[f"top_{cutoff}"] = candidate_hits - baseline_hits

    for key, output_name in (
        ("oracle_any_by_reaction", "oracle_any"),
        ("invalid_at_1_by_reaction", "invalid_at_1"),
    ):
        baseline_values = np.asarray(baseline[key], dtype=float)
        candidate_values = np.asarray(candidate[key], dtype=float)
        if len(baseline_values) != n_reactions or len(candidate_values) != n_reactions:
            raise ValueError(f"Per-reaction values missing or misaligned for {key}")
        metrics[output_name] = candidate_values - baseline_values

    indices = np.random.default_rng(seed).integers(
        0, n_reactions, size=(samples, n_reactions)
    )
    summary = {}
    for name, per_reaction_delta in metrics.items():
        bootstrap_means = per_reaction_delta[indices].mean(axis=1) * 100
        lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
        summary[name] = {
            "candidate_minus_baseline_pp": float(per_reaction_delta.mean() * 100),
            "paired_bootstrap_95_percent_interval_pp": [
                float(lower),
                float(upper),
            ],
        }

    result = {
        "baseline_metrics": str(baseline_path),
        "candidate_metrics": str(candidate_path),
        "reaction_count": n_reactions,
        "samples": samples,
        "seed": seed,
        "interval_method": "paired reaction bootstrap percentile interval",
        "metrics": summary,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    paired_bootstrap(
        args.baseline_metrics,
        args.candidate_metrics,
        args.output,
        args.samples,
        args.seed,
    )


if __name__ == "__main__":
    main()
