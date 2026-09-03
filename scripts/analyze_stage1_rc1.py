#!/usr/bin/env python
"""Summarize the frozen RC1 reaction-center sampling experiment.

This post-processing entry point intentionally never samples a model.  It
reads the first-event records produced by the centre-bias sampler and asks a
well-defined, order-invariant question: did the complete first sampled event
make the token sequence closer to its paired target?  It also derives paired
reaction-level Top-k comparisons from the evaluator diagnostics.

The oracle-centre condition is an upper bound, not a deployable result.  The
script keeps that distinction in the output names so the result cannot be
mistaken for a product-only reaction-centre predictor.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from edit_flows.analysis.first_step import tokenize_smiles
from edit_flows.data.dataset import load_vocab
from edit_flows.utils.tokens import BOS_TOKEN


EXPERIMENTS = {
    "rc1": {
        "performance_groups": {
            "b0": "b0_plain",
            "b1_oracle": "b1_oracle",
            "b2_pseudo": "b2_pseudo",
        },
        "event_groups": {
            "b0": "b0_trace",
            "b1_oracle": "b1_oracle",
            "b2_pseudo": "b2_pseudo",
        },
        "conditions": {
            "b0": (
                "ordinary Euler N=9 (with B0-trace only for identical "
                "event recording)"
            ),
            "b1_oracle": (
                "ORACLE / NOT DEPLOYABLE: true-center first-event position "
                "bias, multiplier=3"
            ),
            "b2_pseudo": (
                "same-product pseudo-center first-event position bias, "
                "multiplier=3"
            ),
        },
        "b0_candidates": ("b1_oracle", "b2_pseudo"),
        "between_key": "paired_bootstrap_b1_minus_b2",
        "between_base": "b2_pseudo",
        "between_candidate": "b1_oracle",
    },
    "rc15": {
        "performance_groups": {
            "b0": "b0_plain",
            "b1_oracle": "b1_oracle",
            "rc15_mixed": "rc15_mixed",
        },
        "event_groups": {
            "b0": "b0_trace",
            "b1_oracle": "b1_oracle",
            "rc15_mixed": "rc15_mixed",
        },
        "conditions": {
            "b0": (
                "ordinary Euler N=9 (with B0-trace only for identical "
                "event recording)"
            ),
            "b1_oracle": (
                "ORACLE / NOT DEPLOYABLE: all nine trajectories use "
                "true-center first-event position bias, multiplier=3"
            ),
            "rc15_mixed": (
                "ORACLE / NOT DEPLOYABLE: three true-center-guided first "
                "events plus six ordinary Euler trajectories, multiplier=3"
            ),
        },
        "b0_candidates": ("b1_oracle", "rc15_mixed"),
        "between_key": "paired_bootstrap_b1_minus_rc15_mixed",
        "between_base": "rc15_mixed",
        "between_candidate": "b1_oracle",
    },
}
TOP_KS = (1, 3, 5, 10)


def _read_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _apply_first_event(
    source_ids: Sequence[int],
    actions: Sequence[Mapping],
    *,
    max_seq_len: int,
) -> tuple[list[int], int]:
    """Apply sampler actions to the *initial* state exactly once.

    ``sample_euler`` first applies substitutions, then calls
    ``apply_ins_del_operations``.  At one position, simultaneous INS+DEL is
    a replacement and wins over a simultaneous SUB.  This compact CPU
    implementation mirrors those semantics and returns the number of
    effective edits (rather than blindly counting redundant masks).
    """
    by_position: dict[int, dict[str, int | None]] = defaultdict(dict)
    for action in actions:
        mode = str(action["mode"]).lower()
        if mode not in {"ins", "sub", "del"}:
            raise ValueError(f"unsupported action mode: {mode!r}")
        position = int(action["position"])
        if position < 0 or position >= len(source_ids):
            raise ValueError(
                f"action position {position} outside source length "
                f"{len(source_ids)}"
            )
        token = action.get("token_id")
        if mode != "del" and token is None:
            raise ValueError(f"{mode} action lacks token_id")
        by_position[position][mode] = None if token is None else int(token)

    output: list[int] = []
    effective_action_count = 0
    for position, source_token in enumerate(source_ids):
        event = by_position.get(position)
        if event is None:
            output.append(int(source_token))
            continue

        has_ins = "ins" in event
        has_sub = "sub" in event
        has_del = "del" in event
        if has_ins and has_del:
            # INS+DEL is replacement inside apply_ins_del_operations.  A SUB
            # at the same location was already applied but is overwritten.
            output.append(int(event["ins"]))
            effective_action_count += 1
            continue
        if has_del:
            # A prior SUB at this location is deleted and has no effect.
            effective_action_count += 1
            continue

        current_token = int(event["sub"]) if has_sub else int(source_token)
        output.append(current_token)
        if has_sub:
            effective_action_count += 1
        if has_ins:
            output.append(int(event["ins"]))
            effective_action_count += 1

    return output[:max_seq_len], effective_action_count


def _percent(value: int | float, total: int | float) -> float:
    return 100.0 * float(value) / float(total) if total else 0.0


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _center_bucket(score: float) -> str:
    if score >= 0.999999:
        return "1.0"
    if score >= 0.499999:
        return "0.5"
    return "0.0"


def _token_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Fast CPU Levenshtein distance for compact token sequences.

    The formal evaluator needs RDKit canonicalization, but this diagnostic
    deliberately measures token-space progress.  Calling the Torch oracle
    once per first event is prohibitively slow for 500k trajectories, so use
    the same ordinary Levenshtein recurrence on Python integers instead.
    Inputs here contain BOS but no padding.
    """
    left_values = left[1:] if left and int(left[0]) == BOS_TOKEN else left
    right_values = right[1:] if right and int(right[0]) == BOS_TOKEN else right
    if len(left_values) < len(right_values):
        left_values, right_values = right_values, left_values
    previous = list(range(len(right_values) + 1))
    for i, left_token in enumerate(left_values, start=1):
        current = [i]
        for j, right_token in enumerate(right_values, start=1):
            deletion = previous[j] + 1
            insertion = current[-1] + 1
            substitution = previous[j - 1] + (left_token != right_token)
            current.append(min(deletion, insertion, substitution))
        previous = current
    return int(previous[-1])


def _event_summary(
    group_dir: Path,
    source_ids: Sequence[Sequence[int]],
    target_ids: Sequence[Sequence[int]],
    before_distances: Sequence[int],
    *,
    max_seq_len: int,
    expected_trajectories: int,
    trajectory_role: str | None = None,
) -> dict:
    diagnostics_path = group_dir / "center_bias_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    all_records = diagnostics["records"]
    if len(all_records) != int(diagnostics["first_event_count"]):
        raise ValueError(
            f"{diagnostics_path}: records={len(all_records)} disagrees with "
            f"first_event_count={diagnostics['first_event_count']}"
        )
    if trajectory_role is None:
        records = all_records
        no_event_count = int(diagnostics["no_event_count"])
    else:
        records = [
            record for record in all_records
            if record.get("row_metadata", {}).get("trajectory_role")
            == trajectory_role
        ]
        no_event_count = int(
            diagnostics.get("no_event_trajectory_role_counts", {}).get(
                trajectory_role, 0
            )
        )

    progress = Counter()
    action_modes = Counter()
    action_scores = Counter()
    event_max_scores = Counter()
    event_steps: list[float] = []
    distance_deltas: list[int] = []
    effective_action_counts: list[int] = []
    per_reaction_deltas: dict[int, list[int]] = defaultdict(list)

    for record in records:
        metadata = record.get("row_metadata") or {}
        row_index = int(metadata["global_input_row"])
        if not 0 <= row_index < len(source_ids):
            raise ValueError(
                f"{diagnostics_path}: global_input_row={row_index} is invalid"
            )
        before = int(before_distances[row_index])
        after_ids, n_effective = _apply_first_event(
            source_ids[row_index],
            record["actions"],
            max_seq_len=max_seq_len,
        )
        after = _token_distance(after_ids, target_ids[row_index])
        delta = before - after
        if delta > 0:
            progress["closer"] += 1
        elif delta == 0:
            progress["unchanged"] += 1
        else:
            progress["farther"] += 1
        distance_deltas.append(delta)
        effective_action_counts.append(n_effective)
        event_steps.append(float(record["first_event_step_idx"]))
        per_reaction_deltas[int(metadata["reaction_position"])].append(delta)

        scores = []
        for action in record["actions"]:
            action_modes[str(action["mode"])] += 1
            score = float(action["center_score"])
            action_scores[_center_bucket(score)] += 1
            scores.append(score)
        event_max_scores[_center_bucket(max(scores))] += 1

    first_event_count = len(records)
    if first_event_count + no_event_count != expected_trajectories:
        raise ValueError(
            f"{diagnostics_path}: first/no-event total "
            f"{first_event_count + no_event_count} != expected "
            f"{expected_trajectories}"
        )
    return {
        "path": str(group_dir),
        "center_source": diagnostics["center_source"],
        "trajectory_role": trajectory_role,
        # Scores are always relative to the region assigned to this group.
        # In particular, B2's score is proximity to its pseudo-center, not
        # proximity to the true reaction center.
        "event_region_description": (
            "assigned true reaction-center component"
            if diagnostics["center_source"] == "oracle"
            else "assigned pseudo-center component"
        ),
        "max_multiplier": float(diagnostics["max_multiplier"]),
        "expected_trajectories": expected_trajectories,
        "first_event_count": first_event_count,
        "first_event_percent": _percent(first_event_count, expected_trajectories),
        "no_event_count": no_event_count,
        "max_hazard_relative_error": float(
            diagnostics["max_hazard_relative_error"]
        ),
        "first_event_distance": {
            "closer_count": int(progress["closer"]),
            "closer_percent": _percent(progress["closer"], first_event_count),
            "unchanged_count": int(progress["unchanged"]),
            "unchanged_percent": _percent(progress["unchanged"], first_event_count),
            "farther_count": int(progress["farther"]),
            "farther_percent": _percent(progress["farther"], first_event_count),
            "mean_delta": float(np.mean(distance_deltas)) if distance_deltas else 0.0,
            "median_delta": _quantile(distance_deltas, 0.5),
            "mean_effective_action_count": (
                float(np.mean(effective_action_counts))
                if effective_action_counts else 0.0
            ),
        },
        "action_modes": dict(sorted(action_modes.items())),
        "action_center_score_histogram": dict(sorted(action_scores.items())),
        "event_max_center_score_histogram": dict(sorted(event_max_scores.items())),
        "event_at_or_near_center_percent": _percent(
            event_max_scores["1.0"] + event_max_scores["0.5"],
            first_event_count,
        ),
        "first_event_step": {
            "mean": float(np.mean(event_steps)) if event_steps else 0.0,
            "median": _quantile(event_steps, 0.5),
            "p90": _quantile(event_steps, 0.9),
        },
        "reaction_mean_distance_delta": {
            "mean": float(np.mean([
                np.mean(values) for values in per_reaction_deltas.values()
            ])) if per_reaction_deltas else 0.0,
            "reaction_count_with_event": len(per_reaction_deltas),
        },
    }


def _performance_rows(group_dir: Path) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    diagnostics = json.loads((group_dir / "diagnostics.json").read_text())
    rows = sorted(diagnostics["per_reaction"], key=lambda row: row["reaction_index"])
    ranks = np.asarray([
        float(row["target_final_rank"])
        if row.get("target_final_rank") is not None else np.inf
        for row in rows
    ])
    oracle = np.asarray([bool(row["oracle_any"]) for row in rows], dtype=float)
    true_unique = np.asarray([
        float(row["true_unique_candidates"]) for row in rows
    ])
    valid = np.asarray([
        float(row["valid_candidate_count"]) for row in rows
    ])
    arrays = {
        **{f"top_{k}": (ranks <= k).astype(float) for k in TOP_KS},
        "oracle_any": oracle,
        "true_unique_candidates": true_unique,
        "valid_candidate_count": valid,
    }
    run_metrics = diagnostics["summary"].get("run_metrics", [])
    invalid_at_1 = (
        float(run_metrics[0]["invalid_rate_percent"])
        if run_metrics else float("nan")
    )
    summary = {
        **{key: 100.0 * float(values.mean()) for key, values in arrays.items()
           if key.startswith("top_") or key == "oracle_any"},
        "mean_true_unique_candidates": float(true_unique.mean()),
        "mean_valid_candidates": float(valid.mean()),
        "invalid_at_1_percent": invalid_at_1,
        "reaction_count": int(len(rows)),
    }
    return summary, arrays


def _paired_bootstrap(
    base: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    draws: int,
    seed: int,
) -> dict[str, dict]:
    keys = tuple(base)
    if set(keys) != set(candidate):
        raise ValueError("paired metric keys differ")
    n_reactions = len(next(iter(base.values())))
    if n_reactions == 0:
        raise ValueError("cannot bootstrap an empty evaluation")
    if any(len(values) != n_reactions for values in candidate.values()):
        raise ValueError("paired reaction counts differ")

    rng = np.random.default_rng(seed)
    # Chunks avoid allocating a large draws x reactions x metrics tensor.
    sampled_deltas = {key: [] for key in keys}
    remaining = draws
    while remaining:
        size = min(remaining, 1000)
        indices = rng.integers(0, n_reactions, size=(size, n_reactions))
        for key in keys:
            sampled_deltas[key].append(
                (candidate[key][indices] - base[key][indices]).mean(axis=1)
            )
        remaining -= size

    output = {}
    for key, chunks in sampled_deltas.items():
        values = np.concatenate(chunks)
        scale = 100.0 if key.startswith("top_") or key == "oracle_any" else 1.0
        observed = float((candidate[key] - base[key]).mean()) * scale
        output[key] = {
            "delta": observed,
            "ci95_low": float(np.quantile(values, 0.025)) * scale,
            "ci95_high": float(np.quantile(values, 0.975)) * scale,
        }
    return output


def _runtime(group_dir: Path) -> dict:
    metadata = json.loads((group_dir / "sampling_metadata.json").read_text())
    runtime = metadata["runtime"]
    return {
        "elapsed_seconds": float(runtime["elapsed_seconds"]),
        "peak_cuda_allocated_bytes": int(runtime["peak_cuda_allocated_bytes"]),
        "peak_cuda_reserved_bytes": int(runtime["peak_cuda_reserved_bytes"]),
        "predictions_sha256": metadata["output_sha256"],
        "git_commit": metadata["git"]["commit"],
        "git_dirty": bool(metadata["git"]["dirty"]),
    }


def analyze(args: argparse.Namespace) -> dict:
    experiment = EXPERIMENTS[args.experiment]
    performance_groups = experiment["performance_groups"]
    event_groups = experiment["event_groups"]
    run_root = Path(args.run_root)
    products = _read_lines(Path(args.products_file))
    targets = _read_lines(Path(args.targets_file))
    if len(products) != len(targets):
        raise ValueError("products and targets have different row counts")
    token2id, _ = load_vocab(args.vocab_file)
    source_ids = [[BOS_TOKEN, *tokenize_smiles(row, token2id)] for row in products]
    target_ids = [[BOS_TOKEN, *tokenize_smiles(row, token2id)] for row in targets]
    before_distances = [
        _token_distance(source, target)
        for source, target in zip(source_ids, target_ids)
    ]

    b0_metadata = json.loads(
        (run_root / performance_groups["b0"] / "sampling_metadata.json").read_text()
    )
    expected_trajectories = int(b0_metadata["product_count"]) * int(
        b0_metadata["sampling"]["n_samples"]
    )
    max_seq_len = int(args.max_seq_len)

    performance = {}
    performance_arrays = {}
    runtimes = {}
    for label, dirname in performance_groups.items():
        performance[label], performance_arrays[label] = _performance_rows(
            run_root / dirname
        )
        runtimes[label] = _runtime(run_root / dirname)

    event_quality = {
        label: _event_summary(
            run_root / dirname,
            source_ids,
            target_ids,
            before_distances,
            max_seq_len=max_seq_len,
            expected_trajectories=expected_trajectories,
        )
        for label, dirname in event_groups.items()
    }
    event_quality_by_trajectory_role = {}
    if args.experiment == "rc15":
        mix_metadata = json.loads(
            (run_root / performance_groups["rc15_mixed"] /
             "sampling_metadata.json").read_text()
        )
        mix_sampling = mix_metadata["sampling"]
        guided_count = int(
            mix_sampling["first_event_center_guided_trajectories"]
        )
        ordinary_count = int(
            mix_sampling["first_event_center_ordinary_trajectories"]
        )
        if guided_count + ordinary_count != int(mix_sampling["n_samples"]):
            raise ValueError("RC1.5 guided/ordinary trajectory counts disagree")
        role_counts = {
            "center_guided": guided_count,
            "ordinary_euler": ordinary_count,
        }
        for role, trajectories_per_product in role_counts.items():
            event_quality_by_trajectory_role[role] = _event_summary(
                run_root / event_groups["rc15_mixed"],
                source_ids,
                target_ids,
                before_distances,
                max_seq_len=max_seq_len,
                expected_trajectories=(
                    int(mix_metadata["product_count"])
                    * trajectories_per_product
                ),
                trajectory_role=role,
            )

    paired_vs_b0 = {
        label: _paired_bootstrap(
            performance_arrays["b0"], performance_arrays[label],
            draws=args.bootstrap_draws, seed=args.seed,
        )
        for label in experiment["b0_candidates"]
    }
    between_base = experiment["between_base"]
    between_candidate = experiment["between_candidate"]
    result = {
        "schema_version": 2,
        "experiment": args.experiment,
        "run_root": str(run_root),
        "input": {
            "products_file": str(Path(args.products_file)),
            "targets_file": str(Path(args.targets_file)),
            "vocab_file": str(Path(args.vocab_file)),
            "source_row_count": len(products),
            "selected_product_count": int(b0_metadata["product_count"]),
            "expected_trajectories": expected_trajectories,
            "max_seq_len": max_seq_len,
        },
        "conditions": experiment["conditions"],
        # RC1.5 can reuse validated B0/B1 outputs through directory links.
        # Store the resolved origin of every group so a committed summary
        # remains auditable even when the run root itself only contains links.
        "group_sources": {
            label: str((run_root / dirname).resolve())
            for label, dirname in performance_groups.items()
        },
        "performance": performance,
        "runtimes": runtimes,
        "event_quality": event_quality,
        "paired_bootstrap_vs_b0": paired_vs_b0,
        experiment["between_key"]: _paired_bootstrap(
            performance_arrays[between_base], performance_arrays[between_candidate],
            draws=args.bootstrap_draws, seed=args.seed,
        ),
    }
    if event_quality_by_trajectory_role:
        result["event_quality_by_trajectory_role"] = (
            event_quality_by_trajectory_role
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--experiment", choices=tuple(EXPERIMENTS), default="rc1",
        help="RC1 all-guided/pseudo control analysis or RC1.5 mixed analysis",
    )
    parser.add_argument("--products-file", required=True)
    parser.add_argument("--targets-file", required=True)
    parser.add_argument("--vocab-file", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        parser.error("--bootstrap-draws must be positive")
    if args.max_seq_len < 1:
        parser.error("--max-seq-len must be positive")

    result = analyze(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
