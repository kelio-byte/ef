#!/usr/bin/env python3
"""Compare two streamed R9K1M2 first-event distance summaries.

This script intentionally reports only trajectory-level local progress.  It
does not score final retrosynthesis candidates and therefore must not be read
as a Top-k evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if value.get("schema_version") != 1:
        raise ValueError(f"unsupported or missing schema_version in {path}")
    for key in ("protocol", "input", "trajectory_counts", "first_event_distance"):
        if key not in value:
            raise ValueError(f"missing {key!r} in {path}")
    return value


def _assert_comparable(baseline: dict, candidate: dict) -> None:
    protocol_fields = (
        "sampler",
        "n_runs",
        "n_branches",
        "n_children",
        "score_mode",
        "child_policy",
        "changed_state_bonus",
        "q_temperature",
        "n_steps",
        "seed",
        "scheduler",
    )
    for field in protocol_fields:
        if baseline["protocol"].get(field) != candidate["protocol"].get(field):
            raise ValueError(
                f"protocol differs for {field}: "
                f"{baseline['protocol'].get(field)!r} != "
                f"{candidate['protocol'].get(field)!r}"
            )
    input_fields = (
        "products_sha256",
        "targets_sha256",
        "sidecar_scores_sha256",
        "checkpoint_sha256",
        "selection_start_product",
        "selection_end_product_exclusive",
    )
    for field in input_fields:
        if baseline["input"].get(field) != candidate["input"].get(field):
            raise ValueError(
                f"input differs for {field}: "
                f"{baseline['input'].get(field)!r} != "
                f"{candidate['input'].get(field)!r}"
            )
    if (
        baseline["trajectory_counts"].get("expected")
        != candidate["trajectory_counts"].get("expected")
    ):
        raise ValueError("expected trajectory count differs")


def _delta(candidate: dict, baseline: dict, field: str) -> float:
    return float(candidate["first_event_distance"][field]) - float(
        baseline["first_event_distance"][field]
    )


def _build_markdown(output: dict) -> str:
    baseline = output["baseline"]
    candidate = output["candidate"]
    baseline_distance = baseline["first_event_distance"]
    candidate_distance = candidate["first_event_distance"]
    delta = output["delta_candidate_minus_baseline"]
    return "\n".join(
        (
            "# R9K1M2 首个非空编辑：距离变化对照",
            "",
            "这是一项逐轨迹局部诊断：把首个非空 Euler 步的全部实际编辑",
            "共同作用到初始 M500 token 序列后，比较其到配对目标的",
            "Levenshtein 距离。它不是最终 Top-k 准确率。",
            "",
            f"- baseline：`{baseline['condition']}`",
            f"- candidate：`{candidate['condition']}`",
            f"- 轨迹数：`{baseline['trajectory_counts']['expected']:,}`",
            "",
            "| 指标（仅有首个非空事件的轨迹） | baseline | candidate | candidate − baseline |",
            "|---|---:|---:|---:|",
            (
                "| 首步后更接近目标 | "
                f"{baseline_distance['closer_percent']:.3f}% | "
                f"{candidate_distance['closer_percent']:.3f}% | "
                f"{delta['closer_percent_pp']:+.3f} pp |"
            ),
            (
                "| 首步后距离不变 | "
                f"{baseline_distance['unchanged_percent']:.3f}% | "
                f"{candidate_distance['unchanged_percent']:.3f}% | "
                f"{delta['unchanged_percent_pp']:+.3f} pp |"
            ),
            (
                "| 首步后更远离目标 | "
                f"{baseline_distance['farther_percent']:.3f}% | "
                f"{candidate_distance['farther_percent']:.3f}% | "
                f"{delta['farther_percent_pp']:+.3f} pp |"
            ),
            (
                "| 平均距离改善（正值更好） | "
                f"{baseline_distance['mean_distance_improvement']:+.5f} | "
                f"{candidate_distance['mean_distance_improvement']:+.5f} | "
                f"{delta['mean_distance_improvement']:+.5f} |"
            ),
            (
                "| 首个非空事件占全部轨迹 | "
                f"{baseline['trajectory_counts']['first_event_percent']:.3f}% | "
                f"{candidate['trajectory_counts']['first_event_percent']:.3f}% | "
                f"{output['first_event_percent_delta_pp']:+.3f} pp |"
            ),
            "",
            "说明：B0-trace 的倍率应为 1，且其首 batch 已进行逐 token",
            "bitwise 中性检查；B1 为真实中心 oracle，不能作为可部署结果。",
            "",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists() or args.output_markdown.exists():
        raise ValueError("refusing to overwrite an existing comparison output")

    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    _assert_comparable(baseline, candidate)
    delta = {
        "closer_percent_pp": _delta(candidate, baseline, "closer_percent"),
        "unchanged_percent_pp": _delta(candidate, baseline, "unchanged_percent"),
        "farther_percent_pp": _delta(candidate, baseline, "farther_percent"),
        "mean_distance_improvement": _delta(
            candidate, baseline, "mean_distance_improvement"
        ),
    }
    output = {
        "schema_version": 1,
        "baseline": baseline,
        "candidate": candidate,
        "delta_candidate_minus_baseline": delta,
        "first_event_percent_delta_pp": float(
            candidate["trajectory_counts"]["first_event_percent"]
        ) - float(baseline["trajectory_counts"]["first_event_percent"]),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(_build_markdown(output))
    print(f"Saved: {args.output_json}")
    print(f"Saved: {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
