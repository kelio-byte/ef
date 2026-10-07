#!/usr/bin/env python
"""Reclassify natural trajectories without assuming one edit order is correct."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from edit_flows.analysis.trajectory_correction import classify_event_progress


CATEGORIES = ("full_progress", "partial_progress", "neutral", "harmful")
METRICS = (
    "full_progress_rate",
    "partial_progress_rate",
    "neutral_rate",
    "harmful_rate",
    "harmful_recovery_rate",
    "final_hit_rate",
    "mean_distance_delta",
)


def _read_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.glob("seed_*/per_trajectory.jsonl")):
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise FileNotFoundError(f"no seed_*/per_trajectory.jsonl under {root}")
    return rows


def _first_progress(row: dict) -> dict | None:
    trace = row.get("trace", [])
    summaries = row.get("events", [])
    first_index = row.get("first_event_index")
    if first_index is None:
        return None
    first_index = int(first_index)
    if first_index >= len(trace) or first_index >= len(summaries):
        return None
    return classify_event_progress(trace[first_index], summaries[first_index])


def _reaction_arrays(rows: list[dict]) -> tuple[list[int], dict[str, np.ndarray]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["reaction_index"]), []).append(row)
    reaction_ids = sorted(grouped)
    values = {
        metric: np.zeros(len(reaction_ids), dtype=np.float64)
        for metric in METRICS
    }

    for offset, reaction_id in enumerate(reaction_ids):
        group = grouped[reaction_id]
        first = []
        for row in group:
            progress = _first_progress(row)
            if progress is not None and progress["category"] != "no_event":
                first.append((row, progress))
        for category in CATEGORIES:
            selected = [item for item in first if item[1]["category"] == category]
            values[f"{category}_rate"][offset] = (
                len(selected) / len(first) if first else 0.0
            )
        harmful = [item for item in first if item[1]["category"] == "harmful"]
        values["harmful_recovery_rate"][offset] = (
            sum(bool(row.get("final_hit")) for row, _ in harmful) / len(harmful)
            if harmful else 0.0
        )
        values["final_hit_rate"][offset] = (
            sum(bool(row.get("final_hit")) for row in group) / len(group)
            if group else 0.0
        )
        values["mean_distance_delta"][offset] = (
            sum(progress["distance_delta"] for _, progress in first) / len(first)
            if first else 0.0
        )
    return reaction_ids, values


def _bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    if left.shape != right.shape:
        raise ValueError(f"paired arrays have different shapes: {left.shape} vs {right.shape}")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(left), size=(n_bootstrap, len(left)), endpoint=False,
    )
    differences = left[indices].mean(axis=1) - right[indices].mean(axis=1)
    return {
        "mean_difference": float(left.mean() - right.mean()),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "n_reactions": int(len(left)),
        "n_bootstrap": int(n_bootstrap),
    }


def _overall(values: dict[str, np.ndarray]) -> dict[str, float]:
    return {metric: float(array.mean()) for metric, array in values.items()}


def _write_markdown(
    path: Path,
    atom: dict[str, float],
    m500: dict[str, float],
    paired: dict[str, dict],
) -> None:
    labels = {
        "full_progress_rate": "完整改善事件比例",
        "partial_progress_rate": "部分改善事件比例",
        "neutral_rate": "距离不变事件比例",
        "harmful_rate": "距离变差事件比例",
        "harmful_recovery_rate": "明显有害事件后的恢复率",
        "final_hit_rate": "最终命中率",
        "mean_distance_delta": "首事件平均距离变化（正数为改善）",
    }
    lines = [
        "# 顺序无关的自然轨迹分析",
        "",
        "本分析不再使用某一条固定的 Levenshtein 对齐路径判断首编辑对错；",
        "而是直接比较首事件前后的 token 编辑距离。bootstrap 以 reaction 为单位。",
        "",
        "| 指标 | Atom@600K | SPE-M500@490K | M500−Atom | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        interval = paired[metric]
        lines.append(
            f"| {labels[metric]} | {atom[metric]:.4f} | {m500[metric]:.4f} | "
            f"{m500[metric] - atom[metric]:+.4f} | "
            f"[{interval['ci95_low']:+.4f}, {interval['ci95_high']:+.4f}] |"
        )
    lines.extend([
        "",
        "定义：完整改善表示一次事件包含 k 个有效编辑且距离恰好减少 k；",
        "部分改善表示距离减少但少于 k；距离不变不被称为错误；距离增加才称为明显有害。",
        "这仍是 token-level 诊断，最终化学正确性仍以 canonical SMILES 命中为准。",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom_dir", type=Path, required=True)
    parser.add_argument("--m500_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    atom_rows = _read_rows(args.atom_dir)
    m500_rows = _read_rows(args.m500_dir)
    atom_ids, atom_values = _reaction_arrays(atom_rows)
    m500_ids, m500_values = _reaction_arrays(m500_rows)
    if atom_ids != m500_ids:
        raise ValueError("Atom and M500 reaction indices do not match")

    paired = {
        metric: _bootstrap_difference(
            m500_values[metric], atom_values[metric],
            n_bootstrap=args.n_bootstrap, seed=args.seed + index,
        )
        for index, metric in enumerate(METRICS)
    }
    result = {
        "protocol": {
            "atom_dir": str(args.atom_dir),
            "m500_dir": str(args.m500_dir),
            "n_reactions": len(atom_ids),
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.seed,
            "cluster_unit": "reaction_index",
            "classification": "order_invariant_distance_progress",
        },
        "atom_overall": _overall(atom_values),
        "m500_overall": _overall(m500_values),
        "paired_bootstrap": paired,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_bootstrap.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _write_markdown(
        args.output_dir / "summary.md",
        result["atom_overall"], result["m500_overall"], paired,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
