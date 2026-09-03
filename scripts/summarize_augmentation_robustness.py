#!/usr/bin/env python
"""Summarize P1 natural trajectories across all R-SMILES augmentation views.

The input is one directory per augmentation view.  Paths, views, and seeds are
repeated observations inside a reaction; all headline estimates and bootstrap
intervals therefore use reaction as the statistical unit.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


METRICS = (
    "full_progress_rate",
    "partial_progress_rate",
    "neutral_rate",
    "harmful_rate",
    "harmful_recovery_rate",
    "final_hit_rate",
    "mean_distance_delta",
)

LABELS = {
    "full_progress_rate": "完整改善事件比例",
    "partial_progress_rate": "部分改善事件比例",
    "neutral_rate": "距离不变事件比例",
    "harmful_rate": "距离变差事件比例",
    "harmful_recovery_rate": "明显有害事件后的恢复率",
    "final_hit_rate": "最终 canonical 命中率",
    "mean_distance_delta": "首事件平均距离变化（正数为改善）",
}


@dataclass
class TrajectoryCounts:
    """Sufficient statistics for one reaction, optionally one view."""

    n_paths: int = 0
    n_first: int = 0
    full_progress: int = 0
    partial_progress: int = 0
    neutral: int = 0
    harmful: int = 0
    harmful_hits: int = 0
    final_hits: int = 0
    distance_delta_sum: float = 0.0

    def add(self, row: dict) -> None:
        self.n_paths += 1
        self.final_hits += int(bool(row.get("final_hit", False)))
        first_index = row.get("first_event_index")
        summaries = row.get("events", [])
        if first_index is None or int(first_index) >= len(summaries):
            return
        summary = summaries[int(first_index)]
        n_actions = int(summary.get("n_actions", 0))
        if n_actions <= 0:
            return
        before = int(summary.get("edit_distance_before", 0))
        after = int(summary.get("edit_distance_after", before))
        delta = before - after
        self.n_first += 1
        self.distance_delta_sum += delta
        if delta == n_actions:
            self.full_progress += 1
        elif delta > 0:
            self.partial_progress += 1
        elif delta == 0:
            self.neutral += 1
        else:
            self.harmful += 1
            self.harmful_hits += int(bool(row.get("final_hit", False)))

    def values(self) -> dict[str, float]:
        def ratio(numerator: int | float, denominator: int) -> float:
            return float(numerator / denominator) if denominator else 0.0

        return {
            "full_progress_rate": ratio(self.full_progress, self.n_first),
            "partial_progress_rate": ratio(self.partial_progress, self.n_first),
            "neutral_rate": ratio(self.neutral, self.n_first),
            "harmful_rate": ratio(self.harmful, self.n_first),
            "harmful_recovery_rate": ratio(self.harmful_hits, self.harmful),
            "final_hit_rate": ratio(self.final_hits, self.n_paths),
            "mean_distance_delta": ratio(self.distance_delta_sum, self.n_first),
        }


def _read_counts(root: Path, *, augmentation: int) -> tuple[dict, dict, set[int]]:
    """Read JSONL incrementally, retaining only reaction-level counters."""
    paths = sorted(root.glob("augmentation_*/per_trajectory.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no augmentation_*/per_trajectory.jsonl under {root}")

    per_view_reaction: dict[tuple[int, int], TrajectoryCounts] = defaultdict(
        TrajectoryCounts,
    )
    per_reaction: dict[int, TrajectoryCounts] = defaultdict(TrajectoryCounts)
    views: set[int] = set()
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "augmentation_index" not in row or "source_row_index" not in row:
                    raise ValueError(f"missing augmentation provenance in {path}")
                reaction = int(row["reaction_index"])
                view = int(row["augmentation_index"])
                source_row = int(row["source_row_index"])
                if not 0 <= view < augmentation:
                    raise ValueError(f"invalid augmentation_index={view} in {path}")
                if source_row != reaction * augmentation + view:
                    raise ValueError(
                        f"source-row mismatch in {path}: reaction={reaction}, "
                        f"view={view}, row={source_row}"
                    )
                per_view_reaction[(view, reaction)].add(row)
                per_reaction[reaction].add(row)
                views.add(view)
    return per_view_reaction, per_reaction, views


def _arrays(
    counts: dict,
    *,
    view: int | None = None,
) -> tuple[list[int], dict[str, np.ndarray]]:
    if view is None:
        grouped = counts
    else:
        grouped = {
            reaction: value
            for (candidate_view, reaction), value in counts.items()
            if candidate_view == view
        }
    reaction_ids = sorted(grouped)
    if not reaction_ids:
        raise ValueError("no reaction-level records found")
    return reaction_ids, {
        metric: np.asarray(
            [grouped[reaction].values()[metric] for reaction in reaction_ids],
            dtype=np.float64,
        )
        for metric in METRICS
    }


def _bootstrap_difference(
    m500: np.ndarray,
    atom: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    if m500.shape != atom.shape:
        raise ValueError(f"unpaired arrays: {m500.shape} vs {atom.shape}")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(atom), size=(n_bootstrap, len(atom)))
    difference = m500[indices].mean(axis=1) - atom[indices].mean(axis=1)
    return {
        "mean_difference": float(m500.mean() - atom.mean()),
        "ci95_low": float(np.quantile(difference, 0.025)),
        "ci95_high": float(np.quantile(difference, 0.975)),
        "n_reactions": int(len(atom)),
        "n_bootstrap": int(n_bootstrap),
    }


def _compare(
    atom_ids: list[int],
    atom_values: dict[str, np.ndarray],
    m500_ids: list[int],
    m500_values: dict[str, np.ndarray],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    if atom_ids != m500_ids:
        raise ValueError("Atom and M500 reaction indices do not match")
    return {
        metric: _bootstrap_difference(
            m500_values[metric], atom_values[metric],
            n_bootstrap=n_bootstrap, seed=seed + index,
        )
        for index, metric in enumerate(METRICS)
    }


def _overall(values: dict[str, np.ndarray]) -> dict[str, float]:
    return {metric: float(array.mean()) for metric, array in values.items()}


def _view_result(
    view: int,
    atom_counts: dict,
    m500_counts: dict,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    atom_ids, atom_values = _arrays(atom_counts, view=view)
    m500_ids, m500_values = _arrays(m500_counts, view=view)
    return {
        "augmentation_index": view,
        "atom": _overall(atom_values),
        "m500": _overall(m500_values),
        "paired_bootstrap": _compare(
            atom_ids, atom_values, m500_ids, m500_values,
            n_bootstrap=n_bootstrap, seed=seed,
        ),
    }


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _write_markdown(path: Path, result: dict) -> None:
    all_views = result["all_views"]
    lines = [
        "# 20 个 augmentation 的自然轨迹稳健性分析",
        "",
        "协议：改进后 global R-SMILES 的同一 1,000 条 reaction；每条使用全部 20 个",
        "R-SMILES augmentation，普通 Euler/cubic、100 steps、N=9、seed=42。",
        "统计时将同一 reaction 的 20×9 条轨迹先聚合，再以 reaction 为 bootstrap 单位。",
        "",
        "## 全部 20 个 view 聚合",
        "",
        "| 指标 | Atom@600K | SPE-M500@490K | M500−Atom | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        interval = all_views["paired_bootstrap"][metric]
        atom = all_views["atom"][metric]
        m500 = all_views["m500"][metric]
        lines.append(
            f"| {LABELS[metric]} | {atom:.4f} | {m500:.4f} | "
            f"{m500 - atom:+.4f} | [{interval['ci95_low']:+.4f}, "
            f"{interval['ci95_high']:+.4f}] |"
        )

    lines.extend([
        "",
        "## 每个 augmentation view 的关键差异",
        "",
        "| View | 完整改善：M500−Atom | 距离变差：M500−Atom | 最终命中：M500−Atom | 平均距离变化：M500−Atom |",
        "|---:|---:|---:|---:|---:|",
    ])
    for item in result["per_augmentation"]:
        atom = item["atom"]
        m500 = item["m500"]
        lines.append(
            f"| {item['augmentation_index']} | "
            f"{100 * (m500['full_progress_rate'] - atom['full_progress_rate']):+.2f} pp | "
            f"{100 * (m500['harmful_rate'] - atom['harmful_rate']):+.2f} pp | "
            f"{100 * (m500['final_hit_rate'] - atom['final_hit_rate']):+.2f} pp | "
            f"{m500['mean_distance_delta'] - atom['mean_distance_delta']:+.4f} |"
        )
    lines.extend([
        "",
        "说明：负的“距离变差差异”表示 M500 更少出现让 token 编辑距离增加的首事件。",
        "最终化学正确性仍以 canonical SMILES 命中为准。",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom_dir", type=Path, required=True)
    parser.add_argument("--m500_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    atom_view_counts, atom_counts, atom_views = _read_counts(
        args.atom_dir, augmentation=args.augmentation,
    )
    m500_view_counts, m500_counts, m500_views = _read_counts(
        args.m500_dir, augmentation=args.augmentation,
    )
    expected_views = set(range(args.augmentation))
    if atom_views != expected_views or m500_views != expected_views:
        raise ValueError(
            f"expected views={sorted(expected_views)}, got atom={sorted(atom_views)}, "
            f"m500={sorted(m500_views)}"
        )

    atom_ids, atom_values = _arrays(atom_counts)
    m500_ids, m500_values = _arrays(m500_counts)
    all_views = {
        "atom": _overall(atom_values),
        "m500": _overall(m500_values),
        "paired_bootstrap": _compare(
            atom_ids, atom_values, m500_ids, m500_values,
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        ),
    }
    per_augmentation = [
        _view_result(
            view, atom_view_counts, m500_view_counts,
            n_bootstrap=args.n_bootstrap, seed=args.seed + 100 * view,
        )
        for view in sorted(expected_views)
    ]
    result = {
        "protocol": {
            "augmentation": args.augmentation,
            "views": sorted(expected_views),
            "cluster_unit": "reaction_index",
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.seed,
        },
        "all_views": all_views,
        "per_augmentation": per_augmentation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _write_markdown(args.output_dir / "summary.md", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
