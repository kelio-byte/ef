#!/usr/bin/env python
"""Aggregate natural trajectory correction traces by reaction cluster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "first_off_oracle_rate",
    "natural_recovery_rate",
    "clean_path_success",
    "final_hit_rate",
    "final_valid_rate",
    "mean_later_sub_del_after_first_off",
    "mean_later_distance_decrease_events_after_first_off",
)


def _read_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("seed_*/per_trajectory.jsonl")):
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise FileNotFoundError(f"no seed_*/per_trajectory.jsonl under {root}")
    return rows


def _reaction_arrays(rows: list[dict]) -> tuple[list[int], dict[str, np.ndarray]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["reaction_index"]), []).append(row)
    reaction_ids = sorted(grouped)
    values = {
        metric: np.zeros(len(reaction_ids), dtype=np.float64)
        for metric in METRICS
    }

    for reaction_offset, reaction_id in enumerate(reaction_ids):
        group = grouped[reaction_id]
        first = [row for row in group if row.get("first_event_found")]
        off = [row for row in first if row.get("first_event_off_oracle")]
        clean = [
            row for row in first
            if row.get("first_event_fully_oracle_consistent")
        ]
        values["first_off_oracle_rate"][reaction_offset] = (
            len(off) / len(first) if first else 0.0
        )
        values["natural_recovery_rate"][reaction_offset] = (
            sum(bool(row.get("final_hit")) for row in off) / len(off)
            if off else 0.0
        )
        values["clean_path_success"][reaction_offset] = (
            sum(bool(row.get("final_hit")) for row in clean) / len(clean)
            if clean else 0.0
        )
        values["final_hit_rate"][reaction_offset] = (
            sum(bool(row.get("final_hit")) for row in group) / len(group)
            if group else 0.0
        )
        values["final_valid_rate"][reaction_offset] = (
            sum(bool(row.get("final_valid")) for row in group) / len(group)
            if group else 0.0
        )
        values["mean_later_sub_del_after_first_off"][reaction_offset] = (
            sum(float(row.get("later_sub_del_actions", 0.0)) for row in off)
            / len(off)
            if off else 0.0
        )
        values[
            "mean_later_distance_decrease_events_after_first_off"
        ][reaction_offset] = (
            sum(
                float(row.get("later_distance_decrease_events", 0.0))
                for row in off
            )
            / len(off)
            if off else 0.0
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
        raise ValueError(
            f"paired arrays have different shapes: {left.shape} vs {right.shape}"
        )
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0, len(left), size=(n_bootstrap, len(left)), endpoint=False,
    )
    differences = (
        left[sample_indices].mean(axis=1)
        - right[sample_indices].mean(axis=1)
    )
    return {
        "m500_minus_atom_mean": float(left.mean() - right.mean()),
        "bootstrap_mean": float(differences.mean()),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "n_reactions": int(len(left)),
        "n_bootstrap": int(n_bootstrap),
    }


def _write_markdown(
    path: Path,
    atom_overall: dict,
    m500_overall: dict,
    paired: dict,
) -> None:
    lines = [
        "# P1 自然轨迹汇总（历史：固定对齐路径版本）",
        "",
        "> **不要把本表作为最终机制结论。** 它以一条固定的 Levenshtein 对齐路径定义“首个偏离”，",
        "> 但同一对序列可能有多条同样合理的编辑顺序。因此本表只保留作历史记录；",
        "> 最终应阅读 `../order_invariant_summary/summary.md` 中按实际距离变化得到的顺序无关结果。",
        "",
        "所有 seed 的 9 条路径合并后，按 reaction block 聚合；"
        "bootstrap 以 reaction 为重采样单位。",
        "",
        "| 指标 | Atom@600K | SPE-M500@490K | M500−Atom | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "first_off_oracle_rate": "首个偏离固定对齐路径的比例（历史）",
        "natural_recovery_rate": "首个偏离后最终命中率（历史）",
        "clean_path_success": "首事件符合固定对齐路径后的最终命中率（历史）",
        "final_hit_rate": "最终 canonical 命中率",
        "final_valid_rate": "最终有效 SMILES 比例",
        "mean_later_sub_del_after_first_off": "首个偏离后的后续 SUB/DEL 次数",
        "mean_later_distance_decrease_events_after_first_off":
            "首个偏离后的后续距离下降事件数",
    }
    for metric in METRICS:
        atom_value = atom_overall[metric]
        m500_value = m500_overall[metric]
        interval = paired[metric]
        lines.append(
            f"| {labels[metric]} | {atom_value:.4f} | {m500_value:.4f} | "
            f"{m500_value - atom_value:+.4f} | "
            f"[{interval['ci95_low']:+.4f}, "
            f"{interval['ci95_high']:+.4f}] |"
        )
    lines.extend([
        "",
        "注意：前两项的分母只包含检测到首个真实事件的路径；",
        "没有首事件的路径另行保留在原始 JSONL 中，未被强行归入恢复率分母。",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom_dir", type=Path, required=True)
    parser.add_argument("--m500_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    atom_rows = _read_rows(args.atom_dir)
    m500_rows = _read_rows(args.m500_dir)
    atom_ids, atom_values = _reaction_arrays(atom_rows)
    m500_ids, m500_values = _reaction_arrays(m500_rows)
    if atom_ids != m500_ids:
        raise ValueError("Atom and M500 reaction indices do not match")

    atom_overall = {
        metric: float(atom_values[metric].mean()) for metric in METRICS
    }
    m500_overall = {
        metric: float(m500_values[metric].mean()) for metric in METRICS
    }
    paired = {
        metric: _bootstrap_difference(
            m500_values[metric],
            atom_values[metric],
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + index,
        )
        for index, metric in enumerate(METRICS)
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": {
            "atom_dir": str(args.atom_dir),
            "m500_dir": str(args.m500_dir),
            "n_reactions": len(atom_ids),
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.seed,
            "cluster_unit": "reaction_index",
        },
        "atom_overall": atom_overall,
        "m500_overall": m500_overall,
        "paired_bootstrap": paired,
    }
    (output_dir / "paired_bootstrap.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _write_markdown(
        output_dir / "summary.md", atom_overall, m500_overall, paired,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
