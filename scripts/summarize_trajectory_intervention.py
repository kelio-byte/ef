#!/usr/bin/env python
"""Aggregate controlled first-completion intervention experiments.

The unit of resampling is the reaction index.  Each intervention condition
contains multiple seeds and paths, so trajectories from the same reaction are
first pooled before the paired bootstrap is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "coverage",
    "forced_final_hit_rate",
    "forced_final_valid_rate",
    "mean_later_sub_del",
    "mean_later_distance_decrease",
)


def _read_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.glob("seed_*/per_trajectory.jsonl")):
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise FileNotFoundError(f"no seed_*/per_trajectory.jsonl under {root}")
    return rows


def _intervention_info(row: dict) -> dict | None:
    trace = row.get("trace", [])
    if not trace:
        return None
    info = trace[0].get("intervention")
    return info if isinstance(info, dict) else None


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
        first = [row for row in group if row.get("first_event_found")]
        applied = [
            row for row in first
            if (_intervention_info(row) or {}).get("applied")
        ]
        values["coverage"][offset] = len(applied) / len(first) if first else 0.0
        values["forced_final_hit_rate"][offset] = (
            sum(bool(row.get("final_hit")) for row in applied) / len(applied)
            if applied else 0.0
        )
        values["forced_final_valid_rate"][offset] = (
            sum(bool(row.get("final_valid")) for row in applied) / len(applied)
            if applied else 0.0
        )
        values["mean_later_sub_del"][offset] = (
            sum(float(row.get("later_sub_del_actions", 0.0)) for row in applied)
            / len(applied)
            if applied else 0.0
        )
        values["mean_later_distance_decrease"][offset] = (
            sum(
                float(row.get("later_distance_decrease_events", 0.0))
                for row in applied
            ) / len(applied)
            if applied else 0.0
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
        "bootstrap_mean": float(differences.mean()),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "n_reactions": int(len(left)),
        "n_bootstrap": int(n_bootstrap),
    }


def _condition_summary(rows: list[dict], mode: str) -> dict:
    first = [row for row in rows if row.get("first_event_found")]
    applied = [
        row for row in first
        if (_intervention_info(row) or {}).get("applied")
    ]
    by_type: dict[str, dict[str, float | int]] = {}
    for action_type in ("ins", "sub"):
        typed = [
            row for row in applied
            if (_intervention_info(row) or {}).get("type") == action_type
        ]
        by_type[action_type] = {
            "n": len(typed),
            "hit_rate": (
                sum(bool(row.get("final_hit")) for row in typed) / len(typed)
                if typed else 0.0
            ),
            "valid_rate": (
                sum(bool(row.get("final_valid")) for row in typed) / len(typed)
                if typed else 0.0
            ),
        }
    return {
        "mode": mode,
        "n_trajectories": len(rows),
        "n_first_event": len(first),
        "n_intervention_applied": len(applied),
        "coverage": len(applied) / len(first) if first else 0.0,
        "forced_final_hit_rate": (
            sum(bool(row.get("final_hit")) for row in applied) / len(applied)
            if applied else 0.0
        ),
        "forced_final_valid_rate": (
            sum(bool(row.get("final_valid")) for row in applied) / len(applied)
            if applied else 0.0
        ),
        "final_hit_rate_all": (
            sum(bool(row.get("final_hit")) for row in rows) / len(rows)
            if rows else 0.0
        ),
        "final_valid_rate_all": (
            sum(bool(row.get("final_valid")) for row in rows) / len(rows)
            if rows else 0.0
        ),
        "mean_later_sub_del": (
            sum(float(row.get("later_sub_del_actions", 0.0)) for row in applied)
            / len(applied)
            if applied else 0.0
        ),
        "mean_later_distance_decrease": (
            sum(
                float(row.get("later_distance_decrease_events", 0.0))
                for row in applied
            ) / len(applied)
            if applied else 0.0
        ),
        "by_type": by_type,
    }


def _write_markdown(
    path: Path,
    summaries: dict[str, dict[str, dict]],
    paired: dict[str, dict],
    causal: dict[str, dict],
) -> None:
    labels = {
        "coverage": "首事件干预覆盖率",
        "forced_final_hit_rate": "强制条件最终命中率",
        "forced_final_valid_rate": "强制条件最终有效率",
        "mean_later_sub_del": "首事件后的 SUB/DEL 平均次数",
        "mean_later_distance_decrease": "首事件后的距离下降事件",
    }
    lines = [
        "# P2 首 completion 因果干预汇总",
        "",
        "所有条件使用相同 reaction 集、100 Euler steps、每个 reaction 9 条轨迹和 seed 42/7/123；",
        "正确/错误条件只替换首个 oracle INS/SUB 位置的 completion token。bootstrap 以 reaction 为单位。",
        "",
        "## 条件结果",
        "",
        "| 模型 | 条件 | 覆盖率 | 最终命中 | 最终有效 | 后续 SUB/DEL | 后续距离下降 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("Atom", "SPE-M500"):
        for mode, label in (
            ("force_correct_completion_first", "force-correct"),
            ("force_wrong_completion_first", "force-wrong"),
        ):
            summary = summaries[model][mode]
            lines.append(
                f"| {model} | {label} | {summary['coverage']:.4f} | "
                f"{summary['forced_final_hit_rate']:.4f} | "
                f"{summary['forced_final_valid_rate']:.4f} | "
                f"{summary['mean_later_sub_del']:.4f} | "
                f"{summary['mean_later_distance_decrease']:.4f} |"
            )

    lines.extend([
        "",
        "## 首 completion 类型分层",
        "",
        "本轮在每条轨迹首个可用 oracle completion 上干预；类型分层用于说明实际覆盖到的是 INS 还是 SUB。",
        "",
        "| 模型 | 条件 | INS 数 | INS 命中 | SUB 数 | SUB 命中 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for model in ("Atom", "SPE-M500"):
        for mode, label in (
            ("force_correct_completion_first", "force-correct"),
            ("force_wrong_completion_first", "force-wrong"),
        ):
            by_type = summaries[model][mode]["by_type"]
            lines.append(
                f"| {model} | {label} | {by_type['ins']['n']} | "
                f"{by_type['ins']['hit_rate']:.4f} | {by_type['sub']['n']} | "
                f"{by_type['sub']['hit_rate']:.4f} |"
            )

    lines.extend([
        "",
        "## M500 − Atom 的配对差异",
        "",
        "| 指标 | force-correct 差异 | 95% CI | force-wrong 差异 | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ])
    for metric in METRICS:
        correct = paired["force_correct_completion_first"][metric]
        wrong = paired["force_wrong_completion_first"][metric]
        lines.append(
            f"| {labels[metric]} | {correct['mean_difference']:+.4f} | "
            f"[{correct['ci95_low']:+.4f}, {correct['ci95_high']:+.4f}] | "
            f"{wrong['mean_difference']:+.4f} | "
            f"[{wrong['ci95_low']:+.4f}, {wrong['ci95_high']:+.4f}] |"
        )

    lines.extend([
        "",
        "## 正确首编辑相对错误首编辑的恢复差距",
        "",
        "| 模型 | force-correct 命中 | force-wrong 命中 | correct−wrong |",
        "|---|---:|---:|---:|",
    ])
    for model in ("Atom", "SPE-M500"):
        item = causal[model]
        lines.append(
            f"| {model} | {item['correct_hit_rate']:.4f} | "
            f"{item['wrong_hit_rate']:.4f} | {item['hit_drop']:+.4f} |"
        )
    diff = causal["SPE-M500_minus_Atom"]
    lines.append(
        f"| M500−Atom 的 hit_drop 差异 |  |  | {diff['mean_difference']:+.4f} "
        f"[{diff['ci95_low']:+.4f}, {diff['ci95_high']:+.4f}] |"
    )
    lines.extend([
        "",
        "解释：force-wrong 不是让模型随机生成任意错误序列，而是在同一首 oracle INS/SUB 位置使用模型认为合法、"
        "且排除 oracle token 的最高概率 token；因此它主要检验错误首 completion 是否能被后续编辑纠正。",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom_correct", type=Path, required=True)
    parser.add_argument("--atom_wrong", type=Path, required=True)
    parser.add_argument("--m500_correct", type=Path, required=True)
    parser.add_argument("--m500_wrong", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    roots = {
        "Atom": {
            "force_correct_completion_first": args.atom_correct,
            "force_wrong_completion_first": args.atom_wrong,
        },
        "SPE-M500": {
            "force_correct_completion_first": args.m500_correct,
            "force_wrong_completion_first": args.m500_wrong,
        },
    }
    rows = {
        model: {mode: _read_rows(root) for mode, root in modes.items()}
        for model, modes in roots.items()
    }
    arrays = {}
    summaries = {}
    reaction_ids = None
    for model, modes in rows.items():
        arrays[model] = {}
        summaries[model] = {}
        for mode, condition_rows in modes.items():
            ids, values = _reaction_arrays(condition_rows)
            if reaction_ids is None:
                reaction_ids = ids
            elif reaction_ids != ids:
                raise ValueError(f"reaction indices do not match for {model}/{mode}")
            arrays[model][mode] = values
            summaries[model][mode] = _condition_summary(condition_rows, mode)

    paired = {}
    for mode_index, mode in enumerate(
        ("force_correct_completion_first", "force_wrong_completion_first")
    ):
        paired[mode] = {
            metric: _bootstrap_difference(
                arrays["SPE-M500"][mode][metric],
                arrays["Atom"][mode][metric],
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + mode_index * 100,
            )
            for metric in METRICS
        }

    causal = {}
    for model in ("Atom", "SPE-M500"):
        correct = arrays[model]["force_correct_completion_first"]["forced_final_hit_rate"]
        wrong = arrays[model]["force_wrong_completion_first"]["forced_final_hit_rate"]
        causal[model] = {
            "correct_hit_rate": float(correct.mean()),
            "wrong_hit_rate": float(wrong.mean()),
            "hit_drop": float((correct - wrong).mean()),
        }
    causal["SPE-M500_minus_Atom"] = _bootstrap_difference(
        arrays["SPE-M500"]["force_correct_completion_first"]["forced_final_hit_rate"]
        - arrays["SPE-M500"]["force_wrong_completion_first"]["forced_final_hit_rate"],
        arrays["Atom"]["force_correct_completion_first"]["forced_final_hit_rate"]
        - arrays["Atom"]["force_wrong_completion_first"]["forced_final_hit_rate"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 200,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": {
            "n_reactions": len(reaction_ids or []),
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.seed,
            "cluster_unit": "reaction_index",
            "atom_correct": str(args.atom_correct),
            "atom_wrong": str(args.atom_wrong),
            "m500_correct": str(args.m500_correct),
            "m500_wrong": str(args.m500_wrong),
        },
        "summaries": summaries,
        "paired_bootstrap": paired,
        "causal": causal,
    }
    (args.output_dir / "paired_bootstrap.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _write_markdown(args.output_dir / "summary.md", summaries, paired, causal)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
