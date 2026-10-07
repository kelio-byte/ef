#!/usr/bin/env python
"""Summarize the matched, order-invariant first-event intervention study.

The control and harmful runs are paired by ``(seed, reaction_index,
path_index)``.  Only rows for which both callbacks actually applied are used
for the causal comparison.  This avoids comparing different coverage pools
or treating an unsuccessful harmful-token search as an intervention.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


MODE_CONTROL = "progress_compatible_first"
MODE_HARMFUL = "force_harmful_completion_first"
MODELS = ("atom", "m500")


def _read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _load_mode(root: Path, model: str, mode: str) -> dict[tuple, dict]:
    mode_root = root / model / mode
    # The first smoke runs used short directory names; accept them so that
    # smoke outputs remain easy to audit while the full study uses the
    # explicit mode names above.
    if not mode_root.exists():
        aliases = {
            MODE_CONTROL: "progress",
            MODE_HARMFUL: "harmful",
        }
        mode_root = root / model / aliases.get(mode, mode)
    rows_by_key: dict[tuple, dict] = {}
    paths = sorted(mode_root.glob("seed_*/per_trajectory.jsonl"))
    direct_path = mode_root / "per_trajectory.jsonl"
    if not paths and direct_path.exists():
        paths = [direct_path]
    for path in paths:
        if path.parent == mode_root:
            summary_path = mode_root / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            seed = int(summary.get("seed", 0))
        else:
            seed = int(path.parent.name.split("_", 1)[1])
        for row in _read_rows(path):
            key = (seed, int(row["reaction_index"]), int(row["path_index"]))
            trace = row.get("trace", [])
            info = trace[0].get("intervention") if trace else None
            rows_by_key[key] = {
                "row": row,
                "info": info,
            }
    if not rows_by_key:
        raise FileNotFoundError(f"no per_trajectory.jsonl under {mode_root}")
    return rows_by_key


def _first_info(item: dict) -> dict:
    info = item.get("info")
    return info if isinstance(info, dict) else {}


def _matched_rows(root: Path, model: str) -> list[dict]:
    control = _load_mode(root, model, MODE_CONTROL)
    harmful = _load_mode(root, model, MODE_HARMFUL)
    matched = []
    for key in sorted(set(control) & set(harmful)):
        left = control[key]
        right = harmful[key]
        left_info = _first_info(left)
        right_info = _first_info(right)
        if not left_info.get("applied") or not right_info.get("applied"):
            continue
        harmful_info = right_info
        matched.append({
            "key": key,
            "reaction_index": key[1],
            "control": left["row"],
            "harmful": right["row"],
            "control_info": left_info,
            "harmful_info": harmful_info,
        })
    return matched


def _mean(rows: list[dict], side: str, field: str) -> float:
    values = [float(row[side].get(field, 0)) for row in rows]
    return float(np.mean(values)) if values else 0.0


def _hit(rows: list[dict], side: str) -> float:
    return _mean(rows, side, "final_hit")


def _valid(rows: list[dict], side: str) -> float:
    return _mean(rows, side, "final_valid")


def _group_by_reaction(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["reaction_index"])].append(row)
    return grouped


def _reaction_values(rows: list[dict], metric: str) -> tuple[list[int], np.ndarray]:
    grouped = _group_by_reaction(rows)
    reaction_ids = sorted(grouped)
    values = []
    for reaction_id in reaction_ids:
        group = grouped[reaction_id]
        if metric == "control_hit":
            value = _hit(group, "control")
        elif metric == "harmful_hit":
            value = _hit(group, "harmful")
        elif metric == "control_valid":
            value = _valid(group, "control")
        elif metric == "harmful_valid":
            value = _valid(group, "harmful")
        elif metric == "hit_drop":
            value = _hit(group, "control") - _hit(group, "harmful")
        elif metric == "valid_drop":
            value = _valid(group, "control") - _valid(group, "harmful")
        elif metric == "control_later_sub_del":
            value = _mean(group, "control", "later_sub_del_actions")
        elif metric == "harmful_later_sub_del":
            value = _mean(group, "harmful", "later_sub_del_actions")
        elif metric == "control_later_distance_decrease":
            value = _mean(group, "control", "later_distance_decrease_events")
        elif metric == "harmful_later_distance_decrease":
            value = _mean(group, "harmful", "later_distance_decrease_events")
        else:
            raise KeyError(metric)
        values.append(value)
    return reaction_ids, np.asarray(values, dtype=np.float64)


def _bootstrap(values: np.ndarray, *, n_bootstrap: int, seed: int) -> dict:
    if len(values) == 0:
        return {
            "mean": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "n_reactions": 0,
            "n_bootstrap": n_bootstrap,
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(values), size=(n_bootstrap, len(values)), endpoint=False,
    )
    samples = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "n_reactions": int(len(values)),
        "n_bootstrap": int(n_bootstrap),
    }


def _coverage(root: Path, model: str, mode: str) -> dict:
    rows = _load_mode(root, model, mode)
    first = 0
    applied = 0
    for item in rows.values():
        row = item["row"]
        info = _first_info(item)
        if row.get("first_event_found"):
            first += 1
        if info.get("applied"):
            applied += 1
    return {
        "n_trajectories": len(rows),
        "n_first_event": first,
        "n_applied": applied,
        "coverage_among_first_events": applied / first if first else 0.0,
    }


def _model_result(
    root: Path,
    model: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    matched = _matched_rows(root, model)
    metrics = (
        "control_hit",
        "harmful_hit",
        "control_valid",
        "harmful_valid",
        "hit_drop",
        "valid_drop",
        "control_later_sub_del",
        "harmful_later_sub_del",
        "control_later_distance_decrease",
        "harmful_later_distance_decrease",
    )
    reaction_ids = None
    bootstrap = {}
    for offset, metric in enumerate(metrics):
        ids, values = _reaction_values(matched, metric)
        if reaction_ids is None:
            reaction_ids = ids
        elif reaction_ids != ids:
            raise RuntimeError("reaction grouping changed across metrics")
        bootstrap[metric] = _bootstrap(
            values, n_bootstrap=n_bootstrap, seed=seed + offset,
        )

    damages = [
        int(item["harmful_info"].get("damage", -1))
        for item in matched
    ]
    action_types = [
        str(item["harmful_info"].get("type", "unknown"))
        for item in matched
    ]
    return {
        "coverage": {
            "control": _coverage(root, model, MODE_CONTROL),
            "harmful": _coverage(root, model, MODE_HARMFUL),
        },
        "n_matched_paths": len(matched),
        "matched_reactions": len(reaction_ids or []),
        "damage_counts": {
            str(value): damages.count(value) for value in sorted(set(damages))
        },
        "harmful_action_type_counts": {
            value: action_types.count(value) for value in sorted(set(action_types))
        },
        "bootstrap_by_reaction": bootstrap,
    }


def _cross_model_bootstrap(
    atom_rows: list[dict],
    m500_rows: list[dict],
    metric: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    atom_ids, atom_values = _reaction_values(atom_rows, metric)
    m500_ids, m500_values = _reaction_values(m500_rows, metric)
    if atom_ids != m500_ids:
        common = sorted(set(atom_ids) & set(m500_ids))
        atom_map = dict(zip(atom_ids, atom_values))
        m500_map = dict(zip(m500_ids, m500_values))
        atom_values = np.asarray([atom_map[i] for i in common])
        m500_values = np.asarray([m500_map[i] for i in common])
    difference = m500_values - atom_values
    return _bootstrap(difference, n_bootstrap=n_bootstrap, seed=seed)


def _write_markdown(path: Path, result: dict) -> None:
    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    lines = [
        "# 顺序无关的首编辑干预实验",
        "",
        "目的：验证“首个错误插入后，模型是否缺少后续 SUB/DEL 纠错能力”这一 Motivation。",
        "控制组保留自然采样中首个**完整改善事件**；干预组只改动其中一个 INS/SUB token，",
        "并要求首事件后的 token 编辑距离比控制组恰好增加 1。两组按 seed、reaction、path 配对，",
        "只统计两边都成功应用干预的路径；因此不会把未找到合适有害 token 的样本混进比较。",
        "",
        "## 覆盖率与配对质量",
        "",
        "| 模型 | 控制首事件数 | 控制应用数 | 有害首事件数 | 有害应用数 | 配对路径数 | 配对反应数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, label in (("atom", "Atom@600K"), ("m500", "SPE-M500@490K")):
        item = result["models"][model]
        control = item["coverage"]["control"]
        harmful = item["coverage"]["harmful"]
        lines.append(
            f"| {label} | {control['n_first_event']} | {control['n_applied']} | "
            f"{harmful['n_first_event']} | {harmful['n_applied']} | "
            f"{item['n_matched_paths']} | {item['matched_reactions']} |"
        )
    lines.extend([
        "",
        "## 配对结果",
        "",
        "命中率/有效率按反应聚类 bootstrap；后续 SUB/DEL 和后续距离下降是每条路径的平均次数。",
        "",
        "| 模型 | 控制最终命中率（逐路径） | 有害最终命中率（逐路径） | 下降 | 控制 Valid | 有害 Valid | 后续 SUB/DEL（控制→有害） | 后续距离下降（控制→有害） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for model, label in (("atom", "Atom@600K"), ("m500", "SPE-M500@490K")):
        item = result["models"][model]["bootstrap_by_reaction"]
        lines.append(
            f"| {label} | {pct(item['control_hit']['mean'])} | "
            f"{pct(item['harmful_hit']['mean'])} | {pct(item['hit_drop']['mean'])} | "
            f"{pct(item['control_valid']['mean'])} | {pct(item['harmful_valid']['mean'])} | "
            f"{item['control_later_sub_del']['mean']:.3f} → {item['harmful_later_sub_del']['mean']:.3f} | "
            f"{item['control_later_distance_decrease']['mean']:.3f} → "
            f"{item['harmful_later_distance_decrease']['mean']:.3f} |"
        )
    lines.extend([
        "",
        "## 反事实干预的跨模型比较",
        "",
        "| 指标（SPE-M500 − Atom） | 差异 | 95% CI |",
        "|---|---:|---:|",
    ])
    for metric, label in (
        ("control_hit", "控制最终命中率（逐路径）"),
        ("harmful_hit", "有害最终命中率（逐路径）"),
        ("hit_drop", "最终命中率下降"),
        ("control_later_sub_del", "控制后续 SUB/DEL"),
        ("harmful_later_sub_del", "有害后续 SUB/DEL"),
    ):
        item = result["cross_model"][metric]
        lines.append(
            f"| {label} | {item['mean']:+.4f} | "
            f"[{item['ci95_low']:+.4f}, {item['ci95_high']:+.4f}] |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "这里的“有害”是一个可控的 token-level 反事实：它只保证编辑距离立即增加 1，",
        "不等同于化学上必然错误。最终结论仍看 canonical SMILES 命中。若有害组命中率下降，",
        "说明首步错误会造成可测的路径损失；若后续 SUB/DEL 次数没有增加或不足以恢复，",
        "才支持“纠错监督不足”这一更具体的机制解释。",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    model_results = {
        model: _model_result(
            args.root, model,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 100 * index,
        )
        for index, model in enumerate(MODELS)
    }
    atom_rows = _matched_rows(args.root, "atom")
    m500_rows = _matched_rows(args.root, "m500")
    cross_metrics = (
        "control_hit",
        "harmful_hit",
        "hit_drop",
        "control_later_sub_del",
        "harmful_later_sub_del",
        "control_later_distance_decrease",
        "harmful_later_distance_decrease",
    )
    cross_model = {
        metric: _cross_model_bootstrap(
            atom_rows, m500_rows, metric,
            n_bootstrap=args.n_bootstrap, seed=args.seed + 1000 + index,
        )
        for index, metric in enumerate(cross_metrics)
    }
    result = {
        "protocol": {
            "control_mode": MODE_CONTROL,
            "harmful_mode": MODE_HARMFUL,
            "pairing_key": "seed/reaction_index/path_index",
            "reaction_cluster_bootstrap": True,
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.seed,
            "harmful_damage": 1,
        },
        "models": model_results,
        "cross_model": cross_model,
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
