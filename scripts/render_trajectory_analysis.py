#!/usr/bin/env python
"""Render one analysis page from an existing compressed trajectory report.

This reads recorded trajectories only. It does not load a checkpoint or rerun
model inference.
"""

from __future__ import annotations

import argparse
import gzip
from html import escape
import json
from pathlib import Path
from statistics import mean

from visualize_trajectory import _render_html


def _quartile_counts(trace: list[dict]) -> list[int]:
    """Count model evaluations by their starting time in four half-open bins."""
    return [
        sum(quarter / 4 <= step["t_start"] < (quarter + 1) / 4 for step in trace)
        for quarter in range(4)
    ]


def _outcome(method: dict) -> str:
    if method["correct"]:
        return "命中目标"
    if method["final_canonical"]:
        return "未命中目标"
    return "无效分子"


def render_analysis(report: dict) -> str:
    """Add a linked comparison table and conclusions to existing case cards."""
    cases = report["cases"]
    if not cases:
        raise ValueError("Trajectory report has no cases")
    reaction_ids = [case["reaction_index"] for case in cases]
    if len(set(reaction_ids)) != len(reaction_ids):
        raise ValueError("Analysis page needs one case per reaction")

    rows = []
    baseline_nfe = []
    accelerated_nfe = []
    before_hits = after_hits = same_outputs = 0
    quarter_totals = [[0, 0] for _ in range(4)]
    differing = []
    late_increases = []
    for case in cases:
        before, after = case["baseline"], case["hazard"]
        before_counts = _quartile_counts(before["trace"])
        after_counts = _quartile_counts(after["trace"])
        if sum(before_counts) != before["nfe"] or sum(after_counts) != after["nfe"]:
            raise ValueError(f"Quartile counts do not match NFE for reaction {case['reaction_index']}")
        for quarter, (before_count, after_count) in enumerate(zip(before_counts, after_counts)):
            quarter_totals[quarter][0] += before_count
            quarter_totals[quarter][1] += after_count
        baseline_nfe.append(before["nfe"])
        accelerated_nfe.append(after["nfe"])
        before_hits += bool(before["correct"])
        after_hits += bool(after["correct"])
        same = before["final_tokens"] == after["final_tokens"]
        same_outputs += same
        if not same:
            differing.append(
                f'{case["reaction_index"]}（{_outcome(before)} → {_outcome(after)}）'
            )
        reaction_id = case["reaction_index"]
        if after_counts[3] > before_counts[3]:
            late_increases.append(
                f'{reaction_id}（{before_counts[3]} → {after_counts[3]} 次）'
            )
        quarters = "".join(
            f"<td>{old} → {new}</td>"
            for old, new in zip(before_counts, after_counts)
        )
        rows.append(
            f'<tr><td><a href="#reaction-{reaction_id}">{reaction_id}</a></td>'
            + quarters
            + f'<td>{before["nfe"]} → {after["nfe"]}</td>'
            + f'<td>{escape(_outcome(before))} → {escape(_outcome(after))}</td>'
            + f'<td>{"是" if same else "否"}</td></tr>'
        )

    quarter_summary = "；".join(
        f"第{quarter + 1}段 {before} → {after} 次"
        for quarter, (before, after) in enumerate(quarter_totals)
    )
    changed_summary = "、".join(differing) if differing else "无"
    late_summary = "、".join(late_increases) if late_increases else "无"
    overview = (
        f'<section id="overview"><h2>{len(cases)} 个案例汇总</h2>'
        '<p>按模型调用开始时的时间 t，把 [0,1) 分成四段：'
        '[0,0.25)、[0.25,0.50)、[0.50,0.75)、[0.75,1)。'
        '表中箭头均表示“加速前 → 加速后”；四段调用数之和等于全程 NFE。'
        '点击反应行号可跳到同一页面中的完整轨迹。</p>'
        '<div class="scroll"><table><thead><tr><th>反应行号</th>'
        '<th>第1段</th><th>第2段</th><th>第3段</th><th>第4段</th>'
        '<th>全程 NFE</th><th>单条轨迹结果</th><th>最终 token 相同</th>'
        '</tr></thead><tbody>'
        + "".join(rows)
        + '</tbody></table></div>'
        '<h3>观察</h3><ul>'
        f'<li>平均 NFE：{mean(baseline_nfe):.1f} → {mean(accelerated_nfe):.1f}；'
        f'加速后范围 {min(accelerated_nfe)}～{max(accelerated_nfe)} 次。'
        'NFE 是模型评估次数，不是单案例的墙钟加速倍数。</li>'
        f'<li>四段调用数合计：{quarter_summary}。'
        f'第 4 段调用变多的案例：{late_summary}。</li>'
        f'<li>最终 token 完全相同：{same_outputs}/{len(cases)}；'
        f'单条轨迹命中目标：{before_hits}/{len(cases)} → {after_hits}/{len(cases)}；'
        f'最终输出不同的反应行号：{changed_summary}。</li>'
        '</ul><p>这些案例只展示轨迹行为；单条轨迹是否命中目标不等于反应级 Top-k，'
        '少量案例不能代替完整数据集的质量结论。</p></section>'
    )

    html = _render_html(report)
    for reaction_id in reaction_ids:
        marker = f'<section><h2>反应 {reaction_id} ·'
        if html.count(marker) != 1:
            raise ValueError(f"Cannot locate case card for reaction {reaction_id}")
        html = html.replace(
            marker, f'<section id="reaction-{reaction_id}"><h2>反应 {reaction_id} ·', 1
        )
    first_case = html.find('<section id="reaction-')
    if first_case < 0:
        raise ValueError("No case cards found in rendered HTML")
    return html[:first_case] + overview + html[first_case:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing trajectory JSON.gz")
    parser.add_argument("--output", type=Path, required=True, help="Standalone analysis HTML")
    parser.add_argument("--force", action="store_true", help="Replace an existing HTML file")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        parser.error("output exists; use --force to replace it")
    with gzip.open(args.input, "rt", encoding="utf-8") as handle:
        report = json.load(handle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_analysis(report), encoding="utf-8")
    print(f"Rendered {len(report['cases'])} recorded cases to {args.output}")


if __name__ == "__main__":
    main()
