#!/usr/bin/env python
"""Visualize inference trajectories before and after acceleration.

The tool reruns complete 32-product batches with R=9 and M=2, preserving the
official per-product seeds. Optional reference predictions verify that a traced
trajectory reproduces the corresponding completed experiment output.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
from html import escape
import json
from pathlib import Path

import torch

from edit_flows.core.scheduler import CubicScheduler
from edit_flows.inference import CHECKPOINT, DATA, load_model, make_batch, sha256
from edit_flows.sampling.branch_sampler import sample_branches
from edit_flows.sampling.branch_sampler_helpers import _mix_child_seed
from edit_flows.scoring import canonicalize_smiles_clear_map
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN, UNK_TOKEN


BATCH_SIZE = 32
N_RUNS = 9
N_CHILDREN = 2
SEED = 42


def _indices(value: str) -> list[int]:
    try:
        indices = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--indices must be comma-separated integers") from exc
    if not indices or any(index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("--indices must be distinct nonnegative integers")
    return sorted(indices)


def _tokens(ids: list[int], id2token: dict[int, str]) -> list[str]:
    return [id2token.get(int(token), "<UNK>") for token in ids if token not in (BOS_TOKEN, PAD_TOKEN)]


def _canonical(tokens: list[str]) -> str:
    try:
        return canonicalize_smiles_clear_map("".join(tokens))[0]
    except Exception:
        return ""


def _reference_lines(path: Path | None, n_products: int, checkpoint_hash: str, source_hash: str) -> list[str] | None:
    if path is None:
        return None
    metadata_path = path.with_name("sampling_metadata.json")
    metadata = json.loads(metadata_path.read_text())
    if metadata["n_products"] != n_products or metadata["n_runs"] != N_RUNS:
        raise ValueError(f"Reference layout differs from R={N_RUNS}: {path}")
    if metadata["checkpoint_sha256"] != checkpoint_hash or metadata["products_sha256"] != source_hash:
        raise ValueError(f"Reference checkpoint or products differ: {path}")
    if sha256(path) != metadata["predictions_sha256"]:
        raise ValueError(f"Reference prediction SHA-256 mismatch: {path}")
    lines = path.read_text().splitlines()
    if len(lines) != n_products * N_RUNS:
        raise ValueError(f"Reference line count differs from metadata: {path}")
    return lines


def _run_batch(model, cfg, token2id, products: list[str], start: int, selected: list[int], run_index: int, device: torch.device, rate_detail_step: int | None):
    product_ids = [[token2id.get(token, UNK_TOKEN) for token in line.split()] for line in products]
    x_unique = make_batch(product_ids, device)
    x_0 = x_unique.repeat_interleave(N_RUNS, dim=0)
    padding = x_unique == PAD_TOKEN
    memory = model.encode_product(x_unique, padding).repeat_interleave(N_RUNS, dim=0)
    padding = padding.repeat_interleave(N_RUNS, dim=0)
    seeds = [
        _mix_child_seed(SEED, start + product_index, run + 1)
        for product_index in range(len(products))
        for run in range(N_RUNS)
    ]
    traced_rows = {(index - start) * N_RUNS + run_index for index in selected}
    results = {}
    for name, time_grid, max_nfe in (("baseline", "adaptive", 100), ("hazard", "hazard", 500)):
        trace_steps = {row: [] for row in traced_rows}
        trace_rate_steps = (
            {row: {rate_detail_step - 1} for row in traced_rows}
            if rate_detail_step is not None else None
        )
        output = sample_branches(
            model,
            x_0,
            CubicScheduler(),
            seeds,
            product_memory=memory,
            product_memory_padding_mask=padding,
            n_steps=max_nfe,
            time_grid=time_grid,
            hazard_max_step=0.04,
            hazard_limit=0.15,
            max_seq_len=cfg["max_seq_len"],
            n_children=N_CHILDREN,
            changed_state_bonus=0.5,
            trace_steps=trace_steps,
            trace_rate_steps=trace_rate_steps,
        )
        results[name] = (output, trace_steps)
    return results


def _method_result(output, trace: list[dict], row: int, id2token: dict[int, str]) -> dict:
    final_tokens = _tokens(output[row].tolist(), id2token)
    if not trace:
        raise RuntimeError(f"No trace steps recorded for batch row {row}")
    for step in trace:
        for position in step.get("position_rates", ()):
            position["token"] = id2token.get(position["token_id"], "<UNK>")
        step["before_tokens"] = _tokens(step.pop("before_ids"), id2token)
        step["after_tokens"] = _tokens(step.pop("after_ids"), id2token)
        step["state_changed"] = step["before_tokens"] != step["after_tokens"]
    if trace[-1]["after_tokens"] != final_tokens or abs(trace[-1]["t_end"] - 1.0) > 1e-5:
        raise RuntimeError(f"Trace and final state disagree for batch row {row}")
    return {
        "nfe": len(trace),
        "changed_steps": sum(step["state_changed"] for step in trace),
        "sampled_event_steps": sum(
            step["insert_events"] + step["substitute_events"] + step["delete_events"] > 0
            for step in trace
        ),
        "final_tokens": final_tokens,
        "final_canonical": _canonical(final_tokens),
        "trace": trace,
    }


def _timeline(case: dict) -> str:
    left, span = 115, 850
    x = lambda t: left + span * max(0.0, min(1.0, t))
    parts = ['<svg viewBox="0 0 1000 160" role="img" aria-label="模型调用时间轴">']
    for tick in (0, 0.25, 0.5, 0.75, 1):
        px = x(tick)
        parts.append(f'<line x1="{px:.1f}" y1="20" x2="{px:.1f}" y2="130" stroke="#dce3eb"/>')
        parts.append(f'<text x="{px:.1f}" y="148" text-anchor="middle" fill="#526174" font-size="12">{tick:g}</text>')
    for name, label, y, color in (("baseline", "加速前", 52, "#3867b0"), ("hazard", "加速后", 104, "#bb5638")):
        parts.append(f'<text x="6" y="{y+4}" fill="{color}" font-size="14">{label}</text>')
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left+span}" y2="{y}" stroke="{color}" opacity=".45"/>')
        for step in case[name]["trace"]:
            px = x(step["t_start"])
            parts.append(f'<line x1="{px:.1f}" y1="{y-10}" x2="{px:.1f}" y2="{y+10}" stroke="{color}" stroke-width="1.4"/>')
            if step["state_changed"]:
                px = x(step["t_end"])
                parts.append(f'<circle cx="{px:.1f}" cy="{y}" r="4" fill="{color}"/>')
    parts.append('</svg>')
    return "".join(parts)


def _step_table(name: str, data: dict) -> str:
    rows = []
    for step in data["trace"]:
        before = escape(" ".join(step["before_tokens"]))
        after = escape(" ".join(step["after_tokens"]))
        changed = "是" if step["state_changed"] else "否"
        events = f'{step["insert_events"]}/{step["substitute_events"]}/{step["delete_events"]}'
        rows.append(
            f'<tr><td>{step["step"]+1}</td><td>{step["t_start"]:.4f}→{step["t_end"]:.4f}</td>'
            f'<td>{step["h"]:.4f}</td><td>{step["edit_hazard"]:.3f}</td>'
            f'<td>{events}</td><td>{changed}</td><td><code>{before}</code></td><td><code>{after}</code></td></tr>'
        )
    return (
        f'<details><summary>{escape(name)}：查看全部 {data["nfe"]} 次模型调用</summary>'
        '<div class="scroll"><table><thead><tr><th>NFE</th><th>时间 t</th><th>步长 h</th>'
        '<th>编辑总强度 Λ</th><th>插/替/删采样标记数</th><th>保留状态变化</th>'
        '<th>步前 token</th><th>步后 token</th></tr></thead><tbody>'
        + "".join(rows) + '</tbody></table></div></details>'
    )


def _render_html(report: dict) -> str:
    cards = []
    for case in report["cases"]:
        base, fast = case["baseline"], case["hazard"]
        same = base["final_canonical"] == fast["final_canonical"] and bool(base["final_canonical"])
        cards.append(
            f'<section><h2>反应 {case["reaction_index"]} · 增强视图 {case["view_index"]} · '
            f'采样 {case["run_index"]+1}</h2>'
            f'<p>输入行号（从 0 起）：{case["product_index"]}。蓝色竖线与红色竖线分别表示一次模型调用；'
            '圆点表示保留下来的分子状态发生变化。</p>'
            f'<p><b>产物：</b><code>{escape(" ".join(case["product_tokens"]))}</code><br>'
            f'<b>目标：</b><code>{escape(" ".join(case["target_tokens"]))}</code></p>'
            + _timeline(case)
            + '<div class="stats">'
            + f'<div><b>加速前</b><br>NFE {base["nfe"]} · 状态变化 {base["changed_steps"]} 次'
            + f'<br>单轨迹命中目标：{"是" if base["correct"] else "否"}'
            + f'<br><code>{escape(" ".join(base["final_tokens"]))}</code></div>'
            + f'<div><b>加速后</b><br>NFE {fast["nfe"]} · 状态变化 {fast["changed_steps"]} 次'
            + f'<br>单轨迹命中目标：{"是" if fast["correct"] else "否"}'
            + f'<br><code>{escape(" ".join(fast["final_tokens"]))}</code></div>'
            + '</div>'
            + f'<p>两种方法的最终规范化分子{"相同" if same else "不同或无效"}。'
            '单条轨迹是否命中目标，不等于反应级 Top-k。</p>'
            + _step_table("加速前", base) + _step_table("加速后", fast)
            + '</section>'
        )
    return (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>推理轨迹可视化分析</title>'
        '<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 22px;color:#172536;background:#f7f9fc}'
        'section{background:white;border:1px solid #dfe7ef;border-radius:12px;padding:22px;margin:24px 0}'
        'h1,h2{line-height:1.25}svg{width:100%;height:auto;background:#fafcff;border:1px solid #e7edf4;border-radius:8px}'
        '.stats{display:grid;grid-template-columns:1fr 1fr;gap:14px}.stats>div{background:#f2f6fa;padding:12px;border-radius:8px;overflow-wrap:anywhere}'
        'code{font:13px/1.4 ui-monospace,monospace;overflow-wrap:anywhere}.scroll{overflow-x:auto}'
        'table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e1e7ef;padding:6px;vertical-align:top;text-align:left}'
        'details{margin:12px 0}summary{cursor:pointer;font-weight:600}</style><body>'
        '<h1>推理轨迹可视化分析</h1>'
        '<p>每条竖线代表一次模型调用；NFE 是模型调用次数，不是单案例的墙钟加速倍数。</p>'
        + "".join(cards) + '</body></html>'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products-file", type=Path, default=DATA / "dev1000/src.txt")
    parser.add_argument("--targets-file", type=Path, default=DATA / "dev1000/tgt.txt")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--vocab", type=Path, default=DATA / "example.vocab.src")
    parser.add_argument("--indices", required=True, type=_indices, help="0-based product-view line indices, comma-separated")
    parser.add_argument("--run-index", type=int, default=0, help="0-based independent run index in 0..8")
    parser.add_argument("--rate-detail-step", type=int, help="1-based model call for saving per-position masked edit rates")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument("--hazard-predictions", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Standalone HTML output; compressed JSON is written beside it")
    parser.add_argument("--per-case-html", action="store_true", help="Also write one reaction-index-trajectory.html file per selected reaction")
    parser.add_argument("--force", action="store_true", help="Replace existing HTML and JSON.gz outputs")
    args = parser.parse_args()
    if not 0 <= args.run_index < N_RUNS:
        parser.error(f"--run-index must be in 0..{N_RUNS-1}")
    if args.rate_detail_step is not None and args.rate_detail_step < 1:
        parser.error("--rate-detail-step must be positive")
    if args.output.suffix.lower() != ".html":
        parser.error("--output must end in .html")
    if args.per_case_html and len({index // 20 for index in args.indices}) != len(args.indices):
        parser.error("--per-case-html requires distinct reaction indices")
    json_path = args.output.with_suffix(".json.gz")
    case_paths = (
        [args.output.with_name(f"{index // 20}-trajectory.html") for index in args.indices]
        if args.per_case_html else []
    )
    if args.output in case_paths:
        parser.error("--output must differ from per-reaction HTML filenames")
    if not args.force and any(path.exists() for path in (args.output, json_path, *case_paths)):
        parser.error("output exists; use --force to replace it")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    model, cfg, token2id = load_model(args.checkpoint, args.vocab, device)
    model.eval()
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    products = args.products_file.read_text().splitlines()
    targets = args.targets_file.read_text().splitlines()
    if len(products) != len(targets) or max(args.indices) >= len(products):
        raise ValueError("Product/target lengths or selected indices are invalid")
    id2token = {index: token for token, index in token2id.items()}
    checkpoint_hash = sha256(args.checkpoint)
    source_hash = sha256(args.products_file)
    references = {
        "baseline": _reference_lines(args.baseline_predictions, len(products), checkpoint_hash, source_hash),
        "hazard": _reference_lines(args.hazard_predictions, len(products), checkpoint_hash, source_hash),
    }
    grouped = defaultdict(list)
    for index in args.indices:
        grouped[index // BATCH_SIZE * BATCH_SIZE].append(index)
    cases = []
    with torch.inference_mode():
        for start, indices in sorted(grouped.items()):
            print(f"Tracing batch {start // BATCH_SIZE + 1}: product lines {indices}")
            batch = products[start : start + BATCH_SIZE]
            results = _run_batch(model, cfg, token2id, batch, start, indices, args.run_index, device, args.rate_detail_step)
            for index in indices:
                row = (index - start) * N_RUNS + args.run_index
                target_tokens = targets[index].split()
                target_canonical = _canonical(target_tokens)
                case = {
                    "product_index": index,
                    "reaction_index": index // 20,
                    "view_index": index % 20,
                    "run_index": args.run_index,
                    "product_tokens": products[index].split(),
                    "target_tokens": target_tokens,
                    "target_canonical": target_canonical,
                }
                for name in ("baseline", "hazard"):
                    output, trace_map = results[name]
                    method = _method_result(output, trace_map[row], row, id2token)
                    method["correct"] = bool(target_canonical and method["final_canonical"] == target_canonical)
                    reference = references[name]
                    if reference is not None:
                        predicted_line = " ".join(method["final_tokens"])
                        if predicted_line != reference[index * N_RUNS + args.run_index]:
                            raise RuntimeError(f"{name} trace does not reproduce recorded prediction at product {index}, run {args.run_index}")
                        method["reference_match"] = True
                    case[name] = method
                cases.append(case)
                print(f'  product {index}: NFE {case["baseline"]["nfe"]} -> {case["hazard"]["nfe"]}; '
                      f'correct {case["baseline"]["correct"]} -> {case["hazard"]["correct"]}')
    report = {
        "protocol": "同条件加速前后推理轨迹对比",
        "rate_detail_step": args.rate_detail_step,
        "checkpoint_sha256": checkpoint_hash,
        "products_sha256": source_hash,
        "targets_sha256": sha256(args.targets_file),
        "baseline_predictions_sha256": sha256(args.baseline_predictions) if args.baseline_predictions else None,
        "hazard_predictions_sha256": sha256(args.hazard_predictions) if args.hazard_predictions else None,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "n_runs": N_RUNS,
        "n_children": N_CHILDREN,
        "hazard_max_step": 0.04,
        "hazard_limit": 0.15,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render_html(report), encoding="utf-8")
    payload = (json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    json_path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    if args.per_case_html:
        for case in cases:
            path = args.output.with_name(f'{case["reaction_index"]}-trajectory.html')
            path.write_text(_render_html({**report, "cases": [case]}), encoding="utf-8")
    print(f"Wrote {args.output}, {json_path}, and {len(case_paths)} per-reaction HTML files")


if __name__ == "__main__":
    main()
