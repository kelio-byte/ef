#!/usr/bin/env python
"""Visualize full Euler sampling trajectory with per-edit-event oracle vs model tables.

For each selected example, runs Euler sampling to completion, records the
model predictions / oracle / actual edits at every step where an edit occurs,
and renders each event as an HTML table in the same column-aligned format as
visualize_first_step.py.
"""

import argparse
import math
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch

from edit_flows.analysis.first_step import (
    build_model_batch,
    extract_oracle_event_set,
    load_parallel_texts,
    tokenize_smiles,
)
from edit_flows.analysis.visualization import (
    CSS,
    del_rate_cell,
    esc,
    fmt_rate,
    ins_rate_cell,
    sub_rate_cell,
    token_dist_cell,
)
from edit_flows.core.rate_scale import get_rate_scale
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.data.dataset import load_vocab
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.sampling.euler import sample_euler
from edit_flows.sampling.euler_beam import sample_euler_beam
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing.global_align import inverse_global_align

lg = None
try:
    from rdkit import Chem, RDLogger
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
except ImportError:
    pass


def _canonicalize_smiles(smiles: str) -> str:
    """#global# SMILES -> RDKit canonical SMILES. Returns '' on failure."""
    try:
        smiles = inverse_global_align(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        pass
    return ""


# ── trajectory HTML builders ──────────────────────────────────────────

def _build_event_table(
    example_idx: int,
    event_idx: int,
    n_events: int,
    step_idx: int,
    t_value: float,
    x_t_tokens: List[str],
    origin_flags: List[bool],
    oracle_log_rates: torch.Tensor,
    oracle_log_ins_probs: torch.Tensor,
    oracle_log_sub_probs: torch.Tensor,
    oracle_event: dict,
    model_log_rates: torch.Tensor,
    model_log_ins_probs: torch.Tensor,
    model_log_sub_probs: torch.Tensor,
    actions: dict,
    id2token: Dict[int, str],
    k_t: float,
    final_correct: bool,
) -> str:
    """Build a single edit-event HTML table block for trajectory visualization."""
    L = len(x_t_tokens)
    oracle_ins_mask = oracle_event["ins_mask"]
    oracle_sub_mask = oracle_event["sub_mask"]
    oracle_del_mask = oracle_event["del_mask"]
    act_ins_mask = actions["ins_mask"]
    act_sub_mask = actions["sub_mask"]
    act_del_mask = actions["del_mask"]
    act_ins_tokens = actions["ins_tokens"]
    act_sub_tokens = actions["sub_tokens"]

    # De-scale oracle rates by k(t) to get raw edit counts (same as first-step viz)
    log_k_t = math.log(max(k_t, 1e-12))

    # ---- x_t token row ----
    xt_cells: List[str] = []
    for j, tok in enumerate(x_t_tokens):
        classes = ["tc"]
        if j == 0:
            classes.append("bos")
        elif origin_flags[j]:
            classes.append("orig")
        else:
            classes.append("inserted")
        if oracle_ins_mask[j].item():
            classes.append("oi")
        elif oracle_sub_mask[j].item():
            classes.append("os")
        elif oracle_del_mask[j].item():
            classes.append("od")
        if act_ins_mask[j].item():
            classes.append("aei")
        elif act_sub_mask[j].item():
            classes.append("aes")
        elif act_del_mask[j].item():
            classes.append("aed")
        display_tok = "BOS" if j == 0 else tok
        xt_cells.append(f'<td class="{" ".join(classes)}">{esc(display_tok)}</td>')

    xt_row = (
        '<tr class="x0-row">'
        f'<th class="lbl">x<sub>t</sub></th>'
        + "".join(xt_cells)
        + "</tr>"
    )

    # ---- oracle section (de-scaled rates) ----
    ora_rates = oracle_log_rates - log_k_t
    oracle_ins_rate = [ins_rate_cell(ora_rates, j) for j in range(L)]
    oracle_ins_tok = [token_dist_cell(oracle_log_ins_probs, j, id2token) for j in range(L)]
    oracle_sub_rate = [sub_rate_cell(ora_rates, j) for j in range(L)]
    oracle_sub_tok = [token_dist_cell(oracle_log_sub_probs, j, id2token) for j in range(L)]
    oracle_del = [del_rate_cell(ora_rates, j) for j in range(L)]

    oracle_html = (
        '<tr class="sec-hdr"><th class="lbl sec-label" colspan="%d">ORACLE</th></tr>' % (L + 1)
        + '<tr class="ins-row">'
        + '<th class="lbl">&lambda;<sub>ins</sub></th>'
        + "".join(oracle_ins_rate)
        + "</tr>"
        + '<tr class="ins-tok-row">'
        + '<th class="lbl">ins top5</th>'
        + "".join(oracle_ins_tok)
        + "</tr>"
        + '<tr class="sub-row">'
        + '<th class="lbl">&lambda;<sub>sub</sub></th>'
        + "".join(oracle_sub_rate)
        + "</tr>"
        + '<tr class="sub-tok-row">'
        + '<th class="lbl">sub top5</th>'
        + "".join(oracle_sub_tok)
        + "</tr>"
        + '<tr class="del-row">'
        + '<th class="lbl">&lambda;<sub>del</sub></th>'
        + "".join(oracle_del)
        + "</tr>"
    )

    # ---- model section ----
    model_ins_rate = [ins_rate_cell(model_log_rates, j) for j in range(L)]
    model_ins_tok = [token_dist_cell(model_log_ins_probs, j, id2token) for j in range(L)]
    model_sub_rate = [sub_rate_cell(model_log_rates, j) for j in range(L)]
    model_sub_tok = [token_dist_cell(model_log_sub_probs, j, id2token) for j in range(L)]
    model_del = [del_rate_cell(model_log_rates, j) for j in range(L)]

    model_html = (
        '<tr class="sec-hdr"><th class="lbl sec-label" colspan="%d">MODEL</th></tr>' % (L + 1)
        + '<tr class="ins-row">'
        + '<th class="lbl">&lambda;<sub>ins</sub></th>'
        + "".join(model_ins_rate)
        + "</tr>"
        + '<tr class="ins-tok-row">'
        + '<th class="lbl">ins top5</th>'
        + "".join(model_ins_tok)
        + "</tr>"
        + '<tr class="sub-row">'
        + '<th class="lbl">&lambda;<sub>sub</sub></th>'
        + "".join(model_sub_rate)
        + "</tr>"
        + '<tr class="sub-tok-row">'
        + '<th class="lbl">sub top5</th>'
        + "".join(model_sub_tok)
        + "</tr>"
        + '<tr class="del-row">'
        + '<th class="lbl">&lambda;<sub>del</sub></th>'
        + "".join(model_del)
        + "</tr>"
    )

    # ---- actual edit row ----
    ae_cells: List[str] = []
    for j in range(L):
        if j == 0:
            ae_cells.append('<td class="ae-row" style="background:#e8e8e8;color:#999">—</td>')
        elif act_ins_mask[j].item():
            tok_id = int(act_ins_tokens[j].item())
            tok_str = id2token.get(tok_id, "?")
            ae_cells.append(f'<td class="ae-row ae-ins">+{esc(tok_str)}</td>')
        elif act_sub_mask[j].item():
            tok_id = int(act_sub_tokens[j].item())
            tok_str = id2token.get(tok_id, "?")
            ae_cells.append(f'<td class="ae-row ae-sub">→{esc(tok_str)}</td>')
        elif act_del_mask[j].item():
            ae_cells.append('<td class="ae-row ae-del">DEL</td>')
        else:
            ae_cells.append('<td class="ae-row"></td>')

    actual_html = (
        '<tr class="sec-hdr"><th class="lbl sec-label" colspan="%d">ACTUAL</th></tr>' % (L + 1)
        + '<tr class="ae-row-tr">'
        + '<th class="lbl">edit</th>'
        + "".join(ae_cells)
        + "</tr>"
    )

    # ---- meta bar ----
    fc_str = "Y" if final_correct else "N"
    meta = (
        f"Ex#{example_idx} Event #{event_idx + 1}/{n_events} &nbsp; "
        f"step={step_idx} &nbsp; t={t_value:.4f} &nbsp; "
        f"Final-Correct: {fc_str}"
    )

    anchor = f"ex{example_idx}_ev{event_idx}"
    return (
        f'<div class="ex" id="{anchor}">'
        f'<h4>{meta}</h4>'
        f'<div class="tbl-wrap"><table class="vt">'
        f"{xt_row}{oracle_html}{model_html}{actual_html}"
        f"</table></div></div>"
    )


def _build_example_section(
    example_idx: int,
    product_str: str,
    target_str: str,
    events: List[dict],
    id2token: Dict[int, str],
    final_correct: bool,
    scheduler,
    use_rate_reparam: bool,
    clamp_kappa: bool,
    clamp_max: float,
    sample_info: str = "",
) -> str:
    """Build the full HTML section for one example with all its edit events."""
    n_events = len(events)

    # Compute correctness of each event's anchor edit vs oracle
    event_correctness: List[bool] = []
    for ev in events:
        if ev.get("oracle_event") is None:
            event_correctness.append(False)
            continue
        oe = ev["oracle_event"]
        actions = ev["actions"]
        any_edit = actions["ins_mask"] | actions["sub_mask"] | actions["del_mask"]
        n_edits = int(any_edit.sum().item())
        if n_edits == 0:
            event_correctness.append(False)
            continue
        all_correct = True
        for pos in range(any_edit.numel()):
            if not any_edit[pos].item():
                continue
            if not oe["pos_mask"][pos].item():
                all_correct = False
                break
            if actions["ins_mask"][pos].item() and not oe["ins_mask"][pos].item():
                all_correct = False
                break
            if actions["sub_mask"][pos].item() and not oe["sub_mask"][pos].item():
                all_correct = False
                break
            if actions["del_mask"][pos].item() and not oe["del_mask"][pos].item():
                all_correct = False
                break
        event_correctness.append(all_correct)

    # Navigation bar
    nav_parts = [f'<div class="event-nav"><b>Events:</b> ']
    for ei, (ev, ec) in enumerate(zip(events, event_correctness)):
        cls = "correct" if ec else "wrong"
        t_val = ev["t"]
        step = ev["step_idx"]
        nav_parts.append(
            f'<a class="{cls}" href="#ex{example_idx}_ev{ei}">'
            f'#{ei + 1} t={t_val:.3f} s={step}</a>'
        )
        if ei < n_events - 1:
            nav_parts.append('<span class="sep">→</span>')
    nav_parts.append("</div>")

    # Summary box — three-line layout for easy product↔target comparison
    fc_str = "MATCH" if final_correct else "MISMATCH"
    fc_cls = "correct" if final_correct else "wrong"
    n_correct_events = sum(event_correctness)
    summary = (
        f'<div class="summary-box">'
        f'<div class="smiles-line"><span class="smiles-label">Product</span>'
        f'<span class="smiles-str">{esc(product_str)}</span></div>'
        f'<div class="smiles-line"><span class="smiles-label">Target </span>'
        f'<span class="smiles-str">{esc(target_str)}</span></div>'
        f'<div class="smiles-meta">'
        f'<b>Result:</b> <span class="{fc_cls}">{fc_str}</span> &nbsp; '
        f'<b>Events:</b> {n_events} total, {n_correct_events} correct'
        + (f' &nbsp; <b>{esc(sample_info)}</b>' if sample_info else "")
        + f"</div></div>"
    )

    # Event tables
    event_tables: List[str] = []
    for ei, ev in enumerate(events):
        if ev["oracle_log_rates"] is None:
            continue

        x_t = ev["x_t"]
        valid_mask = x_t != PAD_TOKEN
        L_valid = int(valid_mask.sum().item())
        xt_tokens = [id2token.get(int(x_t[j].item()), "?") for j in range(L_valid)]

        origin_flags: List[bool] = []
        if ev["origin_mask"] is not None:
            for j in range(L_valid):
                origin_flags.append(bool(ev["origin_mask"][j].item()))
        else:
            origin_flags = [True] * L_valid

        # Compute k(t) for de-scaling oracle rates to raw edit counts
        t_tensor = torch.tensor([[ev["t"]]])
        k_t = get_rate_scale(
            t_tensor, scheduler,
            clamp_kappa=clamp_kappa, clamp_max=clamp_max,
        ).item()

        # Use raw model rates (before k(t) parameterization) for display
        model_log_rates = ev.get("log_rates_raw", ev["log_rates"])

        event_tables.append(
            _build_event_table(
                example_idx=example_idx,
                event_idx=ei,
                n_events=n_events,
                step_idx=ev["step_idx"],
                t_value=ev["t"],
                x_t_tokens=xt_tokens,
                origin_flags=origin_flags,
                oracle_log_rates=ev["oracle_log_rates"],
                oracle_log_ins_probs=ev["oracle_log_ins_probs"],
                oracle_log_sub_probs=ev["oracle_log_sub_probs"],
                oracle_event=ev["oracle_event"],
                model_log_rates=model_log_rates,
                model_log_ins_probs=ev["log_ins_probs"],
                model_log_sub_probs=ev["log_sub_probs"],
                actions=ev["actions"],
                id2token=id2token,
                k_t=k_t,
                final_correct=final_correct,
            )
        )

    return (
        f'<div class="ex-section" id="ex{example_idx}">'
        f"<h2>Example #{example_idx}</h2>"
        f"{summary}"
        f"{''.join(nav_parts)}"
        f"{''.join(event_tables)}"
        f"</div>"
    )


def _build_index_html(
    example_ids: List[int],
    product_strs: List[str],
    target_strs: List[str],
    final_corrects: List[bool],
    checkpoint: str = "",
) -> str:
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Trajectory Visualization</title>",
        f"<style>{CSS}</style></head><body>",
        "<h1>Trajectory Visualization</h1>",
        f'<p style="color:#888;font-size:13px;margin-top:-10px">Checkpoint: <code>{esc(checkpoint)}</code></p>' if checkpoint else "",
        '<div class="legend">',
        '<b>Oracle edit type (top border on x<sub>t</sub>):</b> ',
        '<span><span class="sw oi-sw"></span> INS</span>',
        '<span><span class="sw os-sw"></span> SUB</span>',
        '<span><span class="sw od-sw"></span> DEL</span>',
        '&nbsp;&nbsp;&nbsp;<b>Actual edit (bottom border on x<sub>t</sub>):</b> ',
        '<span><span class="sw aei-sw"></span> INS</span>',
        '<span><span class="sw aes-sw"></span> SUB</span>',
        '<span><span class="sw aed-sw"></span> DEL</span>',
        '&nbsp;&nbsp;&nbsp;<b>Rate magnitude (cell bg):</b> ',
        '<span class="sw-rate" style="background:linear-gradient(to right,#fff,#ffd0d0,#ff4040,#900);"></span>',
        '<span>0 → low → med → high</span>',
        '&nbsp;&nbsp;&nbsp;<b>x<sub>t</sub> origin:</b> ',
        '<span style="background:#fafafa;padding:2px 6px;border:1px solid #ddd">original</span>',
        '<span style="background:#e8f5e9;padding:2px 6px;border:1px solid #ddd">inserted</span>',
        '&nbsp;&nbsp;&nbsp;<b>Actual edit cell:</b> ',
        '<span style="background:#c8e6c9;padding:2px 6px;border:1px solid #ddd">+INS</span>',
        '<span style="background:#ffe0b2;padding:2px 6px;border:1px solid #ddd">→SUB</span>',
        '<span style="background:#ffcdd2;padding:2px 6px;border:1px solid #ddd">DEL</span>',
        "</div>",
        '<div class="nav"><b>Jump to example:</b><br>',
    ]

    for idx, (prod, tgt, fc) in enumerate(zip(product_strs, target_strs, final_corrects)):
        eid = example_ids[idx]
        fc_str = "✓" if fc else "✗"
        lines.append(
            f'<a href="#ex{eid}">Ex#{eid} {fc_str}</a> '
            f'<span style="color:#999;font-size:11px">({esc(prod[:50])} → {esc(tgt[:50])})</span><br>'
        )

    lines.append("</div>")
    return "\n".join(lines)


def _build_footer_html() -> str:
    return "</body></html>"


# ── helpers ────────────────────────────────────────────────────────────

def _decode_to_smiles(token_ids: torch.Tensor, id2token: Dict[int, str]) -> str:
    """Decode token IDs to SMILES string, dropping BOS and PAD."""
    tokens = []
    for tid in token_ids.tolist():
        tid = int(tid)
        if tid in (PAD_TOKEN, BOS_TOKEN):
            continue
        tokens.append(id2token.get(tid, "?"))
    return "".join(tokens)


# ── main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize full Euler sampling trajectory with per-event tables"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--products_file", type=str, required=True)
    parser.add_argument("--targets_file", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="visualizations/trajectory/")
    parser.add_argument("--html", type=lambda x: x.lower() in ("true", "1", "yes"),
                        default=True, help="Generate HTML output (default: True)")
    parser.add_argument("--scheduler", type=str, default=None, choices=["cubic", "linear"])
    parser.add_argument("--deduplicate", type=int, default=0)
    parser.add_argument("--max_lines", type=int, default=0)
    parser.add_argument("--n_examples", type=int, default=5)
    parser.add_argument("--example_ids", type=str, default=None,
                        help="Comma-separated specific example indices to visualize")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of independent Euler samples per example")
    parser.add_argument("--n_branches", type=int, default=0,
                        help="If >1, use Euler-Beam with K parallel branches")
    args = parser.parse_args()

    if args.html:
        os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Load model ──
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model_vocab = ckpt.get("model_vocab")
    scheduler_name = args.scheduler or cfg.get("sample_scheduler", cfg.get("scheduler", "cubic"))
    scheduler = LinearScheduler() if scheduler_name == "linear" else CubicScheduler()
    train_scheduler_name = cfg.get("scheduler", "cubic")
    train_scheduler = LinearScheduler() if train_scheduler_name == "linear" else CubicScheduler()

    if args.vocab_file:
        vocab_path = args.vocab_file
    else:
        vocab_path = os.path.join(cfg["data_dir"], cfg.get("vocab_file", "example.vocab.src"))
    token2id, _ = load_vocab(vocab_path)
    if model_vocab is None:
        model_vocab = len(token2id)
    id2token = {v: k for k, v in token2id.items()}

    use_origin_mask = cfg.get("use_origin_mask", False)
    has_origin_embed = any("origin_embedding" in k for k in ckpt["model_state_dict"])
    if use_origin_mask and not has_origin_embed:
        print("WARNING: config has use_origin_mask=True but checkpoint lacks "
              "origin_embedding weights. Falling back to use_origin_mask=False.")
        use_origin_mask = False

    model = EditFlowsTransformer(
        vocab_size=model_vocab,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        dim_feedforward=cfg["dim_feedforward"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
        attention_dropout=cfg.get("attention_dropout", cfg["dropout"]),
        activation=cfg.get("activation", "relu"),
        pos_encoding_scale=cfg.get("pos_encoding_scale", True),
        use_origin_mask=use_origin_mask,
    ).to(device)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # ── Load data ──
    products, targets = load_parallel_texts(
        args.products_file, args.targets_file,
        deduplicate=args.deduplicate, max_lines=args.max_lines,
    )
    product_ids_all = [tokenize_smiles(line, token2id) for line in products]
    target_ids_all = [tokenize_smiles(line, token2id) for line in targets]

    # ── Select examples ──
    if args.example_ids:
        selected = [int(x) for x in args.example_ids.split(",")]
        selected = [i for i in selected if 0 <= i < len(products)]
    else:
        rng = random.Random(args.seed)
        selected = sorted(rng.sample(range(len(products)), min(args.n_examples, len(products))))

    use_branches = args.n_branches > 1
    n_display = f"n_branches={args.n_branches}" if use_branches else str(args.n_samples)
    print(f"Running trajectory sampling for {len(selected)} examples x {n_display}: {selected}")
    print(f"Scheduler: {scheduler_name}, n_steps: {args.n_steps}")

    # ── Build batch of selected examples ──
    sel_products = [product_ids_all[i] for i in selected]
    sel_targets = [target_ids_all[i] for i in selected]
    x_0, x_1 = build_model_batch(sel_products, sel_targets)
    n_sel = len(selected)

    # ── Run Euler sampling with event recording ──
    use_rate_reparam = cfg.get("use_rate_reparam", False)
    # use_origin_mask already resolved above from checkpoint (may differ from config)
    time_input = cfg.get("time_input", "t")

    print(f"use_rate_reparam={use_rate_reparam}, use_origin_mask={use_origin_mask}, "
          f"time_input={time_input}")

    if use_branches:
        print(f"Sampler: Euler-Beam (K={args.n_branches}, n_steps={args.n_steps})")
        # Quick model forward sanity check
        with torch.no_grad():
            t_test = torch.zeros(1, 1, device=device)
            xp = x_0[:1].to(device)
            pad = xp == 0
            lr, _, _ = model(xp, t_test, pad)
            print(f"Model forward check (t=0): log_rates mean={lr.mean().item():.2f}, "
                  f"std={lr.std().item():.2f}")

        x_final, all_events = sample_euler_beam(
            model, x_0, scheduler,
            n_branches=args.n_branches,
            n_steps=args.n_steps,
            max_seq_len=cfg["max_seq_len"],
            use_rate_reparam=use_rate_reparam,
            clamp_kappa=cfg.get("clamp_kappa", False),
            clamp_max=cfg.get("clamp_max", 50.0),
            time_input=time_input,
            train_scheduler=train_scheduler,
            record_all_events=True,
            x_1=x_1,
            vocab_size=model.vocab_size,
            use_origin_mask=use_origin_mask,
        )
        # Beam returns one result per example (best branch)
        grouped_events = [[ev] for ev in all_events]
        grouped_finals = [x_final[i:i+1] for i in range(n_sel)]
        effective_samples = 1
    else:
        # Repeat for n_samples independent Euler trajectories
        x_0 = x_0.repeat_interleave(args.n_samples, dim=0)
        x_1 = x_1.repeat_interleave(args.n_samples, dim=0)

        # Quick model forward sanity check
        with torch.no_grad():
            t_test = torch.zeros(1, 1, device=device)
            xp = x_0[:1].to(device)
            pad = xp == 0
            lr, _, _ = model(xp, t_test, pad)
            print(f"Model forward check (t=0): log_rates mean={lr.mean().item():.2f}, "
                  f"std={lr.std().item():.2f}")

        x_final, _trajectory, all_events = sample_euler(
            model, x_0, scheduler,
            n_steps=args.n_steps,
            max_seq_len=cfg["max_seq_len"],
            use_rate_reparam=use_rate_reparam,
            clamp_kappa=cfg.get("clamp_kappa", False),
            clamp_max=cfg.get("clamp_max", 50.0),
            time_input=time_input,
            train_scheduler=train_scheduler,
            record_all_events=True,
            x_1=x_1,
            vocab_size=model.vocab_size,
            use_origin_mask=use_origin_mask,
        )
        # Regroup results: (n_sel * n_samples) -> n_sel groups
        grouped_events = []
        grouped_finals = []
        for i in range(n_sel):
            start = i * args.n_samples
            end = start + args.n_samples
            grouped_events.append(all_events[start:end])
            grouped_finals.append(x_final[start:end])
        effective_samples = args.n_samples

    # ── Determine final correctness & pick best sample per example ──
    best_sample_idx: List[int] = []
    final_corrects: List[bool] = []
    for i in range(n_sel):
        tgt_raw = _decode_to_smiles(x_1[i * args.n_samples], id2token) if not use_branches else _decode_to_smiles(x_1[i], id2token)
        tgt_canon = _canonicalize_smiles(tgt_raw)
        best_idx = 0
        best_correct = False
        for s in range(effective_samples):
            pred_raw = _decode_to_smiles(grouped_finals[i][s], id2token)
            pred_canon = _canonicalize_smiles(pred_raw)
            if pred_canon and tgt_canon and pred_canon == tgt_canon:
                best_idx = s
                best_correct = True
                break
        best_sample_idx.append(best_idx)
        final_corrects.append(best_correct)

    # ── Print per-example summary (always) ──
    for bi, idx in enumerate(selected):
        si = best_sample_idx[bi]
        events = grouped_events[bi][si]
        n_match = sum(1 for s in range(effective_samples)
                      if _canonicalize_smiles(_decode_to_smiles(grouped_finals[bi][s], id2token))
                      == _canonicalize_smiles(_decode_to_smiles(x_1[bi * args.n_samples] if not use_branches else x_1[bi], id2token))
                      and _canonicalize_smiles(_decode_to_smiles(grouped_finals[bi][s], id2token)) != "")
        print(f"  Example #{idx}: {n_match}/{effective_samples} match, "
              f"best sample #{si} ({len(events)} edit events)")

    # ── Build HTML (only when --html) ──
    if args.html:
        html_parts = [
            _build_index_html(
                selected,
                [products[i] for i in selected],
                [targets[i] for i in selected],
                final_corrects,
                checkpoint=args.checkpoint,
            ),
        ]

        for bi, idx in enumerate(selected):
            si = best_sample_idx[bi]
            events = grouped_events[bi][si]
            n_match = sum(1 for s in range(effective_samples)
                          if _canonicalize_smiles(_decode_to_smiles(grouped_finals[bi][s], id2token))
                          == _canonicalize_smiles(_decode_to_smiles(x_1[bi * args.n_samples] if not use_branches else x_1[bi], id2token))
                          and _canonicalize_smiles(_decode_to_smiles(grouped_finals[bi][s], id2token)) != "")
            sample_info = f"Sample #{si}, {n_match}/{effective_samples} match" if effective_samples > 1 else ""
            if events:
                html_parts.append(
                    _build_example_section(
                        example_idx=idx,
                        product_str=products[idx],
                        target_str=targets[idx],
                        events=events,
                        id2token=id2token,
                        final_correct=final_corrects[bi],
                        scheduler=scheduler,
                        use_rate_reparam=use_rate_reparam,
                        clamp_kappa=cfg.get("clamp_kappa", False),
                        clamp_max=cfg.get("clamp_max", 50.0),
                        sample_info=sample_info,
                    )
                )

        html_parts.append(_build_footer_html())

        # Build short label from example_ids for filename
        ids_list = [int(x) for x in args.example_ids.split(",")] if args.example_ids else selected
        if len(ids_list) <= 4:
            id_label = "_".join(str(i) for i in ids_list)
        else:
            id_label = "_".join(str(i) for i in ids_list[:3]) + f"_x{len(ids_list)}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output_dir, f"trajectory_{id_label}_{timestamp}.html")
        with open(out_path, "w") as f:
            f.write("\n".join(html_parts))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
