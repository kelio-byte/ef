#!/usr/bin/env python
"""Visualize first-step model predictions vs oracle in HTML table format.

Each example renders as a column-aligned table:
  Row 1: x0 tokens (product), oracle edit positions marked via top-border color
  Row 2-4: Oracle INS / SUB / DEL rates + top-5 token distributions
  Row 5-7: Model INS / SUB / DEL rates + top-5 token distributions

Rate magnitude is encoded as cell background color (white -> red).
"""

import argparse
import math
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from edit_flows.analysis.first_step import (
    build_model_batch,
    extract_oracle_event_set,
    load_parallel_texts,
    parse_time_grid,
    tokenize_smiles,
)
from edit_flows.analysis.visualization import (
    CSS,
    del_rate_cell,
    esc,
    fmt_rate,
    ins_rate_cell,
    rate_cell_style,
    sub_rate_cell,
    token_dist_cell,
)
from edit_flows.core.rate_scale import apply_rate_parameterization, get_rate_scale
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.data.dataset import load_vocab
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.sampling.euler import _compute_model_time
from edit_flows.sampling.oracle import compute_oracle_model_output


# ── HTML builders ──────────────────────────────────────────────────────

def _build_example_html(
    idx: int,
    t_value: float,
    product_str: str,
    target_str: str,
    x0_tokens: List[str],
    oracle_rates: torch.Tensor,        # (L, 3)  [ins, sub, del]
    oracle_ins_probs: torch.Tensor,    # (L, V)
    oracle_sub_probs: torch.Tensor,    # (L, V)
    oracle_event: dict,
    model_rates: torch.Tensor,         # (L, 3)
    model_ins_probs: torch.Tensor,     # (L, V)
    model_sub_probs: torch.Tensor,     # (L, V)
    actions: Optional[dict],           # deterministic per-position argmax
    id2token: Dict[int, str],
    center_hit: bool,
    full_correct: bool,
) -> str:
    """Build a single example's HTML table block."""
    L = len(x0_tokens)
    oracle_ins_mask = oracle_event["ins_mask"]
    oracle_sub_mask = oracle_event["sub_mask"]
    oracle_del_mask = oracle_event["del_mask"]

    # ---- helpers for building cell content ----

    def _ins_rate_cell(rates_3d: torch.Tensor, pos: int) -> str:
        return ins_rate_cell(rates_3d, pos)

    def _sub_rate_cell(rates_3d: torch.Tensor, pos: int) -> str:
        return sub_rate_cell(rates_3d, pos)

    def _del_cell(rates_3d: torch.Tensor, pos: int) -> str:
        return del_rate_cell(rates_3d, pos)

    def _token_cell(probs: torch.Tensor, pos: int) -> str:
        return token_dist_cell(probs, pos, id2token)

    # ---- x0 token row ----
    x0_cells: List[str] = []
    for j, tok in enumerate(x0_tokens):
        classes = ["tc"]
        if j == 0:  # BOS position
            classes.append("bos")
        if oracle_ins_mask[j].item():
            classes.append("oi")  # oracle-ins  → blue top border
        elif oracle_sub_mask[j].item():
            classes.append("os")  # oracle-sub  → orange top border
        elif oracle_del_mask[j].item():
            classes.append("od")  # oracle-del  → red top border
        display_tok = "BOS" if j == 0 else tok
        x0_cells.append(f'<td class="{" ".join(classes)}">{esc(display_tok)}</td>')

    x0_row = (
        '<tr class="x0-row">'
        f'<th class="lbl">x<sub>0</sub></th>'
        + "".join(x0_cells)
        + "</tr>"
    )

    # ---- oracle section ----
    oracle_ins_rate = [_ins_rate_cell(oracle_rates, j) for j in range(L)]
    oracle_ins_tok = [_token_cell(oracle_ins_probs, j) for j in range(L)]
    oracle_sub_rate = [_sub_rate_cell(oracle_rates, j) for j in range(L)]
    oracle_sub_tok = [_token_cell(oracle_sub_probs, j) for j in range(L)]
    oracle_del = [_del_cell(oracle_rates, j) for j in range(L)]

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
    model_ins_rate = [_ins_rate_cell(model_rates, j) for j in range(L)]
    model_ins_tok = [_token_cell(model_ins_probs, j) for j in range(L)]
    model_sub_rate = [_sub_rate_cell(model_rates, j) for j in range(L)]
    model_sub_tok = [_token_cell(model_sub_probs, j) for j in range(L)]
    model_del = [_del_cell(model_rates, j) for j in range(L)]

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

    # ---- deterministic per-position argmax diagnostic ----
    actual_html = ""
    if actions is not None:
        act_ins_mask = actions["ins_mask"][0]
        act_sub_mask = actions["sub_mask"][0]
        act_del_mask = actions["del_mask"][0]
        act_ins_tokens = actions["ins_tokens"][0]
        act_sub_tokens = actions["sub_tokens"][0]
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
            '<tr class="sec-hdr"><th class="lbl sec-label" colspan="%d">MODEL ARGMAX (NOT SAMPLED)</th></tr>' % (L + 1)
            + '<tr class="ae-row-tr">'
            + '<th class="lbl">edit</th>'
            + "".join(ae_cells)
            + "</tr>"
        )

    # ---- heading + summary box (matching trajectory layout) ----
    hit_str = "Y" if center_hit else "N"
    full_str = "Y" if full_correct else "N"
    hit_cls = "correct" if center_hit else "wrong"
    full_cls = "correct" if full_correct else "wrong"

    return (
        f'<div class="ex" id="ex{idx}_t{str(t_value).replace(".", "_")}">'
        f'<h3>Example #{idx} &nbsp; t={t_value}</h3>'
        f'<div class="summary-box">'
        f'<div class="smiles-line"><span class="smiles-label">Product</span>'
        f'<span class="smiles-str">{esc(product_str)}</span></div>'
        f'<div class="smiles-line"><span class="smiles-label">Target </span>'
        f'<span class="smiles-str">{esc(target_str)}</span></div>'
        f'<div class="smiles-meta">'
        f'<b>Center-Hit:</b> <span class="{hit_cls}">{hit_str}</span> &nbsp; '
        f'<b>Full-Correct:</b> <span class="{full_cls}">{full_str}</span>'
        f'</div></div>'
        f'<div class="tbl-wrap"><table class="vt">'
        f"{x0_row}{oracle_html}{model_html}{actual_html}"
        f"</table></div></div>"
    )


def _build_index_html(
    output_dir: str,
    example_ids: List[int],
    t_values: List[float],
    product_strs: List[str],
    target_strs: List[str],
    checkpoint: str = "",
) -> str:
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>First-Step Visualization</title>",
        f"<style>{CSS}</style></head><body>",
        "<h1>First-Step Visualization</h1>",
        f'<p style="color:#888;font-size:13px;margin-top:-10px">Checkpoint: <code>{esc(checkpoint)}</code></p>' if checkpoint else "",
        '<div class="legend">',
        '<b>Oracle edit type (top border on x<sub>0</sub>):</b> ',
        '<span><span class="sw oi-sw"></span> INS</span>',
        '<span><span class="sw os-sw"></span> SUB</span>',
        '<span><span class="sw od-sw"></span> DEL</span>',
        '&nbsp;&nbsp;&nbsp;<b>Rate magnitude (cell bg):</b> ',
        '<span class="sw-rate" style="background:linear-gradient(to right,#fff,#ffd0d0,#ff4040,#900);"></span>',
        '<span>0 → low → med → high</span>',
        '&nbsp;&nbsp;&nbsp;<span style="color:#999">BOS = gray cell</span>',
        "</div>",
        '<div class="nav"><b>Jump to:</b><br>',
    ]

    for idx, (prod, tgt) in enumerate(zip(product_strs, target_strs)):
        tid = example_ids[idx]
        for tv in t_values:
            anchor = f"ex{tid}_t{str(tv).replace('.', '_')}"
            lines.append(
                f'<a href="#{anchor}">Ex#{tid} t={tv}</a> '
                f'<span style="color:#999;font-size:11px">({esc(prod[:50])} → {esc(tgt[:50])})</span><br>'
            )

    lines.append("</div>")
    return "\n".join(lines)


def _build_footer_html() -> str:
    return "</body></html>"


# ── main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize first-step model vs oracle")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--products_file", type=str, required=True)
    parser.add_argument("--targets_file", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scheduler", type=str, default=None, choices=["cubic", "linear"])
    parser.add_argument("--time_grid", type=str, default="0,0.1")
    parser.add_argument("--deduplicate", type=int, default=0)
    parser.add_argument("--max_lines", type=int, default=0)
    parser.add_argument("--n_examples", type=int, default=5)
    parser.add_argument("--example_ids", type=str, default=None,
                        help="Comma-separated specific example indices to visualize")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Load model ──
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    scheduler_name = args.scheduler or cfg.get("sample_scheduler", cfg.get("scheduler", "cubic"))
    scheduler = LinearScheduler() if scheduler_name == "linear" else CubicScheduler()
    train_scheduler_name = cfg.get("scheduler", "cubic")
    train_scheduler = LinearScheduler() if train_scheduler_name == "linear" else CubicScheduler()

    vocab_path = args.vocab_file or os.path.join(
        cfg["data_dir"], cfg.get("vocab_file", "example.vocab.src")
    )
    token2id, _ = load_vocab(vocab_path)
    model_vocab = ckpt.get("model_vocab") or len(token2id)
    id2token = {v: k for k, v in token2id.items()}

    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    use_origin_mask = cfg.get("use_origin_mask", False)
    has_origin_embed = any("origin_embedding" in key for key in state_dict)
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
    model.load_state_dict(state_dict)
    model.eval()

    # ── Load data ──
    products, targets = load_parallel_texts(
        args.products_file, args.targets_file,
        deduplicate=args.deduplicate, max_lines=args.max_lines,
    )
    # ── Select examples ──
    t_values = parse_time_grid(args.time_grid)
    if args.example_ids:
        selected = [int(x) for x in args.example_ids.split(",")]
        selected = [i for i in selected if 0 <= i < len(products)]
    else:
        rng = random.Random(args.seed)
        selected = sorted(rng.sample(range(len(products)), min(args.n_examples, len(products))))

    print(f"Visualizing {len(selected)} examples: {selected}")
    print(f"t values: {t_values}")

    # ── Build HTML ──
    html_parts = [
        _build_index_html(args.output_dir, selected, t_values,
                          [products[i] for i in selected],
                          [targets[i] for i in selected],
                          checkpoint=args.checkpoint),
    ]

    for idx in selected:
        prod_ids = tokenize_smiles(products[idx], token2id)
        tgt_ids = tokenize_smiles(targets[idx], token2id)
        x_0_single, x_1_single = build_model_batch([prod_ids], [tgt_ids])
        x_0_single = x_0_single.to(device)
        x_1_single = x_1_single.to(device)
        x_pad_mask_single = x_0_single == 0

        # token strings for x_0 (skip PAD)
        L_valid = int((x_0_single[0] != 0).sum().item())
        x0_tokens = [id2token.get(x_0_single[0, j].item(), "?") for j in range(L_valid)]

        for t_value in t_values:
            t = torch.full((1, 1), t_value, dtype=torch.float, device=device)
            t_model = _compute_model_time(
                t, scheduler, cfg.get("time_input", "t"), train_scheduler,
            )

            # Model forward
            log_rates, log_ins_probs, log_sub_probs = model(
                x_0_single, t_model, x_pad_mask_single,
                origin_mask=(
                    torch.ones_like(x_0_single, dtype=torch.bool)
                    if use_origin_mask else None
                ),
            )
            log_rates_eff = apply_rate_parameterization(
                log_rates, t, scheduler,
                use_rate_reparam=cfg.get("use_rate_reparam", False),
                clamp_kappa=cfg.get("clamp_kappa", False),
                clamp_max=cfg.get("clamp_max", 50.0),
            )

            # k(t) for de-scaling oracle rates to raw edit counts.
            # clamp_min(1e-2) prevents log(0)→-inf at t=0 (cubic deriv=0),
            # which would falsely inflate displayed oracle rates by +27.6.
            k_t = get_rate_scale(
                t, scheduler,
                clamp_kappa=cfg.get("clamp_kappa", False),
                clamp_max=cfg.get("clamp_max", 50.0),
            )
            log_k_t = torch.log(k_t.clamp_min(1e-2))

            # Oracle forward
            oracle_out = compute_oracle_model_output(
                x_0_single, x_1_single, t, scheduler, model.vocab_size,
            )
            # De-scale oracle rates: oracle has k(t) baked in, divide to get clean integer counts
            oracle_log_rates_raw = oracle_out[0] - log_k_t
            oracle_event = extract_oracle_event_set(
                oracle_out[0], oracle_out[1], oracle_out[2], x_0_single,
            )

            # Per-example metrics (use effective rates for sampling-accurate ranking)
            rates_for_sort = torch.exp(log_rates_eff[0]).sum(dim=-1).tolist()
            # exclude BOS (pos 0) and PAD
            for j in range(L_valid):
                if j == 0 or x_0_single[0, j].item() == 0:
                    rates_for_sort[j] = float("-inf")
            model_top1 = max(range(L_valid), key=lambda j: rates_for_sort[j])

            oracle_positions = [
                j for j in range(L_valid)
                if oracle_event["pos_mask"][0, j].item()
            ]
            center_hit = model_top1 in oracle_positions if oracle_positions else False

            # determine full correct
            pred_type = int(torch.argmax(torch.exp(log_rates_eff[0, model_top1])).item())
            oracle_type_at_top1 = int(oracle_event["type_argmax"][0, model_top1].item())
            type_correct = (
                pred_type == oracle_type_at_top1
                and oracle_event["pos_mask"][0, model_top1].item()
            )
            full_correct = False
            if type_correct:
                if pred_type == 0 and oracle_event["ins_mask"][0, model_top1].item():
                    pred_token = torch.argmax(log_ins_probs[0, model_top1]).item()
                    oracle_token = int(oracle_event["ins_token"][0, model_top1].item())
                    full_correct = pred_token == oracle_token
                elif pred_type == 1 and oracle_event["sub_mask"][0, model_top1].item():
                    pred_token = torch.argmax(log_sub_probs[0, model_top1]).item()
                    oracle_token = int(oracle_event["sub_token"][0, model_top1].item())
                    full_correct = pred_token == oracle_token
                elif pred_type == 2 and oracle_event["del_mask"][0, model_top1].item():
                    full_correct = True

            # Deterministic top-1 edit per position.  This is a rate
            # diagnostic, not a stochastic Euler draw, so the HTML labels it
            # MODEL ARGMAX rather than ACTUAL.
            # Only show edits where max rate exceeds threshold — otherwise
            # the model isn't actually "intending" to edit that position,
            # just like trajectory's probabilistic sampling leaves most cells empty.
            # This is independent of the Center-Hit / Full-Correct metrics.
            _rates = torch.exp(log_rates_eff[0])       # (L, 3)
            _max_rate = _rates.max(dim=-1).values       # (L,)
            _type_best = torch.argmax(_rates, dim=-1)   # 0=INS, 1=SUB, 2=DEL
            _ins_tok = torch.argmax(log_ins_probs[0], dim=-1)  # (L,)
            _sub_tok = torch.argmax(log_sub_probs[0], dim=-1)  # (L,)
            _pad = x_0_single[0] == 0
            _has_edit = (_max_rate > 1e-2) & ~_pad     # only show meaningful edits
            _ins_mask = (_type_best == 0) & _has_edit
            _sub_mask = (_type_best == 1) & _has_edit
            _del_mask = (_type_best == 2) & _has_edit
            actions = {
                "ins_mask": _ins_mask.unsqueeze(0).cpu(),       # (1, L)
                "sub_mask": _sub_mask.unsqueeze(0).cpu(),
                "del_mask": _del_mask.unsqueeze(0).cpu(),
                "ins_tokens": _ins_tok.unsqueeze(0).cpu(),
                "sub_tokens": _sub_tok.unsqueeze(0).cpu(),
            }

            html_parts.append(
                _build_example_html(
                    idx=idx,
                    t_value=t_value,
                    product_str=products[idx],
                    target_str=targets[idx],
                    x0_tokens=x0_tokens,
                    oracle_rates=oracle_log_rates_raw[0],
                    oracle_ins_probs=oracle_out[1][0],
                    oracle_sub_probs=oracle_out[2][0],
                    oracle_event={k: v[0] for k, v in oracle_event.items()},
                    model_rates=log_rates[0],
                    model_ins_probs=log_ins_probs[0],
                    model_sub_probs=log_sub_probs[0],
                    actions=actions,
                    id2token=id2token,
                    center_hit=center_hit,
                    full_correct=full_correct,
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
    out_path = os.path.join(args.output_dir, f"vis_first_step_{id_label}_{timestamp}.html")
    with open(out_path, "w") as f:
        f.write("\n".join(html_parts))

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
