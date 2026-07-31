"""Shared HTML rendering utilities for Edit Flows visualizations.

Used by both visualize_first_step.py (static x_t=x_0 analysis) and
visualize_trajectory.py (dynamic per-event trajectory analysis).
"""

import math
from typing import Dict, Tuple


# ── rate → color mapping ──────────────────────────────────────────────

def rate_to_bg_color(rate: float) -> Tuple[str, str]:
    """Map a scalar rate to (bg_hex, text_color).

    Uses log10 scale clamped to [-8, 2].  Oracle-positive positions
    (rate >= 0.1) produce dark red; near-zero rates stay white.
    """
    if rate < 1e-9:
        return "#ffffff", "#000000"
    log_r = math.log10(rate)
    lo, hi = -8.0, 2.0
    t = (max(lo, min(hi, log_r)) - lo) / (hi - lo)

    if t < 0.25:
        s = t / 0.25
        r, g, b = 255, int(255 - 30 * s), int(255 - 30 * s)
    elif t < 0.55:
        s = (t - 0.25) / 0.30
        r, g, b = 255, int(225 - 155 * s), int(225 - 155 * s)
    elif t < 0.80:
        s = (t - 0.55) / 0.25
        r, g, b = int(255 - 100 * s), int(70 - 70 * s), int(70 - 70 * s)
    else:
        s = (t - 0.80) / 0.20
        r, g, b = int(155 - 55 * s), 0, 0

    text_color = "#ffffff" if t > 0.55 else "#000000"
    return f"{r:02x}{g:02x}{b:02x}", text_color


def rate_cell_style(rate: float) -> str:
    bg, fg = rate_to_bg_color(rate)
    return f"background-color:#{bg};color:{fg};"


def fmt_rate(r: float) -> str:
    if r < 1e-9:
        return "0"
    if r < 1e-3:
        return f"{r:.1e}"
    if r < 10:
        return f"{r:.3f}"
    return f"{r:.2f}"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── cell builders ─────────────────────────────────────────────────────

def ins_rate_cell(log_rates, pos: int) -> str:
    """Build a <td> for the insert rate at a given position."""
    rate_val = math.exp(log_rates[pos, 0].item())
    style = rate_cell_style(rate_val)
    return (
        f'<td class="rc" style="{style}">'
        f'<span class="rv">{fmt_rate(rate_val)}</span></td>'
    )


def sub_rate_cell(log_rates, pos: int) -> str:
    """Build a <td> for the substitute rate at a given position."""
    rate_val = math.exp(log_rates[pos, 1].item())
    style = rate_cell_style(rate_val)
    return (
        f'<td class="rc" style="{style}">'
        f'<span class="rv">{fmt_rate(rate_val)}</span></td>'
    )


def del_rate_cell(log_rates, pos: int) -> str:
    """Build a <td> for the delete rate at a given position."""
    rate_val = math.exp(log_rates[pos, 2].item())
    style = rate_cell_style(rate_val)
    return (
        f'<td class="rc" style="{style}">'
        f'<span class="rv">{fmt_rate(rate_val)}</span></td>'
    )


def token_dist_cell(log_probs, pos: int, id2token: Dict[int, str]) -> str:
    """Build a <td> showing top-5 + other token distribution."""
    import torch

    p = torch.exp(log_probs[pos])
    top5_vals, top5_idx = torch.topk(p, k=min(5, p.numel()))
    lines = ['<td class="tc-tok">']
    for token_id, prob in zip(top5_idx.tolist(), top5_vals.tolist()):
        tok = id2token.get(token_id, f"#{token_id}")
        lines.append(
            f'<span class="tl">{esc(tok)}</span> '
            f'<span class="tp">{prob:.3f}</span><br>'
        )
    other = 1.0 - sum(v.item() for v in top5_vals)
    if other > 1e-6:
        lines.append(
            f'<span class="tl">+</span> '
            f'<span class="tp">{other:.3f}</span>'
        )
    lines.append('</td>')
    return "\n".join(lines)


# ── shared CSS ────────────────────────────────────────────────────────

CSS = """
body{font-family:Menlo,Consolas,monospace;font-size:13px;margin:20px;background:#fafafa}
h1{border-bottom:2px solid #333;padding-bottom:6px}
h2{margin:28px 0 4px 0;border-bottom:1px solid #ccc;padding-bottom:4px}
h3{margin:20px 0 4px 0}
h4{margin:16px 0 2px 0;font-size:13px;color:#555}
.meta{font-weight:normal;font-size:12px;color:#555}
.nav{margin:8px 0 20px 0;line-height:1.8}
.nav a{margin-right:12px}
.tbl-wrap{overflow-x:auto;margin:8px 0 24px 0}
table.vt{border-collapse:collapse;table-layout:auto}
table.vt td,table.vt th{padding:4px 6px;vertical-align:top;text-align:center;min-width:44px;border:1px solid #ddd}
th.lbl{min-width:38px;font-size:11px;color:#666;background:#f5f5f5;text-align:right;padding-right:8px}
.sec-hdr th.sec-label{text-align:left;font-size:11px;font-weight:bold;color:#444;background:#eee;padding:3px 8px}
.tc{font-weight:600;font-size:13px;background:#fafafa}
.tc.bos{background:#e8e8e8;color:#999;font-size:11px}
.tc.oi{border-top:3px solid #2196F3}
.tc.os{border-top:3px solid #FF9800}
.tc.od{border-top:3px solid #F44336}
.tc.oi.bos{border-top:1px solid #ddd}
.tc.os.bos{border-top:1px solid #ddd}
.tc.od.bos{border-top:1px solid #ddd}
/* origin marking for trajectory mode */
.tc.orig{background:#fafafa}
.tc.inserted{background:#e8f5e9}
/* actual-edit markers for trajectory mode */
.tc.aei{border-bottom:3px solid #4CAF50}
.tc.aes{border-bottom:3px solid #FF9800}
.tc.aed{border-bottom:3px solid #F44336}
.rc{font-size:11px;line-height:1.2;min-width:52px}
.rv{font-weight:bold;font-size:12px;display:block}
.tc-tok{font-size:10px;line-height:1.4;background:#fafafa;text-align:left;white-space:nowrap;min-width:52px}
.tc-tok .tl{font-weight:600;color:#444}
.tc-tok .tp{color:#888}
.x0-row td{padding:6px 6px}
/* actual-edit row */
.ae-row td{font-size:12px;font-weight:bold;padding:6px 6px}
.ae-ins{background:#c8e6c9;color:#2e7d32}
.ae-sub{background:#ffe0b2;color:#e65100}
.ae-del{background:#ffcdd2;color:#c62828}
.legend{font-size:11px;color:#666;margin-bottom:20px}
.legend span{margin-right:16px}
.legend .sw{display:inline-block;width:14px;height:14px;border:1px solid #ccc;vertical-align:middle;margin-right:4px}
.legend .sw-rate{display:inline-block;width:40px;height:14px;border:1px solid #ccc;vertical-align:middle;margin:0 2px}
.oi-sw{border-top:3px solid #2196F3;background:#fff}
.os-sw{border-top:3px solid #FF9800;background:#fff}
.od-sw{border-top:3px solid #F44336;background:#fff}
.aei-sw{border-bottom:3px solid #4CAF50;background:#fff}
.aes-sw{border-bottom:3px solid #FF9800;background:#fff}
.aed-sw{border-bottom:3px solid #F44336;background:#fff}
.event-nav{margin:4px 0 12px 0;font-size:12px}
.event-nav a{margin-right:8px;padding:2px 6px;border:1px solid #ccc;border-radius:3px;text-decoration:none;color:#333;background:#f5f5f5}
.event-nav a.correct{background:#c8e6c9;border-color:#a5d6a7}
.event-nav a.wrong{background:#ffcdd2;border-color:#ef9a9a}
.event-nav .sep{color:#999;margin:0 4px}
.correct{color:#2e7d32;font-weight:600}
.wrong{color:#c62828;font-weight:600}
.summary-box{background:#f5f5f5;border:1px solid #ddd;padding:8px 12px;margin:8px 0;font-size:12px;border-radius:4px}
.summary-box b{margin-right:4px}
.smiles-line{display:flex;align-items:flex-start;margin:2px 0}
.smiles-label{display:inline-block;min-width:52px;font-weight:600;color:#666;flex-shrink:0}
.smiles-str{font-family:"Courier New",monospace;font-size:11px;word-break:break-all;line-height:1.5}
.smiles-meta{margin-top:6px;padding-top:4px;border-top:1px solid #ddd}
"""
