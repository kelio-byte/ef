#!/usr/bin/env python
"""Visualize full Euler sampling trajectory with per-edit-event oracle vs model tables.

For each selected example, runs Euler sampling to completion, records the
model predictions / oracle / actual edits at every step where an edit occurs,
and renders each event as an HTML table in the same column-aligned format as
visualize_first_step.py.
"""

import argparse
import json
import math
import os
import random
import subprocess
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
from edit_flows.guidance.model import ProductConditionedGuidance
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.sampling.euler import sample_euler
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


def _load_guidance_model(
    checkpoint_path: str,
    device: torch.device,
    expected_vocab_size: int,
) -> ProductConditionedGuidance:
    """Load an action-level guidance adapter for paired trajectory views."""
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False,
        )
    except TypeError:  # pragma: no cover - older torch fallback
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    vocab_size = int(config.get("model_vocab", 0))
    if vocab_size != expected_vocab_size:
        raise ValueError(
            "guidance/model vocabulary mismatch: guidance checkpoint has "
            f"{vocab_size}, base checkpoint has {expected_vocab_size}"
        )
    model = ProductConditionedGuidance(
        vocab_size=vocab_size,
        hidden_dim=int(config.get("hidden_dim", 256)),
        product_layers=int(config.get("product_layers", 2)),
        state_layers=int(config.get("state_layers", 4)),
        num_heads=int(config.get("num_heads", 8)),
        dim_feedforward=int(config.get("dim_feedforward", 1024)),
        max_seq_len=int(config.get("max_seq_len", 256)),
        dropout=float(config.get("dropout", 0.1)),
        attention_dropout=float(config.get("attention_dropout", 0.1)),
        activation=config.get("activation", "relu"),
        pos_encoding_scale=not bool(config.get("no_pos_encoding_scale", False)),
        pad_token=PAD_TOKEN,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _decode_token_sequence(
    token_ids: torch.Tensor, id2token: Dict[int, str], *, separator: str = " ",
) -> str:
    """Decode a state without hiding token boundaries."""
    tokens = [
        id2token.get(int(token_id), "?")
        for token_id in token_ids.tolist()
        if int(token_id) not in (PAD_TOKEN, BOS_TOKEN)
    ]
    return separator.join(tokens)


def _describe_actions(event: dict, id2token: Dict[int, str]) -> str:
    """Describe every atomic edit in an event, including simultaneous ones."""
    actions = event["actions"]
    x_t = event["x_t"]
    descriptions: List[str] = []
    for pos in range(actions["ins_mask"].numel()):
        if pos == 0 or int(x_t[pos].item()) == PAD_TOKEN:
            continue
        old_token = id2token.get(int(x_t[pos].item()), "?")
        if actions["sub_mask"][pos].item():
            new_token = id2token.get(
                int(actions["sub_tokens"][pos].item()), "?",
            )
            descriptions.append(f"{old_token}→{new_token} @pos {pos}")
        if actions["del_mask"][pos].item():
            descriptions.append(f"-{old_token} @pos {pos}")
        if actions["ins_mask"][pos].item():
            new_token = id2token.get(
                int(actions["ins_tokens"][pos].item()), "?",
            )
            descriptions.append(f"+{new_token} after pos {pos}")
    return "; ".join(descriptions) if descriptions else "no edit"


def _build_sequence_ladder(
    product_str: str,
    target_str: str,
    events: List[dict],
    id2token: Dict[int, str],
) -> str:
    """Render Product -> every post-edit state -> Target for one path."""
    lines = [
        '<div class="smiles-line"><span class="smiles-label">Product</span>'
        f'<span class="smiles-str">{esc(product_str)}</span></div>'
    ]
    for event_idx, event in enumerate(events, start=1):
        x_next = event.get("x_next")
        sequence = (
            _decode_token_sequence(x_next, id2token)
            if x_next is not None else "[post-edit state unavailable]"
        )
        description = _describe_actions(event, id2token)
        lines.append(
            '<div class="smiles-line">'
            f'<span class="smiles-label">Edit {event_idx}</span>'
            f'<span class="smiles-str">{esc(sequence)} '
            f'<span style="color:#8a4b08">--&gt; {esc(description)}</span>'
            '</span></div>'
        )
    lines.append(
        '<div class="smiles-line"><span class="smiles-label">Target</span>'
        f'<span class="smiles-str">{esc(target_str)}</span></div>'
    )
    return "".join(lines)


def _state_key(token_ids: torch.Tensor) -> Tuple[int, ...]:
    """Return the exact non-padding token state used for convergence tests."""
    return tuple(
        int(token_id)
        for token_id in token_ids.tolist()
        if int(token_id) not in (PAD_TOKEN, BOS_TOKEN)
    )


def _reconstruct_post_step_states(
    initial_state: torch.Tensor,
    events: List[dict],
    n_steps: int,
) -> List[Tuple[int, ...]]:
    """Carry event states forward to reconstruct state after every Euler step."""
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    event_states: Dict[int, Tuple[int, ...]] = {}
    for event in events:
        step = int(event["step_idx"])
        if 0 <= step < n_steps and event.get("x_next") is not None:
            event_states[step] = _state_key(event["x_next"])

    current = _state_key(initial_state)
    states: List[Tuple[int, ...]] = []
    for step in range(n_steps):
        current = event_states.get(step, current)
        states.append(current)
    return states


def _find_reconvergence_episodes(
    initial_state: torch.Tensor,
    paths: List[List[dict]],
    n_steps: int,
) -> List[dict]:
    """Find path pairs that differ and later regain the same exact state."""
    states = [
        _reconstruct_post_step_states(initial_state, events, n_steps)
        for events in paths
    ]
    episodes: List[dict] = []
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            divergence_step: Optional[int] = None
            for step in range(n_steps):
                equal = states[left][step] == states[right][step]
                if not equal and divergence_step is None:
                    divergence_step = step
                elif equal and divergence_step is not None:
                    episodes.append({
                        "left_path": left,
                        "right_path": right,
                        "divergence_step": divergence_step,
                        "reconvergence_step": step,
                        "state": states[left][step],
                    })
                    divergence_step = None
    return episodes


def _find_cross_example_collisions(
    initial_states: List[torch.Tensor],
    grouped_events: List[List[List[dict]]],
    n_steps: int,
) -> List[dict]:
    """Report exact same-step states shared by paths from different examples."""
    states = [
        [
            _reconstruct_post_step_states(initial_states[example_idx], path, n_steps)
            for path in example_paths
        ]
        for example_idx, example_paths in enumerate(grouped_events)
    ]
    collisions: List[dict] = []
    seen = set()
    for step in range(n_steps):
        by_state: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
        for example_idx, example_paths in enumerate(states):
            for path_idx, path_states in enumerate(example_paths):
                by_state.setdefault(path_states[step], []).append(
                    (example_idx, path_idx)
                )
        for state, members in by_state.items():
            examples = {example_idx for example_idx, _ in members}
            if len(examples) < 2:
                continue
            identity = (state, tuple(members))
            if identity in seen:
                continue
            seen.add(identity)
            collisions.append({
                "step": step,
                "state": state,
                "members": members,
            })
    return collisions


def _matching_path_view(
    grouped_events: List[List[List[dict]]],
    path_correctness: List[List[bool]],
) -> Tuple[List[List[List[dict]]], List[List[int]]]:
    """Keep Target-matching paths and remember their original path indices."""
    if len(grouped_events) != len(path_correctness):
        raise ValueError("grouped_events and path_correctness must align")

    matched_events: List[List[List[dict]]] = []
    matched_indices: List[List[int]] = []
    for example_paths, example_correctness in zip(
        grouped_events, path_correctness,
    ):
        if len(example_paths) != len(example_correctness):
            raise ValueError("each example must have one correctness flag per path")
        indices = [
            path_idx for path_idx, correct in enumerate(example_correctness)
            if correct
        ]
        matched_indices.append(indices)
        matched_events.append([example_paths[path_idx] for path_idx in indices])
    return matched_events, matched_indices


def _state_key_to_text(
    state: Tuple[int, ...], id2token: Dict[int, str],
) -> str:
    return " ".join(id2token.get(token_id, "?") for token_id in state)


def _build_trajectory_overview(
    example_ids: List[int],
    product_strs: List[str],
    target_strs: List[str],
    initial_states: List[torch.Tensor],
    grouped_events: List[List[List[dict]]],
    grouped_finals: List[torch.Tensor],
    path_correctness: List[List[bool]],
    id2token: Dict[int, str],
    n_steps: int,
) -> str:
    """Build the all-examples path overview before detailed event tables."""
    parts = [
        '<section id="trajectory-overview">',
        '<h1>All Examples — Complete Path Overview</h1>',
        '<p style="color:#666">Reconvergence and cross-example collision '
        'analysis include only paths whose final canonical SMILES matches '
        'Target. State equality itself uses exact token states at the same '
        'post-Euler step (0-based), without chemical canonicalization.</p>',
    ]
    matched_events, matched_indices = _matching_path_view(
        grouped_events, path_correctness,
    )
    for local_idx, example_id in enumerate(example_ids):
        parts.extend([
            f'<div class="ex-section" id="overview-ex{example_id}">',
            f'<h2>Example #{example_id} — all paths</h2>',
        ])
        if len(matched_indices[local_idx]) >= 2:
            episodes = _find_reconvergence_episodes(
                initial_states[local_idx], matched_events[local_idx], n_steps,
            )
        else:
            episodes = []
        if episodes:
            parts.append(
                '<div class="summary-box"><b>Detected divergence → '
                'reconvergence among Target-matching paths:</b><ul>'
            )
            for episode in episodes:
                state_text = _state_key_to_text(episode["state"], id2token)
                left_path = matched_indices[local_idx][episode["left_path"]]
                right_path = matched_indices[local_idx][episode["right_path"]]
                parts.append(
                    '<li>'
                    f'Path #{left_path + 1} vs Path '
                    f'#{right_path + 1}: diverged at step '
                    f'{episode["divergence_step"]}, reconverged at step '
                    f'{episode["reconvergence_step"]} → '
                    f'<code>{esc(state_text)}</code></li>'
                )
            parts.append('</ul></div>')
        elif not matched_indices[local_idx]:
            parts.append(
                '<div class="summary-box">Reconvergence analysis skipped: '
                'no path matches Target.</div>'
            )
        elif len(matched_indices[local_idx]) == 1:
            parts.append(
                '<div class="summary-box">Reconvergence analysis skipped: '
                'fewer than two paths match Target.</div>'
            )
        else:
            parts.append(
                '<div class="summary-box">No exact divergence → '
                'reconvergence detected among Target-matching paths.</div>'
            )

        for path_idx, events in enumerate(grouped_events[local_idx]):
            correct = path_correctness[local_idx][path_idx]
            result = "MATCH" if correct else "MISMATCH"
            result_class = "correct" if correct else "wrong"
            final_text = _decode_token_sequence(
                grouped_finals[local_idx][path_idx], id2token,
            )
            detail_anchor = (
                f"ex{example_id}" if path_idx == 0
                else f"ex{example_id}_path{path_idx}"
            )
            parts.extend([
                '<div class="summary-box" '
                'style="margin-left:16px;border-left:4px solid #78909c">',
                f'<h3>Path #{path_idx + 1} — '
                f'<span class="{result_class}">{result}</span> '
                f'<a href="#{detail_anchor}" '
                'style="font-size:12px">detailed analysis ↓</a></h3>',
                _build_sequence_ladder(
                    product_strs[local_idx], target_strs[local_idx],
                    events, id2token,
                ),
                '<div class="smiles-meta"><b>Final:</b> '
                f'<code>{esc(final_text)}</code></div></div>',
            ])
        parts.append('</div>')

    collisions = _find_cross_example_collisions(
        initial_states, matched_events, n_steps,
    )
    parts.append(
        '<div class="ex-section"><h2>Cross-example state collisions '
        '(Target-matching paths only)</h2>'
    )
    if collisions:
        parts.append('<ul>')
        for collision in collisions:
            members = ", ".join(
                f'Ex#{example_ids[example_idx]} '
                f'Path#{matched_indices[example_idx][path_idx] + 1}'
                for example_idx, path_idx in collision["members"]
            )
            state_text = _state_key_to_text(collision["state"], id2token)
            parts.append(
                f'<li>step {collision["step"]}: {esc(members)} → '
                f'<code>{esc(state_text)}</code></li>'
            )
        parts.append('</ul>')
    else:
        parts.append(
            '<p>No exact token-state collision between Target-matching paths '
            'from different examples.</p>'
        )
    parts.extend(['</div>', '</section>'])
    return "".join(parts)


# ── trajectory HTML builders ──────────────────────────────────────────

def _build_model_distribution_section(
    label: str,
    log_rates: torch.Tensor,
    log_ins_probs: torch.Tensor,
    log_sub_probs: torch.Tensor,
    id2token: Dict[int, str],
    length: int,
) -> str:
    """Render one model distribution block for an event table."""
    ins_rate = [ins_rate_cell(log_rates, j) for j in range(length)]
    ins_tok = [token_dist_cell(log_ins_probs, j, id2token) for j in range(length)]
    sub_rate = [sub_rate_cell(log_rates, j) for j in range(length)]
    sub_tok = [token_dist_cell(log_sub_probs, j, id2token) for j in range(length)]
    delete = [del_rate_cell(log_rates, j) for j in range(length)]
    return (
        f'<tr class="sec-hdr"><th class="lbl sec-label" colspan="{length + 1}">{esc(label)}</th></tr>'
        + '<tr class="ins-row"><th class="lbl">&lambda;<sub>ins</sub></th>'
        + "".join(ins_rate) + "</tr>"
        + '<tr class="ins-tok-row"><th class="lbl">ins top5</th>'
        + "".join(ins_tok) + "</tr>"
        + '<tr class="sub-row"><th class="lbl">&lambda;<sub>sub</sub></th>'
        + "".join(sub_rate) + "</tr>"
        + '<tr class="sub-tok-row"><th class="lbl">sub top5</th>'
        + "".join(sub_tok) + "</tr>"
        + '<tr class="del-row"><th class="lbl">&lambda;<sub>del</sub></th>'
        + "".join(delete) + "</tr>"
    )


def _guidance_factor_cell(
    factors: torch.Tensor,
    pos: int,
    id2token: Dict[int, str],
) -> str:
    """Render the largest action reweighting factors at one position."""
    values = factors[pos].reshape(-1)
    top_values, top_indices = torch.topk(values, k=min(3, values.numel()))
    parts = [f"max {fmt_rate(float(values.max().item()))}"]
    for value, token_id in zip(top_values.tolist(), top_indices.tolist()):
        parts.append(f"{esc(id2token.get(int(token_id), f'#{token_id}'))}:{fmt_rate(float(value))}")
    return '<td class="tc-tok">' + "<br>".join(parts) + "</td>"


def _build_guidance_section(
    guidance_insert: torch.Tensor,
    guidance_substitute: torch.Tensor,
    guidance_delete: torch.Tensor,
    guidance_beta: float,
    id2token: Dict[int, str],
    length: int,
    normalization: str,
) -> str:
    """Render H^beta factors between base and guided model distributions."""
    beta = float(guidance_beta)
    insert = torch.exp(torch.log(guidance_insert.clamp_min(1e-12)) * beta)
    substitute = torch.exp(torch.log(guidance_substitute.clamp_min(1e-12)) * beta)
    delete = torch.exp(torch.log(guidance_delete.clamp_min(1e-12)) * beta)
    return (
        f'<tr class="sec-hdr"><th class="lbl sec-label" colspan="{length + 1}">GUIDANCE H^&beta; '
        f'(beta={beta:.4g}, {esc(normalization)})</th></tr>'
        + '<tr class="ins-tok-row"><th class="lbl">H ins</th>'
        + "".join(_guidance_factor_cell(insert, j, id2token) for j in range(length))
        + "</tr>"
        + '<tr class="sub-tok-row"><th class="lbl">H sub</th>'
        + "".join(_guidance_factor_cell(substitute, j, id2token) for j in range(length))
        + "</tr>"
        + '<tr class="del-row"><th class="lbl">H del</th>'
        + "".join(_guidance_factor_cell(delete, j, id2token) for j in range(length))
        + "</tr>"
    )


def _build_event_table(
    example_idx: int,
    path_idx: int,
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
    pre_model_log_rates: Optional[torch.Tensor] = None,
    pre_model_log_ins_probs: Optional[torch.Tensor] = None,
    pre_model_log_sub_probs: Optional[torch.Tensor] = None,
    guidance_insert: Optional[torch.Tensor] = None,
    guidance_substitute: Optional[torch.Tensor] = None,
    guidance_delete: Optional[torch.Tensor] = None,
    guidance_beta: float = 0.0,
    guidance_rate_normalization: str = "",
    anchor_prefix: str = "",
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

    # ---- base / guidance / guided model sections ----
    has_guidance_trace = (
        pre_model_log_rates is not None
        and pre_model_log_ins_probs is not None
        and pre_model_log_sub_probs is not None
        and guidance_insert is not None
        and guidance_substitute is not None
        and guidance_delete is not None
    )
    if has_guidance_trace:
        model_html = (
            _build_model_distribution_section(
                "BASE MODEL (before guidance)",
                pre_model_log_rates,
                pre_model_log_ins_probs,
                pre_model_log_sub_probs,
                id2token,
                L,
            )
            + _build_guidance_section(
                guidance_insert,
                guidance_substitute,
                guidance_delete,
                guidance_beta,
                id2token,
                L,
                guidance_rate_normalization,
            )
            + _build_model_distribution_section(
                "GUIDED MODEL (used for sampling)",
                model_log_rates,
                model_log_ins_probs,
                model_log_sub_probs,
                id2token,
                L,
            )
        )
    else:
        model_html = _build_model_distribution_section(
            "MODEL", model_log_rates, model_log_ins_probs,
            model_log_sub_probs, id2token, L,
        )

    # ---- actual edit row ----
    ae_cells: List[str] = []
    for j in range(L):
        if j == 0:
            ae_cells.append('<td class="ae-row" style="background:#e8e8e8;color:#999">—</td>')
        else:
            operations: List[str] = []
            classes = ["ae-row"]
            if act_sub_mask[j].item():
                tok_id = int(act_sub_tokens[j].item())
                operations.append(f'→{esc(id2token.get(tok_id, "?"))}')
                classes.append("ae-sub")
            if act_del_mask[j].item():
                operations.append("DEL")
                classes.append("ae-del")
            if act_ins_mask[j].item():
                tok_id = int(act_ins_tokens[j].item())
                operations.append(f'+{esc(id2token.get(tok_id, "?"))}')
                classes.append("ae-ins")
            ae_cells.append(
                f'<td class="{" ".join(classes)}">'
                f'{"; ".join(operations)}</td>'
            )

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
        f"Ex#{example_idx} Path #{path_idx + 1} "
        f"Event #{event_idx + 1}/{n_events} &nbsp; "
        f"step={step_idx} &nbsp; t={t_value:.4f} &nbsp; "
        f"Final-Correct: {fc_str}"
    )

    anchor = f"{anchor_prefix}ex{example_idx}_path{path_idx}_ev{event_idx}"
    return (
        f'<div class="ex" id="{anchor}">'
        f'<h4>{meta}</h4>'
        f'<div class="tbl-wrap"><table class="vt">'
        f"{xt_row}{oracle_html}{model_html}{actual_html}"
        f"</table></div></div>"
    )


def _build_example_section(
    example_idx: int,
    path_idx: int,
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
    anchor_prefix: str = "",
    section_label: str = "",
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
            f'<a class="{cls}" href="#{anchor_prefix}ex{example_idx}_path{path_idx}_ev{ei}">'
            f'#{ei + 1} t={t_val:.3f} s={step}</a>'
        )
        if ei < n_events - 1:
            nav_parts.append('<span class="sep">→</span>')
    nav_parts.append("</div>")

    # Summary box with every concrete intermediate post-edit state.
    fc_str = "MATCH" if final_correct else "MISMATCH"
    fc_cls = "correct" if final_correct else "wrong"
    n_correct_events = sum(event_correctness)
    summary = (
        f'<div class="summary-box">'
        f'{_build_sequence_ladder(product_str, target_str, events, id2token)}'
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
                path_idx=path_idx,
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
                pre_model_log_rates=ev.get("log_rates_pre_guidance"),
                pre_model_log_ins_probs=ev.get("log_ins_probs_pre_guidance"),
                pre_model_log_sub_probs=ev.get("log_sub_probs_pre_guidance"),
                guidance_insert=ev.get("guidance_insert"),
                guidance_substitute=ev.get("guidance_substitute"),
                guidance_delete=ev.get("guidance_delete"),
                guidance_beta=float(ev.get("guidance_beta", 0.0)),
                guidance_rate_normalization=ev.get(
                    "guidance_rate_normalization", "",
                ),
                anchor_prefix=anchor_prefix,
            )
        )

    section_anchor = (
        f"{anchor_prefix}ex{example_idx}" if path_idx == 0
        else f"{anchor_prefix}ex{example_idx}_path{path_idx}"
    )
    heading = f"Example #{example_idx} — Path #{path_idx + 1}"
    if section_label:
        heading = f"{section_label} · {heading}"
    return (
        f'<div class="ex-section" id="{section_anchor}">'
        f"<h2>{esc(heading)}</h2>"
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
            f'<a href="#overview-ex{eid}">Ex#{eid} {fc_str}</a> '
            f'<span style="color:#999;font-size:11px">({esc(prod[:50])} → {esc(tgt[:50])})</span><br>'
        )

    lines.append("</div>")
    return "\n".join(lines)


def _build_footer_html() -> str:
    return "</body></html>"


# ── helpers ────────────────────────────────────────────────────────────

def _decode_to_smiles(token_ids: torch.Tensor, id2token: Dict[int, str]) -> str:
    """Decode token IDs to SMILES string, dropping BOS and PAD."""
    return _decode_token_sequence(token_ids, id2token, separator="")


def _str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "y"):
        return True
    if normalized in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean value, got {value!r}"
    )


def _run_trajectory(
    model,
    guidance_model,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    *,
    scheduler,
    train_scheduler,
    cfg: dict,
    n_steps: int,
    n_samples: int,
    use_origin_mask: bool,
    time_input: str,
    guidance_beta: float,
    guidance_rate_normalization: str,
    id2token: Dict[int, str],
) -> dict:
    """Run one ordinary or guided set of paths and retain paired diagnostics."""
    batch_x0 = x_0.repeat_interleave(n_samples, dim=0)
    batch_x1 = x_1.repeat_interleave(n_samples, dim=0)
    x_final, _trajectory, all_events = sample_euler(
        model,
        batch_x0,
        scheduler,
        n_steps=n_steps,
        max_seq_len=cfg["max_seq_len"],
        use_rate_reparam=cfg.get("use_rate_reparam", False),
        clamp_kappa=cfg.get("clamp_kappa", False),
        clamp_max=cfg.get("clamp_max", 50.0),
        time_input=time_input,
        train_scheduler=train_scheduler,
        record_all_events=True,
        x_1=batch_x1,
        vocab_size=model.vocab_size,
        use_origin_mask=use_origin_mask,
        guidance_model=guidance_model,
        guidance_product=batch_x0 if guidance_model is not None else None,
        guidance_beta=guidance_beta,
        guidance_rate_normalization=guidance_rate_normalization,
    )
    n_sel = x_0.shape[0]
    grouped_events = []
    grouped_finals = []
    for i in range(n_sel):
        start = i * n_samples
        end = start + n_samples
        grouped_events.append(all_events[start:end])
        grouped_finals.append(x_final[start:end])

    path_correctness: List[List[bool]] = []
    final_corrects: List[bool] = []
    for i in range(n_sel):
        tgt_canon = _canonicalize_smiles(_decode_to_smiles(x_1[i], id2token))
        example_correctness: List[bool] = []
        for path in grouped_finals[i]:
            pred_canon = _canonicalize_smiles(_decode_to_smiles(path, id2token))
            example_correctness.append(
                bool(pred_canon and tgt_canon and pred_canon == tgt_canon)
            )
        path_correctness.append(example_correctness)
        final_corrects.append(any(example_correctness))
    return {
        "initial_rows": x_0.clone(),
        "target_rows": x_1.clone(),
        "grouped_events": grouped_events,
        "grouped_finals": grouped_finals,
        "path_correctness": path_correctness,
        "final_corrects": final_corrects,
    }


def _first_divergence(
    initial_left: torch.Tensor,
    events_left: List[dict],
    initial_right: torch.Tensor,
    events_right: List[dict],
    n_steps: int,
) -> Optional[int]:
    """Return the first exact post-step state difference between two paths."""
    left = _reconstruct_post_step_states(initial_left, events_left, n_steps)
    right = _reconstruct_post_step_states(initial_right, events_right, n_steps)
    for step, (left_state, right_state) in enumerate(zip(left, right)):
        if left_state != right_state:
            return step
    return None


def _build_paired_overview(
    example_ids: List[int],
    product_strs: List[str],
    target_strs: List[str],
    base_run: dict,
    guided_run: dict,
    n_steps: int,
) -> str:
    """Render a compact same-page comparison before detailed event tables."""
    parts = [
        '<section id="paired-overview">',
        '<h1>Paired Euler vs Guidance Overview</h1>',
        '<p style="color:#666">Both paths use the same product, augmentation, '
        'sample count and reset seed. Once states diverge, later random events '
        'are reported as separate paths rather than treated as aligned edits.</p>',
        '<table class="vt"><tr><th>Example</th><th>Path</th>'
        '<th>Euler</th><th>Guidance</th><th>First exact divergence step</th>'
        '<th>Euler final</th><th>Guidance final</th></tr>',
    ]
    for local_idx, example_id in enumerate(example_ids):
        for path_idx in range(len(base_run["grouped_events"][local_idx])):
            divergence = _first_divergence(
                base_run["initial_rows"][local_idx],
                base_run["grouped_events"][local_idx][path_idx],
                guided_run["initial_rows"][local_idx],
                guided_run["grouped_events"][local_idx][path_idx],
                n_steps,
            )
            base_ok = base_run["path_correctness"][local_idx][path_idx]
            guided_ok = guided_run["path_correctness"][local_idx][path_idx]
            # The detailed sections contain the readable token sequences; the
            # overview deliberately keeps this row short and target-free.
            parts.append(
                f'<tr><td>Ex#{example_id}</td><td>Path #{path_idx + 1}</td>'
                f'<td class="{"correct" if base_ok else "wrong"}">'
                f'{"MATCH" if base_ok else "MISMATCH"}</td>'
                f'<td class="{"correct" if guided_ok else "wrong"}">'
                f'{"MATCH" if guided_ok else "MISMATCH"}</td>'
                f'<td>{"none" if divergence is None else divergence}</td>'
                f'<td>{"correct" if base_ok else "wrong"}</td>'
                f'<td>{"correct" if guided_ok else "wrong"}</td></tr>'
            )
    parts.extend(['</table>', '</section>'])
    return "".join(parts)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ── main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize full Euler sampling trajectory with per-event tables"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--guidance_checkpoint", type=str, default=None,
        help="Optional action-level guidance checkpoint for paired Euler views",
    )
    parser.add_argument("--guidance_beta", type=float, default=1.0)
    parser.add_argument(
        "--guidance_rate_normalization",
        choices=("per_position", "per_sample"),
        default="per_position",
    )
    parser.add_argument("--products_file", type=str, required=True)
    parser.add_argument("--targets_file", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, default=None)
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory; defaults to visualizations/trajectory-<example_ids>",
    )
    parser.add_argument("--html", type=_str_to_bool,
                        default=True, help="Generate HTML output (default: True)")
    parser.add_argument(
        "--table", type=_str_to_bool, default=True,
        help=("Include per-event oracle/model tables after the all-example "
              "path overview; False keeps only the compact overview"),
    )
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
                        help=("Reserved for a future Euler-Beam branch-tree "
                              "recorder; current complete-path view uses "
                              "--n_samples"))
    args = parser.parse_args()

    if args.n_samples < 1:
        parser.error("--n_samples must be at least 1")
    if args.guidance_beta < 0 or not torch.isfinite(torch.tensor(args.guidance_beta)):
        parser.error("--guidance_beta must be finite and non-negative")
    if args.n_branches:
        parser.error(
            "Euler-Beam does not currently expose branch ancestry/events. "
            "The former --n_branches path attempted to unpack an unsupported "
            "return value. Use --n_samples for complete independent Euler "
            "paths; branch-tree recording must be implemented separately."
        )

    output_dir = args.output_dir or f"visualizations/trajectory-{args.example_ids}"
    if args.html:
        os.makedirs(output_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # ── Load model ──
    try:
        ckpt = torch.load(
            args.checkpoint, map_location=device, weights_only=False,
        )
    except TypeError:  # pragma: no cover - older torch fallback
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
    guidance_model = None
    if args.guidance_checkpoint:
        guidance_model = _load_guidance_model(
            args.guidance_checkpoint, device, model_vocab,
        )

    # ── Load data ──
    products, targets = load_parallel_texts(
        args.products_file, args.targets_file,
        deduplicate=args.deduplicate, max_lines=args.max_lines,
    )
    # ── Select examples ──
    if args.example_ids:
        selected = [int(x) for x in args.example_ids.split(",")]
        selected = [i for i in selected if 0 <= i < len(products)]
    else:
        rng = random.Random(args.seed)
        selected = sorted(rng.sample(range(len(products)), min(args.n_examples, len(products))))

    print(
        f"Running trajectory sampling for {len(selected)} examples x "
        f"{args.n_samples} complete paths: {selected}"
    )
    print(f"Scheduler: {scheduler_name}, n_steps: {args.n_steps}")

    # ── Build batch of selected examples ──
    sel_products = [tokenize_smiles(products[i], token2id) for i in selected]
    sel_targets = [tokenize_smiles(targets[i], token2id) for i in selected]
    x_0, x_1 = build_model_batch(sel_products, sel_targets)
    initial_rows = x_0.clone()
    target_rows = x_1.clone()
    n_sel = len(selected)

    # ── Run ordinary and, when requested, guided trajectories ──
    use_rate_reparam = cfg.get("use_rate_reparam", False)
    time_input = cfg.get("time_input", "t")
    print(
        f"use_rate_reparam={use_rate_reparam}, use_origin_mask={use_origin_mask}, "
        f"time_input={time_input}"
    )

    torch.manual_seed(args.seed)
    base_run = _run_trajectory(
        model, None, x_0, x_1,
        scheduler=scheduler, train_scheduler=train_scheduler, cfg=cfg,
        n_steps=args.n_steps, n_samples=args.n_samples,
        use_origin_mask=use_origin_mask, time_input=time_input,
        guidance_beta=0.0,
        guidance_rate_normalization=args.guidance_rate_normalization,
        id2token=id2token,
    )
    guided_run = None
    if guidance_model is not None:
        torch.manual_seed(args.seed)
        guided_run = _run_trajectory(
            model, guidance_model, x_0, x_1,
            scheduler=scheduler, train_scheduler=train_scheduler, cfg=cfg,
            n_steps=args.n_steps, n_samples=args.n_samples,
            use_origin_mask=use_origin_mask, time_input=time_input,
            guidance_beta=args.guidance_beta,
            guidance_rate_normalization=args.guidance_rate_normalization,
            id2token=id2token,
        )

    runs = [("Euler", base_run, "euler-")]
    if guided_run is not None:
        runs.append(("Guidance", guided_run, "guidance-"))
    for label, run, _prefix in runs:
        print(f"[{label}]")
        for bi, idx in enumerate(selected):
            n_match = sum(run["path_correctness"][bi])
            event_counts = [len(events) for events in run["grouped_events"][bi]]
            print(
                f"  Example #{idx}: {n_match}/{args.n_samples} paths match; "
                f"edit events per path={event_counts}"
            )
        matched_events, _ = _matching_path_view(
            run["grouped_events"], run["path_correctness"],
        )
        reconvergence_count = sum(
            len(_find_reconvergence_episodes(
                run["initial_rows"][bi], matched_events[bi], args.n_steps,
            ))
            for bi in range(n_sel)
            if len(matched_events[bi]) >= 2
        )
        cross_collision_count = len(_find_cross_example_collisions(
            [run["initial_rows"][i] for i in range(n_sel)],
            matched_events,
            args.n_steps,
        ))
        print(
            "  Target-matching trajectory comparison: "
            f"{reconvergence_count} divergence/reconvergence episodes; "
            f"{cross_collision_count} cross-example state collisions"
        )

    # ── Build HTML (only when --html) ──
    if args.html:
        html_parts: List[str] = []
        if guided_run is None:
            html_parts.extend([
                _build_index_html(
                    selected,
                    [products[i] for i in selected],
                    [targets[i] for i in selected],
                    base_run["final_corrects"],
                    checkpoint=args.checkpoint,
                ),
                _build_trajectory_overview(
                    example_ids=selected,
                    product_strs=[products[i] for i in selected],
                    target_strs=[targets[i] for i in selected],
                    initial_states=[base_run["initial_rows"][i] for i in range(n_sel)],
                    grouped_events=base_run["grouped_events"],
                    grouped_finals=base_run["grouped_finals"],
                    path_correctness=base_run["path_correctness"],
                    id2token=id2token,
                    n_steps=args.n_steps,
                ),
            ])
        else:
            html_parts.extend([
                _build_index_html(
                    selected,
                    [products[i] for i in selected],
                    [targets[i] for i in selected],
                    base_run["final_corrects"],
                    checkpoint=(
                        f"Euler={args.checkpoint}; "
                        f"guidance={args.guidance_checkpoint}"
                    ),
                ),
                _build_paired_overview(
                    selected,
                    [products[i] for i in selected],
                    [targets[i] for i in selected],
                    base_run,
                    guided_run,
                    args.n_steps,
                ),
            ])

        if args.table:
            html_parts.append(
                '<h1>Paired per-path Event Analysis</h1>'
                if guided_run is not None else '<h1>Per-path Event Analysis</h1>'
            )
            for bi, idx in enumerate(selected):
                pair_wrapper = guided_run is not None
                if pair_wrapper:
                    html_parts.append(
                        '<div style="display:grid;grid-template-columns:1fr 1fr;'
                        'gap:24px;border-bottom:2px solid #555;padding-bottom:20px;">'
                    )
                for label, run, prefix in runs:
                    if guided_run is None and label != "Euler":
                        continue
                    if pair_wrapper:
                        html_parts.append(f'<div><h2>{label}</h2>')
                    n_match = sum(run["path_correctness"][bi])
                    for path_idx, events in enumerate(run["grouped_events"][bi]):
                        final_prediction = _decode_token_sequence(
                            run["grouped_finals"][bi][path_idx], id2token,
                        )
                        sample_info = (
                            f"{label}; Path #{path_idx + 1}; "
                            f"final={final_prediction}; "
                            f"{n_match}/{args.n_samples} paths match"
                        )
                        html_parts.append(
                            _build_example_section(
                                example_idx=idx,
                                path_idx=path_idx,
                                product_str=products[idx],
                                target_str=targets[idx],
                                events=events,
                                id2token=id2token,
                                final_correct=run["path_correctness"][bi][path_idx],
                                scheduler=scheduler,
                                use_rate_reparam=use_rate_reparam,
                                clamp_kappa=cfg.get("clamp_kappa", False),
                                clamp_max=cfg.get("clamp_max", 50.0),
                                sample_info=sample_info,
                                anchor_prefix=prefix if pair_wrapper else "",
                                section_label=label if pair_wrapper else "",
                            )
                        )
                    if pair_wrapper:
                        html_parts.append('</div>')
                if pair_wrapper:
                    html_parts.append('</div>')

        html_parts.append(_build_footer_html())
        ids_list = (
            [int(x) for x in args.example_ids.split(",")]
            if args.example_ids else selected
        )
        if len(ids_list) <= 4:
            id_label = "_".join(str(i) for i in ids_list)
        else:
            id_label = "_".join(str(i) for i in ids_list[:3]) + f"_x{len(ids_list)}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"trajectory_{id_label}_{timestamp}.html")
        with open(out_path, "w") as f:
            f.write("\n".join(html_parts))
        metadata = {
            "schema_version": 2,
            "script": "scripts/visualize_trajectory.py",
            "git_commit": _git_revision(),
            "argv": list(sys.argv),
            "checkpoint": os.path.abspath(args.checkpoint),
            "guidance_checkpoint": (
                os.path.abspath(args.guidance_checkpoint)
                if args.guidance_checkpoint else None
            ),
            "products_file": os.path.abspath(args.products_file),
            "targets_file": os.path.abspath(args.targets_file),
            "example_ids": selected,
            "seed": args.seed,
            "n_steps": args.n_steps,
            "n_samples": args.n_samples,
            "guidance_beta": args.guidance_beta if guided_run is not None else 0.0,
            "guidance_rate_normalization": args.guidance_rate_normalization,
            "paired": guided_run is not None,
            "device": str(device),
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "html": os.path.abspath(out_path),
        }
        metadata_path = os.path.splitext(out_path)[0] + ".metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Wrote {out_path}")
        print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
