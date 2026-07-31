"""Beam search and greedy single-edit sampling for Edit Flows.

Provides deterministic/semi-deterministic alternatives to stochastic Euler
sampling.  Each step selects and applies exactly one edit (insert, substitute,
or delete), scored by conditional event probability in the Frozen-Hazard
first-event framework:

    p_stop = e^{-U}
    p_e    = (1 - e^{-U}) * u_e / U

where U is the total executable edit mass and u_e is the per-edit rate.

Naming convention:
    κ   — absolute time (per-hypothesis kappa)
    u   — per-action log-rate, shape (L, 2V+1): [0:V) ins, [V:2V) sub, [2V] del
    U   — total executable edit mass (scalar)
    p   — normalized probability (log-space: log_p)
    _e  — edit-specific; _stop — STOP-specific
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn.functional as F
from torch import Tensor

from edit_flows.core.rate_scale import apply_rate_parameterization
from edit_flows.core.scheduler import KappaScheduler
from edit_flows.sampling.euler import _compute_model_time
from edit_flows.sampling.ops import apply_ins_del_operations
from edit_flows.sampling.time_policy import TimePolicy
from edit_flows.utils.tokens import BOS_TOKEN, GAP_TOKEN, PAD_TOKEN, UNK_TOKEN

FORBIDDEN_TOKENS = {PAD_TOKEN, BOS_TOKEN, GAP_TOKEN, UNK_TOKEN}
LOG_NEG_INF = -1e9


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EditCandidate:
    """A single edit operation with its rate-derived score."""

    pos: int
    op: str  # "ins" | "sub" | "del"
    token: Optional[int]  # token id for ins/sub; None for del
    log_u: float          # log-rate of this edit
    score: float          # log_u - log_U (legacy; used in non-FH threshold mode)
    old_token: Optional[int] = None  # pre-edit token at pos (sub only)


@dataclass
class ActionCandidate:
    """Unified action: either STOP or a concrete edit.

    When *kind* is ``"stop"``, *edit* is None and *log_p* = log p_stop.
    When *kind* is ``"edit"``, *edit* is set and *log_p* = log p_e.
    """

    kind: str  # "stop" | "edit"
    log_p: float
    edit: Optional[EditCandidate] = None


@dataclass
class BeamState:
    """One beam-search state: a sequence plus accumulated metadata."""

    x_t: Tensor  # (L,) token ids including BOS prefix, no PAD suffix
    origin_mask: Optional[Tensor]  # (L,) bool, or None
    log_p: float                   # accumulated trajectory log-probability
    last_edit: Optional[EditCandidate] = None
    is_finished: bool = False
    time_policy: Optional[TimePolicy] = None
    kappa: float = 0.0             # per-hypothesis κ (explicit_stop mode)
    _U: float = 0.0                # cached U for finalization of truncated paths
    _key_cache: Optional[Tuple] = None  # cached beam-state identity key


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_reverse_op(edit: EditCandidate, last_edit: Optional[EditCandidate]) -> bool:
    """Return True if *edit* would immediately undo *last_edit*.

    Uses exact token tracking for substitute chains (a→b→a is reverse,
    a→b→c is allowed) and exact position matching for insert/delete pairs.
    """
    if last_edit is None:
        return False

    # sub(i, a→b) then sub(i, b→a)  →  reverse
    # sub(i, a→b) then sub(i, b→c)  →  allowed (multi-step correction)
    if edit.op == "sub" and last_edit.op == "sub" and edit.pos == last_edit.pos:
        return edit.token == last_edit.old_token

    # ins(i, a) then del at same position  →  reverse
    # (after insertion the new token sits at position i)
    if edit.op == "del" and last_edit.op == "ins":
        return edit.pos == last_edit.pos

    # del(i, a) then ins at same position with same token  →  reverse
    if edit.op == "ins" and last_edit.op == "del":
        return edit.pos == last_edit.pos and edit.token == last_edit.token

    return False


def _build_forbidden_mask(vocab_size: int, device: torch.device) -> Tensor:
    """Bool mask: True = allowed, False = forbidden (PAD/BOS/GAP/UNK)."""
    mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
    for tok in FORBIDDEN_TOKENS:
        if tok < vocab_size:
            mask[tok] = False
    return mask


# ---------------------------------------------------------------------------
# GPU: build log_u_edit matrix & compute U
# ---------------------------------------------------------------------------


def _build_log_u_edit(
    log_rates: Tensor,        # (N, L, 3)
    log_ins_probs: Tensor,    # (N, L, V)
    log_sub_probs: Tensor,    # (N, L, V)
    x_t: Tensor,              # (N, L) token ids
    pad_token: int,
    forbidden_mask: Tensor,   # (V,) bool
    bos_pos: int = 0,
) -> Tensor:
    """Build (N, L, 2V+1) log-u matrix on the same device as inputs.

    Layout per position: [0..V-1] insert actions, [V..2V-1] substitute actions,
    [2V] delete action.  Invalid actions are masked to ``LOG_NEG_INF``.

    Invalid actions:
    - Forbidden tokens (PAD, BOS, GAP, UNK) for insert and substitute
    - No-op substitution (token equals current token at that position)
    - Substitution / deletion on BOS position
    - Any action on PAD position
    """
    N, L, _ = log_rates.shape
    V = log_ins_probs.shape[-1]
    device = log_rates.device

    log_lambda_ins = log_rates[:, :, 0:1]  # (N, L, 1)
    log_lambda_sub = log_rates[:, :, 1:2]  # (N, L, 1)
    log_lambda_del = log_rates[:, :, 2:3]  # (N, L, 1)

    # ---- insert: log_u_ins = log_λ_ins + log_p_ins(token|pos) ----
    log_u_ins = log_lambda_ins + log_ins_probs  # (N, L, V)
    log_u_ins = log_u_ins.masked_fill(~forbidden_mask.view(1, 1, -1), LOG_NEG_INF)

    # ---- substitute: log_u_sub = log_λ_sub + log_p_sub(token|pos) ----
    log_u_sub = log_lambda_sub + log_sub_probs  # (N, L, V)
    log_u_sub = log_u_sub.masked_fill(~forbidden_mask.view(1, 1, -1), LOG_NEG_INF)
    # Mask no-op: sub(pos, current_token)
    noop_mask = torch.zeros(N, L, V, dtype=torch.bool, device=device)
    noop_mask[torch.arange(N, device=device)[:, None],
              torch.arange(L, device=device)[None, :],
              x_t] = True
    log_u_sub = log_u_sub.masked_fill(noop_mask, LOG_NEG_INF)

    # ---- delete: log_u_del = log_λ_del ----
    log_u_del = log_lambda_del  # (N, L, 1)

    # ---- position validity ----
    non_pad = x_t != pad_token  # (N, L)
    bos_violation = torch.zeros(N, L, dtype=torch.bool, device=device)
    bos_violation[:, bos_pos] = True

    ins_pos_valid = non_pad                       # insert allowed on BOS
    sub_del_pos_valid = non_pad & ~bos_violation   # sub/del NOT allowed on BOS

    log_u_ins = log_u_ins.masked_fill(~ins_pos_valid.unsqueeze(-1), LOG_NEG_INF)
    log_u_sub = log_u_sub.masked_fill(~sub_del_pos_valid.unsqueeze(-1), LOG_NEG_INF)
    log_u_del = log_u_del.masked_fill(~sub_del_pos_valid.unsqueeze(-1), LOG_NEG_INF)

    return torch.cat([log_u_ins, log_u_sub, log_u_del], dim=-1)  # (N, L, 2V+1)


def _compute_U(log_u_edit: Tensor) -> Tensor:
    """U = sum(exp(log_u)) over all executable actions → (N,).

    Uses logsumexp for numerical stability.  -inf entries (masked actions)
    contribute 0 to the sum.
    """
    N, L, D = log_u_edit.shape
    log_u_flat = log_u_edit.reshape(N, -1)  # (N, L*D)
    return torch.exp(torch.logsumexp(log_u_flat, dim=-1))  # (N,)


# ---------------------------------------------------------------------------
# GPU-batched candidate selection
# ---------------------------------------------------------------------------


def _select_top_edits_batch(
    log_u_edit: Tensor,       # (N, L, 2V+1) on GPU
    V: int,
    x_t: Tensor,              # (N, L) on GPU
    log_U: Tensor,            # (N,) on GPU
    k_ins_token: int,
    k_sub_token: int,
    k_edit_expand: int,
) -> List[List[EditCandidate]]:
    """GPU-batched top-K edit candidate selection.

    All heavy lifting (per-position top-K, global top-K) stays on GPU.
    Only the final ≤ *k_edit_expand* candidates per sample are transferred to CPU.

    Returns one candidate list per sample in the batch.
    """
    N, L, D = log_u_edit.shape
    device = log_u_edit.device

    log_u_ins = log_u_edit[:, :, :V]        # (N, L, V)
    log_u_sub = log_u_edit[:, :, V:2 * V]   # (N, L, V)
    log_u_del = log_u_edit[:, :, 2 * V]      # (N, L)

    # Per-position top-K (GPU, single kernel per op type).
    k_ins = min(k_ins_token, V)
    top_ins_vals, top_ins_idx = torch.topk(log_u_ins, k_ins, dim=-1)  # (N, L, k_ins)

    k_sub = min(k_sub_token, V)
    top_sub_vals, top_sub_idx = torch.topk(log_u_sub, k_sub, dim=-1)  # (N, L, k_sub)

    pos_idx = torch.arange(L, device=device)  # (L,)
    op_names = ["ins", "sub", "del"]

    # Pre-fetch CPU copies for the final candidate construction loop.
    log_U_cpu = log_U.cpu().tolist()
    x_t_cpu = x_t.cpu()

    results: List[List[EditCandidate]] = []

    for n in range(N):
        # ---- build flat tensors for this sample (all on GPU) ----
        ins_vals_flat = top_ins_vals[n].reshape(-1)                              # (L*k_ins,)
        ins_pos_flat = pos_idx.unsqueeze(-1).expand(-1, k_ins).reshape(-1)      # (L*k_ins,)
        ins_tok_flat = top_ins_idx[n].reshape(-1)                                # (L*k_ins,)
        ins_op_flat = torch.zeros(ins_pos_flat.numel(), dtype=torch.long, device=device)

        sub_vals_flat = top_sub_vals[n].reshape(-1)                              # (L*k_sub,)
        sub_pos_flat = pos_idx.unsqueeze(-1).expand(-1, k_sub).reshape(-1)      # (L*k_sub,)
        sub_tok_flat = top_sub_idx[n].reshape(-1)                                # (L*k_sub,)
        sub_op_flat = torch.ones(sub_pos_flat.numel(), dtype=torch.long, device=device)

        del_vals_flat = log_u_del[n]                                            # (L,)
        del_pos_flat = pos_idx                                                  # (L,)
        del_tok_flat = torch.full((L,), -1, dtype=torch.long, device=device)
        del_op_flat = torch.full((L,), 2, dtype=torch.long, device=device)

        # ---- concatenate & filter ----
        all_vals = torch.cat([ins_vals_flat, sub_vals_flat, del_vals_flat])
        all_pos = torch.cat([ins_pos_flat, sub_pos_flat, del_pos_flat])
        all_tok = torch.cat([ins_tok_flat, sub_tok_flat, del_tok_flat])
        all_op = torch.cat([ins_op_flat, sub_op_flat, del_op_flat])

        valid = all_vals > LOG_NEG_INF / 2
        all_vals = all_vals[valid]
        if all_vals.numel() == 0:
            results.append([])
            continue
        all_pos = all_pos[valid]
        all_tok = all_tok[valid]
        all_op = all_op[valid]

        # ---- global top-K (GPU) ----
        if all_vals.numel() > k_edit_expand:
            topk_vals, topk_idx = torch.topk(all_vals, k_edit_expand)
        else:
            topk_vals, topk_idx = all_vals, torch.arange(all_vals.numel(), device=device)

        # ---- single CPU transfer (≤ k_edit_expand items) ----
        vals_cpu = topk_vals.cpu().tolist()
        pos_cpu = all_pos[topk_idx].cpu().tolist()
        tok_cpu = all_tok[topk_idx].cpu().tolist()
        op_cpu = all_op[topk_idx].cpu().tolist()

        x_n_cpu = x_t_cpu[n]
        log_U_n = log_U_cpu[n]

        candidates = []
        for v, p, op_id, tok in zip(vals_cpu, pos_cpu, op_cpu, tok_cpu):
            old_token = int(x_n_cpu[p].item()) if op_id == 1 else None
            token = int(tok) if tok >= 0 else None
            candidates.append(EditCandidate(
                pos=p, op=op_names[op_id], token=token,
                log_u=v, score=v - log_U_n,
                old_token=old_token,
            ))
        results.append(candidates)

    return results


def _collect_edit_candidates_single(
    log_rates: Tensor,       # (L, 3)
    log_ins_probs: Tensor,   # (L, V)
    log_sub_probs: Tensor,   # (L, V)
    x_t: Tensor,             # (L,) current token ids
    non_pad_mask: Tensor,    # (L,) bool
    log_U: float,
    k_ins_token: int,
    k_sub_token: int,
    k_edit_expand: int,
    forbidden_mask: Tensor,  # (V,) bool
    bos_pos: int = 0,
) -> Tuple[List[EditCandidate], float]:
    """Collect top edit candidates for a single sequence.

    Kept for backward compatibility (used by tests).
    Internally delegates to ``_select_top_edits_batch``.
    """
    log_u_single = _build_log_u_edit(
        log_rates.unsqueeze(0),
        log_ins_probs.unsqueeze(0),
        log_sub_probs.unsqueeze(0),
        x_t.unsqueeze(0),
        pad_token=PAD_TOKEN,
        forbidden_mask=forbidden_mask,
        bos_pos=bos_pos,
    )  # (1, L, 2V+1)
    log_U_tensor = torch.tensor([log_U], device=log_u_single.device)
    V = log_ins_probs.shape[-1]
    batch_results = _select_top_edits_batch(
        log_u_single, V, x_t.unsqueeze(0), log_U_tensor,
        k_ins_token, k_sub_token, k_edit_expand,
    )
    return batch_results[0], log_U


# ---------------------------------------------------------------------------
# Rate preparation (kept for backward compatibility)
# ---------------------------------------------------------------------------


def _prepare_log_rates_for_scoring(
    log_rates: Tensor,  # (B, L, 3)
    t: Tensor,  # (B, 1)
    scheduler: KappaScheduler,
    use_rate_reparam: bool,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
    train_scheduler: Optional[KappaScheduler] = None,
    t_model: Optional[Tensor] = None,
) -> Tensor:
    """Return the log-rates used for candidate scoring.

    When ``use_rate_reparam=True`` we keep the model's raw/base rates so
    candidate preferences remain independent of k(t).  Otherwise we recover
    the real rates, including cross-scheduler correction when needed.
    """
    if use_rate_reparam:
        return log_rates

    log_rates_score = apply_rate_parameterization(
        log_rates, t, scheduler,
        use_rate_reparam=False,
        clamp_kappa=clamp_kappa, clamp_max=clamp_max,
    )
    if train_scheduler is not None and scheduler.name != train_scheduler.name:
        if t_model is None:
            raise ValueError("t_model is required for cross-scheduler correction")
        from edit_flows.core.rate_scale import get_rate_scale
        k_sample = get_rate_scale(
            t, scheduler, clamp_kappa=clamp_kappa, clamp_max=clamp_max,
        )
        k_train = get_rate_scale(
            t_model, train_scheduler, clamp_kappa=clamp_kappa, clamp_max=clamp_max,
        )
        log_correction = torch.log(
            k_sample / k_train.clamp_min(1e-12)
        ).unsqueeze(1)
        log_rates_score = log_rates_score + log_correction
    return log_rates_score


def _compute_executable_u_tot(
    log_rates: Tensor,  # (B, L, 3)
    log_ins_probs: Tensor,  # (B, L, V)
    log_sub_probs: Tensor,  # (B, L, V)
    x_t: Tensor,  # (B, L)
    pad_token: int,
    forbidden_mask: Tensor,  # (V,) bool
    bos_pos: int = 0,
) -> Tensor:
    """Total mass of edits that the sampler is actually allowed to execute.

    Kept for backward compatibility.  Internally delegates to
    ``_build_log_u_edit`` + ``_compute_U``.
    """
    log_u_edit = _build_log_u_edit(
        log_rates, log_ins_probs, log_sub_probs, x_t,
        pad_token=pad_token, forbidden_mask=forbidden_mask, bos_pos=bos_pos,
    )
    return _compute_U(log_u_edit)


# ---------------------------------------------------------------------------
# Edit application
# ---------------------------------------------------------------------------


def _apply_single_edit_to_sequence(
    x_t: Tensor,  # (L,)
    origin_mask: Optional[Tensor],  # (L,) bool or None
    edit: EditCandidate,
    max_seq_len: int,
    pad_token: int,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Apply a single edit to *x_t*, returning (x_next, origin_mask_next)."""
    L = x_t.shape[0]
    device = x_t.device

    ins_mask = torch.zeros(1, L, dtype=torch.bool, device=device)
    del_mask = torch.zeros(1, L, dtype=torch.bool, device=device)
    ins_tokens = torch.full((1, L), pad_token, dtype=torch.long, device=device)

    if edit.op == "ins":
        ins_mask[0, edit.pos] = True
        ins_tokens[0, edit.pos] = edit.token
    elif edit.op == "del":
        del_mask[0, edit.pos] = True
    elif edit.op == "sub":
        ins_mask[0, edit.pos] = True
        del_mask[0, edit.pos] = True
        ins_tokens[0, edit.pos] = edit.token
    else:
        raise ValueError(f"Unknown op: {edit.op}")

    x_in = x_t.unsqueeze(0)  # (1, L)

    # Update origin mask via 3-state markers (reuse Euler logic).
    if origin_mask is not None:
        x_pad_mask = x_in == pad_token
        origin_markers = torch.where(
            x_pad_mask,
            torch.full_like(x_in, 2),
            origin_mask.unsqueeze(0).long(),
        )
        origin_markers[del_mask & ins_mask] = 0  # substitution
        origin_ins = torch.zeros_like(ins_tokens, dtype=torch.long)
        origin_markers = apply_ins_del_operations(
            origin_markers, ins_mask, del_mask, origin_ins,
            max_seq_len=max_seq_len, pad_token=2,
        )
        origin_next = (origin_markers == 1).squeeze(0)
    else:
        origin_next = None

    x_next = apply_ins_del_operations(
        x_in, ins_mask, del_mask, ins_tokens,
        max_seq_len=max_seq_len, pad_token=pad_token,
    ).squeeze(0)

    return x_next, origin_next


def _compute_u_tot(log_rates: Tensor) -> Tensor:
    """Total rate per sample: sum over positions of lambda_ins+lambda_sub+lambda_del.

    This is exact — does not require enumerating all O(L*V) candidates — because
    Σ_a Q(a|i) = 1 for both insertion and substitution distributions.
    """
    lambdas = torch.exp(log_rates)  # (B, L, 3)
    return lambdas.sum(dim=(1, 2))  # (B,)


# ---------------------------------------------------------------------------
# Beam dedup helpers
# ---------------------------------------------------------------------------


def _last_edit_key(edit: Optional[EditCandidate]) -> Optional[Tuple[int, str, Optional[int], Optional[int]]]:
    """Hashable summary of the reverse-op-relevant edit metadata."""
    if edit is None:
        return None
    return (edit.pos, edit.op, edit.token, edit.old_token)


def _beam_state_key(state: BeamState, pad_token: int) -> Tuple:
    """Hashable beam-state identity for dedup.

    The key must include every piece of metadata that can change future model
    outputs or legal action sets.  The result is cached on *state* to avoid
    repeated GPU→CPU transfers.
    """
    if state._key_cache is not None:
        return state._key_cache

    seq_key = tuple(int(tok) for tok in state.x_t if int(tok) != pad_token)
    origin_key = None
    if state.origin_mask is not None:
        origin_key = tuple(bool(v) for v in state.origin_mask.detach().cpu().tolist())
    policy_key = state.time_policy.state_key() if state.time_policy is not None else None
    key = (seq_key, origin_key, policy_key, _last_edit_key(state.last_edit), state.is_finished)
    state._key_cache = key
    return key


# ---------------------------------------------------------------------------
# Frozen-Hazard κ update
# ---------------------------------------------------------------------------


def _fh_kappa_next(kappa_cur: float, U: float) -> float:
    """Advance κ via the Frozen-Hazard first-event formula (stateless).

    κ' = κ + (1-κ) * (1/U - e^{-U} / (1 - e^{-U}))
    """
    U_safe = max(U, 1e-12)
    exp_neg_U = math.exp(-U_safe)
    if exp_neg_U >= 1.0 - 1e-12:
        delta = 0.5
    else:
        delta = 1.0 / U_safe - exp_neg_U / (1.0 - exp_neg_U)
    kappa_next = kappa_cur + (1.0 - kappa_cur) * delta
    return max(min(kappa_next, 1.0 - 1e-8), 1e-8)


def _update_fh_kappa(
    fh_kappas: List[float],
    b: int,
    U: float,
) -> None:
    """Advance per-sample kappa via the Frozen-Hazard first-event formula."""
    fh_kappas[b] = _fh_kappa_next(fh_kappas[b], U)


# ---------------------------------------------------------------------------
# Mode-specific helpers
# ---------------------------------------------------------------------------


def _init_kappa(
    scheduler: KappaScheduler,
    max_edits: int,
    device: torch.device,
    explicit_stop: bool,
) -> float:
    """Initial κ for a new hypothesis in explicit_stop mode."""
    if not explicit_stop:
        return 0.0
    depth_t0 = 1.0 / (max_edits + 1.0)
    t0 = torch.full((1,), depth_t0, device=device)
    return float(scheduler(t0).item())


def _advance_kappa(kappa_cur: float, U: float, explicit_stop: bool) -> float:
    """Advance κ after an edit.  No-op in non-explicit_stop modes."""
    if explicit_stop:
        return _fh_kappa_next(kappa_cur, U)
    return kappa_cur


def _check_hard_stop(U: float, stop_u_tot_base: float) -> bool:
    """Safety floor: force-stop when total edit mass is vanishingly small."""
    return stop_u_tot_base > 0 and U < stop_u_tot_base


def _score_actions_fh(
    U: float,
    log_U: float,
    edits: List[EditCandidate],
    stop_u_tot_base: float,
) -> List[ActionCandidate]:
    """Build ActionCandidates with Frozen-Hazard probabilities.

    STOP competes with edits using:
        p_stop = e^{-U}
        log_p_e = log(1 - e^{-U}) + log_u_e - log_U
    """
    actions: List[ActionCandidate] = []

    _p_stop = math.exp(-U)
    log_p_stop = math.log(max(_p_stop, 1e-30))
    actions.append(ActionCandidate(kind="stop", log_p=log_p_stop))

    if _p_stop < 1.0 - 1e-12:
        log_one_minus_p_stop = math.log1p(-_p_stop)
        for c in edits:
            log_p_e = log_one_minus_p_stop + c.log_u - log_U
            actions.append(ActionCandidate(kind="edit", log_p=log_p_e, edit=c))

    # Hard stop: if U below safety threshold, only STOP is valid
    if _check_hard_stop(U, stop_u_tot_base):
        actions = [a for a in actions if a.kind == "stop"]

    return actions


def _score_actions_threshold(
    edits: List[EditCandidate],
) -> List[ActionCandidate]:
    """Build ActionCandidates for non-FH threshold-stop mode.

    No STOP action — stopping is handled externally via TimePolicy or threshold.
    Edit scores use the raw score field (log_u - log_U).
    """
    return [ActionCandidate(kind="edit", log_p=c.score, edit=c) for c in edits]


def _kappa_list_to_batch(kappa_values: List[float], device: torch.device) -> Tensor:
    """Convert per-sample κ values to (N, 1) tensor on *device*."""
    return torch.tensor(kappa_values, device=device, dtype=torch.float).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Shared model-forward helper (GPU → CPU)
# ---------------------------------------------------------------------------


def _model_forward_step(
    model,
    x_batch: Tensor,              # (N, L) on GPU
    kappa_batch: Tensor,          # (N, 1) on GPU
    scheduler: KappaScheduler,
    pad_token: int,
    forbidden_mask: Tensor,       # (V,) on GPU
    V: int,
    use_rate_reparam: bool = False,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
    time_input: str = "t",
    train_scheduler: Optional[KappaScheduler] = None,
    origin_batch: Optional[Tensor] = None,  # (N, L) or None
    k_ins_token: int = 4,
    k_sub_token: int = 4,
    k_edit_expand: int = 16,
) -> Tuple[List[List[EditCandidate]], Tensor, Tensor]:
    """Single GPU forward pass → per-sample edit candidates + U.

    Encapsulates κ→t, model forward, log_u_edit construction, GPU top-K,
    and U computation.  Candidate selection stays on GPU; only the final
    ≤ *k_edit_expand* candidates per sample are transferred to CPU.

    Returns:
        candidates:  list of per-sample candidate lists (CPU)
        U:           (N,) total executable edit mass (GPU)
        log_U:       (N,) log of U (GPU)
    """
    t_batch = scheduler.inverse(kappa_batch)
    t_model = _compute_model_time(t_batch, scheduler, time_input, train_scheduler)

    x_pad_mask = x_batch == pad_token
    log_rates, log_ins_probs, log_sub_probs = model(
        x_batch, t_model, x_pad_mask, origin_mask=origin_batch,
    )

    log_rates_score = _prepare_log_rates_for_scoring(
        log_rates, t_batch, scheduler,
        use_rate_reparam=use_rate_reparam,
        clamp_kappa=clamp_kappa, clamp_max=clamp_max,
        train_scheduler=train_scheduler, t_model=t_model,
    )
    log_u_edit = _build_log_u_edit(
        log_rates_score, log_ins_probs, log_sub_probs, x_batch,
        pad_token=pad_token, forbidden_mask=forbidden_mask,
    )
    U = _compute_U(log_u_edit)
    log_U = torch.log(U.clamp_min(1e-12))

    candidates = _select_top_edits_batch(
        log_u_edit, V, x_batch, log_U,
        k_ins_token, k_sub_token, k_edit_expand,
    )

    return candidates, U, log_U


# ---------------------------------------------------------------------------
# Public API: greedy single-edit sampling
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_greedy_single_edit(
    model,
    x_0: Tensor,  # (B, L_0) — includes BOS prefix, PAD-padded
    scheduler: KappaScheduler,
    time_policy: TimePolicy,
    max_edits: int = 20,
    max_seq_len: int = 256,
    pad_token: int = PAD_TOKEN,
    bos_token: int = BOS_TOKEN,
    use_rate_reparam: bool = False,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
    time_input: str = "t",
    train_scheduler: Optional[KappaScheduler] = None,
    use_origin_mask: bool = False,
    k_ins_token: int = 4,
    k_sub_token: int = 4,
    k_edit_expand: int = 16,
    stop_u_tot_base: float = -1.0,
    verbose: bool = False,
    explicit_stop: bool = False,
) -> Tensor:
    """Greedy single-edit sampling: at each step apply the highest-scoring edit.

    All samples in the batch advance together; finished samples are excluded
    from further editing.  The model forward is batched across active samples
    for GPU efficiency.

    When *explicit_stop* is True, STOP is treated as an explicit candidate
    action alongside edits, scored via the Frozen-Hazard first-event
    approximation::

        p_stop = e^{-U}
        p_e    = (1 - e^{-U}) * u_e / U
        κ'     = κ + (1-κ) * (1/U - e^{-U} / (1 - e^{-U}))

    Otherwise the legacy score-based selection + external threshold is used.

    Args:
        model: EditFlowsTransformer (or compatible .forward).
        x_0: initial sequences (B, L_0) with BOS prefix.
        scheduler: kappa scheduler for rate scaling and time mapping.
        time_policy: TimePolicy instance for κ scheduling (only used when
            ``explicit_stop=False``).
        max_edits: maximum edit steps per sample.
        max_seq_len: hard ceiling on sequence length.
        use_rate_reparam: if True, model outputs base rates v'; real rates
            v = k(t) * v' are recovered before scoring/candidate construction.
        k_ins_token: top-k tokens per position for insertion candidates.
        k_sub_token: top-k tokens per position for substitution candidates.
        k_edit_expand: global top-k edit candidates considered per step.
        stop_u_tot_base: safety threshold on total edit mass.  When > 0
            and ``explicit_stop=True``, serves as a hard floor (recommend
            0.001).  When ``explicit_stop=False``, is the primary stop
            mechanism (recommend 0.05).
        explicit_stop: if True, use Frozen-Hazard explicit STOP framework.
            Default False for backward compatibility.

    Returns:
        x_final: (B, L_out) final sequences, PAD-padded to equal length.
    """
    B, L_0 = x_0.shape
    device = next(model.parameters()).device
    x_t = x_0.to(device)

    if use_origin_mask:
        origin_mask = torch.ones_like(x_t, dtype=torch.bool, device=device)
    else:
        origin_mask = None

    active = torch.ones(B, dtype=torch.bool, device=device)
    last_edits: List[Optional[EditCandidate]] = [None] * B

    vocab_size = model.token_embedding.weight.shape[0]
    forbidden_mask = _build_forbidden_mask(vocab_size, device)
    time_policy.reset(B, device, max_edits)

    # ---- explicit-stop state ----
    fh_kappas: List[float] = []
    if explicit_stop:
        init_kappa = _init_kappa(scheduler, max_edits, device, explicit_stop=True)
        fh_kappas = [init_kappa] * B

    for step in range(max_edits):
        if not active.any():
            break

        # ---- 1. resolve κ ----
        if explicit_stop:
            kappa_vals = [fh_kappas[b] if active[b] else 0.5 for b in range(B)]
            kappa_batch = _kappa_list_to_batch(kappa_vals, device)
        else:
            kappa_batch = time_policy.get_kappa(step)  # (B, 1)

        # ---- 2. GPU forward → per-sample candidates + U ----
        all_candidates, U, log_U = _model_forward_step(
            model, x_t, kappa_batch, scheduler, pad_token, forbidden_mask,
            V=vocab_size,
            use_rate_reparam=use_rate_reparam,
            clamp_kappa=clamp_kappa, clamp_max=clamp_max,
            time_input=time_input, train_scheduler=train_scheduler,
            origin_batch=origin_mask,
            k_ins_token=k_ins_token, k_sub_token=k_sub_token,
            k_edit_expand=k_edit_expand,
        )  # candidates: list[list[EditCandidate]] len=B; U, log_U on GPU

        # ---- 3. TimePolicy feedback (non-explicit_stop only) ----
        policy_stop_list: List[bool] = []
        if not explicit_stop:
            policy_stop = time_policy.update(kappa_batch.squeeze(-1), U)
            policy_stop_list = policy_stop.cpu().tolist()

        # ---- 4. per-sample greedy selection ----
        ins_mask = torch.zeros(B, x_t.shape[1], dtype=torch.bool, device=device)
        del_mask = torch.zeros(B, x_t.shape[1], dtype=torch.bool, device=device)
        ins_tokens = torch.full((B, x_t.shape[1]), pad_token, dtype=torch.long, device=device)

        for b in range(B):
            if not active[b]:
                continue

            U_b = U[b].item()
            log_U_b = log_U[b].item()

            edits = all_candidates[b]

            if not edits:
                active[b] = False
                continue

            # Filter reverse ops; fall back to unfiltered if all filtered.
            valid = [c for c in edits if not _is_reverse_op(c, last_edits[b])]
            if not valid:
                valid = edits

            # ---- score actions (mode-specific) ----
            if explicit_stop:
                actions = _score_actions_fh(U_b, log_U_b, valid, stop_u_tot_base)
            else:
                actions = _score_actions_threshold(valid)

            best_action = max(actions, key=lambda a: a.log_p)

            if best_action.kind == "stop":
                active[b] = False
                continue

            best = best_action.edit
            last_edits[b] = best

            # ---- advance κ (mode-specific) ----
            if explicit_stop:
                _update_fh_kappa(fh_kappas, b, U_b)
                if _check_hard_stop(U_b, stop_u_tot_base):
                    active[b] = False
                    continue

            # ---- non-explicit_stop: external stop check ----
            if not explicit_stop:
                if (b < len(policy_stop_list) and policy_stop_list[b]) or \
                   _check_hard_stop(U_b, stop_u_tot_base):
                    active[b] = False
                    continue

            # Record edit for batch application
            if best.op == "ins":
                ins_mask[b, best.pos] = True
                ins_tokens[b, best.pos] = best.token
            elif best.op == "del":
                del_mask[b, best.pos] = True
            elif best.op == "sub":
                ins_mask[b, best.pos] = True
                del_mask[b, best.pos] = True
                ins_tokens[b, best.pos] = best.token

        if not active.any():
            break

        # ---- 7. batch apply edits on GPU ----
        if use_origin_mask:
            x_pad_mask_curr = x_t == pad_token
            origin_markers = torch.where(
                x_pad_mask_curr,
                torch.full_like(x_t, 2),
                origin_mask.long(),
            )
            origin_markers[del_mask & ins_mask] = 0
            origin_ins = torch.zeros_like(ins_tokens, dtype=torch.long)
            origin_markers = apply_ins_del_operations(
                origin_markers, ins_mask, del_mask, origin_ins,
                max_seq_len=max_seq_len, pad_token=2,
            )
            origin_mask = origin_markers == 1

        x_t = apply_ins_del_operations(
            x_t, ins_mask, del_mask, ins_tokens,
            max_seq_len=max_seq_len, pad_token=pad_token,
        )

        if verbose and step % 5 == 0:
            n_active = active.sum().item()
            avg_len = (~(x_t == pad_token)).sum(dim=1).float().mean().item()
            print(f"  greedy step {step:>3}: active={n_active}/{B}, avg_len={avg_len:.1f}")

    return x_t


# ---------------------------------------------------------------------------
# Public API: beam single-edit sampling
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_beam_single_edit(
    model,
    x_0: Tensor,  # (B, L_0)
    scheduler: KappaScheduler,
    time_policy: TimePolicy,
    beam_size: int = 5,
    max_edits: int = 20,
    max_seq_len: int = 256,
    pad_token: int = PAD_TOKEN,
    bos_token: int = BOS_TOKEN,
    use_rate_reparam: bool = False,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
    time_input: str = "t",
    train_scheduler: Optional[KappaScheduler] = None,
    use_origin_mask: bool = False,
    k_ins_token: int = 4,
    k_sub_token: int = 4,
    k_edit_expand: int = 16,
    stop_u_tot_base: float = -1.0,
    verbose: bool = False,
    explicit_stop: bool = False,
) -> Tensor:
    """Beam-search single-edit sampling.

    Each sample maintains up to *beam_size* active hypotheses.  At every step
    all active states across all samples are flattened into one batch for a
    single model forward pass.  Candidates are then scored, expanded, deduped,
    and pruned per sample independently.

    When *explicit_stop* is True, STOP is treated as an explicit child state
    alongside edit children, scored via the Frozen-Hazard first-event
    approximation::

        p_stop = e^{-U}
        p_e    = (1 - e^{-U}) * u_e / U
        κ'     = κ + (1-κ) * (1/U - e^{-U} / (1 - e^{-U}))

    Each hypothesis maintains its own κ, updated independently.
    Finished states (STOP children) are collected in a separate pool that
    does not occupy active beam slots, matching standard beam-search EOS
    handling.

    Returns:
        x_final: (B, L_out) top-1 sequence per sample, PAD-padded.
    """
    B = x_0.shape[0]
    device = next(model.parameters()).device
    x_0 = x_0.to(device)

    vocab_size = model.token_embedding.weight.shape[0]
    forbidden_mask = _build_forbidden_mask(vocab_size, device)

    # Initialise per-sample beam lists.
    all_beams: List[List[BeamState]] = []
    for b in range(B):
        x_init = x_0[b].clone()
        origin_init = (
            torch.ones_like(x_init, dtype=torch.bool, device=device)
            if use_origin_mask else None
        )
        state_policy = time_policy.clone()
        state_policy.reset(1, device, max_edits)
        init_kappa = _init_kappa(scheduler, max_edits, device, explicit_stop)
        all_beams.append([BeamState(
            x_t=x_init,
            origin_mask=origin_init,
            log_p=0.0,
            time_policy=state_policy,
            kappa=init_kappa,
        )])

    if explicit_stop:
        finished_pool: Dict[int, List[BeamState]] = {b: [] for b in range(B)}
    else:
        finished_pool = None

    for step in range(max_edits):
        # ---- 0. separate finished and active states ----
        finished_states: List[Tuple[int, BeamState]] = []
        active_flat: List[BeamState] = []
        active_sample_of: List[int] = []

        for b, beams in enumerate(all_beams):
            for state in beams:
                if state.is_finished:
                    finished_states.append((b, state))
                else:
                    active_flat.append(state)
                    active_sample_of.append(b)

        # Carry finished states: explicit_stop → separate pool (no beam-slot
        # occupation); otherwise → compete with active children in shared pool.
        sample_candidates: Dict[int, List[BeamState]] = {b: [] for b in range(B)}
        if explicit_stop:
            for b, state in finished_states:
                finished_pool[b].append(state)
        else:
            for b, state in finished_states:
                sample_candidates[b].append(state)

        if not active_flat:
            break

        N = len(active_flat)
        max_len = max(int(s.x_t.shape[0]) for s in active_flat)

        # ---- 1. build padded batch from active states ----
        x_batch = torch.full((N, max_len), pad_token, dtype=torch.long, device=device)
        origin_batch: Optional[Tensor] = (
            torch.zeros(N, max_len, dtype=torch.bool, device=device)
            if use_origin_mask else None
        )
        for i, state in enumerate(active_flat):
            L = state.x_t.shape[0]
            x_batch[i, :L] = state.x_t
            if origin_batch is not None and state.origin_mask is not None:
                origin_batch[i, :L] = state.origin_mask

        # ---- 2. resolve κ ----
        if explicit_stop:
            kappa_vals = [s.kappa for s in active_flat]
        else:
            kappa_vals = [s.time_policy.get_kappa(step).squeeze(-1).item() for s in active_flat]
        kappa_batch = _kappa_list_to_batch(kappa_vals, device)

        # ---- 3. GPU forward → per-sample candidates + U ----
        all_candidates_list, U, log_U = _model_forward_step(
            model, x_batch, kappa_batch, scheduler, pad_token, forbidden_mask,
            V=vocab_size,
            use_rate_reparam=use_rate_reparam,
            clamp_kappa=clamp_kappa, clamp_max=clamp_max,
            time_input=time_input, train_scheduler=train_scheduler,
            origin_batch=origin_batch if use_origin_mask else None,
            k_ins_token=k_ins_token, k_sub_token=k_sub_token,
            k_edit_expand=k_edit_expand,
        )  # candidates: list[list[EditCandidate]] len=N; U, log_U on GPU

        # ---- 4. TimePolicy feedback (non-explicit_stop only) ----
        policy_stop_list: List[bool] = []
        if not explicit_stop:
            for i, state in enumerate(active_flat):
                stop = state.time_policy.update(
                    kappa_batch[i:i + 1].squeeze(-1),
                    U[i:i + 1],
                )
                policy_stop_list.append(bool(stop.item()))

        # ---- 5. expand each active state ----
        for i, state in enumerate(active_flat):
            b = active_sample_of[i]
            U_val = U[i].item()
            log_U_val = log_U[i].item()

            edits = all_candidates_list[i]

            valid_edits = [c for c in edits if not _is_reverse_op(c, state.last_edit)]

            # ---- score actions (mode-specific) ----
            if explicit_stop:
                actions = _score_actions_fh(U_val, log_U_val, valid_edits, stop_u_tot_base)

                # STOP child → finished pool (does not occupy active beam slots)
                # p_stop is always the first action
                p_stop_log_p = actions[0].log_p  # first is always STOP in FH mode
                stop_child = BeamState(
                    x_t=state.x_t.clone(),
                    origin_mask=state.origin_mask.clone() if state.origin_mask is not None else None,
                    log_p=state.log_p + p_stop_log_p,
                    last_edit=state.last_edit,
                    is_finished=True,
                    time_policy=state.time_policy.clone() if state.time_policy is not None else None,
                    kappa=state.kappa,
                    _U=U_val,
                )
                finished_pool[b].append(stop_child)

                # Hard stop: skip edit children
                if _check_hard_stop(U_val, stop_u_tot_base):
                    continue

                # Edit children (only when STOP is not certain)
                _p_stop = math.exp(-U_val)
                if _p_stop >= 1.0 - 1e-12:
                    continue

                if not valid_edits:
                    continue

                log_one_minus_p_stop = math.log1p(-_p_stop)
                for cand in valid_edits:
                    log_p_e = log_one_minus_p_stop + cand.log_u - log_U_val
                    new_x_t, new_origin = _apply_single_edit_to_sequence(
                        state.x_t, state.origin_mask, cand, max_seq_len, pad_token,
                    )
                    new_kappa = _advance_kappa(state.kappa, U_val, explicit_stop)
                    new_state = BeamState(
                        x_t=new_x_t,
                        origin_mask=new_origin,
                        log_p=state.log_p + log_p_e,
                        last_edit=cand,
                        is_finished=False,
                        time_policy=state.time_policy.clone() if state.time_policy is not None else None,
                        kappa=new_kappa,
                        _U=U_val,
                    )
                    sample_candidates[b].append(new_state)

            else:
                # ---- threshold-stop branch ----
                # Stop check BEFORE expansion
                if (i < len(policy_stop_list) and policy_stop_list[i]) or \
                   _check_hard_stop(U_val, stop_u_tot_base):
                    state.is_finished = True
                    sample_candidates[b].append(state)
                    continue

                if not valid_edits:
                    state.is_finished = True
                    sample_candidates[b].append(state)
                    continue

                for cand in valid_edits:
                    new_x_t, new_origin = _apply_single_edit_to_sequence(
                        state.x_t, state.origin_mask, cand, max_seq_len, pad_token,
                    )
                    new_state = BeamState(
                        x_t=new_x_t,
                        origin_mask=new_origin,
                        log_p=state.log_p + cand.score,
                        last_edit=cand,
                        is_finished=False,
                        time_policy=state.time_policy.clone(),
                        kappa=state.kappa,
                    )
                    sample_candidates[b].append(new_state)

        # ---- 8. per-sample dedup + top-K ----
        new_beams: List[List[BeamState]] = []
        for b in range(B):
            states = sample_candidates[b]
            seen: Dict[Tuple, BeamState] = {}
            for st in states:
                key = _beam_state_key(st, pad_token)
                if key not in seen or st.log_p > seen[key].log_p:
                    seen[key] = st
            unique = list(seen.values())
            unique.sort(key=lambda s: s.log_p, reverse=True)
            new_beams.append(unique[:beam_size])

        all_beams = new_beams

        if verbose and step % 5 == 0:
            total_states = sum(len(beams) for beams in all_beams)
            n_finished = sum(1 for beams in all_beams for s in beams if s.is_finished)
            parts = [f"beam step {step:>3}: total_states={total_states}, finished={n_finished}"]
            if explicit_stop and finished_pool is not None:
                total_finished = sum(len(v) for v in finished_pool.values())
                parts.append(f", finished_pool={total_finished}")
            print("".join(parts))

    # ---- collect top-1 per sample ----
    results: List[Tensor] = []
    for b in range(B):
        all_candidates: List[Tuple[BeamState, float]] = []

        if explicit_stop and finished_pool is not None:
            for s in finished_pool.get(b, []):
                all_candidates.append((s, s.log_p))

        for s in (all_beams[b] if all_beams[b] else []):
            score = s.log_p
            if explicit_stop and not s.is_finished and s._U > 0:
                score += math.log(max(math.exp(-s._U), 1e-30))
            all_candidates.append((s, score))

        if all_candidates:
            best_state, _ = max(all_candidates, key=lambda x: x[1])
        else:
            best_state = None
        results.append(best_state.x_t if best_state else x_0[b])

    # Pad to uniform length.
    out_len = max(r.shape[0] for r in results)
    out = torch.full((B, out_len), pad_token, dtype=torch.long, device=device)
    for b, r in enumerate(results):
        L = r.shape[0]
        out[b, :L] = r

    return out