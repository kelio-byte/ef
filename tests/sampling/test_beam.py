"""Tests for edit_flows/sampling/beam.py — greedy/beam single-edit sampling."""

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from edit_flows.sampling.beam import (
    EditCandidate,
    BeamState,
    _is_reverse_op,
    _build_forbidden_mask,
    _collect_edit_candidates_single,
    _apply_single_edit_to_sequence,
    _compute_executable_u_tot,
    _compute_u_tot,
    _prepare_log_rates_for_scoring,
    sample_greedy_single_edit,
    sample_beam_single_edit,
)
from edit_flows.sampling.time_policy import FixedTimePolicy, TimePolicy
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.core.rate_scale import get_rate_scale
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN, GAP_TOKEN, UNK_TOKEN

V = 16  # small test vocab
LOG_NEG_INF = -1e9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_log_rates(L: int, ins_vals=None, sub_vals=None, del_vals=None) -> Tensor:
    """Build (L, 3) log-rates tensor.  Default: all channels near zero."""
    r = torch.full((L, 3), -1.0)
    if ins_vals is not None:
        for pos, val in ins_vals:
            r[pos, 0] = val
    if sub_vals is not None:
        for pos, val in sub_vals:
            r[pos, 1] = val
    if del_vals is not None:
        for pos, val in del_vals:
            r[pos, 2] = val
    return r


def _make_log_probs(L: int, V: int, pos_token_pairs=None) -> Tensor:
    """Build (L, V) log-probs.  Default: uniform (all log(1/V))."""
    lp = torch.full((L, V), LOG_NEG_INF)
    if pos_token_pairs is not None:
        for pos, tok, val in pos_token_pairs:
            lp[pos, tok] = val
    return lp


class _ControlledModel(nn.Module):
    """Model that returns pre-set outputs for beam/greedy integration tests."""

    def __init__(self, vocab_size: int = V):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, 64)
        self._log_rates: Tensor | None = None
        self._log_ins_probs: Tensor | None = None
        self._log_sub_probs: Tensor | None = None

    def set_outputs(self, log_rates: Tensor, log_ins_probs: Tensor,
                    log_sub_probs: Tensor) -> None:
        self._log_rates = log_rates
        self._log_ins_probs = log_ins_probs
        self._log_sub_probs = log_sub_probs

    def forward(self, tokens, time_step, padding_mask, origin_mask=None):
        B = tokens.shape[0]
        # Expand pre-set (L,) outputs to (B, L, ...) by repeating.
        lr = self._log_rates.unsqueeze(0).expand(B, -1, -1).to(tokens.device)
        lip = self._log_ins_probs.unsqueeze(0).expand(B, -1, -1).to(tokens.device)
        lsp = self._log_sub_probs.unsqueeze(0).expand(B, -1, -1).to(tokens.device)
        # Apply PAD masking for realism.
        pad_3d = padding_mask.unsqueeze(-1)
        lr = lr.masked_fill(pad_3d, LOG_NEG_INF)
        lip = lip.masked_fill(pad_3d, LOG_NEG_INF)
        lsp = lsp.masked_fill(pad_3d, LOG_NEG_INF)
        return lr, lip, lsp


class _BranchingTimePolicy(TimePolicy):
    """Test policy whose next kappa depends on the state's u_tot."""

    def reset(self, batch_size: int, device: torch.device, max_edits: int) -> None:
        self._device = device
        self._batch_size = batch_size
        self._next = torch.full((batch_size, 1), 0.5, device=device)

    def get_kappa(self, step: int) -> Tensor:
        return self._next.clone()

    def update(self, kappa: Tensor, u_tot_base: Tensor) -> Tensor:
        self._next = torch.where(
            u_tot_base.unsqueeze(-1) < 0.5,
            torch.full((self._batch_size, 1), 0.2, device=self._device),
            torch.full((self._batch_size, 1), 0.8, device=self._device),
        )
        return torch.zeros(self._batch_size, dtype=torch.bool, device=self._device)

    def state_key(self) -> tuple:
        return ("branching", tuple(float(v) for v in self._next.view(-1).cpu().tolist()))


# ---------------------------------------------------------------------------
# _is_reverse_op
# ---------------------------------------------------------------------------


class TestIsReverseOp:
    def test_none_last_edit_always_false(self):
        cand = EditCandidate(pos=2, op="sub", token=5, log_u_real=0.0, score=0.0)
        assert not _is_reverse_op(cand, None)

    def test_sub_then_different_sub_allowed(self):
        """a→b then b→c should NOT be reverse."""
        last = EditCandidate(pos=3, op="sub", token=7, log_u_real=1.0, score=0.0,
                             old_token=5)
        cand = EditCandidate(pos=3, op="sub", token=9, log_u_real=1.0, score=0.0,
                             old_token=7)
        assert not _is_reverse_op(cand, last)

    def test_sub_then_reverse_sub_blocked(self):
        """a→b then b→a IS reverse."""
        last = EditCandidate(pos=3, op="sub", token=7, log_u_real=1.0, score=0.0,
                             old_token=5)
        cand = EditCandidate(pos=3, op="sub", token=5, log_u_real=1.0, score=0.0,
                             old_token=7)
        assert _is_reverse_op(cand, last)

    def test_sub_at_different_position_allowed(self):
        last = EditCandidate(pos=3, op="sub", token=7, log_u_real=1.0, score=0.0,
                             old_token=5)
        cand = EditCandidate(pos=4, op="sub", token=5, log_u_real=1.0, score=0.0,
                             old_token=7)
        assert not _is_reverse_op(cand, last)

    def test_ins_then_del_same_pos_reverse(self):
        last = EditCandidate(pos=2, op="ins", token=8, log_u_real=1.0, score=0.0)
        cand = EditCandidate(pos=2, op="del", token=None, log_u_real=1.0, score=0.0)
        assert _is_reverse_op(cand, last)

    def test_ins_then_del_different_pos_allowed(self):
        last = EditCandidate(pos=2, op="ins", token=8, log_u_real=1.0, score=0.0)
        cand = EditCandidate(pos=3, op="del", token=None, log_u_real=1.0, score=0.0)
        assert not _is_reverse_op(cand, last)

    def test_del_then_ins_same_pos_same_token_reverse(self):
        last = EditCandidate(pos=2, op="del", token=8, log_u_real=1.0, score=0.0)
        cand = EditCandidate(pos=2, op="ins", token=8, log_u_real=1.0, score=0.0)
        assert _is_reverse_op(cand, last)

    def test_del_then_ins_same_pos_different_token_allowed(self):
        last = EditCandidate(pos=2, op="del", token=8, log_u_real=1.0, score=0.0)
        cand = EditCandidate(pos=2, op="ins", token=9, log_u_real=1.0, score=0.0)
        assert not _is_reverse_op(cand, last)


# ---------------------------------------------------------------------------
# _collect_edit_candidates_single
# ---------------------------------------------------------------------------


class TestCollectCandidatesBOS:
    """2.3: BOS position should not have sub/del candidates."""

    def test_no_sub_on_bos(self):
        x_t = torch.tensor([BOS_TOKEN, 5, 6, 7])
        L = len(x_t)
        log_rates = _make_log_rates(L, sub_vals=[(0, 5.0), (1, 0.0)])
        log_ins = _make_log_probs(L, V)
        log_sub = _make_log_probs(L, V, [(0, 10, 0.0), (1, 10, 0.0)])
        non_pad = torch.ones(L, dtype=torch.bool)
        fmask = _build_forbidden_mask(V, torch.device("cpu"))

        cands, _ = _collect_edit_candidates_single(
            log_rates, log_ins, log_sub, x_t, non_pad, 10.0,
            k_ins_token=2, k_sub_token=2, k_edit_expand=32,
            forbidden_mask=fmask,
        )
        for c in cands:
            if c.op == "sub":
                assert c.pos != 0, f"BOS sub should be filtered, got {c}"

    def test_no_del_on_bos(self):
        x_t = torch.tensor([BOS_TOKEN, 5, 6, 7])
        L = len(x_t)
        log_rates = _make_log_rates(L, del_vals=[(0, 5.0), (1, 0.0)])
        log_ins = _make_log_probs(L, V)
        log_sub = _make_log_probs(L, V)
        non_pad = torch.ones(L, dtype=torch.bool)
        fmask = _build_forbidden_mask(V, torch.device("cpu"))

        cands, _ = _collect_edit_candidates_single(
            log_rates, log_ins, log_sub, x_t, non_pad, 10.0,
            k_ins_token=2, k_sub_token=2, k_edit_expand=32,
            forbidden_mask=fmask,
        )
        for c in cands:
            if c.op == "del":
                assert c.pos != 0, f"BOS del should be filtered, got {c}"

    def test_ins_on_bos_allowed(self):
        """ins(pos=0) means 'insert after BOS' — should be allowed."""
        x_t = torch.tensor([BOS_TOKEN, 5, 6, 7])
        L = len(x_t)
        log_rates = _make_log_rates(L, ins_vals=[(0, 5.0)])
        log_ins = _make_log_probs(L, V, [(0, 10, 0.0)])
        log_sub = _make_log_probs(L, V)
        non_pad = torch.ones(L, dtype=torch.bool)
        fmask = _build_forbidden_mask(V, torch.device("cpu"))

        cands, _ = _collect_edit_candidates_single(
            log_rates, log_ins, log_sub, x_t, non_pad, 10.0,
            k_ins_token=2, k_sub_token=2, k_edit_expand=32,
            forbidden_mask=fmask,
        )
        ins_at_bos = [c for c in cands if c.op == "ins" and c.pos == 0]
        assert len(ins_at_bos) > 0, "ins(pos=0) should be allowed"


class TestCollectCandidatesNoopSub:
    """2.4: no-op substitution (token == current) should be filtered."""

    def test_noop_sub_filtered(self):
        x_t = torch.tensor([BOS_TOKEN, 5, 6, 7])
        L = len(x_t)
        # Position 1: high sub rate, token 5 (== current) gets highest prob.
        log_rates = _make_log_rates(L, sub_vals=[(1, 10.0)])
        log_ins = _make_log_probs(L, V)
        log_sub = _make_log_probs(L, V, [
            (1, 5, 0.0),   # no-op: same as current → should be filtered
            (1, 10, -0.1),  # legitimate candidate
        ])
        non_pad = torch.ones(L, dtype=torch.bool)
        fmask = _build_forbidden_mask(V, torch.device("cpu"))

        cands, _ = _collect_edit_candidates_single(
            log_rates, log_ins, log_sub, x_t, non_pad, 10.0,
            k_ins_token=2, k_sub_token=4, k_edit_expand=32,
            forbidden_mask=fmask,
        )
        sub_at_1 = [c for c in cands if c.op == "sub" and c.pos == 1]
        for c in sub_at_1:
            assert c.token != 5, f"no-op sub(pos=1, token=5) should be filtered, got {c}"

    def test_noop_sub_at_different_position_allowed(self):
        """sub at pos=2 with token that equals pos=1's token is fine."""
        x_t = torch.tensor([BOS_TOKEN, 5, 6, 7])
        L = len(x_t)
        log_rates = _make_log_rates(L, sub_vals=[(2, 10.0)])
        log_ins = _make_log_probs(L, V)
        # pos=2 current token is 6, so sub(pos=2, token=5) is NOT no-op.
        log_sub = _make_log_probs(L, V, [(2, 5, 0.0)])
        non_pad = torch.ones(L, dtype=torch.bool)
        fmask = _build_forbidden_mask(V, torch.device("cpu"))

        cands, _ = _collect_edit_candidates_single(
            log_rates, log_ins, log_sub, x_t, non_pad, 10.0,
            k_ins_token=2, k_sub_token=4, k_edit_expand=32,
            forbidden_mask=fmask,
        )
        sub_at_2 = [c for c in cands if c.op == "sub" and c.pos == 2]
        assert any(c.token == 5 for c in sub_at_2), \
            "sub(pos=2, token=5) is not no-op (current is 6), should be allowed"


class TestCollectCandidatesOldToken:
    """Verify old_token is populated for sub candidates."""

    def test_sub_candidate_has_old_token(self):
        x_t = torch.tensor([BOS_TOKEN, 5, 6, 7])
        L = len(x_t)
        log_rates = _make_log_rates(L, sub_vals=[(1, 5.0)])
        log_ins = _make_log_probs(L, V)
        log_sub = _make_log_probs(L, V, [(1, 10, 0.0)])
        non_pad = torch.ones(L, dtype=torch.bool)
        fmask = _build_forbidden_mask(V, torch.device("cpu"))

        cands, _ = _collect_edit_candidates_single(
            log_rates, log_ins, log_sub, x_t, non_pad, 10.0,
            k_ins_token=2, k_sub_token=2, k_edit_expand=32,
            forbidden_mask=fmask,
        )
        for c in cands:
            if c.op == "sub":
                assert c.old_token is not None, f"sub candidate must have old_token"
                assert c.old_token == x_t[c.pos].item(), \
                    f"old_token={c.old_token}, expected {x_t[c.pos].item()}"
            else:
                assert c.old_token is None, \
                    f"{c.op} candidate should have old_token=None"


# ---------------------------------------------------------------------------
# _apply_single_edit_to_sequence
# ---------------------------------------------------------------------------


class TestApplySingleEdit:
    def test_sub_edit(self):
        x_t = torch.tensor([BOS_TOKEN, 5, 6, 7])
        origin = torch.ones_like(x_t, dtype=torch.bool)
        cand = EditCandidate(pos=1, op="sub", token=10, log_u_real=1.0, score=0.0)

        x_next, origin_next = _apply_single_edit_to_sequence(
            x_t, origin, cand, max_seq_len=32, pad_token=PAD_TOKEN,
        )
        assert x_next[0].item() == BOS_TOKEN
        assert x_next[1].item() == 10  # substituted
        assert x_next[2].item() == 6
        assert origin_next is not None
        assert origin_next[0].item() is True   # BOS unchanged
        assert origin_next[1].item() is False  # substituted position
        assert origin_next[2].item() is True

    def test_ins_edit(self):
        x_t = torch.tensor([BOS_TOKEN, 5, 6, 7])
        origin = torch.ones_like(x_t, dtype=torch.bool)
        cand = EditCandidate(pos=1, op="ins", token=10, log_u_real=1.0, score=0.0)

        x_next, origin_next = _apply_single_edit_to_sequence(
            x_t, origin, cand, max_seq_len=32, pad_token=PAD_TOKEN,
        )
        assert x_next[0].item() == BOS_TOKEN
        assert x_next[1].item() == 5
        assert x_next[2].item() == 10  # inserted
        assert origin_next is not None
        assert origin_next[0].item() is True
        assert origin_next[1].item() is True
        assert origin_next[2].item() is False  # newly inserted

    def test_del_edit(self):
        x_t = torch.tensor([BOS_TOKEN, 5, 6, 7])
        origin = torch.ones_like(x_t, dtype=torch.bool)
        cand = EditCandidate(pos=1, op="del", token=None, log_u_real=1.0, score=0.0)

        x_next, origin_next = _apply_single_edit_to_sequence(
            x_t, origin, cand, max_seq_len=32, pad_token=PAD_TOKEN,
        )
        assert x_next[0].item() == BOS_TOKEN
        assert x_next[1].item() == 6  # 5 deleted, 6 shifted left
        assert origin_next is not None
        assert origin_next[0].item() is True

    def test_no_origin_mask(self):
        x_t = torch.tensor([BOS_TOKEN, 5, 6])
        cand = EditCandidate(pos=1, op="sub", token=10, log_u_real=1.0, score=0.0)
        x_next, origin_next = _apply_single_edit_to_sequence(
            x_t, None, cand, max_seq_len=32, pad_token=PAD_TOKEN,
        )
        assert origin_next is None
        assert x_next[1].item() == 10


# ---------------------------------------------------------------------------
# _compute_u_tot
# ---------------------------------------------------------------------------


class TestComputeUTot:
    def test_sums_all_channels(self):
        log_rates = torch.tensor([[
            [0.0, 0.0, 0.0],  # each exp(0) = 1
            [0.0, 0.0, 0.0],
        ]])  # (1, 2, 3)
        u_tot = _compute_u_tot(log_rates)
        assert u_tot.shape == (1,)
        assert abs(u_tot.item() - 6.0) < 0.01  # 2 pos × 3 channels × exp(0)


class TestComputeExecutableUTot:
    def test_excludes_forbidden_noop_and_bos_mass(self):
        x_t = torch.tensor([[BOS_TOKEN, 5, 6, PAD_TOKEN]])
        B, L = x_t.shape
        log_rates = torch.full((B, L, 3), LOG_NEG_INF)
        log_rates[0, 0, 0] = 0.0   # ins at BOS: valid
        log_rates[0, 0, 1] = 0.0   # sub at BOS: invalid
        log_rates[0, 0, 2] = 0.0   # del at BOS: invalid
        log_rates[0, 1, 1] = 0.0   # sub at pos 1: only noop token mass
        log_rates[0, 1, 2] = 0.0   # del at pos 1: valid
        log_rates[0, 3, 0] = 0.0   # ins on PAD: invalid

        log_ins = torch.full((B, L, V), LOG_NEG_INF)
        log_sub = torch.full((B, L, V), LOG_NEG_INF)

        log_ins[0, 0, 10] = 0.0         # valid ins mass at BOS
        log_ins[0, 3, 10] = 0.0         # would be valid token, but PAD pos invalid
        log_sub[0, 0, 10] = 0.0         # BOS sub invalid regardless of token
        log_sub[0, 1, 5] = 0.0          # noop sub, should not count

        forbidden_mask = _build_forbidden_mask(V, torch.device("cpu"))
        u_tot_exec = _compute_executable_u_tot(
            log_rates, log_ins, log_sub, x_t,
            pad_token=PAD_TOKEN, forbidden_mask=forbidden_mask,
        )
        # Valid executable mass:
        # - ins at BOS: 1
        # - del at pos 1: 1
        # Everything else is invalid / noop.
        assert u_tot_exec.shape == (1,)
        assert abs(u_tot_exec.item() - 2.0) < 1e-6


class TestPrepareLogRatesForScoring:
    def test_cross_scheduler_matches_beam_logic_when_not_reparam(self):
        log_rates = torch.zeros(1, 2, 3)
        t = torch.tensor([[0.5]])
        sample_scheduler = LinearScheduler()
        train_scheduler = CubicScheduler()
        t_model = train_scheduler.inverse(sample_scheduler(t))

        prepared = _prepare_log_rates_for_scoring(
            log_rates, t, sample_scheduler,
            use_rate_reparam=False,
            train_scheduler=train_scheduler,
            t_model=t_model,
        )
        k_sample = get_rate_scale(t, sample_scheduler)
        k_train = get_rate_scale(t_model, train_scheduler)
        expected_shift = torch.log(k_sample / k_train).item()
        assert torch.allclose(
            prepared,
            torch.full_like(prepared, expected_shift),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Greedy integration tests
# ---------------------------------------------------------------------------


class TestGreedyBOSProtection:
    """Greedy must not pick sub/del on BOS even when those score highest."""

    def test_greedy_avoids_bos_sub(self):
        x_t = torch.tensor([BOS_TOKEN, 5, 6, PAD_TOKEN])
        L = len(x_t)
        # BOS sub rate very high, pos=1 ins rate moderate.
        log_rates = _make_log_rates(L, sub_vals=[(0, 20.0)], ins_vals=[(1, 1.0)])
        log_ins = _make_log_probs(L, V, [(1, 10, 0.0)])
        log_sub = _make_log_probs(L, V, [(0, 10, 0.0)])

        model = _ControlledModel(V)
        model.set_outputs(log_rates, log_ins, log_sub)

        result = sample_greedy_single_edit(
            model, x_t.unsqueeze(0), CubicScheduler(),
            max_edits=2, max_seq_len=32, use_rate_reparam=False,
            time_policy=FixedTimePolicy(scheduler=CubicScheduler(), time_const=0.5),
            k_ins_token=4, k_sub_token=4, k_edit_expand=16,
        )
        # BOS should still be at position 0.
        assert result[0, 0].item() == BOS_TOKEN, "BOS should not be substituted"

    def test_greedy_avoids_bos_del(self):
        x_t = torch.tensor([BOS_TOKEN, 5, 6, PAD_TOKEN])
        L = len(x_t)
        log_rates = _make_log_rates(L, del_vals=[(0, 20.0)], ins_vals=[(1, 1.0)])
        log_ins = _make_log_probs(L, V, [(1, 10, 0.0)])
        log_sub = _make_log_probs(L, V)

        model = _ControlledModel(V)
        model.set_outputs(log_rates, log_ins, log_sub)

        result = sample_greedy_single_edit(
            model, x_t.unsqueeze(0), CubicScheduler(),
            max_edits=2, max_seq_len=32, use_rate_reparam=False,
            time_policy=FixedTimePolicy(scheduler=CubicScheduler(), time_const=0.5),
            k_ins_token=4, k_sub_token=4, k_edit_expand=16,
        )
        assert result[0, 0].item() == BOS_TOKEN, "BOS should not be deleted"


class TestGreedyNoopSub:
    """Greedy must not waste a step on no-op substitution."""

    def test_greedy_skips_noop_sub(self):
        x_t = torch.tensor([BOS_TOKEN, 5, 6, PAD_TOKEN])
        L = len(x_t)
        # no-op sub(pos=1, token=5) has highest score; ins(pos=1, 10) second.
        log_rates = _make_log_rates(L, sub_vals=[(1, 10.0)], ins_vals=[(1, 5.0)])
        log_ins = _make_log_probs(L, V, [(1, 10, 0.0)])
        log_sub = _make_log_probs(L, V, [
            (1, 5, 0.0),   # no-op (highest)
            (1, 10, -0.1),
        ])

        model = _ControlledModel(V)
        model.set_outputs(log_rates, log_ins, log_sub)

        result = sample_greedy_single_edit(
            model, x_t.unsqueeze(0), CubicScheduler(),
            max_edits=2, max_seq_len=32, use_rate_reparam=False,
            time_policy=FixedTimePolicy(scheduler=CubicScheduler(), time_const=0.5),
            k_ins_token=4, k_sub_token=4, k_edit_expand=16,
        )
        # If no-op was skipped, the ins(pos=1, 10) should have been applied.
        # Sequence should NOT be unchanged (we max_edits=2 so at least one real edit).
        tokens = [t.item() for t in result[0] if t.item() != PAD_TOKEN]
        # The original: [BOS, 5, 6]
        # After ins(pos=1, 10): [BOS, 5, 10, 6]
        assert 10 in tokens, f"Expected token 10 from non-no-op edit, got {tokens}"


# ---------------------------------------------------------------------------
# Beam integration tests
# ---------------------------------------------------------------------------


class TestBeamStopSemantics:
    """2.1: stop check before expansion — parent preserved as-is, no extra edit."""

    def test_beam_stop_preserves_parent(self):
        x_0 = torch.tensor([[BOS_TOKEN, 5, 6, 7, PAD_TOKEN, PAD_TOKEN]])
        L = x_0.shape[1]
        # Set very low rates so u_tot_base < stop threshold.
        log_rates = torch.full((L, 3), -30.0)
        log_ins = _make_log_probs(L, V, [(1, 10, 0.0)])
        log_sub = _make_log_probs(L, V, [(1, 10, 0.0)])

        model = _ControlledModel(V)
        model.set_outputs(log_rates, log_ins, log_sub)

        result = sample_beam_single_edit(
            model, x_0, CubicScheduler(),
            beam_size=3, max_edits=5, max_seq_len=32,
            use_rate_reparam=False, time_policy=FixedTimePolicy(scheduler=CubicScheduler(), time_const=0.5),
            k_ins_token=2, k_sub_token=2, k_edit_expand=8,
            stop_u_tot_base=1.0,  # u_tot will be ~0, below this threshold
        )
        # The model outputs tiny rates, so stop fires immediately.
        # Parent state should be preserved → output ≈ input.
        result_tokens = [t.item() for t in result[0] if t.item() != PAD_TOKEN]
        expected = [BOS_TOKEN, 5, 6, 7]
        assert result_tokens == expected, \
            f"stop should preserve parent, got {result_tokens}"

    def test_greedy_stop_ignores_non_executable_mass(self):
        x_0 = torch.tensor([[BOS_TOKEN, 5, 6, PAD_TOKEN]])
        L = x_0.shape[1]
        log_rates = torch.full((L, 3), LOG_NEG_INF)
        log_rates[1, 1] = 5.0   # large sub rate, but only noop token below
        log_rates[1, 2] = -10.0  # tiny valid delete mass
        log_ins = torch.full((L, V), LOG_NEG_INF)
        log_sub = torch.full((L, V), LOG_NEG_INF)
        log_sub[1, 5] = 0.0  # noop only

        model = _ControlledModel(V)
        model.set_outputs(log_rates, log_ins, log_sub)

        result = sample_greedy_single_edit(
            model, x_0, CubicScheduler(),
            max_edits=5, max_seq_len=32, use_rate_reparam=False,
            time_policy=FixedTimePolicy(scheduler=CubicScheduler(), time_const=0.5),
            k_ins_token=2, k_sub_token=2, k_edit_expand=8,
            stop_u_tot_base=1e-3,
        )
        result_tokens = [t.item() for t in result[0] if t.item() != PAD_TOKEN]
        expected = [BOS_TOKEN, 5, 6]
        assert result_tokens == expected, \
            f"non-executable noop mass should not block stop, got {result_tokens}"


class TestBeamDeadEnd:
    """2.2: dead-end parent preserved, not lost; no fallback to x_0."""

    def test_beam_dead_end_preserves_parent(self):
        # Simulate: last_edit was sub(pos=1, 5→10), so current token at
        # pos 1 is now 10.  The only good candidate is sub(pos=1, 10→5)
        # which would be a true reversal and must be filtered.
        x_t = torch.tensor([BOS_TOKEN, 10, 6, 7])  # after sub 5→10
        L = len(x_t)
        log_rates = torch.full((L, 3), float("-inf"))
        log_rates[1, 1] = 10.0  # only sub at pos 1 is active
        log_ins = torch.full((L, V), float("-inf"))
        log_sub = torch.full((L, V), float("-inf"))
        log_sub[1, 5] = 0.0  # sub 10→5 (reverse of 5→10)

        non_pad = torch.ones(L, dtype=torch.bool)
        fmask = _build_forbidden_mask(V, torch.device("cpu"))
        fmask_all = torch.ones(V, dtype=torch.bool)  # allow everything for this test
        cands, _ = _collect_edit_candidates_single(
            log_rates, log_ins, log_sub, x_t, non_pad, 10.0,
            k_ins_token=2, k_sub_token=4, k_edit_expand=16,
            forbidden_mask=fmask_all,
        )
        assert len(cands) == 1, f"expected exactly 1 candidate, got {cands}"
        assert cands[0].op == "sub" and cands[0].pos == 1 and cands[0].token == 5

        # This should be detected as reverse of sub(pos=1, 5→10).
        last = EditCandidate(pos=1, op="sub", token=10, log_u_real=1.0, score=0.0,
                             old_token=5)
        assert _is_reverse_op(cands[0], last), \
            "sub(pos=1, 10→5) should be reverse of sub(pos=1, 5→10)"

    def test_beam_dead_end_no_fallback_to_x0(self):
        """When a sample has no candidates, the beam result should be the
        parent state, NOT fall back to the original x_0 (which could differ
        after prior edits)."""
        x_0 = torch.tensor([[BOS_TOKEN, 5, 6, 7, PAD_TOKEN, PAD_TOKEN]])
        L = x_0.shape[1]
        # Tiny rates → stop immediately, parent preserved as finished.
        log_rates = torch.full((L, 3), -30.0)
        log_ins = _make_log_probs(L, V)
        log_sub = _make_log_probs(L, V)

        model = _ControlledModel(V)
        model.set_outputs(log_rates, log_ins, log_sub)

        result = sample_beam_single_edit(
            model, x_0, CubicScheduler(),
            beam_size=3, max_edits=5, max_seq_len=32,
            use_rate_reparam=False, time_policy=FixedTimePolicy(scheduler=CubicScheduler(), time_const=0.5),
            k_ins_token=2, k_sub_token=2, k_edit_expand=8,
            stop_u_tot_base=1.0,
        )
        result_tokens = [t.item() for t in result[0] if t.item() != PAD_TOKEN]
        expected = [BOS_TOKEN, 5, 6, 7]
        assert result_tokens == expected, \
            f"dead-end beam should preserve parent, got {result_tokens}"


class TestBeamFinishedCarryOver:
    """Finished states should be carried to subsequent rounds, not dropped."""

    def test_finished_state_survives_to_final_ranking(self):
        x_0 = torch.tensor([[BOS_TOKEN, 5, 6, 7, PAD_TOKEN, PAD_TOKEN]])
        L0 = x_0.shape[1]
        # Step 0: make one clear best candidate (ins at pos 1, token 10).
        # Step 1+: return tiny rates so stop fires.

        class _TwoStepModel(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.token_embedding = nn.Embedding(vocab_size, 64)
                self.call_count = 0

            def forward(self, tokens, time_step, padding_mask, origin_mask=None):
                B, L = tokens.shape
                lr = torch.full((B, L, 3), -30.0, device=tokens.device)
                lip = torch.full((B, L, V), LOG_NEG_INF, device=tokens.device)
                lsp = torch.full((B, L, V), LOG_NEG_INF, device=tokens.device)
                if self.call_count == 0:
                    # Step 0: ins at pos 1 with token 10 is the best candidate.
                    lr[:, 1, 0] = 5.0  # ins rate at pos 1
                    lip[:, 1, 10] = 0.0  # token 10
                self.call_count += 1
                pad_3d = padding_mask.unsqueeze(-1)
                lr = lr.masked_fill(pad_3d, LOG_NEG_INF)
                lip = lip.masked_fill(pad_3d, LOG_NEG_INF)
                lsp = lsp.masked_fill(pad_3d, LOG_NEG_INF)
                return lr, lip, lsp

        model = _TwoStepModel(V)
        result = sample_beam_single_edit(
            model, x_0, CubicScheduler(),
            beam_size=3, max_edits=5, max_seq_len=32,
            use_rate_reparam=False, time_policy=FixedTimePolicy(scheduler=CubicScheduler(), time_const=0.5),
            k_ins_token=2, k_sub_token=2, k_edit_expand=8,
            stop_u_tot_base=1.0,
        )
        # After step 0 an edit was applied, then stop fired.
        # The edited state (with ins) should have been carried over and selected.
        result_tokens = [t.item() for t in result[0] if t.item() != PAD_TOKEN]
        # Original: [BOS, 5, 6, 7]. After ins(pos=1, 10): [BOS, 5, 10, 6, 7].
        assert 10 in result_tokens, \
            f"finished state with edit should survive, got {result_tokens}"


class TestBeamAdaptiveTimePolicy:
    def test_stateful_policy_updates_per_hypothesis(self):
        x_0 = torch.tensor([[BOS_TOKEN, 4, PAD_TOKEN]])

        class _PolicyAwareModel(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.token_embedding = nn.Embedding(vocab_size, 64)
                self.call_count = 0
                self.time_batches = []

            def forward(self, tokens, time_step, padding_mask, origin_mask=None):
                self.time_batches.append(time_step.squeeze(-1).detach().cpu().tolist())
                B, L = tokens.shape
                lr = torch.full((B, L, 3), LOG_NEG_INF, device=tokens.device)
                lip = torch.full((B, L, V), LOG_NEG_INF, device=tokens.device)
                lsp = torch.full((B, L, V), LOG_NEG_INF, device=tokens.device)

                if self.call_count == 0:
                    lr[:, 1, 1] = 5.0
                    lsp[:, 1, 5] = 0.0
                    lsp[:, 1, 6] = -0.1
                elif self.call_count == 1:
                    for b in range(B):
                        tok = int(tokens[b, 1].item())
                        if tok == 5:
                            lr[b, 1, 1] = -2.5
                            lsp[b, 1, 7] = 0.0
                        else:
                            lr[b, 1, 1] = 1.5
                            lsp[b, 1, 8] = 0.0
                else:
                    for b in range(B):
                        t_val = float(time_step[b, 0].item())
                        if t_val < 0.5:
                            lr[b, 1, 1] = 0.0
                            lsp[b, 1, 9] = 0.0
                        else:
                            lr[b, 1, 1] = 0.0
                            lsp[b, 1, 10] = 0.0

                self.call_count += 1
                pad_3d = padding_mask.unsqueeze(-1)
                lr = lr.masked_fill(pad_3d, LOG_NEG_INF)
                lip = lip.masked_fill(pad_3d, LOG_NEG_INF)
                lsp = lsp.masked_fill(pad_3d, LOG_NEG_INF)
                return lr, lip, lsp

        model = _PolicyAwareModel(V)
        sample_beam_single_edit(
            model, x_0, LinearScheduler(),
            time_policy=_BranchingTimePolicy(),
            beam_size=2, max_edits=3, max_seq_len=16,
            use_rate_reparam=False,
            k_ins_token=1, k_sub_token=2, k_edit_expand=2,
        )
        assert len(model.time_batches) == 3
        third_step_times = sorted(round(v, 4) for v in model.time_batches[2])
        assert third_step_times == [0.2, 0.8], \
            f"expected diverged per-beam times, got {third_step_times}"

    def test_dedup_keeps_same_tokens_with_different_origin_mask(self):
        x_0 = torch.tensor([[BOS_TOKEN, 5, 4, PAD_TOKEN]])

        class _OriginAwareModel(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.token_embedding = nn.Embedding(vocab_size, 64)
                self.call_count = 0
                self.batch_sizes = []

            def forward(self, tokens, time_step, padding_mask, origin_mask=None):
                self.batch_sizes.append(tokens.shape[0])
                B, L = tokens.shape
                lr = torch.full((B, L, 3), LOG_NEG_INF, device=tokens.device)
                lip = torch.full((B, L, V), LOG_NEG_INF, device=tokens.device)
                lsp = torch.full((B, L, V), LOG_NEG_INF, device=tokens.device)

                if self.call_count == 0:
                    lr[:, 2, 2] = 5.0
                    lr[:, 1, 1] = 4.9
                    lsp[:, 1, 6] = 0.0
                elif self.call_count == 1:
                    for b in range(B):
                        if int(tokens[b, 1].item()) == 5:
                            lr[b, 1, 1] = 4.0
                            lsp[b, 1, 6] = 0.0
                        else:
                            lr[b, 2, 2] = 4.0
                else:
                    for b in range(B):
                        if bool(origin_mask[b, 1].item()):
                            lr[b, 1, 1] = 0.0
                            lsp[b, 1, 7] = 0.0
                        else:
                            lr[b, 1, 1] = 0.0
                            lsp[b, 1, 8] = 0.0

                self.call_count += 1
                pad_3d = padding_mask.unsqueeze(-1)
                lr = lr.masked_fill(pad_3d, LOG_NEG_INF)
                lip = lip.masked_fill(pad_3d, LOG_NEG_INF)
                lsp = lsp.masked_fill(pad_3d, LOG_NEG_INF)
                return lr, lip, lsp

        model = _OriginAwareModel(V)
        sample_beam_single_edit(
            model, x_0, CubicScheduler(),
            time_policy=FixedTimePolicy(scheduler=CubicScheduler(), time_const=0.5),
            beam_size=2, max_edits=3, max_seq_len=16,
            use_rate_reparam=False, use_origin_mask=True,
            k_ins_token=1, k_sub_token=2, k_edit_expand=2,
        )
        assert model.batch_sizes[:3] == [1, 2, 2], \
            f"expected both converged states to survive dedup, got batches {model.batch_sizes}"


# ---------------------------------------------------------------------------
# _build_forbidden_mask
# ---------------------------------------------------------------------------


class TestForbiddenMask:
    def test_special_tokens_are_false(self):
        mask = _build_forbidden_mask(V, torch.device("cpu"))
        assert not mask[PAD_TOKEN].item()
        assert not mask[BOS_TOKEN].item()
        assert not mask[GAP_TOKEN].item()
        assert not mask[UNK_TOKEN].item()
        assert mask[5].item()  # normal token should be allowed


# ---------------------------------------------------------------------------
# Origin mask in single-edit apply (same semantics as Euler)
# ---------------------------------------------------------------------------


class TestOriginMaskSingleEdit:
    """4.8: origin mask updates match Euler's 3-state marker semantics."""

    def test_sub_flips_origin_to_false(self):
        x_t = torch.tensor([BOS_TOKEN, 5, 6, 7])
        origin = torch.ones_like(x_t, dtype=torch.bool)
        cand = EditCandidate(pos=1, op="sub", token=10, log_u_real=1.0, score=0.0)
        _, origin_next = _apply_single_edit_to_sequence(
            x_t, origin, cand, max_seq_len=32, pad_token=PAD_TOKEN,
        )
        assert origin_next[0].item() is True   # BOS
        assert origin_next[1].item() is False  # substituted
        assert origin_next[2].item() is True

    def test_ins_new_token_is_false(self):
        x_t = torch.tensor([BOS_TOKEN, 5, 6, 7])
        origin = torch.ones_like(x_t, dtype=torch.bool)
        cand = EditCandidate(pos=1, op="ins", token=10, log_u_real=1.0, score=0.0)
        _, origin_next = _apply_single_edit_to_sequence(
            x_t, origin, cand, max_seq_len=32, pad_token=PAD_TOKEN,
        )
        # New token at pos 2 (after insertion point).
        assert origin_next[2].item() is False  # newly inserted

    def test_del_removes_token_and_mask_together(self):
        x_t = torch.tensor([BOS_TOKEN, 5, 6, 7])
        origin = torch.ones_like(x_t, dtype=torch.bool)
        cand = EditCandidate(pos=1, op="del", token=None, log_u_real=1.0, score=0.0)
        x_next, origin_next = _apply_single_edit_to_sequence(
            x_t, origin, cand, max_seq_len=32, pad_token=PAD_TOKEN,
        )
        # After deletion: [BOS, 6, 7], origin should have len 3 (no PAD).
        assert x_next[1].item() == 6
        assert origin_next[0].item() is True  # BOS
        assert origin_next[1].item() is True  # 6 (shifted from pos 2)
