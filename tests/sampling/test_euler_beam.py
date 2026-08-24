import math

import pytest
import torch

from edit_flows.core.scheduler import LinearScheduler
from edit_flows.sampling.euler_beam import (
    _BranchState,
    _branch_sort_key,
    _sample_actions_per_branch,
    _step_log_p,
    sample_euler_beam,
)
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


def _scalar_triggered_only(actions, log_rates, log_ins, log_sub, adapt_h):
    rates = torch.exp(log_rates[0])
    result = 0.0
    eps = 1e-12
    for pos in actions["ins_mask"][0].nonzero(as_tuple=False).squeeze(-1).tolist():
        token = int(actions["ins_tokens"][0, pos])
        result += math.log(max(1.0 - math.exp(-adapt_h * rates[pos, 0].item()), eps))
        result += math.log(max(torch.exp(log_ins[0, pos, token]).item(), eps))
    for pos in actions["sub_mask"][0].nonzero(as_tuple=False).squeeze(-1).tolist():
        token = int(actions["sub_tokens"][0, pos])
        result += math.log(max(1.0 - math.exp(-adapt_h * rates[pos, 1].item()), eps))
        result += math.log(max(torch.exp(log_sub[0, pos, token]).item(), eps))
    for pos in actions["del_mask"][0].nonzero(as_tuple=False).squeeze(-1).tolist():
        result += math.log(max(1.0 - math.exp(-adapt_h * rates[pos, 2].item()), eps))
    return result


def _scalar_complete_step_log_p(actions, log_rates, log_ins, log_sub, adapt_h):
    rates = torch.exp(log_rates[0])
    result = 0.0
    eps = 1e-12
    for pos in range(rates.shape[0]):
        ins_rate, sub_rate, del_rate = rates[pos].tolist()
        if bool(actions["ins_mask"][0, pos]):
            token = int(actions["ins_tokens"][0, pos])
            result += math.log(max(1.0 - math.exp(-adapt_h * ins_rate), eps))
            result += math.log(max(torch.exp(log_ins[0, pos, token]).item(), eps))
        else:
            result -= adapt_h * ins_rate

        ds_rate = sub_rate + del_rate
        if bool(actions["sub_mask"][0, pos]):
            token = int(actions["sub_tokens"][0, pos])
            result += math.log(max(1.0 - math.exp(-adapt_h * ds_rate), eps))
            result += math.log(max(sub_rate / max(ds_rate, eps), eps))
            result += math.log(max(torch.exp(log_sub[0, pos, token]).item(), eps))
        elif bool(actions["del_mask"][0, pos]):
            result += math.log(max(1.0 - math.exp(-adapt_h * ds_rate), eps))
            result += math.log(max(del_rate / max(ds_rate, eps), eps))
        else:
            result -= adapt_h * ds_rate
    return result


def _random_actions(length, vocab_size):
    sub_mask = torch.rand(1, length) < 0.35
    del_mask = (torch.rand(1, length) < 0.25) & ~sub_mask
    return {
        "ins_mask": torch.rand(1, length) < 0.45,
        "sub_mask": sub_mask,
        "del_mask": del_mask,
        "ins_tokens": torch.randint(vocab_size, (1, length)),
        "sub_tokens": torch.randint(vocab_size, (1, length)),
    }


def test_step_log_p_matches_complete_scalar_implementation():
    torch.manual_seed(7)
    length, vocab_size = 13, 17
    log_rates = torch.randn(1, length, 3)
    log_ins = torch.log_softmax(torch.randn(1, length, vocab_size), dim=-1)
    log_sub = torch.log_softmax(torch.randn(1, length, vocab_size), dim=-1)
    actions = _random_actions(length, vocab_size)
    adapt_h = 0.037
    expected = _scalar_complete_step_log_p(
        actions, log_rates, log_ins, log_sub, adapt_h,
    )
    actual = _step_log_p(actions, log_rates, log_ins, log_sub, adapt_h)
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)


def test_beam_uses_bos_as_an_insert_anchor_only():
    x_t = torch.tensor([[BOS_TOKEN, 7, PAD_TOKEN]])
    log_rates = torch.full((1, 3, 3), 20.0)
    log_probs = torch.log_softmax(torch.zeros(1, 3, 16), dim=-1)
    actions = _sample_actions_per_branch(
        torch.tensor([123]),
        x_t,
        log_rates,
        log_probs,
        log_probs,
        torch.tensor([[0.1]]),
        PAD_TOKEN,
        "poisson",
        step=0,
    )
    assert bool(actions["ins_mask"][0, 0])
    assert not bool(actions["sub_mask"][0, 0])
    assert not bool(actions["del_mask"][0, 0])


def test_beam_filters_special_tokens_and_noop_substitution():
    x_t = torch.tensor([[BOS_TOKEN, 7, PAD_TOKEN]])
    log_rates = torch.full((1, 3, 3), -1e9)
    log_rates[0, 0, 0] = 20.0
    log_rates[0, 1, 1] = 20.0
    log_ins = torch.full((1, 3, 16), -1e9)
    log_ins[:, :, PAD_TOKEN] = 0.0
    log_ins[:, :, 9] = -0.1
    log_sub = torch.full((1, 3, 16), -1e9)
    log_sub[:, :, 7] = 0.0
    log_sub[:, :, PAD_TOKEN] = -0.1
    log_sub[:, :, 10] = -0.2

    actions = _sample_actions_per_branch(
        torch.tensor([123]),
        x_t,
        log_rates,
        log_ins,
        log_sub,
        torch.tensor([[0.1]]),
        PAD_TOKEN,
        "poisson",
        step=0,
    )

    assert bool(actions["ins_mask"][0, 0])
    assert int(actions["ins_tokens"][0, 0].item()) == 9
    assert bool(actions["sub_mask"][0, 1])
    assert int(actions["sub_tokens"][0, 1].item()) == 10


def test_step_score_ignores_bos_substitute_delete_mass():
    x_t = torch.tensor([[BOS_TOKEN, PAD_TOKEN]])
    log_rates = torch.full((1, 2, 3), -1e9)
    log_rates[0, 0, 1] = 0.0
    log_rates[0, 0, 2] = 0.0
    log_probs = torch.log_softmax(torch.zeros(1, 2, 16), dim=-1)
    actions = {
        "ins_mask": torch.zeros(1, 2, dtype=torch.bool),
        "sub_mask": torch.zeros(1, 2, dtype=torch.bool),
        "del_mask": torch.zeros(1, 2, dtype=torch.bool),
        "ins_tokens": torch.zeros(1, 2, dtype=torch.long),
        "sub_tokens": torch.zeros(1, 2, dtype=torch.long),
    }

    score = _step_log_p(
        actions,
        log_rates,
        log_probs,
        log_probs,
        adapt_h=0.1,
        state_tokens=x_t,
    )
    assert abs(score) < 1e-6


def test_step_log_p_no_events_includes_survival_probability():
    length, vocab_size = 5, 8
    actions = {
        "ins_mask": torch.zeros(1, length, dtype=torch.bool),
        "sub_mask": torch.zeros(1, length, dtype=torch.bool),
        "del_mask": torch.zeros(1, length, dtype=torch.bool),
        "ins_tokens": torch.zeros(1, length, dtype=torch.long),
        "sub_tokens": torch.zeros(1, length, dtype=torch.long),
    }
    adapt_h = 0.1
    result = _step_log_p(
        actions,
        torch.zeros(1, length, 3),
        torch.log_softmax(torch.zeros(1, length, vocab_size), dim=-1),
        torch.log_softmax(torch.zeros(1, length, vocab_size), dim=-1),
        adapt_h=adapt_h,
    )
    assert math.isclose(result, -length * 3 * adapt_h, abs_tol=1e-6)


def test_step_log_p_is_finite_for_extreme_rates():
    length, vocab_size = 4, 8
    for raw_log_rate in (-100.0, 20.0):
        log_rates = torch.full((1, length, 3), raw_log_rate)
        log_probs = torch.log_softmax(torch.randn(1, length, vocab_size), dim=-1)
        result = _step_log_p(
            _random_actions(length, vocab_size),
            log_rates, log_probs, log_probs, adapt_h=0.01,
        )
        assert math.isfinite(result)


def test_q_temperature_preserves_t1_and_normalizes_sharpened_logits():
    from edit_flows.sampling.euler_beam import _apply_q_temperature

    log_probs = torch.log_softmax(torch.tensor([[0.0, 1.0, 2.0]]), dim=-1)
    unchanged = _apply_q_temperature(log_probs, 1.0)
    sharpened = _apply_q_temperature(log_probs, 0.8)

    assert unchanged is log_probs
    assert torch.equal(unchanged, log_probs)
    assert torch.allclose(
        torch.exp(sharpened).sum(dim=-1), torch.ones(1), atol=1e-6,
    )
    assert torch.exp(sharpened)[0, -1] > torch.exp(log_probs)[0, -1]


def test_q_temperature_one_is_sampling_compatible():
    model = _StochasticModel()
    x_0 = torch.tensor([[BOS_TOKEN, 4, 5, 6, PAD_TOKEN]])
    common = dict(
        scheduler=LinearScheduler(),
        n_branches=1,
        n_children=2,
        n_steps=4,
        max_seq_len=32,
        base_seed=42,
    )
    implicit = sample_euler_beam(model, x_0, **common)
    explicit = sample_euler_beam(model, x_0, q_temperature=1.0, **common)
    assert torch.equal(implicit, explicit)


def test_branch_sort_prefers_state_mass_then_stable_seed():
    high = _BranchState(
        torch.tensor([[BOS_TOKEN, 4]]), path_log_p=-2.0,
        log_mass=-1.0, seed=10,
    )
    low = _BranchState(
        torch.tensor([[BOS_TOKEN, 5]]), path_log_p=-10.0, log_mass=-3.0,
    )
    assert max([low, high], key=_branch_sort_key) is high
    stable = _BranchState(
        torch.tensor([[BOS_TOKEN, 6]]), path_log_p=-100.0,
        log_mass=-1.0, seed=1,
    )
    assert max([high, stable], key=_branch_sort_key) is stable


def test_m1_trajectory_sort_preserves_pre_task3_behavior():
    from edit_flows.sampling.euler_beam import _trajectory_branch_sort_key

    high = _BranchState(
        torch.tensor([[BOS_TOKEN, 4]]), path_log_p=-2.0, weight=1.0,
    )
    low = _BranchState(
        torch.tensor([[BOS_TOKEN, 5]]), path_log_p=-10.0, weight=100.0,
    )
    assert max([low, high], key=_trajectory_branch_sort_key) is high
    heavier = _BranchState(
        torch.tensor([[BOS_TOKEN, 6]]), path_log_p=-2.0, weight=2.0,
    )
    assert max([high, heavier], key=_trajectory_branch_sort_key) is heavier


class _StochasticModel(torch.nn.Module):
    def __init__(self, vocab_size=16):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, tokens, time_step, padding_mask, origin_mask=None):
        batch, length = tokens.shape
        log_rates = torch.full((batch, length, 3), -0.3, device=tokens.device)
        logits = torch.arange(
            self.vocab_size, dtype=torch.float, device=tokens.device,
        ).view(1, 1, -1).expand(batch, length, -1)
        log_probs = torch.log_softmax(logits / 4.0, dim=-1)
        mask = padding_mask.unsqueeze(-1)
        return (
            log_rates.masked_fill(mask, -1e9),
            log_probs.masked_fill(mask, -1e9),
            log_probs.masked_fill(mask, -1e9),
        )


def test_beam_center_bias_reweights_only_legal_positions_and_preserves_hazard():
    x_t = torch.tensor([[BOS_TOKEN, 4, 5, PAD_TOKEN]])
    log_rates = torch.zeros(1, 4, 3)
    log_probs = torch.log_softmax(torch.zeros(1, 4, 16), dim=-1)
    scores = torch.zeros(1, 4, 3)
    scores[:, 1, 1] = 1.0
    actions = _sample_actions_per_branch(
        torch.tensor([123]),
        x_t,
        log_rates,
        log_probs,
        log_probs,
        torch.tensor([[0.01]]),
        PAD_TOKEN,
        "poisson",
        step=0,
        position_scores=scores,
        position_bias_active=torch.tensor([True]),
        position_bias_max_multiplier=3.0,
    )

    effective = actions["effective_log_rates"]
    diagnostics = actions["position_bias_diagnostics"]
    legal = torch.stack((
        actions["insert_position_mask"],
        actions["sub_del_position_mask"],
        actions["sub_del_position_mask"],
    ), dim=-1)
    assert torch.allclose(
        torch.exp(effective).masked_fill(~legal, 0.0).sum(dim=1),
        torch.exp(log_rates).masked_fill(~legal, 0.0).sum(dim=1),
        atol=1e-6,
    )
    assert diagnostics["relative_error"].max() < 1e-6
    # At the two legal SUB positions, the score-1 position has triple the
    # relative rate of its score-0 neighbour after per-mode renormalization.
    ratio = torch.exp(effective[0, 1, 1] - effective[0, 2, 1])
    assert torch.allclose(ratio, torch.tensor(3.0), atol=1e-6)


class _ForcedFirstSubstitution(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, tokens, time_step, padding_mask, origin_mask=None):
        batch, length = tokens.shape
        log_rates = torch.full(
            (batch, length, 3), -30.0, device=tokens.device,
        )
        log_rates[:, 1, 1] = 20.0
        log_ins = torch.full(
            (batch, length, 16), -1e9, device=tokens.device,
        )
        log_sub = torch.full_like(log_ins, -1e9)
        log_ins[:, :, 9] = 0.0
        log_sub[:, :, 9] = 0.0
        return log_rates, log_ins, log_sub


def test_r9_style_center_bias_tracks_selected_lineage_first_event():
    model = _ForcedFirstSubstitution()
    x_0 = torch.tensor([
        [BOS_TOKEN, 4, 5, PAD_TOKEN],
        [BOS_TOKEN, 6, 7, PAD_TOKEN],
        [BOS_TOKEN, 8, 10, PAD_TOKEN],
    ])
    scores = torch.zeros(3, 4, 3)
    scores[:, 1, 1] = 1.0
    stats = {}
    result = sample_euler_beam(
        model,
        x_0,
        LinearScheduler(),
        n_branches=1,
        n_children=2,
        n_steps=2,
        max_seq_len=16,
        sample_seeds=[101, 202, 303],
        score_mode="full_probability",
        changed_state_bonus=0.5,
        child_policy="stochastic_noop",
        profile_sample_group_size=3,
        first_event_position_scores=scores,
        first_event_position_bias_enabled=torch.tensor([True, False, True]),
        first_event_bias_max_multiplier=3.0,
        first_event_bias_stats=stats,
        first_event_row_metadata=[
            {"trajectory_role": "center_guided"},
            {"trajectory_role": "ordinary_euler"},
            {"trajectory_role": "center_guided"},
        ],
        first_event_bias_record_events=True,
    )

    assert torch.equal(result[:, 1], torch.tensor([9, 9, 9]))
    # All three selected lineages edit at step zero, so the center mechanism
    # cannot remain active during their ordinary R9K1M2 continuation.
    assert stats["first_event_count"] == 3
    assert stats["no_event_count"] == 0
    assert stats["biased_row_steps"] == 3
    assert stats["guided_row_steps"] == 2
    assert stats["max_hazard_relative_error"] < 1e-6
    assert len(stats["records"]) == 3
    assert [record["position_bias_enabled"] for record in stats["records"]] == [
        True, False, True,
    ]
    assert [record["position_bias_reweighted"] for record in stats["records"]] == [
        True, False, True,
    ]
    assert all(record["first_event_step_idx"] == 0 for record in stats["records"])
    assert all(record["beam_branch_rank"] == 0 for record in stats["records"])
    assert all(record["actions"][0]["center_score"] == 1.0
               for record in stats["records"])


def test_r9_style_center_bias_multiplier_one_is_bitwise_neutral():
    model = _StochasticModel()
    x_0 = torch.tensor([
        [BOS_TOKEN, 4, 5, 6, PAD_TOKEN],
        [BOS_TOKEN, 7, 8, PAD_TOKEN, PAD_TOKEN],
        [BOS_TOKEN, 9, 10, 11, PAD_TOKEN],
    ])
    scores = torch.rand(3, 5, 3)
    common = dict(
        scheduler=LinearScheduler(),
        n_branches=1,
        n_children=2,
        n_steps=4,
        max_seq_len=32,
        sample_seeds=[101, 202, 303],
        score_mode="full_probability",
        changed_state_bonus=0.5,
        child_policy="stochastic_noop",
        profile_sample_group_size=3,
    )
    plain = sample_euler_beam(model, x_0, **common)
    stats = {}
    neutral = sample_euler_beam(
        model,
        x_0,
        first_event_position_scores=scores,
        first_event_bias_max_multiplier=1.0,
        first_event_bias_stats=stats,
        **common,
    )
    assert torch.equal(plain, neutral)
    assert stats["first_event_count"] + stats["no_event_count"] == 3
    assert stats["max_hazard_relative_error"] == 0.0
    assert stats["summary_from_final_lineages"] is True


def test_sample_euler_beam_same_seed_is_reproducible():
    model = _StochasticModel()
    x_0 = torch.tensor([
        [BOS_TOKEN, 4, 5, 6, PAD_TOKEN],
        [BOS_TOKEN, 7, 8, PAD_TOKEN, PAD_TOKEN],
    ])
    kwargs = dict(
        scheduler=LinearScheduler(), n_branches=3, n_steps=4,
        max_seq_len=32, base_seed=123,
    )
    assert torch.equal(
        sample_euler_beam(model, x_0, **kwargs),
        sample_euler_beam(model, x_0, **kwargs),
    )


def test_sample_euler_beam_different_seed_changes_output():
    model = _StochasticModel()
    x_0 = torch.tensor([[BOS_TOKEN, 4, 5, 6, 7, 8, PAD_TOKEN]])
    common = dict(
        scheduler=LinearScheduler(), n_branches=3, n_steps=6, max_seq_len=64,
    )
    first = sample_euler_beam(model, x_0, base_seed=100, **common)
    second = sample_euler_beam(model, x_0, base_seed=200, **common)
    assert not torch.equal(first, second)


def test_sample_euler_beam_one_branch_runs():
    model = _StochasticModel()
    x_0 = torch.tensor([
        [BOS_TOKEN, 4, 5, PAD_TOKEN],
        [BOS_TOKEN, 6, PAD_TOKEN, PAD_TOKEN],
    ])
    result = sample_euler_beam(
        model, x_0, LinearScheduler(), n_branches=1, n_steps=3,
        max_seq_len=32, base_seed=42,
    )
    assert result.shape[0] == 2


def test_optional_profile_records_stages_without_changing_output():
    model = _StochasticModel()
    x_0 = torch.tensor([[BOS_TOKEN, 4, 5, 6, PAD_TOKEN]])
    kwargs = dict(
        scheduler=LinearScheduler(),
        n_branches=2,
        n_children=2,
        n_steps=3,
        max_seq_len=32,
        base_seed=42,
    )
    expected = sample_euler_beam(model, x_0, **kwargs)
    profile = {}
    actual = sample_euler_beam(model, x_0, profile=profile, **kwargs)

    assert torch.equal(actual, expected)
    assert profile["steps"] == 3
    assert profile["model_forward_calls"] == 3
    assert profile["uniform_width_fast_path_steps"] == 3
    assert profile["parent_branch_evaluations"] > 0
    assert profile["child_candidate_evaluations"] >= 2
    assert profile["actual_token_count"] > 0
    assert profile["padded_token_count"] >= profile["actual_token_count"]
    assert profile["padded_attention_cost_proxy"] >= (
        profile["actual_attention_cost_proxy"]
    )
    assert sum(profile["active_length_histogram"].values()) == (
        profile["parent_branch_evaluations"]
    )
    for key in (
        "prepare_branches_seconds",
        "model_forward_and_rates_seconds",
        "child_proposal_seconds",
        "step_scoring_seconds",
        "apply_edits_seconds",
        "merge_and_prune_seconds",
        "finalize_output_seconds",
    ):
        assert profile[key] >= 0.0


def test_profile_records_protected_parent_sharing_and_k1_decisions():
    model = _StochasticModel()
    product = torch.tensor([BOS_TOKEN, 4, 5, 6, PAD_TOKEN])
    x_0 = product.unsqueeze(0).repeat(3, 1)
    kwargs = dict(
        scheduler=LinearScheduler(),
        n_branches=1,
        n_children=2,
        n_steps=3,
        max_seq_len=32,
        sample_seeds=[11, 22, 33],
        changed_state_bonus=0.5,
    )
    expected = sample_euler_beam(model, x_0, **kwargs)
    profile = {}
    actual = sample_euler_beam(
        model,
        x_0,
        profile=profile,
        profile_sample_group_size=3,
        **kwargs,
    )

    assert torch.equal(actual, expected)
    assert profile["protected_sample_group_size"] == 3
    assert profile["protected_parent_rows"] == 9
    assert profile["protected_unique_parent_rows"] <= 9
    assert profile["potential_shared_parent_rows"] >= 2
    assert profile["protected_parent_rows_by_step"] == [3, 3, 3]
    assert len(profile["protected_unique_parent_rows_by_step"]) == 3
    assert profile["k1_child_pairs"] == 9
    decisions = (
        profile.get("k1_identical_child_pairs", 0)
        + profile.get("k1_bonus_decisions", 0)
        + profile.get("k1_seed_tiebreak_decisions", 0)
    )
    assert decisions == profile["k1_child_pairs"]


def test_profile_sample_group_size_must_divide_batch():
    model = _StochasticModel()
    x_0 = torch.tensor([
        [BOS_TOKEN, 4, PAD_TOKEN],
        [BOS_TOKEN, 5, PAD_TOKEN],
        [BOS_TOKEN, 6, PAD_TOKEN],
    ])
    with pytest.raises(ValueError, match="profile_sample_group_size"):
        sample_euler_beam(
            model,
            x_0,
            LinearScheduler(),
            profile={},
            profile_sample_group_size=2,
        )


def test_identical_forward_sharing_preserves_independent_lineage_output():
    model = _StochasticModel()
    product = torch.tensor([BOS_TOKEN, 4, 5, 6, PAD_TOKEN])
    x_0 = product.unsqueeze(0).repeat(3, 1)
    common = dict(
        scheduler=LinearScheduler(),
        n_branches=1,
        n_children=2,
        n_steps=4,
        max_seq_len=32,
        sample_seeds=[11, 22, 33],
        changed_state_bonus=0.5,
        profile_sample_group_size=3,
    )
    baseline_stats = {}
    baseline = sample_euler_beam(
        model, x_0, sampling_stats=baseline_stats, **common,
    )
    shared_stats = {}
    shared = sample_euler_beam(
        model,
        x_0,
        share_identical_forwards=True,
        sampling_stats=shared_stats,
        **common,
    )

    assert torch.equal(shared, baseline)
    assert baseline_stats["shared_model_parent_rows"] == 0
    assert shared_stats["shared_model_parent_rows"] > 0
    assert shared_stats["model_forward_parent_rows"] < (
        baseline_stats["model_forward_parent_rows"]
    )
    assert shared_stats["parent_branch_evaluations"] == (
        baseline_stats["parent_branch_evaluations"]
    )


def test_sample_euler_beam_rejects_unsupported_origin_mask():
    model = _StochasticModel()
    x_0 = torch.tensor([[BOS_TOKEN, 4, PAD_TOKEN]])
    with pytest.raises(NotImplementedError, match="origin_mask"):
        sample_euler_beam(model, x_0, LinearScheduler(), use_origin_mask=True)


def test_sample_euler_beam_validates_sizes():
    model = _StochasticModel()
    x_0 = torch.tensor([[BOS_TOKEN, 4, PAD_TOKEN]])
    with pytest.raises(ValueError, match="n_branches"):
        sample_euler_beam(model, x_0, LinearScheduler(), n_branches=0)
    with pytest.raises(ValueError, match="n_steps"):
        sample_euler_beam(model, x_0, LinearScheduler(), n_steps=0)
    with pytest.raises(ValueError, match="n_children"):
        sample_euler_beam(model, x_0, LinearScheduler(), n_children=0)
    with pytest.raises(ValueError, match="q_temperature"):
        sample_euler_beam(model, x_0, LinearScheduler(), q_temperature=0.0)
    with pytest.raises(ValueError, match="changed_state_bonus"):
        sample_euler_beam(
            model, x_0, LinearScheduler(), changed_state_bonus=-0.1,
        )
    with pytest.raises(ValueError, match="child_policy"):
        sample_euler_beam(
            model, x_0, LinearScheduler(), child_policy="unknown",
        )
    with pytest.raises(ValueError, match="n_children=2"):
        sample_euler_beam(
            model, x_0, LinearScheduler(), n_children=3,
            child_policy="stochastic_noop",
        )
    with pytest.raises(ValueError, match="first_edit_diversity"):
        sample_euler_beam(
            model, x_0, LinearScheduler(), n_children=1,
            first_edit_diversity=True,
        )
    with pytest.raises(ValueError, match="profile_sample_group_size"):
        sample_euler_beam(
            model, x_0, LinearScheduler(), n_branches=1, n_children=2,
            first_edit_diversity=True,
        )
    with pytest.raises(ValueError, match="score_mode"):
        sample_euler_beam(
            model, x_0, LinearScheduler(), n_children=2,
            score_mode="legacy_triggered_reverse",
            first_edit_diversity=True,
        )


def test_child_seed_is_stable_distinct_and_m1_compatible():
    from edit_flows.sampling.euler_beam import _mix_child_seed

    assert _mix_child_seed(123, step=7, child_index=0) == 123
    values = {
        _mix_child_seed(parent, step, child)
        for parent in range(100, 200)
        for step in range(4)
        for child in range(1, 5)
    }
    assert len(values) == 100 * 4 * 4
    assert _mix_child_seed(123, 7, 2) == _mix_child_seed(123, 7, 2)
    assert _mix_child_seed(123, 7, 2) != _mix_child_seed(123, 8, 2)


def test_noop_policy_only_clears_second_child_edit_masks():
    from edit_flows.sampling.euler_beam import _set_second_child_noop

    actions = {
        "ins_mask": torch.ones(6, 4, dtype=torch.bool),
        "sub_mask": torch.ones(6, 4, dtype=torch.bool),
        "del_mask": torch.ones(6, 4, dtype=torch.bool),
        "ins_tokens": torch.arange(24).reshape(6, 4),
        "sub_tokens": torch.arange(24).reshape(6, 4),
    }
    original_tokens = {
        name: actions[name].clone() for name in ("ins_tokens", "sub_tokens")
    }
    _set_second_child_noop(actions, n_children=2)

    for name in ("ins_mask", "sub_mask", "del_mask"):
        assert actions[name][::2].all()
        assert not actions[name][1::2].any()
    assert torch.equal(actions["ins_tokens"], original_tokens["ins_tokens"])
    assert torch.equal(actions["sub_tokens"], original_tokens["sub_tokens"])


def test_first_edit_signature_uses_first_active_position_only():
    from edit_flows.sampling.euler_beam import _first_edit_signatures_from_actions

    actions = {
        "ins_mask": torch.tensor([
            [False, False, True, False],
            [False, False, False, False],
            [False, False, False, False],
        ]),
        "sub_mask": torch.tensor([
            [False, False, False, True],
            [False, True, False, False],
            [False, False, False, False],
        ]),
        "del_mask": torch.tensor([
            [False, False, False, False],
            [False, True, False, False],
            [False, False, False, False],
        ]),
        "ins_tokens": torch.tensor([
            [0, 0, 11, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]),
        "sub_tokens": torch.tensor([
            [0, 0, 0, 12],
            [0, 13, 0, 0],
            [0, 0, 0, 0],
        ]),
    }

    assert _first_edit_signatures_from_actions(actions) == [
        (("ins", 2, 11),),
        (("sub", 1, 13), ("del", 1, -1)),
        None,
    ]


def test_first_edit_diversity_reserves_unique_signature_slots():
    from edit_flows.sampling.euler_beam import (
        _merge_state_candidates_with_first_edit_diversity,
    )

    signature_a = (("ins", 1, 7),)
    signature_b = (("sub", 2, 8),)
    candidates = [
        (
            _BranchState(
                torch.tensor([[BOS_TOKEN, 4]]),
                log_mass=-0.1,
                seed=1,
                state_key=(4,),
                first_edit_signature=signature_a,
            ),
            (4,),
        ),
        (
            _BranchState(
                torch.tensor([[BOS_TOKEN, 5]]),
                log_mass=-0.2,
                seed=2,
                state_key=(5,),
                first_edit_signature=signature_a,
            ),
            (5,),
        ),
        (
            _BranchState(
                torch.tensor([[BOS_TOKEN, 6]]),
                log_mass=-0.9,
                seed=3,
                state_key=(6,),
                first_edit_signature=signature_b,
            ),
            (6,),
        ),
    ]

    selected = _merge_state_candidates_with_first_edit_diversity(
        candidates, n_branches=2, origin_key=(4,), changed_state_bonus=0.0,
    )

    assert len(selected) == 2
    assert {
        branch.first_edit_signature for branch in selected
    } == {signature_a, signature_b}


def test_state_merge_uses_logsumexp_mass_not_path_probability():
    from edit_flows.sampling.euler_beam import _merge_state_candidates

    shared_a = _BranchState(
        torch.tensor([[BOS_TOKEN, 4]]), path_log_p=-100.0,
        log_mass=math.log(0.3), seed=1,
    )
    shared_b = _BranchState(
        torch.tensor([[BOS_TOKEN, 4]]), path_log_p=-2.0,
        log_mass=math.log(0.3), seed=2,
    )
    other = _BranchState(
        torch.tensor([[BOS_TOKEN, 5]]), path_log_p=-1.0,
        log_mass=math.log(0.4), seed=3,
    )
    ranked = _merge_state_candidates([
        (shared_a, (4,)), (shared_b, (4,)), (other, (5,)),
    ], n_branches=2)
    assert [tuple(branch.x_t[0, 1:].tolist()) for branch in ranked] == [(4,), (5,)]
    assert math.isclose(ranked[0].log_mass, math.log(0.6), abs_tol=1e-12)
    assert ranked[0].seed == 1
    assert ranked[0].path_log_p == -2.0
    assert ranked[0].weight == 2.0


@pytest.mark.parametrize(
    ("keys", "masses", "paths", "seeds", "bonus"),
    [
        (((4,), (4,)), (-1.0, -2.0), (-5.0, -3.0), (8, 2), 0.5),
        (((4,), (5,)), (-1.0, -1.0), (-5.0, -3.0), (8, 2), 0.5),
        (((5,), (6,)), (-1.0, -1.0), (-5.0, -3.0), (8, 2), 0.5),
    ],
)
def test_k1_m2_fast_selection_matches_general_merge(
    keys, masses, paths, seeds, bonus,
):
    from edit_flows.sampling.euler_beam import (
        _merge_state_candidates,
        _select_k1_m2_children,
    )

    def build():
        return [
            _BranchState(
                torch.tensor([[BOS_TOKEN, key[0]]]),
                log_mass=mass,
                path_log_p=path,
                seed=seed,
                state_key=key,
            )
            for key, mass, path, seed in zip(keys, masses, paths, seeds)
        ]

    general_candidates = build()
    expected = _merge_state_candidates(
        list(zip(general_candidates, keys)),
        n_branches=1,
        origin_key=(4,),
        changed_state_bonus=bonus,
    )[0]
    actual = _select_k1_m2_children(
        build(), list(keys), origin_key=(4,), changed_state_bonus=bonus,
    )

    assert torch.equal(actual.x_t, expected.x_t)
    for attribute in (
        "weight", "path_log_p", "log_mass", "t", "seed", "state_key",
    ):
        assert getattr(actual, attribute) == getattr(expected, attribute)


def test_state_merge_is_order_independent():
    from edit_flows.sampling.euler_beam import _merge_state_candidates

    specs = [
        ((4,), math.log(0.2), -4.0, 8),
        ((5,), math.log(0.3), -3.0, 4),
        ((4,), math.log(0.1), -2.0, 2),
        ((6,), math.log(0.4), -5.0, 6),
    ]

    def build(order):
        return [
            (
                _BranchState(
                    torch.tensor([[BOS_TOKEN, key[0]]]),
                    log_mass=mass, path_log_p=path, seed=seed,
                ),
                key,
            )
            for key, mass, path, seed in (specs[i] for i in order)
        ]

    forward = _merge_state_candidates(build(range(4)), n_branches=3)
    reverse = _merge_state_candidates(build(reversed(range(4))), n_branches=3)
    forward_summary = [
        (branch.x_t[0, 1].item(), branch.log_mass, branch.path_log_p, branch.seed)
        for branch in forward
    ]
    reverse_summary = [
        (branch.x_t[0, 1].item(), branch.log_mass, branch.path_log_p, branch.seed)
        for branch in reverse
    ]
    assert [row[0] for row in forward_summary] == [row[0] for row in reverse_summary]
    for left, right in zip(forward_summary, reverse_summary):
        assert left[0] == right[0]
        assert math.isclose(left[1], right[1], abs_tol=1e-12)
        assert left[2:] == right[2:]


def test_changed_state_bonus_favors_changed_state_without_altering_mass():
    from edit_flows.sampling.euler_beam import _merge_state_candidates

    unchanged = _BranchState(
        torch.tensor([[BOS_TOKEN, 4]]), log_mass=math.log(0.6), seed=1,
    )
    changed = _BranchState(
        torch.tensor([[BOS_TOKEN, 5]]), log_mass=math.log(0.4), seed=2,
    )
    no_bonus = _merge_state_candidates(
        [(unchanged.clone(), (4,)), (changed.clone(), (5,))],
        n_branches=2, origin_key=(4,), changed_state_bonus=0.0,
    )
    with_bonus = _merge_state_candidates(
        [(unchanged.clone(), (4,)), (changed.clone(), (5,))],
        n_branches=2, origin_key=(4,), changed_state_bonus=0.5,
    )
    assert no_bonus[0].seed == 1
    assert with_bonus[0].seed == 2
    assert math.isclose(with_bonus[0].log_mass, math.log(0.4))


def test_m1_default_matches_explicit_and_m4_keeps_parent_forward_batch(monkeypatch):
    import edit_flows.sampling.euler_beam as beam_module

    class CountingModel(_StochasticModel):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, tokens, time_step, padding_mask, origin_mask=None):
            self.batch_sizes.append(tokens.shape[0])
            return super().forward(tokens, time_step, padding_mask, origin_mask)

    x_0 = torch.tensor([[BOS_TOKEN, 4, 5, PAD_TOKEN]])
    common = dict(
        scheduler=LinearScheduler(), n_branches=3, n_steps=2,
        max_seq_len=32, base_seed=77,
    )
    default = sample_euler_beam(_StochasticModel(), x_0, **common)
    explicit = sample_euler_beam(
        _StochasticModel(), x_0, n_children=1, **common,
    )
    assert torch.equal(default, explicit)

    sampled_batch_sizes = []
    original_sampler = beam_module._sample_actions_per_branch

    def recording_sampler(branch_seeds, x_t, *args, **kwargs):
        sampled_batch_sizes.append(x_t.shape[0])
        return original_sampler(branch_seeds, x_t, *args, **kwargs)

    monkeypatch.setattr(
        beam_module, "_sample_actions_per_branch", recording_sampler,
    )
    model = CountingModel()
    sample_euler_beam(model, x_0, n_children=4, **common)
    assert model.batch_sizes[0] == 3
    assert sampled_batch_sizes[0] == 3 * 4
    assert all(size <= 3 for size in model.batch_sizes)


def test_stateless_uniform_is_reproducible_distinct_and_order_independent():
    from edit_flows.sampling.euler_beam import _stateless_uniform

    seeds = torch.tensor([101, 202, 303], dtype=torch.int64)
    forward = _stateless_uniform(seeds, step=7, seq_len=64, stream=2, dtype=torch.float32)
    repeat = _stateless_uniform(seeds, step=7, seq_len=64, stream=2, dtype=torch.float32)
    reverse = _stateless_uniform(seeds.flip(0), step=7, seq_len=64, stream=2, dtype=torch.float32)
    other_stream = _stateless_uniform(seeds, step=7, seq_len=64, stream=3, dtype=torch.float32)

    assert torch.equal(forward, repeat)
    assert torch.equal(forward, reverse.flip(0))
    assert not torch.equal(forward[0], forward[1])
    assert not torch.equal(forward, other_stream)
    assert bool(((forward > 0) & (forward < 1)).all())


def test_stateless_uniform_has_reasonable_first_moments():
    from edit_flows.sampling.euler_beam import _stateless_uniform

    seeds = torch.arange(1, 20001, dtype=torch.int64)
    values = _stateless_uniform(
        seeds, step=11, seq_len=4, stream=0, dtype=torch.float64,
    )
    assert abs(values.mean().item() - 0.5) < 0.01
    assert abs(values.var().item() - (1.0 / 12.0)) < 0.005


def test_batch_step_log_p_matches_single_branch_wrapper():
    from edit_flows.sampling.euler_beam import _step_log_p, _step_log_p_batch

    torch.manual_seed(19)
    batch, length, vocab_size = 4, 9, 13
    log_rates = torch.randn(batch, length, 3)
    log_ins = torch.log_softmax(torch.randn(batch, length, vocab_size), dim=-1)
    log_sub = torch.log_softmax(torch.randn(batch, length, vocab_size), dim=-1)
    actions = _random_actions(length, vocab_size)
    actions = {key: value.expand(batch, -1).clone() for key, value in actions.items()}
    h = torch.tensor([[0.01], [0.03], [0.07], [0.11]])
    batch_result = _step_log_p_batch(actions, log_rates, log_ins, log_sub, h)
    scalar_result = torch.tensor([
        _step_log_p(
            {key: value[i:i + 1] for key, value in actions.items()},
            log_rates[i:i + 1], log_ins[i:i + 1], log_sub[i:i + 1],
            float(h[i, 0]),
        )
        for i in range(batch)
    ])
    assert torch.allclose(batch_result, scalar_result, rtol=1e-6, atol=1e-6)


def test_batch_apply_edits_matches_individual_application():
    from edit_flows.sampling.euler_beam import _apply_edits_batch

    x = torch.tensor([
        [1, 4, 5, 6, 0],
        [1, 7, 8, 0, 0],
        [1, 9, 10, 11, 12],
    ])
    actions = {
        "ins_mask": torch.tensor([
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, False, False],
        ]),
        "del_mask": torch.tensor([
            [False, False, True, False, False],
            [False, True, False, False, False],
            [False, False, False, True, False],
        ]),
        "sub_mask": torch.tensor([
            [False, False, False, True, False],
            [False, False, False, False, False],
            [False, True, False, False, False],
        ]),
        "ins_tokens": torch.tensor([
            [0, 13, 0, 0, 0],
            [0, 0, 14, 0, 0],
            [0, 0, 0, 0, 0],
        ]),
        "sub_tokens": torch.tensor([
            [0, 0, 0, 15, 0],
            [0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0],
        ]),
    }
    batch_result = _apply_edits_batch(x, actions, max_seq_len=32, pad_token=0)
    rows = []
    for i in range(x.shape[0]):
        row_actions = {key: value[i:i + 1] for key, value in actions.items()}
        rows.append(_apply_edits_batch(
            x[i:i + 1], row_actions, max_seq_len=32, pad_token=0,
        ))
    max_len = max(row.shape[1] for row in rows)
    expected = torch.zeros(len(rows), max_len, dtype=x.dtype)
    for i, row in enumerate(rows):
        expected[i, :row.shape[1]] = row
    assert torch.equal(batch_result, expected)


@pytest.mark.parametrize("n_children", [1, 4])
def test_sample_seeds_match_individual_runs_and_validate_length(n_children):
    model = _StochasticModel()
    x_single = torch.tensor([[BOS_TOKEN, 4, 5, 6, PAD_TOKEN]])
    x_batch = x_single.repeat(3, 1)
    seeds = [42, 1042, 2042]
    common = dict(
        scheduler=LinearScheduler(), n_branches=3, n_children=n_children,
        n_steps=4, max_seq_len=32,
    )
    batched = sample_euler_beam(
        model, x_batch, sample_seeds=seeds, **common,
    )
    individual_rows = [
        sample_euler_beam(model, x_single, base_seed=seed, **common)
        for seed in seeds
    ]
    max_len = max(row.shape[1] for row in individual_rows)
    individual = torch.full(
        (len(seeds) * common["n_branches"], max_len),
        PAD_TOKEN, dtype=torch.long,
    )
    output_index = 0
    for run_rows in individual_rows:
        for row in run_rows:
            individual[output_index, :row.shape[0]] = row
            output_index += 1
    assert torch.equal(batched, individual)

    with pytest.raises(ValueError, match="sample_seeds length"):
        sample_euler_beam(model, x_batch, sample_seeds=[42], **common)


def test_make_batch_is_product_major():
    from scripts.sample_retro import _make_batch

    batch = _make_batch([[4, 5], [7]], n_samples=3, pad_token=PAD_TOKEN)
    assert torch.equal(batch[0], batch[1])
    assert torch.equal(batch[1], batch[2])
    assert torch.equal(batch[3], batch[4])
    assert torch.equal(batch[4], batch[5])
    assert not torch.equal(batch[2], batch[3])


def test_euler_beam_output_rows_are_branch_rank_major_then_run_minor():
    from scripts.sample_retro import _euler_beam_output_row_indices

    assert _euler_beam_output_row_indices(
        product_index=0, n_runs=3, n_branches=3,
    ) == [0, 3, 6, 1, 4, 7, 2, 5, 8]
    assert _euler_beam_output_row_indices(
        product_index=1, n_runs=2, n_branches=3,
    ) == [6, 9, 7, 10, 8, 11]


def test_cli_sample_seeds_are_independent_of_batching_and_branch_count():
    from scripts.sample_retro import _make_euler_beam_sample_seeds

    whole = _make_euler_beam_sample_seeds(42, 0, 7, 3)
    split = (
        _make_euler_beam_sample_seeds(42, 0, 4, 3)
        + _make_euler_beam_sample_seeds(42, 4, 3, 3)
    )
    assert whole == split
    assert len(set(whole)) == len(whole)


def test_grouped_branch_seeds_recreate_three_run_three_branch_streams():
    from scripts.sample_retro import (
        _make_euler_beam_sample_seeds,
        _make_grouped_euler_beam_branch_seeds,
    )

    grouped = _make_grouped_euler_beam_branch_seeds(
        42, 20, n_products=2, n_seed_groups=3, branches_per_group=3,
    )
    run_seeds = _make_euler_beam_sample_seeds(42, 20, 2, 3)
    expected = [
        [seed + branch for seed in run_seeds[start:start + 3]
         for branch in range(3)]
        for start in (0, 3)
    ]
    assert grouped == expected


def test_all_final_branches_are_returned_with_fixed_layout():
    model = _StochasticModel()
    x_0 = torch.tensor([[BOS_TOKEN, 4, 5, 6, PAD_TOKEN]])
    common = dict(
        scheduler=LinearScheduler(), n_branches=5, n_children=2,
        n_steps=4, max_seq_len=32, base_seed=91,
    )
    stats = {}
    all_branches = sample_euler_beam(
        model, x_0, sampling_stats=stats, **common,
    )

    assert all_branches.shape[0] == common["n_branches"]
    assert stats["final_branch_shortfall_samples"] == 0
    assert stats["final_branch_shortfall_outputs"] == 0
    assert stats["parent_branch_evaluations"] > 0
    assert stats["child_candidate_evaluations"] == (
        2 * stats["parent_branch_evaluations"]
    )


def test_first_edit_diversity_keeps_fixed_output_budget_and_records_signatures():
    model = _StochasticModel()
    x_0 = torch.tensor([[BOS_TOKEN, 4, 5, 6, PAD_TOKEN]])
    stats = {}
    result = sample_euler_beam(
        model,
        x_0,
        LinearScheduler(),
        n_branches=3,
        n_children=2,
        n_steps=4,
        max_seq_len=32,
        base_seed=91,
        first_edit_diversity=True,
        sampling_stats=stats,
    )

    assert result.shape[0] == 3
    assert stats["first_edit_signature_groups"] > 0
    assert stats["first_edit_diverse_slots"] > 0
    assert stats["child_candidate_evaluations"] == (
        2 * stats["parent_branch_evaluations"]
    )


def test_first_edit_diversity_coordinates_r9_style_run_group():
    model = _StochasticModel()
    x_0 = torch.tensor([
        [BOS_TOKEN, 4, 5, PAD_TOKEN],
        [BOS_TOKEN, 4, 5, PAD_TOKEN],
        [BOS_TOKEN, 4, 5, PAD_TOKEN],
    ])
    stats = {}
    result = sample_euler_beam(
        model,
        x_0,
        LinearScheduler(),
        n_branches=1,
        n_children=2,
        n_steps=4,
        max_seq_len=32,
        base_seed=91,
        profile_sample_group_size=3,
        first_edit_diversity=True,
        sampling_stats=stats,
    )

    assert result.shape[0] == 3
    assert stats["first_edit_signature_groups"] > 0
    assert stats["first_edit_diverse_slots"] == 3 * stats["steps"]
    assert stats["child_candidate_evaluations"] == (
        2 * stats["parent_branch_evaluations"]
    )


def test_final_branch_shortfall_is_duplicate_filled_and_recorded(monkeypatch):
    import edit_flows.sampling.euler_beam as beam_module

    def no_op_sampler(branch_seeds, x_t, *args, **kwargs):
        shape = x_t.shape
        return {
            "ins_mask": torch.zeros(shape, dtype=torch.bool),
            "sub_mask": torch.zeros(shape, dtype=torch.bool),
            "del_mask": torch.zeros(shape, dtype=torch.bool),
            "ins_tokens": torch.zeros(shape, dtype=torch.long),
            "sub_tokens": torch.zeros(shape, dtype=torch.long),
        }

    monkeypatch.setattr(
        beam_module, "_sample_actions_per_branch", no_op_sampler,
    )
    stats = {}
    result = sample_euler_beam(
        _StochasticModel(),
        torch.tensor([[BOS_TOKEN, 4, PAD_TOKEN]]),
        LinearScheduler(),
        n_branches=3,
        n_children=2,
        n_steps=2,
        sampling_stats=stats,
    )

    assert result.shape[0] == 3
    assert torch.equal(result[0], result[1])
    assert torch.equal(result[1], result[2])
    assert stats["final_branch_shortfall_samples"] == 1
    assert stats["final_branch_shortfall_outputs"] == 2

def test_initial_branch_seed_layout_is_validated():
    model = _StochasticModel()
    x_0 = torch.tensor([[BOS_TOKEN, 4, PAD_TOKEN]])
    with pytest.raises(ValueError, match="mutually exclusive"):
        sample_euler_beam(
            model, x_0, LinearScheduler(), n_branches=2,
            sample_seeds=[7], initial_branch_seeds=[[7, 8]],
        )
    with pytest.raises(ValueError, match="row must contain"):
        sample_euler_beam(
            model, x_0, LinearScheduler(), n_branches=2,
            initial_branch_seeds=[[7]],
        )


def test_legacy_score_mode_matches_triggered_only_reference():
    from edit_flows.sampling.euler_beam import (
        _legacy_branch_sort_key,
        _step_log_p_batch,
    )

    torch.manual_seed(29)
    batch, length, vocab_size = 3, 11, 15
    log_rates = torch.randn(batch, length, 3)
    log_ins = torch.log_softmax(torch.randn(batch, length, vocab_size), dim=-1)
    log_sub = torch.log_softmax(torch.randn(batch, length, vocab_size), dim=-1)
    actions = _random_actions(length, vocab_size)
    actions = {key: value.expand(batch, -1).clone() for key, value in actions.items()}
    h = torch.tensor([[0.01], [0.05], [0.1]])
    actual = _step_log_p_batch(
        actions, log_rates, log_ins, log_sub, h,
        score_mode="legacy_triggered_reverse",
    )
    expected = torch.tensor([
        _scalar_triggered_only(
            {key: value[i:i + 1] for key, value in actions.items()},
            log_rates[i:i + 1], log_ins[i:i + 1], log_sub[i:i + 1],
            float(h[i, 0]),
        )
        for i in range(batch)
    ])
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)

    a = _BranchState(torch.tensor([[BOS_TOKEN, 4]]), path_log_p=-2.0)
    b = _BranchState(torch.tensor([[BOS_TOKEN, 5]]), path_log_p=-10.0)
    assert max([a, b], key=_legacy_branch_sort_key) is b
