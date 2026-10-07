import torch
import torch.nn as nn

from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.sampling.center_bias import (
    align_position_scores,
    renormalize_position_biased_log_rates,
)
from edit_flows.sampling.euler import sample_euler
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


def test_center_bias_preserves_each_mode_total_hazard():
    log_rates = torch.log(
        torch.tensor(
            [[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 8.0, 9.0]]]
        )
    )
    scores = torch.tensor(
        [[[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 1.0]]]
    )
    legal = torch.ones_like(log_rates, dtype=torch.bool)
    output, diagnostics = renormalize_position_biased_log_rates(
        log_rates, scores, legal, torch.tensor([True]), max_multiplier=3.0
    )
    assert torch.allclose(
        torch.exp(output).sum(dim=1),
        torch.exp(log_rates).sum(dim=1),
        rtol=1e-6,
        atol=1e-6,
    )
    assert diagnostics["relative_error"].max() < 1e-6


def test_center_bias_gives_score_one_three_times_relative_weight():
    log_rates = torch.zeros(1, 2, 3)
    scores = torch.zeros_like(log_rates)
    scores[:, 0, :] = 1.0
    legal = torch.ones_like(log_rates, dtype=torch.bool)
    output, _ = renormalize_position_biased_log_rates(
        log_rates, scores, legal, torch.tensor([True]), max_multiplier=3.0
    )
    ratio = torch.exp(output[:, 0]) / torch.exp(output[:, 1])
    assert torch.allclose(ratio, torch.full_like(ratio, 3.0))


def test_constant_scores_and_inactive_rows_are_bitwise_unchanged():
    torch.manual_seed(2)
    log_rates = torch.randn(2, 4, 3)
    scores = torch.full_like(log_rates, 0.5)
    scores[1, 0, 0] = 1.0
    legal = torch.ones_like(log_rates, dtype=torch.bool)
    output, _ = renormalize_position_biased_log_rates(
        log_rates, scores, legal, torch.tensor([True, False])
    )
    assert torch.equal(output, log_rates)


def test_multiplier_one_is_bitwise_neutral_with_variable_scores():
    torch.manual_seed(3)
    log_rates = torch.randn(2, 4, 3)
    scores = torch.rand_like(log_rates)
    legal = torch.rand_like(log_rates) > 0.25
    output, diagnostics = renormalize_position_biased_log_rates(
        log_rates,
        scores,
        legal,
        torch.tensor([True, True]),
        max_multiplier=1.0,
    )
    assert torch.equal(output, log_rates)
    assert not diagnostics["changed"].any()
    assert torch.equal(
        diagnostics["before_hazard"], diagnostics["after_hazard"]
    )


def test_illegal_positions_do_not_enter_normalization():
    log_rates = torch.log(
        torch.tensor([[[1.0, 1.0, 1.0], [4.0, 4.0, 4.0]]])
    )
    scores = torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]])
    legal = torch.tensor([[[True, True, True], [False, False, False]]])
    output, diagnostics = renormalize_position_biased_log_rates(
        log_rates, scores, legal, torch.tensor([True])
    )
    assert torch.equal(output, log_rates)
    assert torch.equal(diagnostics["before_hazard"], torch.ones(1, 3))


def test_align_position_scores_only_changes_batch_width():
    scores = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    assert torch.equal(align_position_scores(scores, 2), scores[:, :2])
    padded = align_position_scores(scores, 5)
    assert torch.equal(padded[:, :3], scores)
    assert torch.equal(padded[:, 3:], torch.zeros(2, 2, 3))


def test_constant_first_event_scores_match_plain_euler(dummy_model):
    x_0 = torch.tensor([[BOS_TOKEN, 4, 5, PAD_TOKEN]])
    scores = torch.full((1, 4, 3), 0.5)
    torch.manual_seed(812)
    plain, _ = sample_euler(
        dummy_model, x_0, CubicScheduler(), n_steps=5, max_seq_len=32
    )
    torch.manual_seed(812)
    biased, _ = sample_euler(
        dummy_model,
        x_0,
        CubicScheduler(),
        n_steps=5,
        max_seq_len=32,
        first_event_position_scores=scores,
    )
    assert torch.equal(plain, biased)


class _ForcedFirstSubstitution(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, tokens, time_step, padding_mask, origin_mask=None):
        batch, length = tokens.shape
        log_rates = torch.full((batch, length, 3), -30.0)
        log_rates[:, 1, 1] = 20.0
        log_ins = torch.full((batch, length, 16), -1e9)
        log_sub = torch.full_like(log_ins, -1e9)
        log_ins[:, :, 9] = 0.0
        log_sub[:, :, 9] = 0.0
        return log_rates, log_ins, log_sub


def test_first_event_bias_deactivates_each_row_after_its_first_edit():
    model = _ForcedFirstSubstitution()
    x_0 = torch.tensor(
        [
            [BOS_TOKEN, 4, 5, PAD_TOKEN],
            [BOS_TOKEN, 6, 7, PAD_TOKEN],
        ]
    )
    scores = torch.zeros(2, 4, 3)
    scores[:, 1, 1] = 1.0
    stats = {}
    result, _ = sample_euler(
        model,
        x_0,
        LinearScheduler(),
        n_steps=2,
        max_seq_len=16,
        first_event_position_scores=scores,
        first_event_bias_stats=stats,
        first_event_row_metadata=[{"trajectory": 0}, {"trajectory": 1}],
    )
    assert torch.equal(result[:, 1], torch.tensor([9, 9]))
    assert stats["first_event_count"] == 2
    assert stats["no_event_count"] == 0
    assert stats["biased_row_steps"] == 2
    assert stats["guided_row_steps"] == 2
    assert len(stats["records"]) == 2
    assert all(record["action_count"] == 1 for record in stats["records"])
    assert all(
        record["actions"][0]["center_score"] == 1.0
        for record in stats["records"]
    )
    assert all(
        record["actions"][0]["token_id"] == 9
        for record in stats["records"]
    )
    assert stats["max_hazard_relative_error"] < 1e-6


def test_first_event_bias_can_leave_some_rows_as_ordinary_euler():
    model = _ForcedFirstSubstitution()
    x_0 = torch.tensor(
        [
            [BOS_TOKEN, 4, 5, PAD_TOKEN],
            [BOS_TOKEN, 6, 7, PAD_TOKEN],
        ]
    )
    scores = torch.zeros(2, 4, 3)
    scores[:, 1, 1] = 1.0
    stats = {}
    result, _ = sample_euler(
        model,
        x_0,
        LinearScheduler(),
        n_steps=2,
        max_seq_len=16,
        first_event_position_scores=scores,
        first_event_position_bias_enabled=torch.tensor([True, False]),
        first_event_bias_stats=stats,
        first_event_row_metadata=[
            {"trajectory_role": "center_guided"},
            {"trajectory_role": "ordinary_euler"},
        ],
    )
    assert torch.equal(result[:, 1], torch.tensor([9, 9]))
    assert stats["first_event_count"] == 2
    assert stats["biased_row_steps"] == 2
    assert stats["guided_row_steps"] == 1
    assert [record["position_bias_enabled"] for record in stats["records"]] == [
        True, False,
    ]
    assert stats["records"][1]["position_bias_reweighted"] is False
