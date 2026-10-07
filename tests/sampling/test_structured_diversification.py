import torch
import torch.nn as nn

from edit_flows.core.scheduler import LinearScheduler
from edit_flows.sampling.structured_diversification import (
    sample_structured_diversification,
)
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


class DirectionModel(nn.Module):
    """Make the first three directions at position 1 unambiguous."""

    def __init__(self, vocab_size=16):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, tokens, time_step, padding_mask, origin_mask=None):
        batch, length = tokens.shape
        log_rates = torch.full(
            (batch, length, 3), -20.0, device=tokens.device,
        )
        log_rates[:, 1, 0] = 3.0  # INS
        log_rates[:, 1, 1] = 2.0  # SUB
        log_rates[:, 1, 2] = 1.0  # DEL
        log_rates = log_rates.masked_fill(
            padding_mask.unsqueeze(-1), -20.0,
        )
        logits = torch.full(
            (batch, length, self.vocab_size), -10.0, device=tokens.device,
        )
        logits[:, :, 7] = 2.0
        logits[:, :, 8] = 1.0
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_rates, log_probs, log_probs.clone()


class BosAnchorModel(nn.Module):
    """Expose only a leading insertion, which must be anchored at BOS."""

    def __init__(self, vocab_size=16):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, tokens, time_step, padding_mask, origin_mask=None):
        batch, length = tokens.shape
        log_rates = torch.full(
            (batch, length, 3), -20.0, device=tokens.device,
        )
        log_rates[:, 0, 0] = 12.0  # INS immediately after BOS
        logits = torch.full(
            (batch, length, self.vocab_size), -10.0, device=tokens.device,
        )
        logits[:, :, 7] = 5.0
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_rates, log_probs, log_probs.clone()


def test_structured_sampler_selects_distinct_directions_without_competition():
    model = DirectionModel()
    x_0 = torch.tensor([[BOS_TOKEN, 4, 5, 6, PAD_TOKEN]])
    stats = {}
    records = []

    results, returned_records = sample_structured_diversification(
        model,
        x_0,
        LinearScheduler(),
        n_trajectories=3,
        n_steps=1,
        max_seq_len=32,
        action_records=records,
        sampling_stats=stats,
    )

    assert results.shape[0] == 3
    assert len(returned_records) == 1
    assert records == returned_records
    selected = returned_records[0]["selected_actions"]
    assert [item["trajectory"] for item in selected] == [1, 2, 3]
    assert [
        (item["position"], item["operation"])
        for item in selected
    ] == [(1, "INS"), (1, "SUB"), (1, "DEL")]
    assert returned_records[0]["unique_direction_count"] == 3
    assert returned_records[0]["direction_duplicate_rate"] == 0.0
    assert returned_records[0]["final_unique_count"] == 3
    assert stats["trajectory_count"] == 3
    assert stats["final_unique_candidates"] == 3

    final_keys = {
        tuple(row[row != PAD_TOKEN].tolist()) for row in results
    }
    assert len(final_keys) == 3


def test_structured_sampler_uses_concrete_token_fallback():
    model = DirectionModel()
    x_0 = torch.tensor([[BOS_TOKEN, 4, PAD_TOKEN]])

    results, records = sample_structured_diversification(
        model,
        x_0,
        LinearScheduler(),
        n_trajectories=5,
        n_steps=1,
        max_seq_len=32,
    )

    assert results.shape[0] == 5
    assert len(records[0]["selected_actions"]) == 5
    assert records[0]["unique_action_count"] == 5
    assert records[0]["action_duplicate_rate"] == 0.0
    assert any(
        item["selection_mode"] == "concrete_fallback"
        for item in records[0]["selected_actions"]
    )


def test_structured_sampler_uses_bos_as_leading_insert_anchor():
    model = BosAnchorModel()
    x_0 = torch.tensor([[BOS_TOKEN, 4, PAD_TOKEN]])

    results, records = sample_structured_diversification(
        model,
        x_0,
        LinearScheduler(),
        n_trajectories=1,
        n_steps=1,
        max_seq_len=32,
    )

    action = records[0]["selected_actions"][0]
    assert (action["position"], action["operation"], action["token"]) == (
        0, "INS", 7,
    )
    assert results[0].tolist() == [BOS_TOKEN, 7, 4]
