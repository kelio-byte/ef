import torch
import torch.nn as nn

from edit_flows.core.scheduler import LinearScheduler
from edit_flows.sampling.structured_diversification_v2 import (
    sample_delayed_structured_diversification,
)
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


class DelayedModeModel(nn.Module):
    """No event before t=.25; three clear modes at the first trigger."""

    def __init__(self, vocab_size=16):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, tokens, time_step, padding_mask, origin_mask=None):
        batch, length = tokens.shape
        log_rates = torch.full(
            (batch, length, 3), -20.0, device=tokens.device,
        )
        active = (time_step[:, 0] >= 0.25) & (time_step[:, 0] < 0.5)
        log_rates[active, 1, 0] = 12.0  # INS anchor
        log_rates[active, 2, 1] = 11.0  # SUB alternative
        log_rates[active, 3, 2] = 10.0  # DEL alternative
        log_rates = log_rates.masked_fill(
            padding_mask.unsqueeze(-1), -20.0,
        )
        logits = torch.full(
            (batch, length, self.vocab_size), -10.0,
            device=tokens.device,
        )
        logits[:, :, 7] = 3.0
        logits[:, :, 8] = 2.0
        logits[:, :, 9] = 1.0
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_rates, log_probs, log_probs.clone()


class DelayedBosAnchorModel(nn.Module):
    """Trigger one certain leading insertion after the initial Euler step."""

    def __init__(self, vocab_size=16):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, tokens, time_step, padding_mask, origin_mask=None):
        batch, length = tokens.shape
        log_rates = torch.full(
            (batch, length, 3), -20.0, device=tokens.device,
        )
        active = (time_step[:, 0] >= 0.25) & (time_step[:, 0] < 0.5)
        log_rates[active, 0, 0] = 30.0  # INS immediately after BOS
        logits = torch.full(
            (batch, length, self.vocab_size), -10.0, device=tokens.device,
        )
        logits[:, :, 7] = 5.0
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_rates, log_probs, log_probs.clone()


def test_delayed_structured_uses_first_event_and_3x3_budget():
    model = DelayedModeModel()
    x_0 = torch.tensor([[BOS_TOKEN, 4, 5, 6, PAD_TOKEN]])
    stats = {}
    records = []

    results, returned = sample_delayed_structured_diversification(
        model,
        x_0,
        LinearScheduler(),
        k_mode=3,
        k_completion=3,
        n_steps=4,
        max_seq_len=32,
        mode_pool_size=3,
        base_seed=123,
        action_records=records,
        sampling_stats=stats,
    )

    assert results.shape[0] == 9
    assert returned == records
    record = returned[0]
    assert record["trigger_t"] == 0.25
    assert record["trigger_event_count"] >= 1
    selected_modes = {
        (item["position"], item["operation"])
        for item in record["selected_actions"][::3]
    }
    assert selected_modes == {(1, "INS"), (2, "SUB"), (3, "DEL")}
    assert record["selected_actions"][0]["mode_selection"] == "anchor_top1"
    assert [
        item["token"] for item in record["selected_actions"][:3]
    ] == [7, 8, 9]
    for offset in (0, 3, 6):
        group = record["selected_actions"][offset:offset + 3]
        if group[0]["operation"] != "DEL":
            assert [item["token"] for item in group] == [7, 8, 9]
        else:
            assert [item["token"] for item in group] == [-1, -1, -1]
    assert len({
        item["continuation_seed"]
        for item in record["selected_actions"]
    }) == 9
    assert stats["products"] == 1
    assert stats["trajectory_count"] == 9
    assert stats["fallback_trigger_count"] == 0
    assert stats["mean_trigger_t"] == 0.25
    assert stats["final_slots"] == 9


def test_delayed_structured_accumulates_stats_across_batch_rows():
    model = DelayedModeModel()
    x_0 = torch.tensor([
        [BOS_TOKEN, 4, 5, 6, PAD_TOKEN],
        [BOS_TOKEN, 6, 5, 4, PAD_TOKEN],
    ])
    stats = {}
    results, records = sample_delayed_structured_diversification(
        model,
        x_0,
        LinearScheduler(),
        k_mode=3,
        k_completion=3,
        n_steps=4,
        max_seq_len=32,
        mode_pool_size=3,
        sampling_stats=stats,
    )
    assert results.shape[0] == 18
    assert len(records) == 2
    assert stats["products"] == 2
    assert stats["trajectory_count"] == 18
    assert stats["final_slots"] == 18
    assert stats["selected_mode_rank_count"] == 6


def test_delayed_structured_uses_bos_as_leading_insert_anchor():
    model = DelayedBosAnchorModel()
    x_0 = torch.tensor([[BOS_TOKEN, 4, PAD_TOKEN]])

    results, records = sample_delayed_structured_diversification(
        model,
        x_0,
        LinearScheduler(),
        k_mode=1,
        k_completion=1,
        n_steps=4,
        max_seq_len=32,
        mode_pool_size=1,
        base_seed=123,
    )

    action = records[0]["selected_actions"][0]
    assert records[0]["trigger_t"] == 0.25
    assert (action["position"], action["operation"], action["token"]) == (
        0, "INS", 7,
    )
    assert results[0].tolist() == [BOS_TOKEN, 7, 4]
