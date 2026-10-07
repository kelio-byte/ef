from __future__ import annotations

import torch

from edit_flows.analysis.trajectory_correction import (
    action_is_oracle_consistent,
    classify_event_progress,
    canonicalize_global_smiles,
    event_is_fully_oracle_consistent,
)


def test_event_progress_is_order_invariant_and_not_binary_wrong():
    event = {"actions": [{"type": "ins", "position": 1, "token": 4}]}
    assert classify_event_progress(
        event,
        {"n_actions": 1, "edit_distance_before": 5, "edit_distance_after": 4},
    )["category"] == "full_progress"
    assert classify_event_progress(
        event,
        {"n_actions": 1, "edit_distance_before": 5, "edit_distance_after": 5},
    )["category"] == "neutral"
    assert classify_event_progress(
        event,
        {"n_actions": 1, "edit_distance_before": 5, "edit_distance_after": 6},
    )["category"] == "harmful"
    assert classify_event_progress(
        {"actions": [{}, {}]},
        {"n_actions": 2, "edit_distance_before": 5, "edit_distance_after": 4},
    )["category"] == "partial_progress"
from edit_flows.core.scheduler import CubicScheduler
from edit_flows.sampling.euler import sample_euler
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


def test_token_support_is_required_for_oracle_consistency() -> None:
    oracle_row = {
        "position": 2,
        "types": ["ins"],
        "ins_token_support": [17],
        "sub_token_support": [],
    }
    assert action_is_oracle_consistent(
        {"position": 2, "type": "ins", "token": 17}, oracle_row,
    )
    assert not action_is_oracle_consistent(
        {"position": 2, "type": "ins", "token": 18}, oracle_row,
    )
    assert not action_is_oracle_consistent(
        {"position": 1, "type": "ins", "token": 17}, oracle_row,
    )


def test_full_event_requires_all_actions_to_match() -> None:
    event = {
        "oracle_available": True,
        "oracle": [
            {"position": 1, "types": ["ins"], "ins_token_support": [8]},
            {"position": 2, "types": ["sub"], "sub_token_support": [9]},
        ],
        "actions": [
            {"position": 1, "type": "ins", "token": 8},
            {"position": 2, "type": "sub", "token": 10},
        ],
    }
    assert not event_is_fully_oracle_consistent(event)
    event["actions"][1]["token"] = 9
    assert event_is_fully_oracle_consistent(event)


def test_canonicalizer_collapses_equivalent_smiles() -> None:
    assert canonicalize_global_smiles("C(C)O") == canonicalize_global_smiles("CCO")


def test_compact_recording_does_not_change_sampling(dummy_model) -> None:
    x_0 = torch.tensor([[BOS_TOKEN, 3, 4, PAD_TOKEN]])
    scheduler = CubicScheduler()
    torch.manual_seed(12345)
    plain, _ = sample_euler(
        dummy_model, x_0, scheduler, n_steps=5, max_seq_len=32,
    )
    torch.manual_seed(12345)
    compact, _, events = sample_euler(
        dummy_model,
        x_0,
        scheduler,
        n_steps=5,
        max_seq_len=32,
        record_compact_events=True,
    )
    assert torch.equal(plain, compact)
    assert len(events) == 1
    for event in events[0]:
        assert "actions" in event
        assert "x_t" in event
        assert "x_next" in event
