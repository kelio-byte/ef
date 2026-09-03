from scripts.diagnose_r9k1m2_first_event_distance import (
    _FirstEventDistanceCollector,
    _token_distance,
)
from edit_flows.utils.tokens import BOS_TOKEN


def test_streaming_collector_replays_and_aggregates_first_event_progress():
    source = [[BOS_TOKEN, 10, 11]]
    target = [[BOS_TOKEN, 10, 12]]
    collector = _FirstEventDistanceCollector(
        source,
        target,
        [_token_distance(source[0], target[0])],
        global_start=40,
        max_seq_len=16,
    )

    collector.consume(
        {
            "first_event_step_idx": 7,
            "position_bias_enabled": True,
            "position_bias_reweighted": True,
            "row_metadata": {"global_input_row": 40},
            "actions": [
                {
                    "mode": "SUB",
                    "position": 2,
                    "token_id": 12,
                    "center_score": 1.0,
                }
            ],
        }
    )

    summary = collector.summary()
    assert summary["closer_count"] == 1
    assert summary["unchanged_count"] == 0
    assert summary["farther_count"] == 0
    assert summary["mean_distance_improvement"] == 1.0
    assert summary["mean_effective_action_count"] == 1.0
    assert summary["mean_first_event_step_index"] == 7.0
    assert summary["guided_event_count"] == 1
    assert summary["reweighted_event_count"] == 1
