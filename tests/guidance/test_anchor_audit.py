from scripts.audit_guidance_anchors import summarize_anchor_sharing


def _record(source_index, state_tokens, time):
    return {
        "source_index": source_index,
        "state_tokens": state_tokens,
        "time": time,
    }


def test_anchor_audit_detects_non_shared_states_and_times():
    records = [
        _record(0, [1, 2], 0.2),
        _record(0, [1, 2], 0.4),
        _record(0, [1, 3], 0.4),
        _record(0, [1, 2], 0.8),
        _record(1, [1], 0.5),
        _record(1, [1], 0.5),
    ]
    summary = summarize_anchor_sharing(records)
    assert summary["record_count"] == 6
    assert summary["group_count"] == 2
    assert summary["group_size_distribution"] == {"2": 1, "4": 1}
    assert summary["groups_all_states_equal"] == 1
    assert summary["groups_all_times_equal"] == 1
    assert summary["same_time_pair_count"] == 2
    assert summary["same_time_state_equal_pair_count"] == 1
    assert summary["same_time_state_equal_fraction"] == 0.5


def test_anchor_audit_handles_no_same_time_pairs():
    summary = summarize_anchor_sharing([
        _record(0, [1], 0.1),
        _record(0, [2], 0.2),
    ])
    assert summary["same_time_pair_count"] == 0
    assert summary["same_time_state_equal_fraction"] is None
