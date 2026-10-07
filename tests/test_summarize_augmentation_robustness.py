from __future__ import annotations

from scripts.summarize_augmentation_robustness import TrajectoryCounts


def test_trajectory_counts_uses_first_event_distance_progress() -> None:
    counts = TrajectoryCounts()
    counts.add({
        "final_hit": True,
        "first_event_index": 0,
        "events": [{
            "n_actions": 1,
            "edit_distance_before": 4,
            "edit_distance_after": 3,
        }],
    })
    counts.add({
        "final_hit": False,
        "first_event_index": 0,
        "events": [{
            "n_actions": 1,
            "edit_distance_before": 4,
            "edit_distance_after": 5,
        }],
    })
    values = counts.values()
    assert values["full_progress_rate"] == 0.5
    assert values["harmful_rate"] == 0.5
    assert values["harmful_recovery_rate"] == 0.0
    assert values["final_hit_rate"] == 0.5
    assert values["mean_distance_delta"] == 0.0
