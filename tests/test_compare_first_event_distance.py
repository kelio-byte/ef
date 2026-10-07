from scripts.compare_first_event_distance import _assert_comparable, _build_markdown


def _summary(condition: str, multiplier: float, closer: float) -> dict:
    return {
        "schema_version": 1,
        "condition": condition,
        "protocol": {
            "sampler": "R9K1M2",
            "n_runs": 9,
            "n_branches": 1,
            "n_children": 2,
            "score_mode": "full_probability",
            "child_policy": "stochastic_noop",
            "changed_state_bonus": 0.5,
            "q_temperature": 1.0,
            "n_steps": 100,
            "seed": 42,
            "scheduler": "cubic",
            "max_multiplier": multiplier,
        },
        "input": {
            "products_sha256": "product",
            "targets_sha256": "target",
            "sidecar_scores_sha256": "sidecar",
            "checkpoint_sha256": "checkpoint",
            "selection_start_product": 0,
            "selection_end_product_exclusive": 20,
        },
        "trajectory_counts": {
            "expected": 180,
            "first_event_percent": 97.0,
        },
        "first_event_distance": {
            "closer_percent": closer,
            "unchanged_percent": 15.0,
            "farther_percent": 100.0 - closer - 15.0,
            "mean_distance_improvement": 0.5,
        },
    }


def test_comparison_accepts_same_protocol_and_renders_markdown():
    baseline = _summary("b0_trace", 1.0, 65.0)
    candidate = _summary("b1_oracle", 3.0, 67.0)
    _assert_comparable(baseline, candidate)
    output = {
        "baseline": baseline,
        "candidate": candidate,
        "delta_candidate_minus_baseline": {
            "closer_percent_pp": 2.0,
            "unchanged_percent_pp": 0.0,
            "farther_percent_pp": -2.0,
            "mean_distance_improvement": 0.1,
        },
        "first_event_percent_delta_pp": 0.0,
    }
    markdown = _build_markdown(output)
    assert "67.000%" in markdown
    assert "+2.000 pp" in markdown
