import pytest

from edit_flows.guidance.data import make_guidance_record
from edit_flows.utils.tokens import BOS_TOKEN
from scripts.audit_local_action_credit import summarize_local_action_credit


def _record(sample_index, transition_tokens, reward):
    return make_guidance_record(
        product_tokens=[BOS_TOKEN, 4],
        state_tokens=[BOS_TOKEN, 4],
        terminal_tokens=[BOS_TOKEN, 7],
        transition_tokens=transition_tokens,
        time_step=0.5,
        reward=reward,
        source_index=3,
        sample_index=sample_index,
        time_index=50,
        sample_seed=sample_index + 1,
        coupling_seed=sample_index + 10,
    )


def test_local_credit_audit_requires_distinct_first_actions():
    records = [
        _record(0, [BOS_TOKEN, 5], 1.0),
        _record(1, [BOS_TOKEN, 6], 0.25),
        _record(2, [BOS_TOKEN, 4], 0.0),  # genuine first-step no-op
        _record(3, [BOS_TOKEN, 5], 0.5),  # duplicate first action
    ]
    summary = summarize_local_action_credit(records, vocab_size=16)
    transition = summary["transition_actions"]
    groups = summary["local_credit_groups"]
    by_time = summary["by_time_index"]["50"]

    assert transition["nonempty_row_count"] == 3
    assert transition["action_type_counts"] == {
        "insert": 0,
        "substitute": 3,
        "delete": 0,
    }
    assert groups["structurally_valid_group_count"] == 1
    assert groups["groups_with_two_nonempty_children_and_reward_variation"] == 1
    assert groups["groups_with_two_distinct_actions_and_reward_variation"] == 1
    assert groups["locally_discriminative_group_fraction"] == 1.0
    assert by_time["locally_discriminative_group_count"] == 1


def test_local_credit_audit_rejects_missing_transition_tokens():
    record = make_guidance_record(
        product_tokens=[BOS_TOKEN, 4],
        state_tokens=[BOS_TOKEN, 4],
        terminal_tokens=[BOS_TOKEN, 5],
        time_step=0.5,
        reward=1.0,
        source_index=0,
        sample_index=0,
        time_index=50,
        sample_seed=1,
        coupling_seed=2,
    )
    with pytest.raises(ValueError, match="transition_tokens"):
        summarize_local_action_credit([record], vocab_size=16)


def test_local_credit_audit_validates_atomic_proposal_transition_metadata():
    records = [
        _record(0, [BOS_TOKEN, 4, 9], 1.0),
        _record(1, [BOS_TOKEN, 4, 8], 0.0),
        _record(2, [BOS_TOKEN, 4, 9], 0.5),
        _record(3, [BOS_TOKEN, 4, 8], 0.25),
    ]
    for record in records:
        record.update({
            "proposal_operation": "insert",
            "proposal_position": 1,
            "proposal_token": record["transition_tokens"][-1],
        })
    summary = summarize_local_action_credit(records, vocab_size=16)
    metadata = summary["proposal_metadata"]
    assert metadata["available"] is True
    assert metadata["exact_transition_match_count"] == 4
    assert metadata["mismatch_count"] == 0

    records[0]["transition_tokens"] = [BOS_TOKEN, 4, 7]
    mismatched = summarize_local_action_credit(records, vocab_size=16)
    assert mismatched["proposal_metadata"]["mismatch_count"] == 1
