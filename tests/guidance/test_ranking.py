import pytest
import torch

from edit_flows.guidance.ranking import (
    _within_group_pearson,
    score_masked_action_sets,
    shared_anchor_pairwise_loss,
)
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


def _ranking_batch():
    state = torch.tensor([
        [BOS_TOKEN, 4],
        [BOS_TOKEN, 4],
        [BOS_TOKEN, 4],
        [BOS_TOKEN, 4],
    ], dtype=torch.long)
    terminal = torch.tensor([
        [BOS_TOKEN, 5, PAD_TOKEN],
        [BOS_TOKEN, 6, PAD_TOKEN],
        [BOS_TOKEN, 7, PAD_TOKEN],
        [BOS_TOKEN, 4, 8],
    ], dtype=torch.long)
    reward = torch.tensor([1.0, 0.5, 0.25, 0.0])
    source_index = torch.tensor([10, 10, 10, 10], dtype=torch.long)
    guidance_insert = torch.ones(4, 2, 16)
    guidance_substitute = torch.ones(4, 2, 16)
    guidance_substitute[:, 1, 5] = 16.0
    guidance_substitute[:, 1, 6] = 4.0
    guidance_substitute[:, 1, 7] = 2.0
    guidance_insert[:, 1, 8] = 1.0
    guidance_delete = torch.ones(4, 2, 1)
    guidance_insert.requires_grad_()
    guidance_substitute.requires_grad_()
    guidance_delete.requires_grad_()
    return (
        (guidance_insert, guidance_substitute, guidance_delete),
        state,
        terminal,
        source_index,
        reward,
    )


def test_score_masked_action_sets_is_length_normalized():
    guidance = (
        torch.full((2, 1, 3), 4.0),
        torch.full((2, 1, 3), 4.0),
        torch.full((2, 1, 1), 4.0),
    )
    insert_mask = torch.tensor([[[True, False, False]], [[True, True, False]]])
    substitute_mask = torch.zeros_like(insert_mask)
    delete_mask = torch.zeros(2, 1, 1, dtype=torch.bool)
    scores, counts = score_masked_action_sets(
        guidance, insert_mask, substitute_mask, delete_mask,
    )
    assert torch.equal(counts, torch.tensor([1.0, 2.0]))
    assert torch.allclose(scores[0], scores[1])
    assert torch.allclose(scores, torch.full((2,), torch.log(torch.tensor(4.0))))


def test_shared_anchor_pairwise_loss_prefers_higher_reward_action_sets():
    guidance, state, terminal, source_index, reward = _ranking_batch()
    loss, metrics = shared_anchor_pairwise_loss(
        guidance, state, terminal, source_index, reward,
        vocab_size=16, group_size=4,
    )
    assert torch.isfinite(loss)
    assert metrics["candidate_pair_count"].item() == 6
    assert metrics["pair_count"].item() == 6
    assert metrics["pair_accuracy_tie_half"].item() > 0.99
    assert metrics["no_action_candidate_fraction"].item() == 0.0
    assert torch.isfinite(metrics["reward_score_pearson_within_group"])
    loss.backward()
    assert all(value.grad is not None for value in guidance)
    assert all(torch.isfinite(value.grad).all() for value in guidance)


def test_within_group_pearson_removes_group_score_offsets():
    values = torch.tensor([10.0, 11.0, 20.0, 21.0])
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])
    groups = torch.tensor([0, 0, 1, 1])
    assert _within_group_pearson(values, rewards, groups).item() > 0.99


def test_shared_anchor_pairwise_loss_penalizes_reversed_scores():
    guidance, state, terminal, source_index, reward = _ranking_batch()
    bad_insert = guidance[0].detach().clone()
    bad_substitute = guidance[1].detach().clone()
    bad_delete = guidance[2].detach().clone()
    bad_substitute[:, 1, 5] = 1.0
    bad_substitute[:, 1, 6] = 2.0
    bad_substitute[:, 1, 7] = 4.0
    bad_insert[:, 1, 8] = 16.0
    bad_insert.requires_grad_()
    bad_substitute.requires_grad_()
    bad_delete.requires_grad_()
    bad_loss, bad_metrics = shared_anchor_pairwise_loss(
        (bad_insert, bad_substitute, bad_delete),
        state, terminal, source_index, reward, vocab_size=16, group_size=4,
    )
    good_loss, _ = shared_anchor_pairwise_loss(
        guidance, state, terminal, source_index, reward,
        vocab_size=16, group_size=4,
    )
    assert bad_metrics["pair_accuracy_tie_half"].item() < 0.01
    assert bad_loss > good_loss


def test_shared_anchor_pairwise_loss_ignores_equal_rewards_and_handles_no_pair():
    guidance, state, terminal, source_index, reward = _ranking_batch()
    equal_reward = torch.ones_like(reward)
    loss, metrics = shared_anchor_pairwise_loss(
        guidance, state, terminal, source_index, equal_reward,
        vocab_size=16, group_size=4,
    )
    assert loss.item() == 0.0
    assert metrics["pair_count"].item() == 0.0
    loss.backward()
    assert all(value.grad is not None for value in guidance)


def test_shared_anchor_pairwise_loss_skips_zero_action_candidates():
    guidance, state, terminal, source_index, reward = _ranking_batch()
    terminal[0] = torch.tensor([BOS_TOKEN, 4, PAD_TOKEN])
    reward[0] = 1.0
    loss, metrics = shared_anchor_pairwise_loss(
        guidance, state, terminal, source_index, reward,
        vocab_size=16, group_size=4,
    )
    assert torch.isfinite(loss)
    assert metrics["no_action_candidate_fraction"].item() > 0.0
    assert metrics["pair_count"].item() < metrics["candidate_pair_count"].item()


def test_shared_anchor_pairwise_loss_requires_complete_groups():
    guidance, state, terminal, source_index, reward = _ranking_batch()
    with pytest.raises(ValueError, match="complete product groups"):
        shared_anchor_pairwise_loss(
            guidance[:], state[:3], terminal[:3], source_index[:3], reward[:3],
            vocab_size=16, group_size=4,
        )
