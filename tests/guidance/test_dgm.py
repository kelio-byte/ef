import pytest
import torch

from edit_flows.guidance.dgm import (
    guided_log_probs,
    positive_guidance,
    positive_guidance_bregman_loss,
)
from edit_flows.guidance.rewards import rdkit_validity_reward


def test_guided_posterior_matches_reward_tilting():
    base = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float32).log()
    reward = torch.tensor([[1.0, 5.0, 0.5]], dtype=torch.float32)
    expected = (base.exp() * reward)
    expected = expected / expected.sum(dim=-1, keepdim=True)

    actual = guided_log_probs(base, reward).exp()
    torch.testing.assert_close(actual, expected)


def test_constant_guidance_preserves_base_distribution():
    base = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float32).log()
    actual = guided_log_probs(base, torch.full_like(base, 7.0)).exp()
    torch.testing.assert_close(actual, base.exp())


def test_positive_guidance_and_bregman_minimum():
    raw = torch.tensor([-1.0, 0.0, 2.0])
    positive = positive_guidance(raw)
    assert torch.all(positive > 0)

    reward = torch.tensor([0.5, 1.0, 3.0])
    prediction = reward.clone().requires_grad_()
    loss = positive_guidance_bregman_loss(prediction, reward)
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(reward), atol=1e-6, rtol=1e-6)


def test_bregman_mask_rejects_empty_selection():
    with pytest.raises(ValueError, match="selects no"):
        positive_guidance_bregman_loss(
            torch.ones(2, 3), torch.ones(2), mask=torch.zeros(2, 3, dtype=torch.bool),
        )


def test_rdkit_validity_reward_does_not_use_targets():
    values = rdkit_validity_reward(["CCO", "not-a-smiles", "", "C.C"])
    torch.testing.assert_close(values, torch.tensor([1.0, 0.0, 0.0, 1.0]))
