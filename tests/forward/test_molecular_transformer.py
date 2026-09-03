from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from edit_flows.forward import (
    MolecularTransformerScorer,
    forward_beam_reconstruction_rank,
    forward_beam_reconstruction_reward,
    forward_log_likelihood_reward,
    load_molecular_transformer,
    positive_forward_reward,
    smi_tokenize,
)


class _ToyForwardEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, source):
        length, batch = source.shape[:2]
        memory = torch.zeros(length, batch, self.hidden_dim, device=source.device)
        return memory, memory


class _ToyForwardDecoder(nn.Module):
    def __init__(self, vocab_size: int, bos_id: int, eos_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id

    def forward(self, target, memory, source):
        length, batch = target.shape[:2]
        output = torch.full(
            (length, batch, self.vocab_size), -20.0, device=target.device,
        )
        last_token = target[-1, :, 0]
        first = last_token.eq(self.bos_id)
        output[-1, first, 4] = 0.0  # C
        output[-1, first, 5] = -0.1  # O
        output[-1, ~first, self.eos_id] = 0.0
        return output


class _ToyForwardModel(nn.Module):
    def __init__(self, vocab_size: int, bos_id: int, eos_id: int):
        super().__init__()
        self.encoder = _ToyForwardEncoder(vocab_size)
        self.decoder = _ToyForwardDecoder(vocab_size, bos_id, eos_id)


def _toy_scorer() -> MolecularTransformerScorer:
    vocab = ["<blank>", "<unk>", "<s>", "</s>", "C", "O"]
    return MolecularTransformerScorer(
        _ToyForwardModel(len(vocab), bos_id=2, eos_id=3),
        nn.Identity(),
        vocab,
        torch.device("cpu"),
    )


def test_official_smi_tokenizer_round_trip() -> None:
    value = "N[C@@H](C)C(=O)O"
    tokens = smi_tokenize(value)
    assert "[C@@H]" in tokens
    assert "".join(tokens) == value


def test_official_smi_tokenizer_rejects_unmatched_text() -> None:
    with pytest.raises(ValueError):
        smi_tokenize("CC<bad>")


def test_positive_forward_reward_is_finite_and_monotone() -> None:
    values = positive_forward_reward(torch.tensor([-10.0, -2.0, 0.0]))
    assert torch.isfinite(values).all()
    assert torch.all(values[1:] > values[:-1])
    assert torch.all((values > 0) & (values <= 1))


def test_forward_reward_invalid_pair_gets_finite_floor() -> None:
    class FakeScorer:
        def score_batch(self, *_args, **_kwargs):
            raise AssertionError("invalid pair should not reach the model")

    values = forward_log_likelihood_reward(
        FakeScorer(), ["CC<bad>"], ["CCO"]
    )
    assert values.tolist() == [-20.0]


def test_forward_beam_generation_returns_ranked_completed_smiles() -> None:
    predictions, scores = _toy_scorer().generate_batch(
        ["C"], beam_size=2, max_length=4, batch_size=1,
    )
    assert predictions == [["C", "O"]]
    assert scores.shape == (1, 2)
    assert torch.isfinite(scores).all()
    assert scores[0, 0] > scores[0, 1]


def test_forward_beam_reconstruction_rank_and_reward_use_source_cache() -> None:
    class FakeBeamScorer:
        calls = 0

        def generate_batch(self, sources, **_kwargs):
            self.calls += 1
            assert sources == ["CC"]
            return [["CO", "C"]], torch.tensor([[0.0, -1.0]])

    scorer = FakeBeamScorer()
    cache = {}
    stats = {}
    ranks = forward_beam_reconstruction_rank(
        scorer,
        ["C C", "C C"],
        ["C O", "C"],
        beam_size=2,
        cache=cache,
        stats=stats,
    )
    rewards = forward_beam_reconstruction_reward(
        scorer,
        ["C C", "C C"],
        ["C O", "C"],
        beam_size=2,
        cache=cache,
    )
    assert ranks.tolist() == [1, 2]
    assert rewards.tolist() == [1.0, 0.5]
    assert scorer.calls == 1
    assert stats["generated_source_count"] == 1
    assert stats["deduplicated_input_count"] == 1


def test_forward_beam_reconstruction_can_canonicalize_equivalent_sources() -> None:
    class FakeBeamScorer:
        calls = 0

        def generate_batch(self, sources, **_kwargs):
            self.calls += 1
            assert sources == ["CO"]
            return [["CO"]], torch.tensor([[0.0]])

    scorer = FakeBeamScorer()
    ranks = forward_beam_reconstruction_rank(
        scorer,
        ["O C", "C O"],
        ["C O", "C O"],
        beam_size=1,
        canonicalize_source=True,
    )
    assert ranks.tolist() == [1, 1]
    assert scorer.calls == 1


def test_forward_beam_generation_validates_limits() -> None:
    scorer = _toy_scorer()
    with pytest.raises(ValueError, match="min_length"):
        scorer.generate_batch(["C"], max_length=1, min_length=1)


@pytest.mark.skipif(
    not Path("new_checkpoints/MIT_mixed_augm_model_average_20.pt").exists(),
    reason="legacy Molecular Transformer checkpoint is an external experiment asset",
)
def test_legacy_checkpoint_loads_and_scores() -> None:
    scorer = load_molecular_transformer(
        "new_checkpoints/MIT_mixed_augm_model_average_20.pt", device="cpu"
    )
    assert len(scorer.vocab) == 297
    scores = scorer.score_batch(["CCO"], ["CC=O"], batch_size=1)
    assert scores.shape == (1,)
    assert torch.isfinite(scores).all()
