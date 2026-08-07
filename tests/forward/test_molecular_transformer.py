from __future__ import annotations

from pathlib import Path

import pytest
import torch

from edit_flows.forward import (
    load_molecular_transformer,
    positive_forward_reward,
    smi_tokenize,
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
