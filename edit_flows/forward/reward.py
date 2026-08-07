"""Batch forward-consistency rewards for DGM experiments."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence

import torch
from torch import Tensor

from .molecular_transformer import (
    MolecularTransformerScorer,
    retro_global_to_smiles,
    smi_tokenize,
)


def forward_log_likelihood_reward(
    scorer: MolecularTransformerScorer,
    reactants_global: Sequence[str],
    products_global: Sequence[str],
    *,
    batch_size: int = 64,
    cache: MutableMapping[tuple[str, str], float] | None = None,
    invalid_score: float = -20.0,
) -> Tensor:
    """Score Edit Flows candidates with a frozen forward model.

    Inputs are space-tokenized ``#global#`` strings.  The return value is the
    length-normalized teacher-forced log-likelihood of the product given the
    candidate reactants/reagents; larger is better.  Unlike an RDKit validity
    check, syntactically malformed candidates are not discarded before the
    forward model sees them: the model's tokenizer/UNK path supplies a finite
    low score whenever the token sequence is representable.

    ``cache`` is keyed by the normalized ordinary `(reactants, product)` pair,
    so augmentation duplicates can be scored once.  The cache is caller-owned
    to make its size and hit rate auditable.
    """

    if len(reactants_global) != len(products_global):
        raise ValueError("reactants_global and products_global must have equal length")
    if not torch.isfinite(torch.tensor(float(invalid_score))):
        raise ValueError("invalid_score must be finite")
    normalized: list[tuple[str, str] | None] = []
    scores = torch.full(
        (len(reactants_global),), float(invalid_score), dtype=torch.float32
    )
    pending: dict[tuple[str, str], list[int]] = {}
    for index, (reactant, product) in enumerate(zip(reactants_global, products_global)):
        try:
            key = (retro_global_to_smiles(reactant), retro_global_to_smiles(product))
            # Validate with the same tokenizer used by the scorer.  This keeps
            # one malformed terminal state from aborting an entire batch.
            if not key[0] or not key[1]:
                raise ValueError("empty normalized SMILES")
            smi_tokenize(key[0])
            smi_tokenize(key[1])
        except (TypeError, ValueError):
            normalized.append(None)
            continue
        normalized.append(key)
        if cache is not None and key in cache:
            scores[index] = float(cache[key])
        else:
            pending.setdefault(key, []).append(index)
    if pending:
        keys = list(pending)
        values = scorer.score_batch(
            [key[0] for key in keys],
            [key[1] for key in keys],
            batch_size=batch_size,
            reduction="mean",
        )
        for key, value in zip(keys, values.tolist()):
            value = float(value)
            if cache is not None:
                cache[key] = value
            for index in pending[key]:
                scores[index] = value
    return scores


def positive_forward_reward(
    log_likelihood: Tensor,
    *,
    temperature: float = 1.0,
    min_log_likelihood: float = -20.0,
) -> Tensor:
    """Map forward log-likelihood to a finite positive DGM reward.

    The exponential is the simplest monotone density-ratio proxy.  It is
    clipped only on the lower side to avoid underflow for malformed candidates;
    all values remain in `(0, 1]` because the normalized log-likelihood is
    non-positive for a valid probability model.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.isfinite(log_likelihood).all():
        raise ValueError("log_likelihood contains non-finite values")
    clipped = log_likelihood.clamp_min(min_log_likelihood)
    return torch.exp(clipped / temperature)


__all__ = ["forward_log_likelihood_reward", "positive_forward_reward"]
