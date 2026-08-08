"""Batch forward-consistency rewards for DGM experiments."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence

import torch
from torch import Tensor
from rdkit import Chem

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


def _canonical_smiles_clear_map(smiles: str) -> str:
    """Return an isomeric canonical SMILES without atom-map annotations."""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def forward_beam_reconstruction_rank(
    scorer: MolecularTransformerScorer,
    reactants_global: Sequence[str],
    products_global: Sequence[str],
    *,
    beam_size: int = 5,
    max_length: int = 200,
    min_length: int = 1,
    batch_size: int = 16,
    forbid_unk: bool = False,
    canonicalize_source: bool = False,
    cache: MutableMapping[str, Sequence[str]] | None = None,
) -> Tensor:
    """Return the 1-based forward-beam rank of each requested product.

    A rank of zero means that the product was not reconstructed or that the
    input pair was malformed.  The forward generator depends only on the
    candidate reactants, so ``cache`` is keyed by normalized reactant SMILES
    and stores the generated product beam.  With ``canonicalize_source=True``,
    atom maps are removed and chemically equivalent SMILES share one forward
    input/cache entry.  True retrosynthesis targets are never consumed by this
    function.
    """

    if len(reactants_global) != len(products_global):
        raise ValueError("reactants_global and products_global must have equal length")
    ranks = torch.zeros(len(reactants_global), dtype=torch.long)
    normalized: list[tuple[str, str] | None] = []
    pending: dict[str, list[int]] = {}
    generated: dict[str, Sequence[str]] = {}
    for index, (reactants, product) in enumerate(
        zip(reactants_global, products_global)
    ):
        try:
            source = retro_global_to_smiles(reactants)
            target = _canonical_smiles_clear_map(retro_global_to_smiles(product))
            if canonicalize_source:
                source = _canonical_smiles_clear_map(source)
            if not source or not target:
                raise ValueError("empty normalized reaction side")
            smi_tokenize(source)
        except (TypeError, ValueError):
            normalized.append(None)
            continue
        normalized.append((source, target))
        if cache is not None and source in cache:
            generated[source] = cache[source]
        else:
            pending.setdefault(source, []).append(index)

    if pending:
        sources = list(pending)
        predictions, _ = scorer.generate_batch(
            sources,
            beam_size=beam_size,
            max_length=max_length,
            min_length=min_length,
            batch_size=batch_size,
            forbid_unk=forbid_unk,
        )
        for source, beam in zip(sources, predictions):
            canonical_beam = [
                _canonical_smiles_clear_map(prediction)
                for prediction in beam
            ]
            generated[source] = canonical_beam
            if cache is not None:
                cache[source] = canonical_beam

    for index, pair in enumerate(normalized):
        if pair is None:
            continue
        source, target = pair
        for rank, prediction in enumerate(generated[source], start=1):
            if prediction and prediction == target:
                ranks[index] = rank
                break
    return ranks


def forward_beam_reconstruction_reward(
    scorer: MolecularTransformerScorer,
    reactants_global: Sequence[str],
    products_global: Sequence[str],
    *,
    beam_size: int = 5,
    max_length: int = 200,
    min_length: int = 1,
    batch_size: int = 16,
    forbid_unk: bool = False,
    canonicalize_source: bool = False,
    reciprocal_rank: bool = True,
    miss_reward: float = 0.0,
    cache: MutableMapping[str, Sequence[str]] | None = None,
) -> Tensor:
    """Map forward product reconstruction ranks to a finite reward."""

    if miss_reward < 0 or not torch.isfinite(torch.tensor(miss_reward)):
        raise ValueError("miss_reward must be finite and non-negative")
    ranks = forward_beam_reconstruction_rank(
        scorer,
        reactants_global,
        products_global,
        beam_size=beam_size,
        max_length=max_length,
        min_length=min_length,
        batch_size=batch_size,
        forbid_unk=forbid_unk,
        canonicalize_source=canonicalize_source,
        cache=cache,
    )
    reward = torch.full(ranks.shape, float(miss_reward), dtype=torch.float32)
    matched = ranks > 0
    if reciprocal_rank:
        reward[matched] = ranks[matched].float().reciprocal()
    else:
        reward[matched] = 1.0
    return reward


__all__ = [
    "forward_beam_reconstruction_rank",
    "forward_beam_reconstruction_reward",
    "forward_log_likelihood_reward",
    "positive_forward_reward",
]
