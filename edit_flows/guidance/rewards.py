"""Reward functions used by the first DGM mechanics experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def rdkit_validity_reward(smiles: Sequence[str]) -> Tensor:
    """Return 1 for RDKit-parsable SMILES and 0 otherwise.

    This is deliberately a weak terminal reward.  It does not compare against
    a target product and therefore is safe for the first guidance smoke tests.
    Chemical plausibility, atom conservation and forward consistency are left
    to later reward implementations.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "RDKit is required for rdkit_validity_reward"
        ) from exc

    values = []
    for value in smiles:
        if not isinstance(value, str) or not value.strip():
            values.append(0.0)
            continue
        try:
            molecule = Chem.MolFromSmiles(value)
        except Exception:
            molecule = None
        values.append(1.0 if molecule is not None else 0.0)
    return torch.tensor(values, dtype=torch.float32)
