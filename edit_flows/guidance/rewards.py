"""Reward functions used by the first DGM mechanics experiments."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Callable, Iterator, MutableMapping

import torch
from torch import Tensor


@contextmanager
def _suppress_rdkit_parse_logs() -> Iterator[None]:
    """Temporarily silence RDKit parser diagnostics.

    Invalid intermediate SMILES are expected while evaluating a reward.  RDKit
    otherwise writes one diagnostic per invalid string, which can dominate the
    runtime/logs of a rollout benchmark.  The reward evaluator is currently
    single-threaded, so a short process-wide logging toggle is sufficient; the
    ``finally`` block guarantees that the caller's logging state is restored.
    """
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.error")
    rdBase.DisableLog("rdApp.warning")
    try:
        yield
    finally:
        rdBase.EnableLog("rdApp.error")
        rdBase.EnableLog("rdApp.warning")


def rdkit_validity_reward(
    smiles: Sequence[str],
    *,
    cache: MutableMapping[str, float] | None = None,
    normalize: Callable[[str], str] | None = None,
) -> Tensor:
    """Return 1 for RDKit-parsable SMILES and 0 otherwise.

    This is deliberately a weak terminal reward.  It does not compare against
    a target product and therefore is safe for the first guidance smoke tests.
    Chemical plausibility, atom conservation and forward consistency are left
    to later reward implementations.

    ``cache`` is an optional caller-owned mapping from SMILES to reward.  It is
    useful for augmentation-heavy validation, where the same candidate often
    occurs many times.  ``normalize`` can convert a serialized/tokenized
    representation to ordinary SMILES before parsing; normalized strings are
    used as cache keys.  The default remains uncached and unnormalized.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "RDKit is required for rdkit_validity_reward"
        ) from exc

    values = []
    with _suppress_rdkit_parse_logs():
        for value in smiles:
            if not isinstance(value, str):
                values.append(0.0)
                continue
            if normalize is not None:
                try:
                    value = normalize(value)
                except Exception:
                    value = ""
            if cache is not None and value in cache:
                values.append(float(cache[value]))
                continue
            if not value.strip():
                result = 0.0
                if cache is not None:
                    cache[value] = result
                values.append(result)
                continue
            try:
                molecule = Chem.MolFromSmiles(value)
            except Exception:
                molecule = None
            result = 1.0 if molecule is not None else 0.0
            if cache is not None:
                cache[value] = result
            values.append(result)
    return torch.tensor(values, dtype=torch.float32)


def retro_tokenized_validity_reward(
    smiles: Sequence[str],
    *,
    cache: MutableMapping[str, float] | None = None,
) -> Tensor:
    """Evaluate Edit Flows' space-tokenized/global-aligned SMILES.

    Sampling writes one token per space (and the training representation uses
    global-alignment parentheses).  RDKit must see the compact inverse-aligned
    string, otherwise serialization spaces and alignment markers can be
    mistaken for chemical invalidity.  This wrapper keeps that conversion
    explicit instead of changing the generic RDKit reward's input contract.
    """
    try:
        from scripts.preprocessing.global_align import inverse_global_align
    except ImportError:
        # When this function is called from ``python scripts/foo.py``, Python
        # puts ``scripts/`` (rather than the repository root) on sys.path.
        # Support that documented invocation as well as library imports from
        # the repository root without duplicating the alignment code.
        try:
            from preprocessing.global_align import inverse_global_align
        except ImportError as exc:  # pragma: no cover - package-layout dependent
            raise RuntimeError(
                "preprocessing.global_align is required for "
                "retro_tokenized_validity_reward"
            ) from exc

    def normalize(value: str) -> str:
        compact = "".join(value.split())
        if not compact:
            return ""
        return inverse_global_align(compact)

    return rdkit_validity_reward(smiles, cache=cache, normalize=normalize)
