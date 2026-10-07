#!/usr/bin/env python
"""Evaluate Euler samples from the SPE tokenizer branch.

The historical ``score_#global#.py`` scorer always applies the global-alignment
inverse transform.  SPE changes the token granularity but inherits the same
``#global#`` representation, so this script joins token fragments, applies
that inverse transform, and then canonicalizes with RDKit.  It keeps the
independent-Euler sample order:

    reaction -> augmentation -> sample

Candidates are equally weighted.  Within a reaction, the ranked list is the
first-seen list after RDKit canonicalization and deduplication, matching the
Edit Flows semantics used by ``scripts/score.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rdkit import Chem
from rdkit import RDLogger

from preprocessing.global_align import inverse_global_align


RDLogger.logger().setLevel(RDLogger.CRITICAL)
TOP_K = (1, 3, 5, 10)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_smiles_line(line: str) -> str | None:
    """Return canonical SMILES, or ``None`` for an invalid prediction."""
    smiles = "".join(line.strip().split())
    if not smiles:
        return None
    try:
        smiles = inverse_global_align(smiles)
        molecule = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if molecule is None:
        return None
    try:
        canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
        return canonical or None
    except Exception:
        return None


def _first_seen_unique(values: list[str | None]) -> tuple[list[str], int]:
    unique: list[str] = []
    seen: set[str] = set()
    invalid = 0
    for value in values:
        if value is None:
            invalid += 1
        elif value not in seen:
            seen.add(value)
            unique.append(value)
    return unique, invalid


def evaluate(
    predictions_path: Path,
    targets_path: Path,
    *,
    augmentation: int,
    beam_size: int,
    length: int | None = None,
    target_offset: int = 0,
) -> dict:
    if augmentation < 1 or beam_size < 1:
        raise ValueError("augmentation and beam_size must be positive")
    if target_offset < 0:
        raise ValueError("target_offset must be non-negative")

    prediction_lines = predictions_path.read_text().splitlines()
    target_lines = targets_path.read_text().splitlines()
    block_size = augmentation * beam_size
    if not prediction_lines:
        raise ValueError("prediction file is empty")
    if len(prediction_lines) % block_size:
        raise ValueError(
            "prediction count is not divisible by augmentation*beam_size: "
            f"{len(prediction_lines)} % ({augmentation}*{beam_size})"
        )
    available_reactions = len(prediction_lines) // block_size
    n_reactions = available_reactions if length is None else length
    if n_reactions < 1 or n_reactions > available_reactions:
        raise ValueError(
            f"length must be in [1, {available_reactions}], got {n_reactions}"
        )
    target_start = target_offset * augmentation
    required_targets = target_start + n_reactions * augmentation
    if len(target_lines) < required_targets:
        raise ValueError(
            f"target file has {len(target_lines)} lines, but {required_targets} "
            "are required for the requested prefix/offset"
        )

    oracle_count = 0
    top_k_counts = {str(k): 0 for k in TOP_K}
    invalid_count = 0
    valid_count = 0
    duplicate_count = 0
    unique_count_total = 0
    unique_slot_count = 0
    invalid_by_sample = [0] * beam_size
    per_reaction = []

    for reaction_index in range(n_reactions):
        candidates: list[str | None] = []
        for augmentation_index in range(augmentation):
            product_index = reaction_index * augmentation + augmentation_index
            start = product_index * beam_size
            candidates.extend(
                canonicalize_smiles_line(line)
                for line in prediction_lines[start:start + beam_size]
            )
        unique_candidates, reaction_invalid = _first_seen_unique(candidates)
        target = canonicalize_smiles_line(
            target_lines[target_start + reaction_index * augmentation]
        )
        if target is None:
            raise ValueError(f"invalid target at reaction {reaction_index}")
        try:
            target_rank = unique_candidates.index(target) + 1
        except ValueError:
            target_rank = None

        oracle = target_rank is not None
        oracle_count += int(oracle)
        for k in TOP_K:
            top_k_counts[str(k)] += int(target_rank is not None and target_rank <= k)
        invalid_count += reaction_invalid
        valid_reaction_count = len(candidates) - reaction_invalid
        valid_count += valid_reaction_count
        duplicate_count += valid_reaction_count - len(unique_candidates)
        unique_count_total += len(unique_candidates)
        unique_slot_count += len(candidates)
        for sample_index in range(beam_size):
            invalid_by_sample[sample_index] += sum(
                candidates[augmentation_index * beam_size + sample_index] is None
                for augmentation_index in range(augmentation)
            )
        per_reaction.append({
            "reaction_index": reaction_index + target_offset,
            "oracle": oracle,
            "target_rank": target_rank,
            "valid_candidates": valid_reaction_count,
            "unique_candidates": len(unique_candidates),
            "invalid_candidates": reaction_invalid,
        })

    total_slots = n_reactions * augmentation * beam_size
    total_unique_rate = 100.0 * unique_count_total / total_slots
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "aggregation": "first_seen_equal_weight_after_canonicalization",
            "augmentation": augmentation,
            "beam_size": beam_size,
            "top_k": list(TOP_K),
            "target_offset": target_offset,
            "reaction_count": n_reactions,
            "candidate_slots": total_slots,
            "canonicalizer": (
                "join_tokens -> inverse_global_align -> "
                "RDKit MolToSmiles(isomericSmiles=True)"
            ),
        },
        "inputs": {
            "predictions": {"path": str(predictions_path), "sha256": _sha256(predictions_path)},
            "targets": {"path": str(targets_path), "sha256": _sha256(targets_path)},
        },
        "metrics": {
            "top_k_percent": {
                key: 100.0 * value / n_reactions
                for key, value in top_k_counts.items()
            },
            "oracle_count": oracle_count,
            "oracle_percent": 100.0 * oracle_count / n_reactions,
            "invalid_count": invalid_count,
            "invalid_rate_percent": 100.0 * invalid_count / total_slots,
            "valid_count": valid_count,
            "valid_rate_percent": 100.0 * valid_count / total_slots,
            "duplicate_count_among_valid": duplicate_count,
            "duplicate_rate_among_valid_percent": (
                100.0 * duplicate_count / valid_count if valid_count else 0.0
            ),
            "unique_candidate_count": unique_count_total,
            "unique_candidate_rate_percent": total_unique_rate,
            "mean_unique_candidates_per_reaction": unique_count_total / n_reactions,
            "mean_valid_candidates_per_reaction": valid_count / n_reactions,
            "invalid_count_by_sample_rank": invalid_by_sample,
            "invalid_rate_by_sample_rank_percent": [
                100.0 * value / (n_reactions * augmentation)
                for value in invalid_by_sample
            ],
        },
        "per_reaction": per_reaction,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--beam-size", type=int, required=True)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--target-offset", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)
    result = evaluate(
        args.predictions,
        args.targets,
        augmentation=args.augmentation,
        beam_size=args.beam_size,
        length=args.length,
        target_offset=args.target_offset,
    )
    output_path = args.output_json or args.predictions.with_name("spe_evaluation.json")
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
