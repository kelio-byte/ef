#!/usr/bin/env python3
"""Evaluate Molecular Transformer forward-beam reconstruction on a split.

This is a reward-model diagnostic, not a retrosynthesis sampler evaluation.
It reads known reactants only to measure the frozen forward model itself and
must not use test results to tune DGM hyperparameters.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from rdkit import Chem, RDLogger
import torch

from edit_flows.forward import load_molecular_transformer
from edit_flows.forward.molecular_transformer import retro_global_to_smiles


RDLogger.DisableLog("rdApp.*")


def _read_unique(path: str, augmentation: int) -> list[str]:
    lines = Path(path).read_text().splitlines()
    if augmentation < 1:
        raise ValueError("augmentation must be positive")
    if len(lines) % augmentation:
        raise ValueError(
            f"{path} has {len(lines)} lines, not divisible by {augmentation}"
        )
    return [lines[index] for index in range(0, len(lines), augmentation)]


def _canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def run(args: argparse.Namespace) -> dict:
    if args.beam_size < 1 or args.batch_size < 1:
        raise ValueError("beam_size and batch_size must be positive")
    if args.start_reaction < 0:
        raise ValueError("start_reaction must be non-negative")
    if args.max_reactions is not None and args.max_reactions < 1:
        raise ValueError("max_reactions must be positive")
    products = _read_unique(args.products_file, args.augmentation)
    reactants = _read_unique(args.reactants_file, args.augmentation)
    if len(products) != len(reactants):
        raise ValueError("product and reactant files contain different reaction counts")
    end = len(products) if args.max_reactions is None else min(
        len(products), args.start_reaction + args.max_reactions,
    )
    if args.start_reaction >= end:
        raise ValueError("selected reaction interval is empty")
    products = products[args.start_reaction:end]
    reactants = reactants[args.start_reaction:end]
    products = [retro_global_to_smiles(value) for value in products]
    reactants = [retro_global_to_smiles(value) for value in reactants]

    load_started = time.perf_counter()
    scorer = load_molecular_transformer(args.checkpoint, device=args.device)
    load_seconds = time.perf_counter() - load_started
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generate_started = time.perf_counter()
    predictions, scores = scorer.generate_batch(
        reactants,
        beam_size=args.beam_size,
        max_length=args.max_length,
        min_length=args.min_length,
        batch_size=args.batch_size,
        forbid_unk=args.forbid_unk,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - generate_started

    ranks = []
    invalid = 0
    examples = []
    for index, (target, beam, beam_scores) in enumerate(
        zip(products, predictions, scores.tolist())
    ):
        target_canonical = _canonical(target)
        beam_canonical = [_canonical(candidate) for candidate in beam]
        invalid += sum(not candidate for candidate in beam_canonical)
        rank = 0
        for candidate_rank, candidate in enumerate(beam_canonical, start=1):
            if candidate and candidate == target_canonical:
                rank = candidate_rank
                break
        ranks.append(rank)
        if len(examples) < args.example_count:
            examples.append({
                "reaction_index": args.start_reaction + index,
                "reactants": reactants[index],
                "target_product": target,
                "rank": rank,
                "beam": [
                    {"product": value, "log_probability": score}
                    for value, score in zip(beam, beam_scores)
                ],
            })

    count = len(ranks)
    hit_rates = {
        f"hit_at_{rank}": sum(0 < value <= rank for value in ranks) / count
        for rank in range(1, args.beam_size + 1)
    }
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "products_file": str(Path(args.products_file).resolve()),
        "reactants_file": str(Path(args.reactants_file).resolve()),
        "augmentation": args.augmentation,
        "reaction_interval": [args.start_reaction, end],
        "reaction_count": count,
        "beam_size": args.beam_size,
        "max_length": args.max_length,
        "min_length": args.min_length,
        "batch_size": args.batch_size,
        "forbid_unk": args.forbid_unk,
        **hit_rates,
        "mean_reciprocal_rank": sum(
            1.0 / rank for rank in ranks if rank > 0
        ) / count,
        "invalid_prediction_rate": invalid / (count * args.beam_size),
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "reactions_per_second": count / generation_seconds,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda" else 0
        ),
        "examples": examples,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--products_file", required=True)
    parser.add_argument("--reactants_file", required=True)
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--start_reaction", type=int, default=0)
    parser.add_argument("--max_reactions", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--forbid_unk", action="store_true")
    parser.add_argument("--example_count", type=int, default=3)
    parser.add_argument("--output_json", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
