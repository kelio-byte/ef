#!/usr/bin/env python3
"""Run a small, reproducible Molecular Transformer compatibility smoke.

The input files may be the project's space-tokenized ``#global#`` files.  By
default both sides are compacted and inverse-aligned before the official
Molecular Transformer tokenizer is applied.  The script compares the expected
forward direction (reactants -> product) with the swapped direction; it is a
direction/weight-loading smoke, not a DGM accuracy claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from edit_flows.forward import load_molecular_transformer
from edit_flows.forward.molecular_transformer import retro_global_to_smiles


def _read_examples(
    products_file: Path,
    reactants_file: Path,
    *,
    n_examples: int,
    augmentation: int,
    raw_smiles: bool,
) -> tuple[list[str], list[str]]:
    products = products_file.read_text().splitlines()
    reactants = reactants_file.read_text().splitlines()
    if len(products) != len(reactants):
        raise ValueError("products/reactants line counts differ")
    if augmentation < 1:
        raise ValueError("augmentation must be positive")
    chosen = list(range(0, min(len(products), n_examples * augmentation), augmentation))
    if not raw_smiles:
        products = [retro_global_to_smiles(products[i]) for i in chosen]
        reactants = [retro_global_to_smiles(reactants[i]) for i in chosen]
    else:
        products = [products[i].strip() for i in chosen]
        reactants = [reactants[i].strip() for i in chosen]
    return products, reactants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--products_file", required=True)
    parser.add_argument("--reactants_file", required=True)
    parser.add_argument("--n_examples", type=int, default=25)
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--raw_smiles",
        action="store_true",
        help="inputs are already compact ordinary SMILES; skip inverse global alignment",
    )
    args = parser.parse_args()

    products, reactants = _read_examples(
        Path(args.products_file),
        Path(args.reactants_file),
        n_examples=args.n_examples,
        augmentation=args.augmentation,
        raw_smiles=args.raw_smiles,
    )
    scorer = load_molecular_transformer(args.checkpoint, device=args.device)
    forward = scorer.score_batch(reactants, products, batch_size=args.batch_size)
    reverse = scorer.score_batch(products, reactants, batch_size=args.batch_size)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": args.device,
        "n_examples": len(products),
        "vocab_size": len(scorer.vocab),
        "direction": {
            "forward_reactants_to_product_mean": float(forward.mean()),
            "reverse_product_to_reactants_mean": float(reverse.mean()),
            "forward_better_fraction": float((forward > reverse).float().mean()),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
