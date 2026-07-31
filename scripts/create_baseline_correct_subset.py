#!/usr/bin/env python3
"""
Create a subset of test samples where the R-SMILES baseline model predicts correctly.

For each of the 100140 test inputs (5007 products x 20 augmentations):
1. Take the top-1 beam prediction from the baseline model
2. Canonicalize both prediction and target (inverse_global_align + RDKit)
3. If they match exactly, mark as "baseline correct"
4. Randomly sample N from the correct set

Output:
    src-test.txt      — original product SMILES (tokenized)
    tgt-test.txt      — original target SMILES (tokenized)
    baseline_top1.txt — canonicalized top-1 prediction from baseline model
    meta.json         — metadata about the subset
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, "/data6/duanbh/desktop/retrosynthesis")
from preprocessing.global_align import inverse_global_align  # noqa: E402
from rdkit import Chem  # noqa: E402


def canonicalize_smiles(smi_tokens: str) -> str:
    """Convert tokenized #global# SMILES to canonical SMILES (atom maps cleared)."""
    smi = "".join(smi_tokens.split())
    try:
        smi = inverse_global_align(smi)
    except Exception:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Create baseline-correct subset")
    parser.add_argument(
        "--src_file",
        default="/data6/duanbh/desktop/retrosynthesis/dataset/USPTO_50K_PtoR_aug20_#global#/test/src-test.txt",
    )
    parser.add_argument(
        "--tgt_file",
        default="/data6/duanbh/desktop/retrosynthesis/dataset/USPTO_50K_PtoR_aug20_#global#/test/tgt-test.txt",
    )
    parser.add_argument(
        "--pred_file",
        default="/data6/duanbh/desktop/retrosynthesis/exp/USPTO_50K_PtoR_aug20_#global#/average_model_56-60-results.txt",
    )
    parser.add_argument(
        "--output_dir",
        default="analysis_subsets/USPTO_50K_PtoR_aug20_#global#/baseline_correct_seed42_1000",
    )
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam_size", type=int, default=10)
    args = parser.parse_args()

    # Read all lines
    print(f"Reading sources from {args.src_file} ...")
    with open(args.src_file) as f:
        src_lines = [line.rstrip("\n") for line in f]
    print(f"Reading targets from {args.tgt_file} ...")
    with open(args.tgt_file) as f:
        tgt_lines = [line.rstrip("\n") for line in f]
    print(f"Reading predictions from {args.pred_file} ...")
    with open(args.pred_file) as f:
        pred_lines = [line.rstrip("\n") for line in f]

    n_total = len(src_lines)
    assert len(tgt_lines) == n_total, "src/tgt line count mismatch"
    assert len(pred_lines) == n_total * args.beam_size, (
        f"Expected {n_total * args.beam_size} pred lines, got {len(pred_lines)}"
    )

    print(f"Total inputs: {n_total}, beam_size: {args.beam_size}")

    # Find correct predictions (top-1 = first of each beam group)
    correct_indices = []
    correct_canonical = []
    n_valid_pred = 0
    for i in range(n_total):
        pred_tokens = pred_lines[i * args.beam_size]  # top-1
        pred_can = canonicalize_smiles(pred_tokens)
        tgt_can = canonicalize_smiles(tgt_lines[i])

        if pred_can:
            n_valid_pred += 1

        if pred_can and tgt_can and pred_can == tgt_can:
            correct_indices.append(i)
            correct_canonical.append(pred_can)

    print(f"Valid predictions: {n_valid_pred}/{n_total}")
    print(f"Baseline-correct: {len(correct_indices)}/{n_total} "
          f"({100 * len(correct_indices) / n_total:.1f}%)")

    # Sample
    rng = random.Random(args.seed)
    if len(correct_indices) < args.n_samples:
        print(f"WARNING: only {len(correct_indices)} correct, sampling all of them")
        sampled = list(range(len(correct_indices)))
    else:
        sampled = sorted(rng.sample(range(len(correct_indices)), args.n_samples))

    print(f"Sampled {len(sampled)} from {len(correct_indices)} correct")

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)
    sampled_indices = [correct_indices[i] for i in sampled]

    with open(os.path.join(args.output_dir, "src-test.txt"), "w") as f:
        for idx in sampled_indices:
            f.write(src_lines[idx] + "\n")

    with open(os.path.join(args.output_dir, "tgt-test.txt"), "w") as f:
        for idx in sampled_indices:
            f.write(tgt_lines[idx] + "\n")

    with open(os.path.join(args.output_dir, "baseline_top1.txt"), "w") as f:
        for i in sampled:
            f.write(correct_canonical[i] + "\n")

    meta = {
        "description": "Test samples where R-SMILES baseline Top-1 prediction is correct",
        "source_src": args.src_file,
        "source_tgt": args.tgt_file,
        "source_pred": args.pred_file,
        "beam_size": args.beam_size,
        "total_inputs": n_total,
        "total_correct": len(correct_indices),
        "n_valid_predictions": n_valid_pred,
        "n_sampled": len(sampled),
        "random_seed": args.seed,
        "sampled_original_indices": sampled_indices,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Output written to {args.output_dir}/")
    print(f"  src-test.txt: {len(sampled)} lines")
    print(f"  tgt-test.txt: {len(sampled)} lines")
    print(f"  baseline_top1.txt: {len(sampled)} lines")
    print(f"  meta.json")


if __name__ == "__main__":
    main()
