#!/usr/bin/env python
"""Oracle generation: sample with theoretically optimal edit rates.

Instead of using a trained model, this script computes the optimal edit rates
at each Euler step by dynamically aligning the current state with the known
target.  This reveals the best-case performance of the Edit Flows formulation.

Usage:
  # Standard dataset
  PYTHONPATH=. python scripts/oracle_sample.py \
      --products_file train_subsets/USPTO_50K_PtoR_aug20/test/src-test.txt \
      --targets_file train_subsets/USPTO_50K_PtoR_aug20/test/tgt-test.txt \
      --vocab_file /data6/duanbh/desktop/retrosynthesis/dataset/USPTO_50K_PtoR_aug20/example.vocab.src \
      --output_dir train_subsets/eval/oracle_standard \
      --n_samples 10 --n_steps 100 --batch_size 32 \
      --augmentation 20 --score_script scripts/score.py

  # #global# dataset
  PYTHONPATH=. python scripts/oracle_sample.py \
      --products_file train_subsets/USPTO_50K_PtoR_aug20_#global#/test/src-test.txt \
      --targets_file train_subsets/USPTO_50K_PtoR_aug20_#global#/test/tgt-test.txt \
      --vocab_file /data6/duanbh/desktop/retrosynthesis/dataset/USPTO_50K_PtoR_aug20_#global#/example.vocab.src \
      --output_dir train_subsets/eval/oracle_global \
      --n_samples 10 --n_steps 100 --batch_size 32 \
      --augmentation 20 --score_script scripts/score_#global#.py
"""

import argparse
import math
import os
import subprocess
import sys
import time

import torch
from torch import Tensor
from tqdm import tqdm

from edit_flows.data.dataset import load_vocab
from edit_flows.sampling.euler import sample_euler_oracle
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN, UNK_TOKEN


def tokenize_smiles(smiles: str, token2id: dict) -> list:
    tokens = smiles.strip().split()
    unk_id = token2id.get("<unk>", UNK_TOKEN)
    return [token2id.get(t, unk_id) for t in tokens]


def _ids_to_str(ids: list, id2token: dict) -> str:
    return " ".join(id2token[tid] for tid in ids
                    if tid not in (PAD_TOKEN, BOS_TOKEN))


def _make_batch(product_ids: list[list[int]], target_ids: list[list[int]],
                n_samples: int, pad_token: int,
                bos_token: int = BOS_TOKEN) -> tuple[Tensor, Tensor]:
    """Build x_0 and x_1 batches with BOS prefix and n_samples repeats."""
    B = len(product_ids)
    max_src_len = max(len(ids) for ids in product_ids)
    max_tgt_len = max(len(ids) for ids in target_ids)

    x_0 = torch.full((B, max_src_len + 1), pad_token, dtype=torch.long)
    x_1 = torch.full((B, max_tgt_len + 1), pad_token, dtype=torch.long)
    x_0[:, 0] = bos_token
    x_1[:, 0] = bos_token
    for i, (src_ids, tgt_ids) in enumerate(zip(product_ids, target_ids)):
        x_0[i, 1:1 + len(src_ids)] = torch.tensor(src_ids, dtype=torch.long)
        x_1[i, 1:1 + len(tgt_ids)] = torch.tensor(tgt_ids, dtype=torch.long)

    return x_0.repeat_interleave(n_samples, dim=0), \
        x_1.repeat_interleave(n_samples, dim=0)


def main():
    parser = argparse.ArgumentParser(
        description="Oracle generation for Edit Flows retrosynthesis")
    parser.add_argument("--products_file", type=str, required=True,
                        help="File with tokenized product SMILES (one per line)")
    parser.add_argument("--targets_file", type=str, required=True,
                        help="File with tokenized target SMILES (one per line)")
    parser.add_argument("--vocab_file", type=str, required=True,
                        help="Path to vocabulary file (example.vocab.src)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save predictions.txt and eval.log")
    parser.add_argument("--n_steps", type=int, default=100,
                        help="Euler sampling steps (default 100)")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Independent samples per product (default 10)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Products per GPU batch (default 32)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--augmentation", type=int, default=20,
                        help="Augmentation factor for scoring (default 20)")
    parser.add_argument("--score_script", type=str, default="scripts/score.py",
                        help="Path to score script for evaluation")
    parser.add_argument("--n_best", type=int, default=10,
                        help="Top-N accuracy to report (default 10)")
    parser.add_argument("--skip_scoring", action="store_true",
                        help="Skip scoring step (sampling only)")
    parser.add_argument("--deduplicate", type=int, default=0,
                        help="Take every Nth line (e.g. 20 for aug20 datasets)")
    parser.add_argument("--max_lines", type=int, default=0,
                        help="Only process first N lines (after dedup)")
    parser.add_argument("--scheduler", type=str, default="cubic",
                        choices=["cubic", "linear"],
                        help="Kappa scheduler (default: cubic)")
    parser.add_argument("--clamp_kappa", action="store_true",
                        help="Clamp on 1/(1-kappa) instead of full k(t)")
    parser.add_argument("--clamp_max", type=float, default=50.0,
                        help="Max clamp value (default: 50.0)")
    parser.add_argument("--record_trajectory", action="store_true",
                        help="Record per-step edit distances and save as .pt file")
    parser.add_argument("--event_prob_mode", type=str, default="poisson",
                        choices=["poisson", "linear"],
                        help="Event probability discretization: "
                             "'poisson' uses 1-exp(-h*lambda), "
                             "'linear' uses min(h*lambda, 1)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load vocabulary
    token2id, model_vocab = load_vocab(args.vocab_file)
    id2token = {v: k for k, v in token2id.items()}

    # Load data
    with open(args.products_file) as f:
        products = [line.strip() for line in f]
    with open(args.targets_file) as f:
        targets = [line.strip() for line in f]
    assert len(products) == len(targets), \
        f"Product/target count mismatch: {len(products)} vs {len(targets)}"

    # Deduplicate: take every Nth line (for aug20 datasets)
    if args.deduplicate > 0:
        products = products[::args.deduplicate]
        targets = targets[::args.deduplicate]
        # Adjust augmentation for scoring
        if args.augmentation == args.deduplicate:
            args.augmentation = 1
        print(f"After dedup (stride={args.deduplicate}): {len(products)} products")

    # Limit number of lines
    if args.max_lines > 0 and len(products) > args.max_lines:
        products = products[:args.max_lines]
        targets = targets[:args.max_lines]
        print(f"Limited to first {args.max_lines} products")

    # Write the matching targets subset for scoring
    targets_subset_file = os.path.join(args.output_dir, "targets_subset.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(targets_subset_file, "w") as f:
        for t in targets:
            f.write(t + "\n")

    n_total = len(products)
    product_ids = [tokenize_smiles(s, token2id) for s in products]
    target_ids = [tokenize_smiles(s, token2id) for s in targets]

    if args.scheduler == "linear":
        scheduler = LinearScheduler()
    else:
        scheduler = CubicScheduler()

    os.makedirs(args.output_dir, exist_ok=True)
    pred_file = os.path.join(args.output_dir, "predictions.txt")

    print(f"Oracle generation: {n_total} products x {args.n_samples} samples")
    print(f"  n_steps: {args.n_steps}, batch_size: {args.batch_size}")
    print(f"  scheduler: {args.scheduler}, clamp_kappa: {args.clamp_kappa}, clamp_max: {args.clamp_max}")
    print(f"  event_prob_mode: {args.event_prob_mode}")
    print(f"  vocab size: {model_vocab}, device: {device}")
    print(f"  output: {pred_file}")

    batch_size = args.batch_size
    n_batches = math.ceil(n_total / batch_size)
    t_start = time.time()

    all_ts, all_dists = [], []
    all_trajectories = []  # per-batch: [{"ts": [...], "dists": [...]}, ...]

    with open(pred_file, "w") as f_out:
        for batch_idx in tqdm(range(n_batches), desc="Oracle batches"):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_total)
            batch_products = product_ids[start:end]
            batch_targets = target_ids[start:end]

            x_0, x_1 = _make_batch(
                batch_products, batch_targets, args.n_samples, PAD_TOKEN,
            )
            x_0 = x_0.to(device)
            x_1 = x_1.to(device)

            result = sample_euler_oracle(
                x_0, x_1, scheduler,
                vocab_size=model_vocab,
                n_steps=args.n_steps,
                max_seq_len=256,
                record_edit_distances=args.record_trajectory,
                event_prob_mode=args.event_prob_mode,
            )

            if args.record_trajectory:
                results, _, (ts_list, dists_list) = result
                all_trajectories.append({"ts": ts_list, "dists": dists_list})
            else:
                results, _ = result

            results = results.cpu()
            B = end - start
            for i in range(B):
                for s in range(args.n_samples):
                    row = results[i * args.n_samples + s]
                    line = _ids_to_str(row.tolist(), id2token)
                    f_out.write(line + "\n")

    elapsed = time.time() - t_start
    print(f"Sampling done in {elapsed:.1f}s "
          f"({elapsed / n_total:.3f}s per product, "
          f"{elapsed / (n_total * args.n_samples):.3f}s per sample)")

    if args.record_trajectory:
        traj_path = os.path.join(args.output_dir, "trajectory.pt")
        torch.save({"trajectories": all_trajectories}, traj_path)
        total_steps = sum(len(t["ts"]) for t in all_trajectories)
        print(f"Trajectory data saved to: {traj_path} "
              f"({len(all_trajectories)} batches, {total_steps} total steps)")

    # Scoring
    if not args.skip_scoring:
        print(f"\nRunning scoring with: {args.score_script}")
        log_file = os.path.join(args.output_dir, "eval.log")
        cmd = [
            sys.executable, args.score_script,
            "--predictions", pred_file,
            "--targets", targets_subset_file,
            "--beam_size", str(args.n_samples),
            "--augmentation", str(args.augmentation),
            "--n_best", str(args.n_best),
            "--edit_flows",
        ]
        print(f"  {' '.join(cmd)}")
        with open(log_file, "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        print(f"Scoring done, results in: {log_file}")

        # Print key results
        with open(log_file) as f:
            for line in f:
                if line.startswith("Top-") or "Invalid SMILES" in line or \
                   "Unique Rates" in line:
                    print(f"  {line.rstrip()}")

    print(f"\nDone. Predictions: {pred_file}")


if __name__ == "__main__":
    main()
