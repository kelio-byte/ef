#!/usr/bin/env python
"""Pre-compute Levenshtein DP alignments for retro training data.

Works directly on string tokens — no vocab dependency.
Outputs two text files per split (*_aligned_src.txt, *_aligned_tgt.txt)
with <GAP> as the gap marker.

Usage:
  PYTHONPATH=. python scripts/precompute_alignments.py \
      --data_dir /path/to/dataset/USPTO_50K_PtoR_aug20
"""

import argparse
import os
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

from tqdm import tqdm

GAP_STR = "<GAP>"


def _align_strings(seq_0: List[str], seq_1: List[str]) -> Tuple[List[str], List[str]]:
    """Levenshtein DP alignment on string token sequences."""
    m, n = len(seq_0), len(seq_1)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_0[i - 1] == seq_1[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    aligned_0, aligned_1 = [], []
    i, j = m, n
    while i or j:
        if i and j and seq_0[i - 1] == seq_1[j - 1]:
            aligned_0.append(seq_0[i - 1])
            aligned_1.append(seq_1[j - 1])
            i, j = i - 1, j - 1
        elif i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
            aligned_0.append(seq_0[i - 1])
            aligned_1.append(seq_1[j - 1])
            i, j = i - 1, j - 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            aligned_0.append(seq_0[i - 1])
            aligned_1.append(GAP_STR)
            i -= 1
        else:
            aligned_0.append(GAP_STR)
            aligned_1.append(seq_1[j - 1])
            j -= 1

    return aligned_0[::-1], aligned_1[::-1]


def _process_pair(args: Tuple[str, str]) -> Tuple[List[str], List[str]]:
    src_line, tgt_line = args
    src_tokens = src_line.strip().split()
    tgt_tokens = tgt_line.strip().split()
    return _align_strings(src_tokens, tgt_tokens)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute DP alignments for retro data")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["train", "val", "test"],
                        help="Dataset splits to process")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count)")
    args = parser.parse_args()

    data_dir = args.data_dir
    num_workers = args.num_workers or cpu_count()
    print(f"Using {num_workers} workers")

    for split in args.splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Split dir {split_dir} not found, skipping")
            continue

        src_path = os.path.join(split_dir, f"src-{split}.txt")
        tgt_path = os.path.join(split_dir, f"tgt-{split}.txt")
        out_src = os.path.join(split_dir, f"{split}_aligned_src.txt")
        out_tgt = os.path.join(split_dir, f"{split}_aligned_tgt.txt")

        if not os.path.exists(src_path) or not os.path.exists(tgt_path):
            print(f"Missing {src_path} or {tgt_path}, skipping")
            continue

        if os.path.exists(out_src) and os.path.exists(out_tgt):
            print(f"{out_src} and {out_tgt} already exist, skipping "
                  f"(delete to re-compute)")
            continue

        print(f"Reading {split} split...")
        with open(src_path) as f_src, open(tgt_path) as f_tgt:
            lines = list(zip(f_src, f_tgt))

        print(f"Aligning {len(lines):,} pairs...")
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_pair, lines, chunksize=2000),
                total=len(lines),
                desc=f"Aligning {split}",
            ))

        print(f"Writing aligned files...")
        with open(out_src, 'w') as f0, open(out_tgt, 'w') as f1:
            for z0, z1 in results:
                f0.write(' '.join(z0) + '\n')
                f1.write(' '.join(z1) + '\n')

        # Report size
        src_size = os.path.getsize(out_src)
        tgt_size = os.path.getsize(out_tgt)
        print(f"Saved {len(results):,} aligned pairs:")
        print(f"  {out_src} ({src_size / 1024 / 1024:.1f} MB)")
        print(f"  {out_tgt} ({tgt_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
