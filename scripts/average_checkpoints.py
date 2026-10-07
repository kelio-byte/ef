#!/usr/bin/env python
"""Average model weights across multiple checkpoints.

Checkpoints must share the same model architecture (config). The output is a
single checkpoint file that can be used directly by sample_retro.py /
first_step_forward_analysis.py / etc.

Usage examples:

  # Explicit checkpoint list
  python scripts/average_checkpoints.py \
      --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt \
      --output averaged.pt

  # Average last N checkpoints from a directory
  python scripts/average_checkpoints.py \
      --checkpoint_dir checkpoints/.../run_dir/ \
      --last_n 5 \
      --output averaged.pt

  # Average checkpoints in a step range
  python scripts/average_checkpoints.py \
      --checkpoint_dir checkpoints/.../run_dir/ \
      --step_range 1650000 1680000 \
      --output averaged.pt

  # Weighted average (exponential decay, newest gets higher weight)
  python scripts/average_checkpoints.py \
      --checkpoint_dir checkpoints/.../run_dir/ \
      --last_n 5 \
      --weight_scheme exp_decay \
      --exp_decay_factor 0.5 \
      --output averaged.pt
"""

import argparse
import glob
import os
import re
import torch
from edit_flows.utils.checkpoint import average_model_checkpoints, atomic_torch_save


def _extract_step(filename: str) -> int:
    m = re.search(r"step(\d+)", filename)
    if m:
        return int(m.group(1))
    return 0


def gather_checkpoints(
    checkpoint_dir: str | None,
    checkpoints: list[str] | None,
    last_n: int | None,
    step_range: tuple[int, int] | None,
) -> list[str]:
    if last_n is not None and last_n <= 0:
        raise ValueError("last_n must be > 0")
    if step_range is not None and step_range[0] > step_range[1]:
        raise ValueError("step_range must be ordered low to high")
    if checkpoints:
        # Keep manual weights paired with the caller's explicit path order.
        return list(checkpoints)

    if not checkpoint_dir:
        raise ValueError("Either --checkpoints or --checkpoint_dir is required.")

    pattern = os.path.join(checkpoint_dir, "checkpoint_step*.pt")
    all_ckpts = sorted(glob.glob(pattern), key=_extract_step)
    if not all_ckpts:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    if last_n is not None:
        return all_ckpts[-last_n:]

    if step_range is not None:
        lo, hi = step_range
        return [p for p in all_ckpts if lo <= _extract_step(p) <= hi]

    return all_ckpts


def compute_weights(
    ckpt_paths: list[str],
    weight_scheme: str,
    manual_weights: list[float] | None,
    exp_decay_factor: float,
) -> torch.Tensor:
    n = len(ckpt_paths)
    if n == 0:
        raise ValueError("No checkpoints selected")

    if manual_weights is not None:
        if len(manual_weights) != n:
            raise ValueError(
                f"Got {n} checkpoints but {len(manual_weights)} manual weights."
            )
        w = torch.tensor(manual_weights, dtype=torch.float32)
        if (not torch.isfinite(w).all() or (w < 0).any()
                or not torch.isfinite(w.sum()) or w.sum() <= 0):
            raise ValueError("Manual weights must be finite, non-negative, with positive sum")
        return w / w.sum()

    if weight_scheme == "uniform":
        return torch.full((n,), 1.0 / n)

    if weight_scheme == "exp_decay":
        if not 0 < exp_decay_factor <= 1:
            raise ValueError("exp_decay_factor must be in (0, 1]")
        # newest last: weight = factor^(n-1-i)
        indices = torch.arange(n, dtype=torch.float32)
        w = exp_decay_factor ** (n - 1 - indices)
        return w / w.sum()

    if weight_scheme == "linear_decay":
        # newest last: weight = i+1
        w = torch.arange(1, n + 1, dtype=torch.float32)
        return w / w.sum()

    raise ValueError(f"Unknown weight_scheme: {weight_scheme}")


def main():
    parser = argparse.ArgumentParser(
        description="Average model weights across checkpoints"
    )
    # Input: explicit paths
    parser.add_argument(
        "--checkpoints", type=str, nargs="*", default=None,
        help="Explicit list of checkpoint paths",
    )
    # Input: directory-based
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="Directory containing checkpoint_step*.pt files",
    )
    parser.add_argument(
        "--last_n", type=int, default=None,
        help="Use the last N checkpoints from the directory",
    )
    parser.add_argument(
        "--step_range", type=int, nargs=2, default=None,
        help="Min/max step range (inclusive) to include from the directory",
    )
    parser.add_argument(
        "--step_interval", type=int, default=1,
        help="Only use every k-th checkpoint (default: 1 = all)",
    )
    # Weighting
    parser.add_argument(
        "--weight_scheme", type=str, default="uniform",
        choices=["uniform", "exp_decay", "linear_decay"],
        help="Weighting scheme (default: uniform)",
    )
    parser.add_argument(
        "--weights", type=float, nargs="*", default=None,
        help="Manual per-checkpoint weights (will be normalized)",
    )
    parser.add_argument(
        "--exp_decay_factor", type=float, default=0.5,
        help="Decay factor for exp_decay scheme (default: 0.5)",
    )
    # Output
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path for the averaged checkpoint",
    )
    parser.add_argument(
        "--save_optimizer", action="store_true",
        help="Also save optimizer state from the first checkpoint (rarely needed)",
    )
    args = parser.parse_args()
    if args.step_interval <= 0:
        parser.error("--step_interval must be > 0")

    # --- gather ---
    ckpt_paths = gather_checkpoints(
        args.checkpoint_dir,
        args.checkpoints,
        args.last_n,
        tuple(args.step_range) if args.step_range else None,
    )
    ckpt_paths = ckpt_paths[:: args.step_interval]
    if not ckpt_paths:
        raise ValueError("No checkpoints selected.")
    print(f"Averaging {len(ckpt_paths)} checkpoints:")
    for p in ckpt_paths:
        print(f"  {p}")

    # --- compute weights ---
    weights = compute_weights(
        ckpt_paths, args.weight_scheme, args.weights, args.exp_decay_factor
    )
    print(f"Weights ({args.weight_scheme}): {weights.tolist()}")

    out_ckpt = average_model_checkpoints(
        ckpt_paths, weights, save_optimizer=args.save_optimizer,
    )

    # --- save ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    atomic_torch_save(out_ckpt, args.output)
    print(f"Saved averaged checkpoint to {args.output}")


if __name__ == "__main__":
    main()
