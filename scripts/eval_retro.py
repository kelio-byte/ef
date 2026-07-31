#!/usr/bin/env python
"""End-to-end sampling + evaluation for Edit Flows retrosynthesis.

Given a checkpoint, automatically:
  1. Detects dataset type (standard / #global#) from config
  2. Runs sampling on the test set
  3. Runs the appropriate scoring script
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import yaml
import torch


def detect_dataset_type(data_dir: str) -> str:
    """Return 'global' if dataset uses #global# format, else 'standard'."""
    if "#global#" in data_dir:
        return "global"
    return "standard"


def detect_augmentation(data_dir: str) -> int:
    """Parse augmentation factor from dataset name (e.g. 'aug20' -> 20). Default 1."""
    m = re.search(r"aug(\d+)", data_dir)
    if m:
        return int(m.group(1))
    return 1


def _extract_step(filename: str) -> int:
    m = re.search(r"step(\d+)", filename)
    if m:
        return int(m.group(1))
    return 0


def average_checkpoints_in_dir(checkpoint_dir: str, output_path: str) -> str:
    """Equal-weight average of all checkpoint_step*.pt in *checkpoint_dir*.

    Returns *output_path* if averaging succeeded, or the path to an
    already-existing averaged checkpoint.
    """
    if os.path.exists(output_path):
        print(f"Averaged checkpoint already exists: {output_path}")
        return output_path

    pattern = os.path.join(checkpoint_dir, "checkpoint_step*.pt")
    ckpt_paths = sorted(glob.glob(pattern), key=_extract_step)
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No checkpoint_step*.pt files found in {checkpoint_dir}"
        )

    print(f"Average checkpoint dir:      {checkpoint_dir}")
    print(f"Averaging {len(ckpt_paths)} checkpoints (equal weight):")
    for p in ckpt_paths:
        print(f"  {os.path.basename(p)}")

    device = torch.device("cpu")
    avg_state = None
    reference_config = None
    reference_vocab_info = {}
    n = len(ckpt_paths)

    for i, path in enumerate(ckpt_paths):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"]

        if i == 0:
            avg_state = {k: v.float() / n for k, v in sd.items()}
            reference_config = ckpt["config"]
            reference_vocab_info = {
                k: ckpt[k]
                for k in ["real_vocab_size", "model_vocab"]
                if k in ckpt
            }
        else:
            if ckpt["config"] != reference_config:
                print(f"  [WARN] config mismatch for {os.path.basename(path)}")
            for k, v in sd.items():
                avg_state[k] += v.float() / n

    # Restore original dtypes
    ref_sd = torch.load(ckpt_paths[0], map_location=device, weights_only=False)["model_state_dict"]
    avg_state = {k: v.to(dtype=ref_sd[k].dtype) for k, v in avg_state.items()}

    out_ckpt = {
        "model_state_dict": avg_state,
        "config": reference_config,
        "step": -1,
        **reference_vocab_info,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(out_ckpt, output_path)
    print(f"Saved averaged checkpoint to {output_path}\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end sampling + evaluation for Edit Flows retrosynthesis")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing checkpoint_step*.pt files. "
                             "All checkpoints are equal-weight averaged into "
                             "checkpoint_averaged.pt (skipped if it already exists), "
                             "then used for evaluation.")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of independent samples per product")
    parser.add_argument("--n_steps", type=int, default=100,
                        help="Euler sampling steps")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="GPU batch size for sampling (products per batch)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scheduler", type=str, default=None,
                        choices=["cubic", "linear"],
                        help="Override sampling scheduler")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: <ckpt_dir>/eval_<timestamp>)")
    parser.add_argument("--n_best", type=int, default=10,
                        help="Top-N accuracy to report")
    parser.add_argument("--process_number", type=int, default=None,
                        help="Number of CPU processes for canonicalization")
    parser.add_argument("--detailed", action="store_true", default=False,
                        help="Enable detailed per-category accuracy")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data dir (default: from checkpoint config)")
    parser.add_argument("--test_src", type=str, default="test/src-test.txt",
                        help="Test source file relative to data_dir")
    parser.add_argument("--test_tgt", type=str, default="test/tgt-test.txt",
                        help="Test target file relative to data_dir")
    parser.add_argument("--augmentation", type=int, default=None,
                        help="Test-set augmentation factor (default: auto-detect from dataset name)")
    args = parser.parse_args()

    if not args.checkpoint and not args.checkpoint_dir:
        parser.error("Either --checkpoint or --checkpoint_dir is required.")
    if args.checkpoint and args.checkpoint_dir:
        parser.error("Use --checkpoint or --checkpoint_dir, not both.")

    # If checkpoint_dir, average checkpoints (or use existing averaged)
    if args.checkpoint_dir:
        args.checkpoint = average_checkpoints_in_dir(
            args.checkpoint_dir,
            os.path.join(args.checkpoint_dir, "checkpoint_averaged.pt"),
        )

    # Load checkpoint config
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]

    data_dir = args.data_dir or cfg["data_dir"]
    dataset_type = detect_dataset_type(data_dir)
    augmentation = args.augmentation if args.augmentation is not None else detect_augmentation(data_dir)

    if not os.path.isdir(data_dir):
        print(f"ERROR: data_dir not found: {data_dir}")
        sys.exit(1)

    test_src = os.path.join(data_dir, args.test_src)
    test_tgt = os.path.join(data_dir, args.test_tgt)
    if not os.path.exists(test_src):
        print(f"ERROR: test src not found: {test_src}")
        sys.exit(1)
    if not os.path.exists(test_tgt):
        print(f"ERROR: test tgt not found: {test_tgt}")
        sys.exit(1)

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ckpt_dir = os.path.dirname(args.checkpoint)
        output_dir = os.path.join(ckpt_dir, "eval")
    os.makedirs(output_dir, exist_ok=True)

    pred_file = os.path.join(output_dir, "predictions.txt")
    log_file = os.path.join(output_dir, "eval.log")

    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Data dir:      {data_dir}")
    print(f"Dataset type:  {dataset_type}")
    print(f"Augmentation:  {augmentation}")
    print(f"Test src:      {test_src}")
    print(f"Test tgt:      {test_tgt}")
    print(f"Output dir:    {output_dir}")
    print(f"n_samples:     {args.n_samples}")
    print(f"n_steps:       {args.n_steps}")

    # Step 1: Sampling
    print("\n" + "=" * 55)
    print("Step 1/2: Sampling ...")
    print("=" * 55)

    sample_cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "sample_retro.py"),
        "--checkpoint", args.checkpoint,
        "--products_file", test_src,
        "--n_samples", str(args.n_samples),
        "--n_steps", str(args.n_steps),
        "--batch_size", str(args.batch_size),
        "--output_dir", output_dir,
        "--device", args.device,
    ]
    if args.scheduler:
        sample_cmd += ["--scheduler", args.scheduler]
    print(f"Running: {' '.join(sample_cmd)}")
    result = subprocess.run(sample_cmd, capture_output=False)
    if result.returncode != 0:
        print("ERROR: Sampling failed.")
        sys.exit(1)

    # Step 2: Scoring
    print("\n" + "=" * 55)
    print("Step 2/2: Scoring ...")
    print("=" * 55)

    if dataset_type == "global":
        score_script = os.path.join(os.path.dirname(__file__), "score_#global#.py")
    else:
        score_script = os.path.join(os.path.dirname(__file__), "score.py")

    score_cmd = [
        sys.executable, score_script,
        "--predictions", pred_file,
        "--targets", test_tgt,
        "--beam_size", str(args.n_samples),
        "--augmentation", str(augmentation),
        "--n_best", str(args.n_best),
        "--edit_flows",
    ]
    if args.process_number:
        score_cmd += ["--process_number", str(args.process_number)]
    if args.detailed:
        score_cmd.append("--detailed")
        if os.path.exists(os.path.join(data_dir, args.test_src)):
            score_cmd += ["--sources", test_src]

    print(f"Running: {' '.join(score_cmd)}")

    with open(log_file, "w") as log_f:
        proc = subprocess.Popen(score_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        proc.wait()
        if proc.returncode != 0:
            print("ERROR: Scoring failed.")
            sys.exit(1)

    print("\n" + "=" * 55)
    print(f"Done. Results in: {output_dir}")
    print(f"  predictions.txt  - raw sampled SMILES")
    print(f"  eval.log          - evaluation metrics")


if __name__ == "__main__":
    main()
