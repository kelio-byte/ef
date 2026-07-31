#!/usr/bin/env python
"""Sampling script for Edit Flows."""

import argparse
import yaml
import torch
from tqdm import tqdm

from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.sampling.euler import sample_euler
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN
from edit_flows.utils.helpers import pretty_print


def main():
    parser = argparse.ArgumentParser(description="Sample from Edit Flows model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--empty_prior", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint["config"]

    model = EditFlowsTransformer(
        vocab_size=cfg["vocab_size"] + 3,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        dim_feedforward=cfg["dim_feedforward"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scheduler = CubicScheduler() if cfg["scheduler"] == "cubic" else LinearScheduler()

    if args.empty_prior:
        x_0 = torch.empty((args.n_samples, 0), dtype=torch.long)
    else:
        x_0 = torch.randint(
            3, cfg["vocab_size"] + 3,
            (args.n_samples, cfg["min_seq_len"]),
            dtype=torch.long,
        )
        bos_col = torch.full((args.n_samples, 1), BOS_TOKEN, dtype=torch.long)
        x_0 = torch.cat([bos_col, x_0], dim=1)

    print(f"Sampling {args.n_samples} sequences for {args.n_steps} steps...")
    results, _ = sample_euler(
        model, x_0, scheduler,
        n_steps=args.n_steps,
        max_seq_len=args.max_seq_len,
        record_trajectory=False,
    )

    print("\nGenerated sequences:")
    for i in range(args.n_samples):
        seq = results[i][results[i] != PAD_TOKEN]
        print(f"  [{i}] (len={len(seq)}): {seq.tolist()}")


if __name__ == "__main__":
    main()
