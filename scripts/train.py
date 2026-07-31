#!/usr/bin/env python
"""Training script for Edit Flows."""

import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.core.coupling import EmptyCoupling, UniformCoupling, GeneratorCoupling
from edit_flows.core.alignment import opt_align_xs_to_zs, naive_align_xs_to_zs, shifted_align_xs_to_zs
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.training.trainer import prepare_batch, train_step
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN


def main():
    parser = argparse.ArgumentParser(description="Train Edit Flows model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cfg = config["default"]
    device = torch.device(args.device)

    torch.manual_seed(42)
    np.random.seed(42)

    model = EditFlowsTransformer(
        vocab_size=cfg["vocab_size"] + 3,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        dim_feedforward=cfg["dim_feedforward"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    scheduler = CubicScheduler() if cfg["scheduler"] == "cubic" else LinearScheduler()

    if cfg["coupling"] == "empty":
        coupling = EmptyCoupling()
    elif cfg["coupling"] == "uniform":
        coupling = UniformCoupling(
            min_len=cfg["min_seq_len"],
            max_len=cfg["max_seq_len"],
            vocab_size=cfg["vocab_size"],
            pad_token=PAD_TOKEN,
        )
    else:
        raise ValueError(f"Unknown coupling: {cfg['coupling']}")

    align_fn = {
        "opt": opt_align_xs_to_zs,
        "naive": naive_align_xs_to_zs,
        "shifted": shifted_align_xs_to_zs,
    }[cfg["align_fn"]]

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"Total steps: {cfg['total_steps']}")

    model.train()
    pbar = tqdm(range(cfg["total_steps"]), desc="Training")
    for step in pbar:
        lengths = np.random.randint(
            cfg["min_seq_len"], cfg["max_seq_len"] + 1,
            size=cfg["batch_size"],
        )
        max_len = lengths.max()
        x_1 = torch.full(
            (cfg["batch_size"], max_len + 1), PAD_TOKEN, dtype=torch.long,
        )
        for b, length in enumerate(lengths):
            x_1[b, 0] = BOS_TOKEN
            x_1[b, 1:length + 1] = torch.randint(
                3, cfg["vocab_size"] + 3, (length,),
            )

        x_0, x_1_batch = coupling.sample(x_1.to(device))
        batch = prepare_batch(
            x_0, x_1_batch, scheduler, align_fn,
            vocab_size=cfg["vocab_size"],
        )
        metrics = train_step(model, batch, scheduler, optimizer)

        if step % 100 == 0:
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "u_ins": f"{metrics['u_ins']:.4f}",
                "u_del": f"{metrics['u_del']:.4f}",
            })

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }, "checkpoint.pt")
    print("Model saved to checkpoint.pt")


if __name__ == "__main__":
    main()
