#!/usr/bin/env python
"""Training script for Edit Flows on retrosynthesis data."""

import argparse
import glob
import os
import re
import sys
import shutil
import yaml
import torch
from datetime import datetime
from torch.utils.data import DataLoader

from edit_flows.data.dataset import (
    RetroDataset, PreAlignedDataset, load_vocab, collate_fn,
)
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.training.trainer import prepare_batch, train_step
from edit_flows.training.schedulers import NoamScheduler
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.core.alignment import (
    opt_align_xs_to_zs, naive_align_xs_to_zs, shifted_align_xs_to_zs,
    identity_align_xs_to_zs,
)


class Tee:
    def __init__(self, filepath: str):
        self.file = open(filepath, "a", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, text: str):
        self.file.write(text)
        self.stdout.write(text)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, *args):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()


def extract_dataset_name(data_dir: str) -> str:
    return os.path.basename(data_dir.rstrip("/"))


def prune_checkpoints(save_dir: str, keep: int) -> None:
    ckpts = sorted(
        glob.glob(os.path.join(save_dir, "checkpoint_step*.pt")),
        key=lambda p: int(re.search(r"step(\d+)", p).group(1)),
    )
    while len(ckpts) > keep:
        old = ckpts.pop(0)
        os.remove(old)
        print(f"Pruned old checkpoint: {os.path.basename(old)}")


def save_checkpoint(save_dir: str, step: int, model, optimizer,
                    lr_scheduler, cfg, real_vocab_size, model_vocab,
                    keep: int) -> None:
    ckpt_path = os.path.join(save_dir, f"checkpoint_step{step}.pt")
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict(),
        "step": step,
        "config": cfg,
        "real_vocab_size": real_vocab_size,
        "model_vocab": model_vocab,
    }
    torch.save(state, ckpt_path)
    prune_checkpoints(save_dir, keep)


def main():
    parser = argparse.ArgumentParser(description="Train Edit Flows for retrosynthesis")
    parser.add_argument("--config", type=str, default="configs/retro.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint (.pt path)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override save directory")
    parser.add_argument("--keep_checkpoints", type=int, default=None,
                        help="Max checkpoints to keep (default 10)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cfg = config["retro"]
    device = torch.device(args.device)

    data_dir = cfg["data_dir"]
    dataset_name = extract_dataset_name(data_dir)
    vocab_path = os.path.join(data_dir, cfg["vocab_file"])
    token2id, model_vocab = load_vocab(vocab_path)
    real_vocab_size = model_vocab - 4

    keep_checkpoints = args.keep_checkpoints if args.keep_checkpoints is not None else cfg.get("keep_checkpoints", 10)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.save_dir:
        save_dir = os.path.join(args.save_dir, dataset_name, timestamp)
    elif args.checkpoint:
        save_dir = os.path.dirname(args.checkpoint)
    else:
        save_dir = os.path.join("checkpoints", dataset_name, timestamp)

    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "train.log")
    with Tee(log_path):
        config_dst = os.path.join(save_dir, "config.yaml")
        shutil.copy(args.config, config_dst)
        print(f"Config saved to {config_dst}")

        print(f"Checkpoint dir: {save_dir}")
        print(f"Dataset: {dataset_name}")
        print(f"Vocab: {real_vocab_size} real tokens, {model_vocab} model tokens")

        train_aligned_src = os.path.join(data_dir, "train", "train_aligned_src.txt")
        train_aligned_tgt = os.path.join(data_dir, "train", "train_aligned_tgt.txt")
        if os.path.exists(train_aligned_src) and os.path.exists(train_aligned_tgt):
            print(f"Using pre-aligned data: {train_aligned_src}, {train_aligned_tgt}")
            train_dataset = PreAlignedDataset(train_aligned_src, train_aligned_tgt, token2id)
            train_loader = DataLoader(
                train_dataset, batch_size=cfg["batch_size"], shuffle=True,
                collate_fn=collate_fn, num_workers=cfg.get("num_workers", 2),
                drop_last=True, pin_memory=True,
            )
            align_fn = identity_align_xs_to_zs
        else:
            print("WARNING: Pre-aligned data not found. Run scripts/precompute_alignments.py first.")
            print("Falling back to on-the-fly DP alignment (slow).")
            train_dataset = RetroDataset(
                src_path=os.path.join(data_dir, "train", "src-train.txt"),
                tgt_path=os.path.join(data_dir, "train", "tgt-train.txt"),
                token2id=token2id,
            )
            train_loader = DataLoader(
                train_dataset, batch_size=cfg["batch_size"], shuffle=True,
                collate_fn=collate_fn, num_workers=cfg.get("num_workers", 2),
                drop_last=True, pin_memory=True,
            )
            align_fn = {
                "opt": opt_align_xs_to_zs,
                "naive": naive_align_xs_to_zs,
                "shifted": shifted_align_xs_to_zs,
            }[cfg["align_fn"]]

        model = EditFlowsTransformer(
            vocab_size=model_vocab,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            dim_feedforward=cfg["dim_feedforward"],
            max_seq_len=cfg["max_seq_len"],
            dropout=cfg["dropout"],
            attention_dropout=cfg["attention_dropout"],
            activation=cfg["activation"],
            pos_encoding_scale=cfg["pos_encoding_scale"],
            use_origin_mask=cfg.get("use_origin_mask", False),
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), betas=(0.9, 0.998), eps=1e-8,
        )

        lr_scheduler = NoamScheduler(
            optimizer,
            d_model=cfg["hidden_dim"],
            warmup_steps=cfg["warmup_steps"],
            factor=cfg["learning_rate_factor"],
        )

        kappa_scheduler = CubicScheduler() if cfg["scheduler"] == "cubic" else LinearScheduler()

        print(f"Align: {'identity (pre-aligned)' if os.path.exists(train_aligned_src) else cfg['align_fn']}")
        print(f"Train: {len(train_dataset):,} pairs, {len(train_loader):,} batches/epoch")
        print(f"Rate reparam: {cfg.get('use_rate_reparam', False)}")
        print(f"Time input: {cfg.get('time_input', 't')}")
        print(f"Clamp kappa: {cfg.get('clamp_kappa', False)} (max={cfg.get('clamp_max', 50.0)})")
        print(f"Origin mask: {cfg.get('use_origin_mask', False)}")

        start_step = 0
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            lr_scheduler.load_state_dict(ckpt.get("lr_scheduler_state", {}))
            start_step = ckpt.get("step", 0)
            print(f"Resumed from step {start_step}")

        print(f"Keep: {keep_checkpoints} latest checkpoints")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Device: {device}")
        print(f"Total steps: {cfg['total_steps']}")
        print(f"Log: {log_path}")
        print("=" * 55)

        total_steps = cfg["total_steps"]
        model.train()
        train_iter = iter(train_loader)

        for step in range(start_step, total_steps):
            try:
                x_0, x_1 = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x_0, x_1 = next(train_iter)

            batch = prepare_batch(
                x_0, x_1, kappa_scheduler, align_fn,
                model_vocab_size=model_vocab,
                use_origin_mask=cfg.get("use_origin_mask", False),
            )
            batch = {k: v.to(device) for k, v in batch.items()}

            metrics = train_step(
                model, batch, kappa_scheduler, optimizer,
                max_grad_norm=cfg["max_grad_norm"],
                use_rate_reparam=cfg.get("use_rate_reparam", False),
                clamp_kappa=cfg.get("clamp_kappa", False),
                clamp_max=cfg.get("clamp_max", 50.0),
                time_input=cfg.get("time_input", "t"),
            )
            lr_scheduler.step()

            if step % 100 == 0:
                print(
                    f"step {step:>8}/{total_steps} | "
                    f"loss: {metrics['loss']:.4f} | "
                    f"lr: {lr_scheduler.get_lr():.2e} | "
                    f"u_tot: {metrics['u_tot']:6.2f} | "
                    f"ins: {metrics['u_ins']:6.2f} | "
                    f"del: {metrics['u_del']:6.2f} | "
                    f"sub: {metrics['u_sub']:6.2f}"
                )

            if step > 0 and step % 10000 == 0:
                save_checkpoint(
                    save_dir, step, model, optimizer, lr_scheduler,
                    cfg, real_vocab_size, model_vocab, keep_checkpoints,
                )
                print(f"--- Checkpoint saved (step={step})")

        save_checkpoint(
            save_dir, total_steps, model, optimizer, lr_scheduler,
            cfg, real_vocab_size, model_vocab, keep_checkpoints,
        )
        print("=" * 55)
        print(f"Training complete. Final model saved to {save_dir}")


if __name__ == "__main__":
    main()
