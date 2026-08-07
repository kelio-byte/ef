#!/usr/bin/env python
"""Train the small action-level DGM guidance adapter.

The base Edit Flows model is never loaded or modified here.  This script only
consumes guidance-record ``.pt`` files, trains ``ProductConditionedGuidance``,
and writes an independent checkpoint plus TensorBoard scalars.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader

from edit_flows.guidance.data import GuidanceDataset, collate_guidance_records
from edit_flows.guidance.model import ProductConditionedGuidance
from edit_flows.guidance.training import (
    evaluate_guidance_step,
    train_guidance_step,
)
from edit_flows.utils.tokens import PAD_TOKEN


def _set_seed(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(args: argparse.Namespace, vocab_size: int) -> ProductConditionedGuidance:
    return ProductConditionedGuidance(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        product_layers=args.product_layers,
        state_layers=args.state_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation=args.activation,
        pos_encoding_scale=not args.no_pos_encoding_scale,
        pad_token=PAD_TOKEN,
    )


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {
        key: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _write_scalars(writer, prefix: str, metrics: dict[str, float], step: int) -> None:
    for name, value in metrics.items():
        writer.add_scalar(f"{prefix}/{name}", value, step)


def run(args: argparse.Namespace) -> dict:
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("batch_size and epochs must be positive")
    if args.max_steps < 0 or args.val_interval < 1 or args.log_interval < 1:
        raise ValueError("max_steps must be non-negative and intervals positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    _set_seed(args.seed)
    device = torch.device(args.device)
    train_dataset = GuidanceDataset(args.train_data)
    val_dataset = GuidanceDataset(args.val_data) if args.val_data else None
    metadata_vocab = train_dataset.metadata.get("model_vocab")
    vocab_size = args.model_vocab or metadata_vocab
    if vocab_size is None:
        raise ValueError("model vocab is missing; pass --model_vocab")
    vocab_size = int(vocab_size)
    if vocab_size < 1:
        raise ValueError("model vocab must be positive")
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_guidance_records,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_guidance_records,
            pin_memory=device.type == "cuda",
        )

    model = _build_model(args, vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["device"] = str(device)
    config["model_vocab"] = vocab_size
    config["train_records"] = len(train_dataset)
    config["val_records"] = len(val_dataset) if val_dataset is not None else 0
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "TensorBoard is required for train_guidance.py; install tensorboard"
        ) from exc
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    started = time.perf_counter()
    global_step = 0
    epochs_completed = 0
    last_train_metrics: dict[str, float] = {}
    last_val_metrics: dict[str, float] = {}
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        for raw_batch in train_loader:
            batch = _move_batch(raw_batch, device)
            metrics = train_guidance_step(
                model,
                batch,
                optimizer,
                background=args.background,
                max_grad_norm=args.max_grad_norm,
            )
            global_step += 1
            last_train_metrics = metrics
            _write_scalars(writer, "train", metrics, global_step)
            if global_step == 1 or global_step % args.log_interval == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"step {global_step} | epoch {epoch + 1} | "
                    f"loss {metrics['loss']:.6f} | reward {metrics['reward_mean']:.4f} | "
                    f"elapsed {elapsed:.1f}s",
                    flush=True,
                )
            if val_loader is not None and global_step % args.val_interval == 0:
                val_totals: dict[str, float] = {}
                val_count = 0
                for val_index, val_raw_batch in enumerate(val_loader):
                    if args.val_batches > 0 and val_index >= args.val_batches:
                        break
                    val_batch = _move_batch(val_raw_batch, device)
                    val_metrics = evaluate_guidance_step(
                        model, val_batch, background=args.background,
                    )
                    val_batch_size = val_batch["reward"].shape[0]
                    val_count += val_batch_size
                    for name, value in val_metrics.items():
                        val_totals[name] = val_totals.get(name, 0.0) + value * val_batch_size
                if val_count:
                    last_val_metrics = {
                        name: value / val_count
                        for name, value in val_totals.items()
                    }
                    _write_scalars(writer, "validation", last_val_metrics, global_step)
                    print(
                        f"validation step {global_step} | "
                        f"loss {last_val_metrics['loss']:.6f}", flush=True,
                    )
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop = True
                break
        epochs_completed = epoch + 1

    wall_seconds = time.perf_counter() - started
    checkpoint = {
        "schema_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "train_metadata": train_dataset.metadata,
        "val_metadata": val_dataset.metadata if val_dataset is not None else None,
        "global_step": global_step,
        "epochs_completed": epochs_completed,
        "wall_seconds": wall_seconds,
        "last_train_metrics": last_train_metrics,
        "last_val_metrics": last_val_metrics,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_path = output_dir / "guidance_final.pt"
    torch.save(checkpoint, checkpoint_path)
    writer.flush()
    writer.close()
    summary = {
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "steps": global_step,
        "epochs": epochs_completed,
        "wall_seconds": wall_seconds,
        "last_train_loss": last_train_metrics.get("loss"),
        "last_validation_loss": last_val_metrics.get("loss"),
    }
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--val_batches", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--background", type=float, default=1e-4)
    parser.add_argument("--model_vocab", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--product_layers", type=int, default=2)
    parser.add_argument("--state_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--no_pos_encoding_scale", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
