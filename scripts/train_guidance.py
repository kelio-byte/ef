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

from edit_flows.guidance.data import (
    GuidanceDataset,
    ProductGroupBatchSampler,
    collate_guidance_records,
)
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


def _load_control_bregman_loss(path: str | None) -> float | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    for key in ("best_validation_bregman_loss", "best_validation_loss"):
        value = payload.get(key)
        if value is not None and torch.isfinite(torch.tensor(float(value))):
            return float(value)
    raise ValueError(
        f"control metrics JSON {path} has no finite best validation Bregman loss"
    )


def _pairwise_metric_weight(name: str, metrics: dict[str, float]) -> float:
    if name in {
        "pair_accuracy_strict",
        "pair_accuracy_tie_half",
        "pair_tie_fraction",
        "pair_margin_mean",
    }:
        return max(float(metrics.get("pair_count", 0.0)), 0.0)
    if name == "reward_score_pearson":
        return max(float(metrics.get("candidate_pair_count", 0.0)), 0.0)
    return 0.0


def run(args: argparse.Namespace) -> dict:
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("batch_size and epochs must be positive")
    if args.max_steps < 0 or args.val_interval < 1 or args.log_interval < 1:
        raise ValueError("max_steps must be non-negative and intervals positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.group_size < 1:
        raise ValueError("group_size must be positive")
    if args.pairwise_loss_weight < 0 or not torch.isfinite(
        torch.tensor(args.pairwise_loss_weight)
    ):
        raise ValueError("pairwise_loss_weight must be finite and non-negative")
    if args.pairwise_temperature <= 0 or not torch.isfinite(
        torch.tensor(args.pairwise_temperature)
    ):
        raise ValueError("pairwise_temperature must be finite and positive")
    if args.pairwise_equal_tolerance < 0 or not torch.isfinite(
        torch.tensor(args.pairwise_equal_tolerance)
    ):
        raise ValueError("pairwise_equal_tolerance must be finite and non-negative")
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
    grouped_batches = (
        args.use_grouped_batches
        or args.pairwise_loss_weight > 0
        or args.pairwise_all_val_anchors
        or args.checkpoint_selection == "pairwise_guarded"
    )
    train_batch_sampler = None
    if grouped_batches:
        train_batch_sampler = ProductGroupBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
            group_size=args.group_size,
            shuffle=True,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_guidance_records,
            pin_memory=device.type == "cuda",
        )
    else:
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
    val_batch_sampler = None
    if val_dataset is not None:
        if grouped_batches:
            val_batch_sampler = ProductGroupBatchSampler(
                val_dataset,
                batch_size=args.batch_size,
                group_size=args.group_size,
                shuffle=False,
                seed=args.seed,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_sampler=val_batch_sampler,
                num_workers=args.num_workers,
                collate_fn=collate_guidance_records,
                pin_memory=device.type == "cuda",
            )
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_guidance_records,
                pin_memory=device.type == "cuda",
            )

    model = _build_model(args, vocab_size).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
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
    config["grouped_batches"] = grouped_batches
    config["train_group_count"] = (
        train_batch_sampler.group_count if train_batch_sampler is not None else None
    )
    config["val_group_count"] = (
        val_batch_sampler.group_count if val_batch_sampler is not None else None
    )
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
    best_validation_loss = float("inf")
    best_validation_bregman_loss = float("inf")
    best_pairwise_accuracy = float("-inf")
    best_pairwise_pearson = float("-inf")
    best_validation_step = 0
    control_bregman_loss = _load_control_bregman_loss(args.control_metrics_json)
    if args.checkpoint_selection == "pairwise_guarded" and control_bregman_loss is None:
        raise ValueError(
            "pairwise_guarded selection requires --control_metrics_json"
        )
    pairwise_bregman_guard = (
        1.15 * control_bregman_loss
        if control_bregman_loss is not None else None
    )
    selected_checkpoint_eligible = False
    best_checkpoint_path = output_dir / "guidance_best.pt"
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        for raw_batch in train_loader:
            batch = _move_batch(raw_batch, device)
            metrics = train_guidance_step(
                model,
                batch,
                optimizer,
                background=args.background,
                background_loss_weight=args.background_loss_weight,
                max_grad_norm=args.max_grad_norm,
                pairwise_loss_weight=args.pairwise_loss_weight,
                pairwise_temperature=args.pairwise_temperature,
                pairwise_equal_tolerance=args.pairwise_equal_tolerance,
                pairwise_group_size=args.group_size,
                pairwise_anchor_rotation=global_step % args.group_size,
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
                val_pairwise_totals: dict[str, float] = {}
                val_pairwise_weights: dict[str, float] = {}
                val_count = 0
                for val_index, val_raw_batch in enumerate(val_loader):
                    if args.val_batches > 0 and val_index >= args.val_batches:
                        break
                    val_batch = _move_batch(val_raw_batch, device)
                    val_metrics = evaluate_guidance_step(
                        model,
                        val_batch,
                        background=args.background,
                        background_loss_weight=args.background_loss_weight,
                        pairwise_loss_weight=args.pairwise_loss_weight,
                        pairwise_temperature=args.pairwise_temperature,
                        pairwise_equal_tolerance=args.pairwise_equal_tolerance,
                        pairwise_group_size=args.group_size,
                        pairwise_all_anchors=args.pairwise_all_val_anchors,
                    )
                    val_batch_size = val_batch["reward"].shape[0]
                    val_count += val_batch_size
                    for name, value in val_metrics.items():
                        pairwise_weight = _pairwise_metric_weight(name, val_metrics)
                        if pairwise_weight > 0:
                            val_pairwise_totals[name] = (
                                val_pairwise_totals.get(name, 0.0)
                                + value * pairwise_weight
                            )
                            val_pairwise_weights[name] = (
                                val_pairwise_weights.get(name, 0.0)
                                + pairwise_weight
                            )
                        elif name == "pair_count":
                            val_totals[name] = val_totals.get(name, 0.0) + value
                        elif name == "candidate_pair_count":
                            val_totals[name] = val_totals.get(name, 0.0) + value
                        else:
                            val_totals[name] = val_totals.get(name, 0.0) + value * val_batch_size
                if val_count:
                    last_val_metrics = {
                        name: value / val_count
                        for name, value in val_totals.items()
                    }
                    for name, total in val_pairwise_totals.items():
                        weight = val_pairwise_weights[name]
                        if weight > 0:
                            last_val_metrics[name] = total / weight
                    for name in ("pair_count", "candidate_pair_count"):
                        if name in val_totals:
                            last_val_metrics[name] = val_totals[name]
                    _write_scalars(writer, "validation", last_val_metrics, global_step)
                    print(
                        f"validation step {global_step} | "
                        f"loss {last_val_metrics['loss']:.6f}", flush=True,
                    )
                    validation_bregman = last_val_metrics.get(
                        "loss_bregman", last_val_metrics["loss"],
                    )
                    validation_pairwise_accuracy = last_val_metrics.get(
                        "pair_accuracy_tie_half", float("-inf"),
                    )
                    validation_pairwise_pearson = last_val_metrics.get(
                        "reward_score_pearson", float("-inf"),
                    )
                    if args.checkpoint_selection == "pairwise_guarded":
                        eligible = (
                            pairwise_bregman_guard is not None
                            and validation_bregman <= pairwise_bregman_guard
                        )
                        better = False
                        if eligible:
                            accuracy_delta = validation_pairwise_accuracy - best_pairwise_accuracy
                            if accuracy_delta > 0.005:
                                better = True
                            elif abs(accuracy_delta) <= 0.005:
                                pearson_delta = validation_pairwise_pearson - best_pairwise_pearson
                                if pearson_delta > 1e-9:
                                    better = True
                                elif abs(pearson_delta) <= 1e-9 and validation_bregman < best_validation_bregman_loss:
                                    better = True
                        should_save = better
                    else:
                        eligible = True
                        should_save = last_val_metrics["loss"] < best_validation_loss
                    if should_save:
                        best_validation_loss = last_val_metrics["loss"]
                        best_validation_bregman_loss = validation_bregman
                        best_pairwise_accuracy = validation_pairwise_accuracy
                        best_pairwise_pearson = validation_pairwise_pearson
                        best_validation_step = global_step
                        selected_checkpoint_eligible = eligible
                        torch.save({
                            "schema_version": 1,
                            "checkpoint_type": "best_pairwise_guarded" if args.checkpoint_selection == "pairwise_guarded" else "best_validation",
                            "selection_metric": "validation/pair_accuracy_tie_half" if args.checkpoint_selection == "pairwise_guarded" else "validation/loss",
                            "selection_rule": args.checkpoint_selection,
                            "selection_eligible": eligible,
                            "control_bregman_loss": control_bregman_loss,
                            "bregman_guard": pairwise_bregman_guard,
                            "model_state_dict": model.state_dict(),
                            "config": config,
                            "train_metadata": train_dataset.metadata,
                            "val_metadata": val_dataset.metadata,
                            "global_step": global_step,
                            "epochs_completed": epoch + 1,
                            "best_validation_loss": best_validation_loss,
                            "best_validation_bregman_loss": best_validation_bregman_loss,
                            "best_pairwise_accuracy": best_pairwise_accuracy,
                            "best_pairwise_pearson": best_pairwise_pearson,
                            "best_validation_step": best_validation_step,
                            "last_train_metrics": last_train_metrics,
                            "last_val_metrics": last_val_metrics,
                            "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        }, best_checkpoint_path)
                    writer.add_scalar(
                        "validation/best_loss", best_validation_loss, global_step,
                    )
                    writer.add_scalar(
                        "validation/best_bregman_loss",
                        best_validation_bregman_loss,
                        global_step,
                    )
                    if best_pairwise_accuracy != float("-inf"):
                        writer.add_scalar(
                            "validation/best_pairwise_accuracy",
                            best_pairwise_accuracy,
                            global_step,
                        )
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop = True
                break
        epochs_completed = epoch + 1

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    peak_memory_allocated = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda" else 0
    )
    peak_memory_reserved = (
        int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda" else 0
    )
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
        "peak_memory_allocated_bytes": peak_memory_allocated,
        "peak_memory_reserved_bytes": peak_memory_reserved,
        "last_train_metrics": last_train_metrics,
        "last_val_metrics": last_val_metrics,
        "best_validation_loss": (
            best_validation_loss if best_validation_step else None
        ),
        "best_validation_bregman_loss": (
            best_validation_bregman_loss if best_validation_step else None
        ),
        "best_pairwise_accuracy": (
            best_pairwise_accuracy if best_validation_step and best_pairwise_accuracy != float("-inf") else None
        ),
        "best_pairwise_pearson": (
            best_pairwise_pearson if best_validation_step and best_pairwise_pearson != float("-inf") else None
        ),
        "best_validation_step": best_validation_step or None,
        "selection_rule": args.checkpoint_selection,
        "selection_eligible": selected_checkpoint_eligible,
        "control_bregman_loss": control_bregman_loss,
        "bregman_guard": pairwise_bregman_guard,
        "best_checkpoint": (
            str(best_checkpoint_path) if best_validation_step else None
        ),
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
        "peak_memory_allocated_bytes": peak_memory_allocated,
        "peak_memory_reserved_bytes": peak_memory_reserved,
        "last_train_loss": last_train_metrics.get("loss"),
        "last_validation_loss": last_val_metrics.get("loss"),
        "last_validation_bregman_loss": last_val_metrics.get("loss_bregman"),
        "last_validation_pairwise_accuracy": last_val_metrics.get(
            "pair_accuracy_tie_half"
        ),
        "best_validation_bregman_loss": (
            best_validation_bregman_loss if best_validation_step else None
        ),
        "best_pairwise_accuracy": (
            best_pairwise_accuracy if best_validation_step and best_pairwise_accuracy != float("-inf") else None
        ),
        "best_pairwise_pearson": (
            best_pairwise_pearson if best_validation_step and best_pairwise_pearson != float("-inf") else None
        ),
        "best_validation_loss": (
            best_validation_loss if best_validation_step else None
        ),
        "best_validation_step": best_validation_step or None,
        "selection_rule": args.checkpoint_selection,
        "selection_eligible": selected_checkpoint_eligible,
        "control_bregman_loss": control_bregman_loss,
        "bregman_guard": pairwise_bregman_guard,
        "best_checkpoint": (
            str(best_checkpoint_path) if best_validation_step else None
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
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
    parser.add_argument("--background_loss_weight", type=float, default=0.01)
    parser.add_argument(
        "--use_grouped_batches",
        action="store_true",
        help="Keep all records with one source_index in the same batch",
    )
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--pairwise_loss_weight", type=float, default=0.0)
    parser.add_argument("--pairwise_temperature", type=float, default=1.0)
    parser.add_argument("--pairwise_equal_tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--pairwise_all_val_anchors",
        action="store_true",
        help="Evaluate all records in each product group as validation anchors",
    )
    parser.add_argument(
        "--checkpoint_selection",
        choices=("validation_loss", "pairwise_guarded"),
        default="validation_loss",
    )
    parser.add_argument(
        "--control_metrics_json",
        default=None,
        help="Control summary JSON used for pairwise Bregman guardrail",
    )
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
