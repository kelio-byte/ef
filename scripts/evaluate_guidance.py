#!/usr/bin/env python
"""Evaluate a guidance checkpoint with the shared-anchor diagnostics.

This is a read-only evaluator.  It never updates model parameters, creates a
new checkpoint, or reads reaction targets.  It is intended to put old and new
guidance checkpoints on the same product-group/pairwise metric scale before
any sampling experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from edit_flows.guidance.data import (
    GuidanceDataset,
    ProductGroupBatchSampler,
    collate_guidance_records,
)
from edit_flows.guidance.model import ProductConditionedGuidance
from edit_flows.guidance.training import evaluate_guidance_step
from edit_flows.utils.tokens import PAD_TOKEN


_PAIR_WEIGHTED_METRICS = {
    "pair_accuracy_strict",
    "pair_accuracy_tie_half",
    "pair_tie_fraction",
    "pair_margin_mean",
}
_GROUP_WEIGHTED_METRICS = {
    "score_calibration_loss",
}


def _load_model(checkpoint_path: str, device: torch.device) -> ProductConditionedGuidance:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False,
    )
    config = checkpoint.get("config", {})
    vocab_size = int(config.get("model_vocab", 0))
    if vocab_size < 1:
        raise ValueError("guidance checkpoint is missing a positive model_vocab")
    model = ProductConditionedGuidance(
        vocab_size=vocab_size,
        hidden_dim=int(config.get("hidden_dim", 256)),
        product_layers=int(config.get("product_layers", 2)),
        state_layers=int(config.get("state_layers", 4)),
        num_heads=int(config.get("num_heads", 8)),
        dim_feedforward=int(config.get("dim_feedforward", 1024)),
        max_seq_len=int(config.get("max_seq_len", 256)),
        dropout=float(config.get("dropout", 0.1)),
        attention_dropout=float(config.get("attention_dropout", 0.1)),
        activation=config.get("activation", "relu"),
        pos_encoding_scale=not bool(config.get("no_pos_encoding_scale", False)),
        pad_token=PAD_TOKEN,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {
        key: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def evaluate(args: argparse.Namespace) -> dict:
    if args.batch_size < 1 or args.group_size < 1:
        raise ValueError("batch_size and group_size must be positive")
    device = torch.device(args.device)
    dataset = GuidanceDataset(args.data)
    sampler = ProductGroupBatchSampler(
        dataset,
        batch_size=args.batch_size,
        group_size=args.group_size,
        shuffle=False,
        seed=0,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_guidance_records,
        pin_memory=device.type == "cuda",
    )
    model = _load_model(args.checkpoint, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    totals: dict[str, float] = {}
    weights: dict[str, float] = {}
    record_count = 0
    batch_count = 0
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            batch = _move_batch(raw_batch, device)
            metrics = evaluate_guidance_step(
                model,
                batch,
                pairwise_group_size=args.group_size,
                pairwise_all_anchors=args.all_anchors,
                score_calibration_weight=args.score_calibration_weight,
            )
            batch_size = int(batch["reward"].shape[0])
            record_count += batch_size
            batch_count += 1
            pair_count = max(float(metrics.get("pair_count", 0.0)), 0.0)
            candidate_count = max(float(metrics.get("candidate_pair_count", 0.0)), 0.0)
            calibration_group_count = max(
                float(metrics.get("score_calibration_group_count", 0.0)), 0.0,
            )
            for name, value in metrics.items():
                if name in _PAIR_WEIGHTED_METRICS and pair_count > 0:
                    totals[name] = totals.get(name, 0.0) + value * pair_count
                    weights[name] = weights.get(name, 0.0) + pair_count
                elif name in _GROUP_WEIGHTED_METRICS and calibration_group_count > 0:
                    totals[name] = totals.get(name, 0.0) + value * calibration_group_count
                    weights[name] = weights.get(name, 0.0) + calibration_group_count
                elif name in {
                    "reward_score_pearson",
                    "reward_score_pearson_within_group",
                } and candidate_count > 0:
                    totals[name] = totals.get(name, 0.0) + value * candidate_count
                    weights[name] = weights.get(name, 0.0) + candidate_count
                elif name in {"pair_count", "candidate_pair_count"}:
                    totals[name] = totals.get(name, 0.0) + value
                else:
                    totals[name] = totals.get(name, 0.0) + value * batch_size
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    metrics = {
        name: value / weights[name]
        if name in weights and weights[name] > 0
        else value / max(record_count, 1)
        for name, value in totals.items()
    }
    for name in ("pair_count", "candidate_pair_count"):
        if name in totals:
            metrics[name] = totals[name]
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": str(Path(args.data).resolve()),
        "device": str(device),
        "batch_size": args.batch_size,
        "group_size": args.group_size,
        "all_anchors": args.all_anchors,
        "records": record_count,
        "groups": record_count // args.group_size,
        "batches": batch_count,
        "wall_seconds": wall_seconds,
        "records_per_second": record_count / max(wall_seconds, 1e-9),
        "peak_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0
        ),
        "peak_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda" else 0
        ),
        "metrics": metrics,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--all_anchors", action="store_true")
    parser.add_argument("--score_calibration_weight", type=float, default=0.0)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
