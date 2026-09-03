#!/usr/bin/env python
"""Run compact natural-trajectory diagnostics for the correction study.

This entry point deliberately does not change the formal evaluator.  It
selects one augmentation view per reaction, runs ordinary Euler sampling, and
stores compact event traces suitable for checking whether a later SUB/DEL
action follows an initially off-oracle edit.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

from edit_flows.analysis.first_step import (
    build_model_batch,
    load_parallel_texts,
    tokenize_smiles,
)
from edit_flows.analysis.trajectory_correction import (
    aggregate_trace_summaries,
    summarize_trace,
)
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.data.dataset import load_vocab
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.sampling.euler import sample_euler


def _load_checkpoint(path: str, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, map_location=device)


def _scheduler(name: str):
    if name == "linear":
        return LinearScheduler()
    if name == "cubic":
        return CubicScheduler()
    raise ValueError(f"Unsupported scheduler: {name}")


def _load_model(
    checkpoint: dict,
    device: torch.device,
    vocab_file: str | None,
) -> tuple[torch.nn.Module, dict[int, str], dict]:
    cfg = checkpoint["config"]
    data_dir = cfg["data_dir"]
    vocab_path = vocab_file or os.path.join(
        data_dir, cfg.get("vocab_file", "example.vocab.src")
    )
    token2id, model_vocab = load_vocab(vocab_path)
    id2token = {int(value): key for key, value in token2id.items()}
    model = EditFlowsTransformer(
        vocab_size=checkpoint.get("model_vocab", model_vocab),
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        dim_feedforward=cfg["dim_feedforward"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
        attention_dropout=cfg.get("attention_dropout", cfg["dropout"]),
        activation=cfg.get("activation", "relu"),
        pos_encoding_scale=cfg.get("pos_encoding_scale", True),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, id2token, cfg


def _first_augmentation(
    products: list[str],
    targets: list[str],
    *,
    augmentation: int,
    max_reactions: int,
) -> tuple[list[str], list[str], list[int]]:
    if augmentation < 1:
        raise ValueError("augmentation must be positive")
    if len(products) != len(targets):
        raise ValueError("products and targets must have the same number of rows")
    if len(products) % augmentation:
        raise ValueError(
            f"input rows={len(products)} is not divisible by augmentation={augmentation}"
        )
    reaction_indices = list(range(len(products) // augmentation))[:max_reactions]
    selected = [index * augmentation for index in reaction_indices]
    return (
        [products[index] for index in selected],
        [targets[index] for index in selected],
        reaction_indices,
    )



def _select_augmentation_view(
    products: list[str],
    targets: list[str],
    *,
    augmentation: int,
    max_reactions: int,
    augmentation_index: int,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Select one fixed augmentation view and preserve source-row provenance."""
    if augmentation < 1:
        raise ValueError("augmentation must be positive")
    if augmentation_index < 0 or augmentation_index >= augmentation:
        raise ValueError(
            f"augmentation_index must be in [0, {augmentation - 1}], "
            f"got {augmentation_index}"
        )
    if len(products) != len(targets):
        raise ValueError("products and targets must have the same number of rows")
    if len(products) % augmentation:
        raise ValueError(
            f"input rows={len(products)} is not divisible by augmentation={augmentation}"
        )
    reaction_indices = list(range(len(products) // augmentation))[:max_reactions]
    source_row_indices = [
        reaction_index * augmentation + augmentation_index
        for reaction_index in reaction_indices
    ]
    return (
        [products[index] for index in source_row_indices],
        [targets[index] for index in source_row_indices],
        reaction_indices,
        source_row_indices,
    )


def _sampling_kwargs(model, cfg: dict, scheduler, train_scheduler, args: argparse.Namespace):
    return {
        "scheduler": scheduler,
        "n_steps": args.n_steps,
        "max_seq_len": int(cfg["max_seq_len"]),
        "use_rate_reparam": bool(cfg.get("use_rate_reparam", False)),
        "clamp_kappa": bool(cfg.get("clamp_kappa", False)),
        "clamp_max": float(cfg.get("clamp_max", 50.0)),
        "time_input": cfg.get("time_input", "t"),
        "train_scheduler": train_scheduler,
        "event_prob_mode": args.event_prob_mode,
        "x_1": None,
        "vocab_size": model.vocab_size,
    }


def run_analysis(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    checkpoint = _load_checkpoint(args.checkpoint, device)
    model, id2token, cfg = _load_model(
        checkpoint, device, args.vocab_file,
    )

    all_products, all_targets = load_parallel_texts(
        args.products_file, args.targets_file,
    )
    products, targets, reaction_indices, source_row_indices = _select_augmentation_view(
        all_products,
        all_targets,
        augmentation=args.augmentation,
        max_reactions=args.max_reactions,
        augmentation_index=args.augmentation_index,
    )
    product_ids = [tokenize_smiles(value, {token: index for index, token in id2token.items()}) for value in products]
    target_ids = [tokenize_smiles(value, {token: index for index, token in id2token.items()}) for value in targets]

    scheduler_name = args.scheduler or cfg.get(
        "sample_scheduler", cfg.get("scheduler", "cubic")
    )
    scheduler = _scheduler(scheduler_name)
    train_scheduler = _scheduler(cfg.get("scheduler", "cubic"))
    all_rows: list[dict] = []
    verification: list[dict] = []
    start_time = time.perf_counter()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    for start in range(0, len(products), args.batch_size):
        end = min(start + args.batch_size, len(products))
        x_0, x_1 = build_model_batch(
            product_ids[start:end], target_ids[start:end],
        )
        batch_size = x_0.shape[0]
        batch_x0 = x_0.repeat_interleave(args.n_samples, dim=0).to(device)
        batch_x1 = x_1.repeat_interleave(args.n_samples, dim=0).to(device)
        kwargs = _sampling_kwargs(model, cfg, scheduler, train_scheduler, args)
        kwargs["x_1"] = batch_x1

        if args.verify_no_record_change:
            batch_seed = args.seed + start
            torch.manual_seed(batch_seed)
            plain, _ = sample_euler(model, batch_x0, **{
                key: value for key, value in kwargs.items()
                if key not in {"x_1", "vocab_size"}
            })
            torch.manual_seed(batch_seed)
            compact, _, events = sample_euler(
                model,
                batch_x0,
                record_compact_events=True,
                **kwargs,
            )
            identical = bool(torch.equal(plain, compact))
            verification.append({
                "batch_start": start,
                "batch_size": batch_size,
                "identical": identical,
            })
            if not identical:
                raise RuntimeError(
                    f"compact recording changed output for batch starting at {start}"
                )
        else:
            compact, _, events = sample_euler(
                model,
                batch_x0,
                record_compact_events=True,
                **kwargs,
            )

        compact_cpu = compact.detach().cpu()
        for local_index in range(batch_size):
            for path_index in range(args.n_samples):
                row_index = local_index * args.n_samples + path_index
                trace = events[row_index]
                final_ids = compact_cpu[row_index].tolist()
                target_row = target_ids[start + local_index]
                metrics = summarize_trace(
                    trace,
                    final_ids,
                    target_row,
                    id2token,
                )
                all_rows.append({
                    "reaction_index": reaction_indices[start + local_index],
                    "augmentation_index": args.augmentation_index,
                    "source_row_index": source_row_indices[start + local_index],
                    "path_index": path_index,
                    "product": products[start + local_index],
                    "target": targets[start + local_index],
                    "final_ids": final_ids,
                    "trace": trace,
                    **metrics,
                })

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    summary = aggregate_trace_summaries(all_rows)
    summary.update({
        "checkpoint": os.path.abspath(args.checkpoint),
        "products_file": os.path.abspath(args.products_file),
        "targets_file": os.path.abspath(args.targets_file),
        "scheduler": scheduler_name,
        "n_steps": args.n_steps,
        "n_samples": args.n_samples,
        "augmentation": args.augmentation,
        "augmentation_index": args.augmentation_index,
        "n_reactions": len(products),
        "selection_layout": "reaction-major, path-minor",
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "verification": verification,
    })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (output_dir / "per_trajectory.jsonl").open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--products_file", required=True)
    parser.add_argument("--targets_file", required=True)
    parser.add_argument("--vocab_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scheduler", choices=["cubic", "linear"], default=None)
    parser.add_argument("--event_prob_mode", choices=["poisson", "linear"], default="poisson")
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument(
        "--augmentation_index",
        type=int,
        default=0,
        help="zero-based augmentation view selected from every reaction block",
    )
    parser.add_argument("--max_reactions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify_no_record_change",
        action="store_true",
        help="run each batch once without and once with compact recording",
    )
    args = parser.parse_args()
    summary = run_analysis(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
