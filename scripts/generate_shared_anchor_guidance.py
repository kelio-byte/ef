#!/usr/bin/env python
"""Generate guidance records with a genuinely shared intermediate anchor.

For each product this script first samples one ordinary Euler prefix to a
fixed interior time.  The exact prefix state is then repeated ``n_children``
times and all rows are continued in one vectorized Euler call.  Consequently
records in one ``source_index`` group share both ``state_tokens`` and ``time``;
their terminal continuations remain stochastic and independent in the batched
RNG stream.

The output initially carries the cheap RDKit validity reward.  Use
``scripts/generate_forward_guidance_data.py`` afterwards to attach the audited
Molecular Transformer forward-beam reward without changing the anchor fields.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from edit_flows.guidance.data import make_guidance_record, save_guidance_dataset
from edit_flows.guidance.rewards import retro_tokenized_validity_reward
from edit_flows.sampling.euler import sample_euler

try:  # Works both as ``python scripts/foo.py`` and as a test import.
    from scripts.generate_guidance_data import (
        _decode_rows,
        _load_model,
        _make_initial_batch,
        _mix_seed,
        _read_original_products,
        _trim_pad,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI execution
    from generate_guidance_data import (
        _decode_rows,
        _load_model,
        _make_initial_batch,
        _mix_seed,
        _read_original_products,
        _trim_pad,
    )


def validate_shared_anchor_config(
    n_steps: int, anchor_time: float, n_children: int,
) -> int:
    """Validate CLI values and return the integer anchor step."""
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    if n_children < 2:
        raise ValueError("n_children must be >= 2 for shared-anchor data")
    if not 0.0 < float(anchor_time) < 1.0:
        raise ValueError("anchor_time must be strictly inside (0, 1)")
    anchor_index = int(round(float(anchor_time) * n_steps))
    if anchor_index < 1 or anchor_index >= n_steps:
        raise ValueError(
            "anchor_time must map to an interior Euler step; "
            f"got step {anchor_index} for n_steps={n_steps}"
        )
    return anchor_index


def generate(args: argparse.Namespace) -> dict:
    anchor_index = validate_shared_anchor_config(
        args.n_steps, args.anchor_time, args.n_children,
    )
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {output}; pass --overwrite to replace it"
        )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    (
        checkpoint_data, cfg, model, token2id, id2token,
        sample_scheduler, train_scheduler, use_origin_mask,
    ) = _load_model(
        args.checkpoint, device, args.data_dir, args.vocab_file,
    )
    if use_origin_mask:
        raise ValueError(
            "shared-anchor generation currently requires use_origin_mask=False; "
            "the checkpoint must provide an origin-mask continuation state"
        )
    products = _read_original_products(args.products_file, args.augmentation)
    if args.max_products is not None:
        if args.max_products < 1:
            raise ValueError("max_products must be >= 1")
        products = products[:args.max_products]
    unk_id = token2id.get("<unk>", 2)
    product_ids = [
        [token2id.get(token, unk_id) for token in product.split()]
        for product in products
    ]

    records = []
    started = time.perf_counter()
    total_batches = (len(products) + args.batch_size - 1) // args.batch_size
    for batch_index, batch_start in enumerate(
        range(0, len(products), args.batch_size), start=1,
    ):
        batch_products = products[batch_start:batch_start + args.batch_size]
        batch_ids = product_ids[batch_start:batch_start + args.batch_size]
        batch_seed = _mix_seed(args.seed, batch_start, 0)
        torch.manual_seed(batch_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(batch_seed)

        # One prefix per product creates the common state.  The trajectory
        # contains the exact post-edit state at each Euler step.
        x0_cpu = _make_initial_batch(batch_ids, 1)
        x0 = x0_cpu.to(device)
        _, trajectory = sample_euler(
            model, x0, sample_scheduler,
            n_steps=args.n_steps,
            max_seq_len=cfg["max_seq_len"],
            use_rate_reparam=cfg.get("use_rate_reparam", False),
            clamp_kappa=cfg.get("clamp_kappa", False),
            clamp_max=cfg.get("clamp_max", 50.0),
            time_input=cfg.get("time_input", "t"),
            train_scheduler=train_scheduler,
            record_trajectory=True,
            use_origin_mask=False,
        )
        actual_steps = len(trajectory) - 1
        if anchor_index >= actual_steps:
            raise RuntimeError(
                f"Euler produced only {actual_steps} steps, cannot use "
                f"anchor step {anchor_index}"
            )
        anchor_state = trajectory[anchor_index].to(device)
        anchor_time = anchor_index / actual_steps

        # Repeat the anchor and product in one batch.  This is the critical
        # vectorization point: model calls are per continuation batch, not per
        # child or per CPU record.
        continuation_state = anchor_state.repeat_interleave(args.n_children, dim=0)
        continuation_product = x0.repeat_interleave(args.n_children, dim=0)
        continuation_seed = _mix_seed(
            args.seed, batch_start, anchor_index + 100003,
        )
        torch.manual_seed(continuation_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(continuation_seed)
        terminal_states, _ = sample_euler(
            model, continuation_state, sample_scheduler,
            n_steps=args.n_steps,
            max_seq_len=cfg["max_seq_len"],
            use_rate_reparam=cfg.get("use_rate_reparam", False),
            clamp_kappa=cfg.get("clamp_kappa", False),
            clamp_max=cfg.get("clamp_max", 50.0),
            time_input=cfg.get("time_input", "t"),
            train_scheduler=train_scheduler,
            use_origin_mask=False,
            start_time=anchor_time,
        )
        terminal_text = _decode_rows(terminal_states, id2token)
        rewards = retro_tokenized_validity_reward(terminal_text)

        for product_offset, _ in enumerate(batch_products):
            source_index = batch_start + product_offset
            product_row = x0_cpu[product_offset]
            state_row = anchor_state[product_offset].detach().cpu()
            for child_index in range(args.n_children):
                row_index = product_offset * args.n_children + child_index
                child_seed = _mix_seed(
                    args.seed, source_index, child_index + 1,
                )
                coupling_seed = _mix_seed(
                    args.seed, source_index, 100000 + child_index + 1,
                )
                records.append(make_guidance_record(
                    product_tokens=_trim_pad(product_row.tolist()),
                    state_tokens=_trim_pad(state_row.tolist()),
                    terminal_tokens=_trim_pad(terminal_states[row_index].detach().cpu().tolist()),
                    time_step=anchor_time,
                    reward=float(rewards[row_index].item()),
                    source_index=source_index,
                    sample_index=child_index,
                    time_index=anchor_index,
                    sample_seed=child_seed,
                    coupling_seed=coupling_seed,
                ))

        if batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0:
            elapsed = time.perf_counter() - started
            rate = batch_index / max(elapsed, 1e-9)
            eta = (total_batches - batch_index) / max(rate, 1e-9)
            print(
                f"Shared-anchor batches {batch_index}/{total_batches} | "
                f"products {min(batch_start + args.batch_size, len(products))}/"
                f"{len(products)} | elapsed {elapsed:.1f}s | ETA {eta:.1f}s",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    metadata = {
        "schema_version": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "products_file": str(Path(args.products_file).resolve()),
        "product_count": len(products),
        "record_count": len(records),
        "augmentation": args.augmentation,
        "n_steps": args.n_steps,
        "n_children": args.n_children,
        "anchor_time_requested": float(args.anchor_time),
        "anchor_time": anchor_time,
        "anchor_time_index": anchor_index,
        "shared_anchor": True,
        "rng_scope": "product-batch prefix and vectorized continuation seeds",
        "sampler": "euler_shared_anchor_continuation",
        "reward": "rdkit_validity",
        "sample_scheduler": sample_scheduler.name,
        "train_scheduler": train_scheduler.name,
        "model_vocab": checkpoint_data.get("model_vocab", len(token2id)),
        "generation_wall_seconds": wall,
        "generation_records_per_second": len(records) / max(wall, 1e-9),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "batch_count": total_batches,
    }
    save_guidance_dataset(output, records, metadata=metadata)
    return {
        "output": str(output),
        "products": len(products),
        "records": len(records),
        "reward_mean": float(torch.tensor([r["reward"] for r in records]).mean()),
        "wall_seconds": wall,
        "records_per_second": len(records) / max(wall, 1e-9),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--products_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--vocab_file", default=None)
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--max_products", type=int, default=None)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--n_children", type=int, default=4)
    parser.add_argument("--anchor_time", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(generate(args))


if __name__ == "__main__":
    main()
