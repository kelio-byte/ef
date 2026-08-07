#!/usr/bin/env python
"""Generate product-conditioned DGM guidance records from ordinary Euler.

This script deliberately does not use Euler-Beam and never reads target files.
It samples terminal candidates from a frozen Edit Flows checkpoint, evaluates a
terminal reward, then samples a small number of aligned intermediate states for
guidance training.  The output is a CPU ``.pt`` artifact consumed by the
guidance-data utilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import time
from typing import Sequence

import torch

from edit_flows.core.alignment import opt_align_xs_to_zs
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.data.dataset import load_vocab
from edit_flows.guidance.data import (
    make_guidance_record,
    sample_intermediate_states,
    save_guidance_dataset,
)
from edit_flows.guidance.rewards import retro_tokenized_validity_reward
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.sampling.euler import sample_euler
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN, UNK_TOKEN


def _mix_seed(base_seed: int, index: int, salt: int) -> int:
    if base_seed < 0 or index < 0 or salt < 0:
        raise ValueError("seed components must be non-negative")
    mask = (1 << 64) - 1
    value = (
        (int(base_seed) & mask)
        ^ (((int(index) + 1) * 0x9E3779B97F4A7C15) & mask)
        ^ (((int(salt) + 1) * 0xD1B54A32D192ED03) & mask)
    )
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & mask
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & mask
    value ^= value >> 31
    return value & ((1 << 63) - 1)


def _trim_pad(row: Sequence[int]) -> list[int]:
    result = []
    for token in row:
        token = int(token)
        if token == PAD_TOKEN:
            break
        result.append(token)
    return result


def _read_original_products(path: str, augmentation: int) -> list[str]:
    if augmentation < 1:
        raise ValueError("augmentation must be >= 1")
    lines = Path(path).read_text().splitlines()
    if len(lines) % augmentation:
        raise ValueError(
            f"{path} has {len(lines)} lines, which is not divisible by "
            f"augmentation={augmentation}"
        )
    return [lines[index].strip() for index in range(0, len(lines), augmentation)]


def _make_initial_batch(
    product_ids: Sequence[Sequence[int]],
    n_samples: int,
) -> torch.Tensor:
    if not product_ids:
        raise ValueError("product_ids cannot be empty")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    max_len = max(len(ids) for ids in product_ids)
    batch = torch.full(
        (len(product_ids) * n_samples, max_len + 1),
        PAD_TOKEN, dtype=torch.long,
    )
    row = 0
    for ids in product_ids:
        for _ in range(n_samples):
            batch[row, 0] = BOS_TOKEN
            batch[row, 1:1 + len(ids)] = torch.tensor(ids, dtype=torch.long)
            row += 1
    return batch


def _load_model(
    checkpoint: str,
    device: torch.device,
    data_dir_override: str | None,
    vocab_file_override: str | None,
):
    try:
        checkpoint_data = torch.load(
            checkpoint, map_location=device, weights_only=False,
        )
    except TypeError:  # pragma: no cover - older torch fallback
        checkpoint_data = torch.load(checkpoint, map_location=device)
    cfg = checkpoint_data["config"]
    data_dir = data_dir_override or cfg["data_dir"]
    vocab_file = vocab_file_override or cfg.get("vocab_file", "example.vocab.src")
    vocab_path = Path(vocab_file)
    if not vocab_path.is_absolute():
        vocab_path = Path(data_dir) / vocab_path
    token2id, _ = load_vocab(str(vocab_path))
    model_vocab = checkpoint_data.get("model_vocab", len(token2id))
    use_origin_mask = bool(cfg.get("use_origin_mask", False))
    has_origin_embedding = any(
        "origin_embedding" in key
        for key in checkpoint_data["model_state_dict"]
    )
    if use_origin_mask and not has_origin_embedding:
        use_origin_mask = False
    model = EditFlowsTransformer(
        vocab_size=model_vocab,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        dim_feedforward=cfg["dim_feedforward"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
        attention_dropout=cfg.get("attention_dropout", cfg["dropout"]),
        activation=cfg.get("activation", "relu"),
        pos_encoding_scale=cfg.get("pos_encoding_scale", True),
        use_origin_mask=use_origin_mask,
    ).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()
    sample_name = cfg.get("sample_scheduler", cfg.get("scheduler", "cubic"))
    train_name = cfg.get("scheduler", "cubic")
    sample_scheduler = (
        CubicScheduler() if sample_name == "cubic" else LinearScheduler()
    )
    train_scheduler = (
        CubicScheduler() if train_name == "cubic" else LinearScheduler()
    )
    id2token = {value: key for key, value in token2id.items()}
    return (
        checkpoint_data, cfg, model, token2id, id2token,
        sample_scheduler, train_scheduler, use_origin_mask,
    )


def _decode_rows(rows: torch.Tensor, id2token: dict[int, str]) -> list[str]:
    decoded = []
    for row in rows.detach().cpu().tolist():
        decoded.append(" ".join(
            id2token.get(int(token), "<unk>")
            for token in row
            if int(token) not in (PAD_TOKEN, BOS_TOKEN)
        ))
    return decoded


def generate(args: argparse.Namespace) -> dict:
    if args.n_steps < 2:
        raise ValueError("n_steps must be >= 2 to sample an interior guidance state")
    if args.time_samples < 1:
        raise ValueError("time_samples must be >= 1")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {output}; pass --overwrite to replace it"
        )
    device = torch.device(args.device)
    (
        checkpoint_data, cfg, model, token2id, id2token,
        sample_scheduler, train_scheduler, use_origin_mask,
    ) = _load_model(
        args.checkpoint, device, args.data_dir, args.vocab_file,
    )
    products = _read_original_products(args.products_file, args.augmentation)
    if args.max_products is not None:
        if args.max_products < 1:
            raise ValueError("max_products must be >= 1")
        products = products[:args.max_products]
    unk_id = token2id.get("<unk>", UNK_TOKEN)
    product_ids = [
        [token2id.get(token, unk_id) for token in product.split()]
        for product in products
    ]
    records = []
    generation_started = time.perf_counter()
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
        x0_cpu = _make_initial_batch(batch_ids, args.n_samples)
        x0 = x0_cpu.to(device)
        final_states, trajectory = sample_euler(
            model, x0, sample_scheduler,
            n_steps=args.n_steps,
            max_seq_len=cfg["max_seq_len"],
            use_rate_reparam=cfg.get("use_rate_reparam", False),
            clamp_kappa=cfg.get("clamp_kappa", False),
            clamp_max=cfg.get("clamp_max", 50.0),
            time_input=cfg.get("time_input", "t"),
            train_scheduler=train_scheduler,
            record_trajectory=True,
            use_origin_mask=use_origin_mask,
        )
        final_states = final_states.detach().cpu()
        actual_steps = len(trajectory) - 1
        if actual_steps < 1:
            raise RuntimeError("Euler returned an empty trajectory")
        final_text = _decode_rows(final_states, id2token)
        reward = retro_tokenized_validity_reward(final_text)
        for product_offset, _ in enumerate(batch_products):
            for sample_index in range(args.n_samples):
                row_index = product_offset * args.n_samples + sample_index
                sample_seed = batch_seed
                time_rng = random.Random(
                    _mix_seed(args.seed, batch_start + product_offset, sample_index + 1)
                )
                candidate_indices = list(range(1, actual_steps))
                if args.time_samples <= len(candidate_indices):
                    time_indices = sorted(time_rng.sample(
                        candidate_indices, args.time_samples,
                    ))
                else:
                    time_indices = [
                        candidate_indices[time_rng.randrange(len(candidate_indices))]
                        for _ in range(args.time_samples)
                    ]
                product_row = x0_cpu[row_index]
                terminal_row = final_states[row_index]
                product_batch = product_row.unsqueeze(0).repeat(
                    args.time_samples, 1,
                )
                terminal_batch = terminal_row.unsqueeze(0).repeat(
                    args.time_samples, 1,
                )
                coupling_seed = _mix_seed(
                    args.seed,
                    batch_start + product_offset,
                    (sample_index + 1) * 100000 + 17,
                )
                torch.manual_seed(coupling_seed)
                states = sample_intermediate_states(
                    product_batch,
                    terminal_batch,
                    [index / actual_steps for index in time_indices],
                    vocab_size=checkpoint_data.get("model_vocab", len(token2id)),
                    scheduler=sample_scheduler,
                    align_fn=opt_align_xs_to_zs,
                )
                for local_time, (time_index, state) in enumerate(
                    zip(time_indices, states),
                ):
                    records.append(make_guidance_record(
                        product_tokens=_trim_pad(product_row.tolist()),
                        state_tokens=_trim_pad(state.tolist()),
                        terminal_tokens=_trim_pad(terminal_row.tolist()),
                        time_step=time_index / actual_steps,
                        reward=float(reward[row_index].item()),
                        source_index=batch_start + product_offset,
                        sample_index=sample_index,
                        time_index=time_index,
                        sample_seed=sample_seed,
                        coupling_seed=coupling_seed + local_time,
                    ))
        if batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0:
            elapsed = time.perf_counter() - generation_started
            rate = batch_index / max(elapsed, 1e-9)
            eta = (total_batches - batch_index) / max(rate, 1e-9)
            print(
                f"Guidance batches {batch_index}/{total_batches} | "
                f"products {min(batch_start + args.batch_size, len(products))}/"
                f"{len(products)} | elapsed {elapsed:.1f}s | ETA {eta:.1f}s",
                flush=True,
            )
    generation_elapsed = time.perf_counter() - generation_started
    metadata = {
        "schema_version": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "products_file": str(Path(args.products_file).resolve()),
        "product_count": len(products),
        "record_count": len(records),
        "augmentation": args.augmentation,
        "n_steps": args.n_steps,
        "n_samples": args.n_samples,
        "time_samples": args.time_samples,
        "seed": args.seed,
        "rng_scope": "product-batch Euler seed; coupling seed per product/sample",
        "sampler": "euler",
        "reward": "rdkit_validity",
        "sample_scheduler": sample_scheduler.name,
        "train_scheduler": train_scheduler.name,
        "model_vocab": checkpoint_data.get("model_vocab", len(token2id)),
        "generation_wall_seconds": generation_elapsed,
        "batch_count": total_batches,
    }
    save_guidance_dataset(output, records, metadata=metadata)
    reward_values = torch.tensor([record["reward"] for record in records])
    summary = {
        "output": str(output),
        "products": len(products),
        "records": len(records),
        "reward_mean": float(reward_values.mean().item()) if records else 0.0,
        "reward_positive": int(reward_values.sum().item()) if records else 0,
        "wall_seconds": generation_elapsed,
    }
    print(summary)
    return summary


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
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--time_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
