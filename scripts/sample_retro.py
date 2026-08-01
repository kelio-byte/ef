#!/usr/bin/env python
"""Sampling script for Edit Flows retrosynthesis.

Products are processed in GPU batches.  Each product produces consecutive
outputs: ``n_samples`` for Euler or ``n_runs`` for Euler-Beam.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import subprocess
import time
import torch
from torch import Tensor
from tqdm import tqdm

from edit_flows.data.dataset import load_vocab
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.sampling.euler import sample_euler
from edit_flows.sampling.euler_beam import _mix_child_seed, sample_euler_beam
from edit_flows.sampling.beam import sample_greedy_single_edit, sample_beam_single_edit
from edit_flows.sampling.time_policy import (
    DepthTimePolicy, FixedTimePolicy, RatioTimePolicy, KappaTimePolicy,
)
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN, UNK_TOKEN


def tokenize_smiles(smiles: str, token2id: dict) -> list:
    tokens = smiles.strip().split()
    unk_id = token2id.get("<unk>", UNK_TOKEN)
    return [token2id.get(t, unk_id) for t in tokens]


def _ids_to_str(ids: list, id2token: dict) -> str:
    return " ".join(id2token[tid] for tid in ids
                    if tid not in (PAD_TOKEN, BOS_TOKEN))


def _make_batch(product_ids: list[list[int]], n_samples: int,
                pad_token: int, bos_token: int = BOS_TOKEN) -> Tensor:
    B = len(product_ids)
    max_len = max(len(ids) for ids in product_ids)
    x_0 = torch.full((B, max_len + 1), pad_token, dtype=torch.long)
    x_0[:, 0] = bos_token
    for i, ids in enumerate(product_ids):
        x_0[i, 1:1 + len(ids)] = torch.tensor(ids, dtype=torch.long)
    return x_0.repeat_interleave(n_samples, dim=0)


def _make_euler_beam_sample_seeds(
    base_seed: int,
    global_start: int,
    n_products: int,
    n_runs: int,
) -> list[int]:
    """构造不依赖 n_branches 和 batch 划分的 product/run seeds。"""
    return [
        _mix_child_seed(base_seed, global_start + i, r + 1)
        for i in range(n_products)
        for r in range(n_runs)
    ]


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_metadata(path: str, include_sha256: bool = False) -> dict:
    absolute_path = os.path.abspath(path)
    stat = os.stat(absolute_path)
    metadata = {
        "path": absolute_path,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        metadata["sha256"] = _sha256_file(absolute_path)
    return metadata


def _infer_augmentation(*values: str | None) -> tuple[int | None, str | None]:
    """Infer augmentation only from an explicit ``augN`` path component."""
    matches = []
    for value in values:
        if not value:
            continue
        match = re.search(r"(?:^|[/_\-])aug(?:mentation)?[_\-]?(\d+)", value,
                          flags=re.IGNORECASE)
        if match:
            matches.append((int(match.group(1)), value))
    unique = {number for number, _ in matches}
    if len(unique) == 1:
        number = next(iter(unique))
        source = next(value for candidate, value in matches
                      if candidate == number)
        return number, source
    return None, None


def _git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout)
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _outputs_per_product(args) -> int:
    if args.sampler == "euler_beam":
        return args.n_runs
    return args.n_samples


def _select_products(
    products: list[str],
    start_product: int,
    max_products: int | None,
    augmentation: int | None,
) -> tuple[list[str], int]:
    """Select a reproducible, augmentation-aligned input interval."""
    if start_product < 0:
        raise ValueError(
            f"start_product must be >= 0, got {start_product}"
        )
    if max_products is not None and max_products <= 0:
        raise ValueError(
            f"max_products must be > 0, got {max_products}"
        )
    end_product = (
        len(products)
        if max_products is None else start_product + max_products
    )
    if start_product >= len(products) or end_product > len(products):
        raise ValueError(
            "requested product interval is outside the input file: "
            f"[{start_product}, {end_product}) for {len(products)} lines"
        )
    if augmentation is not None and (
        start_product % augmentation != 0
        or (end_product - start_product) % augmentation != 0
    ):
        raise ValueError(
            "product interval must preserve complete augmentation blocks: "
            f"start={start_product}, count={end_product - start_product}, "
            f"augmentation={augmentation}"
        )
    return products[start_product:end_product], end_product


def _build_sampling_metadata(
    args,
    cfg: dict,
    *,
    prediction_path: str,
    source_product_count: int,
    selection_start_product: int,
    product_count: int,
    output_line_count: int,
    n_sampling_steps: int,
    sample_scheduler_name: str,
    train_scheduler_name: str,
    use_origin_mask: bool,
    elapsed_seconds: float,
    peak_cuda_allocated_bytes: int | None = None,
    peak_cuda_reserved_bytes: int | None = None,
    euler_beam_profile: dict | None = None,
) -> dict:
    # Augmentation describes the actual input layout, so it must not be
    # inferred from the checkpoint's training data directory.  A single
    # --product has no augmentation layout to infer.
    augmentation, augmentation_source = _infer_augmentation(
        args.products_file,
    )
    input_metadata = {
        "kind": "products_file" if args.products_file else "single_product",
        "product_count": product_count,
        "source_product_count": source_product_count,
        "selection_start_product": selection_start_product,
        "selection_end_product_exclusive": (
            selection_start_product + product_count
        ),
    }
    if args.products_file:
        input_metadata.update(_path_metadata(
            args.products_file, include_sha256=True,
        ))

    sampling = {
        "n_steps": n_sampling_steps,
        "sample_scheduler": sample_scheduler_name,
        "train_scheduler": train_scheduler_name,
        "seed": args.seed,
        "seed_applied_to_sampler": args.sampler == "euler_beam",
    }
    if args.sampler == "euler_beam":
        sampling.update({
            "n_branches": args.n_branches,
            "n_children": args.n_children,
            "n_runs": args.n_runs,
            "score_mode": args.euler_beam_score_mode,
            "changed_state_bonus": args.euler_beam_changed_state_bonus,
            "matmul_precision": args.euler_beam_matmul_precision,
            "child_policy": args.euler_beam_child_policy,
            "seed_scope": "stable product/run streams",
        })
    else:
        sampling["n_samples"] = args.n_samples
        sampling["seed_scope"] = (
            "not applied by sample_retro.py"
            if args.sampler == "euler" else "sampler-specific"
        )

    runtime = {
        "elapsed_seconds": elapsed_seconds,
        "batch_size": args.batch_size,
        "device": args.device,
    }
    if peak_cuda_allocated_bytes is not None:
        runtime["peak_cuda_allocated_bytes"] = peak_cuda_allocated_bytes
        runtime["peak_cuda_reserved_bytes"] = peak_cuda_reserved_bytes
    if euler_beam_profile is not None:
        runtime["euler_beam_profile"] = euler_beam_profile

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "layout": "input-product-major, output-minor",
        "sampler": args.sampler,
        "augmentation": augmentation,
        "augmentation_inferred_from": augmentation_source,
        "checkpoint": _path_metadata(args.checkpoint),
        "input": input_metadata,
        "product_count": product_count,
        "output_beam_size": _outputs_per_product(args),
        "output_line_count": output_line_count,
        "output_sha256": _sha256_file(prediction_path),
        "sampling": sampling,
        "runtime": runtime,
        "model": {
            "configured_use_origin_mask": cfg.get("use_origin_mask", False),
            "effective_use_origin_mask": use_origin_mask,
        },
        "git": _git_state(),
    }


def main():
    parser = argparse.ArgumentParser(description="Sample Edit Flows retrosynthesis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--product", type=str, default=None,
                        help="Product SMILES string (tokenized, space-separated)")
    parser.add_argument("--products_file", type=str, default=None,
                        help="File with one tokenized product SMILES per line")
    parser.add_argument("--start_product", type=int, default=0,
                        help="0-based products_file line offset")
    parser.add_argument("--max_products", type=int, default=None,
                        help="Optional number of products_file lines to sample")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data dir (for vocab)")
    parser.add_argument("--vocab_file", type=str, default=None,
                        help="Override vocab file path")
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of independent samples per product")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="GPU batch size (number of products per batch)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save predictions")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (used as base_seed for euler_beam)")
    parser.add_argument("--scheduler", type=str, default=None,
                        choices=["cubic", "linear"],
                        help="Override sampling scheduler (default: use config sample_scheduler or training scheduler)")
    parser.add_argument("--sampler", type=str, default="euler",
                        choices=["euler", "euler_beam", "greedy_edit", "beam_edit"],
                        help="Sampling algorithm (default: euler)")
    parser.add_argument("--n_branches", type=int, default=5,
                        help="并行分支数 (euler_beam)")
    parser.add_argument("--n_children", type=int, default=1,
                        help="每个父分支每步生成的后继数 (euler_beam)")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="每个产物独立运行次数 (euler_beam, 等价于 Euler 的 --n_samples)")
    parser.add_argument("--euler_beam_score_mode", type=str,
                        default="full_probability",
                        choices=["full_probability", "legacy_triggered_reverse"],
                        help="Euler-Beam scoring mode; legacy is only for ablation")
    parser.add_argument("--euler_beam_changed_state_bonus", type=float,
                        default=0.0,
                        help="Fixed search bonus for states changed from the product")
    parser.add_argument("--euler_beam_matmul_precision", type=str,
                        default="highest", choices=["highest", "high"],
                        help=("Float32 matmul precision for CUDA Euler-Beam; "
                              "'high' enables TF32 on supported GPUs"))
    parser.add_argument("--euler_beam_child_policy", type=str,
                        default="stochastic",
                        choices=["stochastic", "stochastic_noop"],
                        help="Euler-Beam child proposal policy")
    parser.add_argument(
        "--euler_beam_profile",
        action="store_true",
        default=False,
        help=(
            "Synchronize CUDA between Euler-Beam stages and record a "
            "timing breakdown; use only for short profiling runs"
        ),
    )
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam size for beam_edit sampler")
    parser.add_argument("--max_edits", type=int, default=20,
                        help="Max edit steps for greedy/beam samplers")
    parser.add_argument("--time_policy", type=str, default="depth",
                        choices=["depth", "fixed", "ratio", "kappa"],
                        help="Time policy for greedy/beam: depth, fixed, ratio, kappa")
    parser.add_argument("--time_const", type=float, default=0.5,
                        help="Fixed t value when time_policy=fixed")
    parser.add_argument("--k_ins_token", type=int, default=4,
                        help="Top-k insert tokens per position")
    parser.add_argument("--k_sub_token", type=int, default=4,
                        help="Top-k substitute tokens per position")
    parser.add_argument("--k_edit_expand", type=int, default=16,
                        help="Global top-k edit candidates per step")
    parser.add_argument("--stop_u_tot_base", type=float, default=-1.0,
                        help="Stop threshold on executable edit mass in scoring-rate space (< 0 disables)")
    parser.add_argument("--explicit_stop", action="store_true", default=False,
                        help="Treat STOP as an explicit candidate action alongside edits")
    parser.add_argument("--kappa_mode", type=str, default="ratio",
                        choices=["ratio", "frozen_hazard", "poisson"],
                        help="Kappa update mode when explicit_stop=True")
    parser.add_argument("--p_stop_mode", type=str, default="absolute",
                        choices=["absolute", "normalized"],
                        help="p_stop formula: e^{-U} (absolute) or e^{-U/U_init} (normalized)")
    parser.add_argument("--fh_warmup_steps", type=int, default=0,
                        help="Warmup steps using depth kappa before frozen-hazard kappa kicks in")
    args = parser.parse_args()

    if args.euler_beam_profile and args.sampler != "euler_beam":
        raise ValueError("euler_beam_profile requires --sampler euler_beam")

    device = torch.device(args.device)
    if args.sampler == "euler_beam" and device.type == "cuda":
        torch.set_float32_matmul_precision(
            args.euler_beam_matmul_precision
        )

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model_vocab = ckpt.get("model_vocab")

    if args.vocab_file:
        vocab_path = args.vocab_file
    elif args.data_dir:
        vocab_path = os.path.join(args.data_dir, cfg.get("vocab_file", "example.vocab.src"))
    else:
        vocab_path = os.path.join(cfg["data_dir"], cfg.get("vocab_file", "example.vocab.src"))

    token2id, _ = load_vocab(vocab_path)
    if model_vocab is None:
        model_vocab = len(token2id)

    id2token = {v: k for k, v in token2id.items()}

    use_origin_mask = cfg.get("use_origin_mask", False)
    has_origin_embed = any("origin_embedding" in k for k in ckpt["model_state_dict"])
    if use_origin_mask and not has_origin_embed:
        print("WARNING: config has use_origin_mask=True but checkpoint lacks "
              "origin_embedding weights. Falling back to use_origin_mask=False.")
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if args.scheduler:
        sample_scheduler_name = args.scheduler
    else:
        sample_scheduler_name = cfg.get("sample_scheduler", cfg.get("scheduler", "cubic"))
    train_scheduler_name = cfg.get("scheduler", "cubic")
    kappa_scheduler = CubicScheduler() if sample_scheduler_name == "cubic" else LinearScheduler()
    train_scheduler = CubicScheduler() if train_scheduler_name == "cubic" else LinearScheduler()
    time_input = cfg.get("time_input", "t")
    clamp_kappa = cfg.get("clamp_kappa", False)
    clamp_max = cfg.get("clamp_max", 50.0)
    n_sampling_steps = args.n_steps or cfg.get("n_sampling_steps", 100)

    if args.product and args.products_file:
        raise ValueError("Provide only one of --product or --products_file")
    if args.product:
        if args.start_product != 0 or args.max_products is not None:
            raise ValueError(
                "start_product/max_products require --products_file"
            )
        products = [args.product]
    elif args.products_file:
        with open(args.products_file) as f:
            products = [line.strip() for line in f]
    else:
        raise ValueError("Provide --product or --products_file")

    source_product_count = len(products)
    input_augmentation, _ = _infer_augmentation(args.products_file)
    products, selection_end_product = _select_products(
        products,
        start_product=args.start_product,
        max_products=args.max_products,
        augmentation=input_augmentation,
    )
    n_products = len(products)
    if args.start_product or args.max_products is not None:
        print(
            "Selected product interval: "
            f"[{args.start_product}, {selection_end_product}) from "
            f"{source_product_count} input lines"
        )
    product_ids = [tokenize_smiles(s, token2id) for s in products]
    outputs_per_product = _outputs_per_product(args)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        pred_file = os.path.join(args.output_dir, "predictions.txt")
        f_out = open(pred_file, "w")
        print(
            f"Sampling {n_products} products x {outputs_per_product} "
            "outputs"
        )
        print(f"Batch size: {args.batch_size}")
        print(f"Predictions will be saved to: {pred_file}")
    else:
        f_out = None

    batch_size = args.batch_size
    n_batches = math.ceil(n_products / batch_size)

    use_greedy_beam = args.sampler in ("greedy_edit", "beam_edit")
    print(f"Sampler: {args.sampler}")
    euler_beam_profile = {} if args.euler_beam_profile else None
    if use_greedy_beam:
        # Build time policy.
        if args.time_policy == "depth":
            time_policy = DepthTimePolicy(scheduler=kappa_scheduler)
        elif args.time_policy == "fixed":
            time_policy = FixedTimePolicy(scheduler=kappa_scheduler, time_const=args.time_const)
        elif args.time_policy == "ratio":
            time_policy = RatioTimePolicy(scheduler=kappa_scheduler)
        elif args.time_policy == "kappa":
            time_policy = KappaTimePolicy(scheduler=kappa_scheduler)
        else:
            raise ValueError(f"Unknown time_policy: {args.time_policy}")
        print(f"  max_edits={args.max_edits}, time_policy={args.time_policy}, "
              f"k_ins={args.k_ins_token}, k_sub={args.k_sub_token}, "
              f"k_edit_expand={args.k_edit_expand}")
    if args.sampler == "beam_edit":
        print(f"  beam_size={args.beam_size}")
    if args.sampler == "euler_beam":
        print(f"  n_branches={args.n_branches}, "
              f"n_children={args.n_children}, n_runs={args.n_runs}, "
              f"matmul_precision={args.euler_beam_matmul_precision}, "
              f"child_policy={args.euler_beam_child_policy}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sampling_started_at = time.perf_counter()
    written_predictions = 0
    try:
        for batch_idx in tqdm(range(n_batches), desc="Batches"):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_products)
            batch_products = product_ids[start:end]

            if use_greedy_beam:
                n_rep = 1
            elif args.sampler == "euler_beam":
                n_rep = args.n_runs
            else:
                n_rep = args.n_samples

            x_0 = _make_batch(batch_products, n_rep, PAD_TOKEN)
            x_0 = x_0.to(device)

            if args.sampler == "euler":
                results, _ = sample_euler(
                    model, x_0, kappa_scheduler,
                    n_steps=n_sampling_steps,
                    max_seq_len=cfg["max_seq_len"],
                    use_rate_reparam=cfg.get("use_rate_reparam", False),
                    clamp_kappa=clamp_kappa,
                    clamp_max=clamp_max,
                    time_input=time_input,
                    train_scheduler=train_scheduler,
                    use_origin_mask=use_origin_mask,
                )
            elif args.sampler == "euler_beam":
                B_prod = end - start
                # _make_batch uses repeat_interleave, so rows are product-major:
                # P0R0, P0R1, ..., P1R0, P1R1, ...
                sample_seeds = _make_euler_beam_sample_seeds(
                    args.seed, args.start_product + start,
                    B_prod, args.n_runs,
                )
                results = sample_euler_beam(
                    model, x_0, kappa_scheduler,
                    n_branches=args.n_branches,
                    n_children=args.n_children,
                    n_steps=n_sampling_steps,
                    max_seq_len=cfg["max_seq_len"],
                    use_rate_reparam=cfg.get("use_rate_reparam", False),
                    clamp_kappa=clamp_kappa,
                    clamp_max=clamp_max,
                    time_input=time_input,
                    train_scheduler=train_scheduler,
                    use_origin_mask=use_origin_mask,
                    sample_seeds=sample_seeds,
                    score_mode=args.euler_beam_score_mode,
                    changed_state_bonus=args.euler_beam_changed_state_bonus,
                    child_policy=args.euler_beam_child_policy,
                    profile=euler_beam_profile,
                )
            elif args.sampler == "greedy_edit":
                results = sample_greedy_single_edit(
                    model, x_0, kappa_scheduler,
                    time_policy,
                    max_edits=args.max_edits,
                    max_seq_len=cfg["max_seq_len"],
                    use_rate_reparam=cfg.get("use_rate_reparam", False),
                    clamp_kappa=clamp_kappa,
                    clamp_max=clamp_max,
                    time_input=time_input,
                    train_scheduler=train_scheduler,
                    use_origin_mask=use_origin_mask,
                    k_ins_token=args.k_ins_token,
                    k_sub_token=args.k_sub_token,
                    k_edit_expand=args.k_edit_expand,
                    stop_u_tot_base=args.stop_u_tot_base,
                    explicit_stop=args.explicit_stop,
                    kappa_mode=args.kappa_mode,
                    p_stop_mode=args.p_stop_mode,
                    fh_warmup_steps=args.fh_warmup_steps,
                )
            elif args.sampler == "beam_edit":
                results = sample_beam_single_edit(
                    model, x_0, kappa_scheduler,
                    time_policy,
                    beam_size=args.beam_size,
                    max_edits=args.max_edits,
                    max_seq_len=cfg["max_seq_len"],
                    use_rate_reparam=cfg.get("use_rate_reparam", False),
                    clamp_kappa=clamp_kappa,
                    clamp_max=clamp_max,
                    time_input=time_input,
                    train_scheduler=train_scheduler,
                    use_origin_mask=use_origin_mask,
                    k_ins_token=args.k_ins_token,
                    k_sub_token=args.k_sub_token,
                    k_edit_expand=args.k_edit_expand,
                    stop_u_tot_base=args.stop_u_tot_base,
                )
            else:
                raise ValueError(f"Unknown sampler: {args.sampler}")

            results = results.cpu()
            B = end - start
            for i in range(B):
                n_out = args.n_samples if use_greedy_beam else n_rep
                for s in range(n_out):
                    row_idx = i if use_greedy_beam else i * n_rep + s
                    row = results[row_idx]
                    line = _ids_to_str(row.tolist(), id2token)
                    if f_out:
                        f_out.write(line + "\n")
                        written_predictions += 1
                    else:
                        print(line)
    finally:
        if f_out:
            f_out.close()

    if args.output_dir:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - sampling_started_at
        peak_cuda_allocated_bytes = (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else None
        )
        peak_cuda_reserved_bytes = (
            torch.cuda.max_memory_reserved(device)
            if device.type == "cuda" else None
        )
        expected_predictions = n_products * outputs_per_product
        if written_predictions != expected_predictions:
            raise RuntimeError(
                "sampler output count mismatch: expected "
                f"{expected_predictions}, wrote {written_predictions}"
            )
        metadata = _build_sampling_metadata(
            args,
            cfg,
            prediction_path=pred_file,
            source_product_count=source_product_count,
            selection_start_product=args.start_product,
            product_count=n_products,
            output_line_count=written_predictions,
            n_sampling_steps=n_sampling_steps,
            sample_scheduler_name=sample_scheduler_name,
            train_scheduler_name=train_scheduler_name,
            use_origin_mask=use_origin_mask,
            elapsed_seconds=elapsed_seconds,
            peak_cuda_allocated_bytes=peak_cuda_allocated_bytes,
            peak_cuda_reserved_bytes=peak_cuda_reserved_bytes,
            euler_beam_profile=euler_beam_profile,
        )
        metadata_path = os.path.join(
            args.output_dir, "sampling_metadata.json",
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")
        if args.sampler == "euler_beam":
            print(f"Done. Total predictions: {n_products * args.n_runs} "
                  f"(n_branches={args.n_branches}, "
                  f"n_children={args.n_children}, n_runs={args.n_runs})")
        else:
            print(f"Done. Total predictions: {n_products * args.n_samples}")
        print(f"Saved to: {pred_file}")
        print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
