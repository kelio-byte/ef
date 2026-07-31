#!/usr/bin/env python
"""Sampling script for Edit Flows retrosynthesis.

Products are processed in GPU batches.  Each product is independently sampled
n_samples times, producing n_samples consecutive lines in the output.
"""

import argparse
import math
import os
import torch
from torch import Tensor
from tqdm import tqdm

from edit_flows.data.dataset import load_vocab
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.sampling.euler import sample_euler
from edit_flows.sampling.euler_beam import sample_euler_beam
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


def main():
    parser = argparse.ArgumentParser(description="Sample Edit Flows retrosynthesis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--product", type=str, default=None,
                        help="Product SMILES string (tokenized, space-separated)")
    parser.add_argument("--products_file", type=str, default=None,
                        help="File with one tokenized product SMILES per line")
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
    parser.add_argument("--n_runs", type=int, default=1,
                        help="每个产物独立运行次数 (euler_beam, 等价于 Euler 的 --n_samples)")
    parser.add_argument("--euler_beam_score_mode", type=str,
                        default="full_probability",
                        choices=["full_probability", "legacy_triggered_reverse"],
                        help="Euler-Beam scoring mode; legacy is only for ablation")
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

    device = torch.device(args.device)

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

    if args.product:
        products = [args.product]
    elif args.products_file:
        with open(args.products_file) as f:
            products = [line.strip() for line in f]
    else:
        raise ValueError("Provide --product or --products_file")

    n_products = len(products)
    product_ids = [tokenize_smiles(s, token2id) for s in products]

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        pred_file = os.path.join(args.output_dir, "predictions.txt")
        f_out = open(pred_file, "w")
        print(f"Sampling {n_products} products x {args.n_samples} samples")
        print(f"Batch size: {args.batch_size}")
        print(f"Predictions will be saved to: {pred_file}")
    else:
        f_out = None

    batch_size = args.batch_size
    n_batches = math.ceil(n_products / batch_size)

    use_greedy_beam = args.sampler in ("greedy_edit", "beam_edit")
    print(f"Sampler: {args.sampler}")
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
        print(f"  n_branches={args.n_branches}, n_runs={args.n_runs}")

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
                sample_seeds = [
                    args.seed + r * 1000 + i * args.n_branches
                    for i in range(B_prod)
                    for r in range(args.n_runs)
                ]
                results = sample_euler_beam(
                    model, x_0, kappa_scheduler,
                    n_branches=args.n_branches,
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
                    else:
                        print(line)
    finally:
        if f_out:
            f_out.close()

    if args.output_dir:
        if args.sampler == "euler_beam":
            print(f"Done. Total predictions: {n_products * args.n_runs} "
                  f"(n_branches={args.n_branches}, n_runs={args.n_runs})")
        else:
            print(f"Done. Total predictions: {n_products * args.n_samples}")
        print(f"Saved to: {pred_file}")


if __name__ == "__main__":
    main()