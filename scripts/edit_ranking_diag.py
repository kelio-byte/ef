#!/usr/bin/env python
"""Experiment 2: Edit Ranking Diagnostic.

For a subset of test samples, at each step of the oracle-guided trajectory,
measure where the oracle-preferred edit ranks in the model's candidate list.

This directly quantifies the model's edit ranking quality, independent of
search strategy.
"""

import argparse
import math
import os
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from edit_flows.data.dataset import load_vocab
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.sampling.beam import (
    _collect_edit_candidates_single,
    _build_forbidden_mask,
    _compute_u_tot,
)
from edit_flows.sampling.oracle import compute_oracle_model_output
from edit_flows.sampling.euler import _compute_model_time
from edit_flows.core.rate_scale import apply_rate_parameterization
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN


def _depth_time_value(step: int, max_edits: int) -> float:
    """Match beam.py depth-time mapping: keep t strictly inside (0, 1)."""
    if max_edits <= 0:
        raise ValueError("max_edits must be positive")
    return (step + 1) / (max_edits + 1)


def tokenize_smiles(smiles: str, token2id: dict) -> list:
    tokens = smiles.strip().split()
    unk_id = token2id.get("<unk>", 3)
    return [token2id.get(t, unk_id) for t in tokens]


def make_sequence(ids: list[int], bos_token: int = BOS_TOKEN) -> Tensor:
    x = torch.full((len(ids) + 1,), PAD_TOKEN, dtype=torch.long)
    x[0] = bos_token
    x[1:1 + len(ids)] = torch.tensor(ids, dtype=torch.long)
    return x


def _oracle_candidates_for_state(
    x_t: Tensor, x_1: Tensor, t_val: float,
    scheduler, vocab_size: int, device: torch.device,
    k_ins_token: int, k_sub_token: int, k_edit_expand: int,
    forbidden_mask: Tensor,
) -> tuple:
    """Get oracle top-1 candidate at state (x_t, x_1, t_val)."""
    L = x_t.shape[0]
    t_tensor = torch.tensor([[t_val]], device=device)
    x_t_batch = x_t.unsqueeze(0).to(device)
    x_1_batch = x_1.unsqueeze(0).to(device)

    log_rates, log_ins_probs, log_sub_probs, _ = compute_oracle_model_output(
        x_t_batch, x_1_batch, t_tensor, scheduler, vocab_size,
        pad_token=PAD_TOKEN, bos_token=BOS_TOKEN,
    )
    # Oracle output includes k(t) scaling already; compute cond_prob from it.
    log_rates_real = log_rates[0]  # (L_t, 3)
    log_ins = log_ins_probs[0]     # (L_t, V)
    log_sub = log_sub_probs[0]     # (L_t, V)
    u_tot = _compute_u_tot(log_rates_real.unsqueeze(0))[0].item()
    log_u_tot = math.log(max(u_tot, 1e-12))

    non_pad = x_t != PAD_TOKEN

    cands, _ = _collect_edit_candidates_single(
        log_rates_real, log_ins, log_sub, x_t, non_pad, log_u_tot,
        k_ins_token, k_sub_token, k_edit_expand, forbidden_mask,
    )
    return cands


def _model_candidates_for_state(
    model, x_t: Tensor, t_val: float,
    scheduler, train_scheduler, time_input: str,
    use_rate_reparam: bool, device: torch.device,
    k_ins_token: int, k_sub_token: int, k_edit_expand: int,
    forbidden_mask: Tensor,
) -> tuple:
    """Get model candidates at state (x_t, t_val)."""
    L = x_t.shape[0]
    t_tensor = torch.tensor([[t_val]], device=device)
    x_t_batch = x_t.unsqueeze(0).to(device)
    x_pad_mask = x_t_batch == PAD_TOKEN

    t_model = _compute_model_time(t_tensor, scheduler, time_input, train_scheduler)

    log_rates, log_ins_probs, log_sub_probs = model(
        x_t_batch, t_model, x_pad_mask,
    )

    # Score from base rates as beam.py does.
    if use_rate_reparam:
        log_rates_score = log_rates
    else:
        log_rates_score = apply_rate_parameterization(
            log_rates, t_tensor, scheduler, use_rate_reparam=False,
        )

    log_rates_score = log_rates_score[0]  # (L, 3)
    log_ins = log_ins_probs[0]  # (L, V)
    log_sub = log_sub_probs[0]  # (L, V)

    u_tot = _compute_u_tot(log_rates_score.unsqueeze(0))[0].item()
    log_u_tot = math.log(max(u_tot, 1e-12))

    non_pad = x_t != PAD_TOKEN

    cands, _ = _collect_edit_candidates_single(
        log_rates_score, log_ins, log_sub, x_t, non_pad, log_u_tot,
        k_ins_token, k_sub_token, k_edit_expand, forbidden_mask,
    )
    return cands


def find_candidate_rank(
    targets: set[tuple[int, str, int | None]],
    candidates: list,
) -> int:
    """Return best 0-based rank among valid targets; -1 if not found."""
    for rank, c in enumerate(candidates):
        if (c.pos, c.op, c.token) in targets:
            return rank
    return -1


def main():
    parser = argparse.ArgumentParser(
        description="Edit Ranking Diagnostic: model vs oracle edit ranking")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str,
                        default="analysis_subsets/USPTO_50K_PtoR_aug20_#global#/"
                                "test_dedup_seed42_1000")
    parser.add_argument("--vocab_file", type=str,
                        default="/data6/duanbh/desktop/retrosynthesis/dataset/"
                                "USPTO_50K_PtoR_aug20_#global#/example.vocab.src")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--max_edits", type=int, default=15)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--k_ins_token", type=int, default=4)
    parser.add_argument("--k_sub_token", type=int, default=4)
    parser.add_argument("--k_edit_expand", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint.
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model_vocab = ckpt.get("model_vocab")

    # Load vocab.
    if args.vocab_file:
        vocab_path = args.vocab_file
    else:
        vocab_path = os.path.join(cfg["data_dir"], cfg.get("vocab_file", "example.vocab.src"))
    token2id, _ = load_vocab(vocab_path)
    if model_vocab is None:
        model_vocab = len(token2id)
    id2token = {v: k for k, v in token2id.items()}

    # Load model.
    use_origin_mask = cfg.get("use_origin_mask", False)
    has_origin_embed = any("origin_embedding" in k for k in ckpt["model_state_dict"])
    if use_origin_mask and not has_origin_embed:
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

    use_rate_reparam = cfg.get("use_rate_reparam", False)
    train_scheduler_name = cfg.get("scheduler", "cubic")
    train_scheduler = CubicScheduler() if train_scheduler_name == "cubic" else LinearScheduler()
    sample_scheduler_name = cfg.get("sample_scheduler", cfg.get("scheduler", "cubic"))
    scheduler = CubicScheduler() if sample_scheduler_name == "cubic" else LinearScheduler()
    time_input = cfg.get("time_input", "t")

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Rate reparam: {use_rate_reparam}")
    print(f"Origin mask: {use_origin_mask}")
    print(f"Train scheduler: {train_scheduler_name}, Sample scheduler: {sample_scheduler_name}")
    print(f"Time input: {time_input}")

    # Load data.
    src_path = os.path.join(args.data_dir, "src-test.txt")
    tgt_path = os.path.join(args.data_dir, "tgt-test.txt")
    with open(src_path) as f:
        products = [line.strip() for line in f]
    with open(tgt_path) as f:
        targets = [line.strip() for line in f]

    n_eval = min(args.n_samples, len(products))
    products = products[:n_eval]
    targets = targets[:n_eval]

    product_ids = [tokenize_smiles(s, token2id) for s in products]
    target_ids = [tokenize_smiles(s, token2id) for s in targets]

    forbidden_mask = _build_forbidden_mask(model_vocab, device)

    # Statistics accumulators.
    total_steps = 0
    rank_counts = {1: 0, 5: 0, 16: 0, -1: 0}  # top-1, top-5, top-16, not-in-candidates
    score_gaps = []  # model top-1 score - target score
    oracle_in_model_top1 = 0
    model_matches_oracle = 0

    detail_lines = []

    for sample_idx in tqdm(range(n_eval), desc="Diagnostic"):
        x_t = make_sequence(product_ids[sample_idx]).to(device)
        x_1 = make_sequence(target_ids[sample_idx]).to(device)

        for step in range(args.max_edits):
            t_val = _depth_time_value(step, args.max_edits)

            # Get oracle candidates for this state.
            oracle_cands = _oracle_candidates_for_state(
                x_t, x_1, t_val, scheduler, model_vocab, device,
                args.k_ins_token, args.k_sub_token, args.k_edit_expand,
                forbidden_mask,
            )

            if not oracle_cands:
                break  # oracle thinks we're done

            oracle_top = oracle_cands[0]
            oracle_best_score = oracle_top.score
            oracle_targets = {
                (c.pos, c.op, c.token)
                for c in oracle_cands
                if abs(c.score - oracle_best_score) <= 1e-6
            }

            # Get model candidates for this state.
            model_cands = _model_candidates_for_state(
                model, x_t, t_val, scheduler, train_scheduler,
                time_input, use_rate_reparam, device,
                args.k_ins_token, args.k_sub_token, args.k_edit_expand,
                forbidden_mask,
            )

            if not model_cands:
                break

            rank = find_candidate_rank(oracle_targets, model_cands)

            total_steps += 1
            if rank == 0:
                rank_counts[1] += 1
                oracle_in_model_top1 += 1
                if len(model_cands) > 0:
                    model_matches_oracle += 1
            elif 0 < rank < 5:
                rank_counts[5] += 1
            elif 5 <= rank < 16:
                rank_counts[16] += 1
            else:
                rank_counts[-1] += 1

            # Score gap: model top-1 score minus oracle-top's score in model list.
            if rank >= 0:
                gap = model_cands[0].score - model_cands[rank].score
            else:
                # Oracle's preferred edit not in model candidates at all.
                # Approximate gap as large.
                gap = 999.0
            score_gaps.append(gap)

            # Record detail for first 10 samples.
            if sample_idx < 10:
                oracle_token_str = id2token.get(oracle_top.token, f"<{oracle_top.token}>")
                model_top1_str = id2token.get(model_cands[0].token, f"<{model_cands[0].token}>") \
                    if model_cands[0].token is not None else "-"
                detail_lines.append(
                    f"sample={sample_idx:>3} step={step:>2} "
                    f"oracle=({oracle_top.op},{oracle_top.pos},{oracle_token_str}) "
                    f"model_top1=({model_cands[0].op},{model_cands[0].pos},{model_top1_str}) "
                    f"oracle_rank={rank} gap={gap:.4f}"
                )

            # Apply oracle's top edit to advance the state.
            from edit_flows.sampling.beam import _apply_single_edit_to_sequence
            x_t, _ = _apply_single_edit_to_sequence(
                x_t, None, oracle_top, args.max_seq_len, PAD_TOKEN,
            )
            x_t = x_t.to(device)

    # ---- Report ----
    report_path = os.path.join(args.output_dir, "ranking_report.txt")
    with open(report_path, "w") as f:
        f.write("=== Edit Ranking Diagnostic Report ===\n\n")
        f.write(f"Samples: {n_eval}\n")
        f.write(f"Total oracle steps: {total_steps}\n")
        f.write(f"k_ins_token={args.k_ins_token}, k_sub_token={args.k_sub_token}, "
                f"k_edit_expand={args.k_edit_expand}\n\n")

        f.write("--- Oracle Top-1 Edit Rank in Model Candidates ---\n")
        for label, count in [("top-1", rank_counts[1]), ("top-5", rank_counts[5]),
                              ("top-16", rank_counts[16]), ("not in candidates", rank_counts[-1])]:
            pct = count / total_steps * 100 if total_steps > 0 else 0
            f.write(f"  {label:>20s}: {count:>5}  ({pct:5.1f}%)\n")

        f.write(f"\n  oracle's top-1 == model's top-1: {oracle_in_model_top1}/{total_steps}"
                f" ({oracle_in_model_top1/total_steps*100:.1f}%)\n"
                if total_steps > 0 else "\n")

        # Score gap stats.
        finite_gaps = [g for g in score_gaps if g < 900]
        if finite_gaps:
            avg_gap = sum(finite_gaps) / len(finite_gaps)
            f.write(f"\n--- Score Gap (model_top1.score - oracle_edit.score in model list) ---\n")
            f.write(f"  Mean gap: {avg_gap:.4f} nats\n")
            f.write(f"  Min gap:  {min(finite_gaps):.4f} nats\n")
            f.write(f"  Max gap:  {max(finite_gaps):.4f} nats\n")
            f.write(f"  Steps where oracle edit not in model candidates: "
                    f"{len(score_gaps) - len(finite_gaps)}\n")

        f.write("\n--- First 10 Samples Detail ---\n")
        for line in detail_lines:
            f.write(line + "\n")

    # Print summary to stdout.
    print(f"\n=== Results ===")
    print(f"Total oracle steps: {total_steps}")
    print(f"Oracle top-1 in model top-1:  {rank_counts[1]:>5} ({rank_counts[1]/total_steps*100:5.1f}%)" if total_steps > 0 else "")
    print(f"Oracle top-1 in model top-5:  {rank_counts[1]+rank_counts[5]:>5} ({(rank_counts[1]+rank_counts[5])/total_steps*100:5.1f}%)" if total_steps > 0 else "")
    print(f"Oracle top-1 in model top-16: {rank_counts[1]+rank_counts[5]+rank_counts[16]:>5} ({(rank_counts[1]+rank_counts[5]+rank_counts[16])/total_steps*100:5.1f}%)" if total_steps > 0 else "")
    print(f"Oracle top-1 NOT in model top-16: {rank_counts[-1]:>5} ({rank_counts[-1]/total_steps*100:5.1f}%)" if total_steps > 0 else "")
    if finite_gaps:
        print(f"Mean score gap: {sum(finite_gaps)/len(finite_gaps):.4f} nats")
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
