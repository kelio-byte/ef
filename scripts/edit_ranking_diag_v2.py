#!/usr/bin/env python
"""Edit Ranking Diagnostic v2: with step decomposition (Experiments A + B).

For a subset of test samples, at each step of the oracle-guided trajectory,
measure where the oracle-preferred edit ranks in the model's candidate list.

New in v2:
  - Per-step breakdown of ranking metrics
  - Tie-aware oracle target matching (carried over from v1 fix)
  - Interior depth-time mapping (carried over from v1 fix)
"""

import argparse
import json
import math
import os
import torch
import torch.nn as nn
from collections import defaultdict
from torch import Tensor
from tqdm import tqdm

from edit_flows.data.dataset import load_vocab
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.sampling.beam import (
    _collect_edit_candidates_single,
    _build_forbidden_mask,
    _compute_executable_u_tot,
    _compute_u_tot,
    _prepare_log_rates_for_scoring,
)
from edit_flows.sampling.oracle import compute_oracle_model_output
from edit_flows.sampling.euler import _compute_model_time
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN


def _depth_time_value(step: int, max_edits: int) -> float:
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
    L = x_t.shape[0]
    t_tensor = torch.tensor([[t_val]], device=device)
    x_t_batch = x_t.unsqueeze(0).to(device)
    x_1_batch = x_1.unsqueeze(0).to(device)

    log_rates, log_ins_probs, log_sub_probs, _ = compute_oracle_model_output(
        x_t_batch, x_1_batch, t_tensor, scheduler, vocab_size,
        pad_token=PAD_TOKEN, bos_token=BOS_TOKEN,
    )
    log_rates_real = log_rates[0]
    log_ins = log_ins_probs[0]
    log_sub = log_sub_probs[0]
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
    L = x_t.shape[0]
    t_tensor = torch.tensor([[t_val]], device=device)
    x_t_batch = x_t.unsqueeze(0).to(device)
    x_pad_mask = x_t_batch == PAD_TOKEN

    t_model = _compute_model_time(t_tensor, scheduler, time_input, train_scheduler)

    log_rates, log_ins_probs, log_sub_probs = model(
        x_t_batch, t_model, x_pad_mask,
    )

    log_rates_score = _prepare_log_rates_for_scoring(
        log_rates, t_tensor, scheduler,
        use_rate_reparam=use_rate_reparam,
        train_scheduler=train_scheduler,
        t_model=t_model,
    )

    log_rates_score = log_rates_score[0]
    log_ins = log_ins_probs[0]
    log_sub = log_sub_probs[0]

    u_tot = _compute_executable_u_tot(
        log_rates_score.unsqueeze(0),
        log_ins.unsqueeze(0),
        log_sub.unsqueeze(0),
        x_t.unsqueeze(0),
        pad_token=PAD_TOKEN,
        forbidden_mask=forbidden_mask,
    )[0].item()
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
    for rank, c in enumerate(candidates):
        if (c.pos, c.op, c.token) in targets:
            return rank
    return -1


def step_bin(step: int) -> str:
    """Map step index to a bin label."""
    if step == 0:
        return "step=0"
    elif step <= 3:
        return "step=1-3"
    elif step <= 7:
        return "step=4-7"
    else:
        return "step>=8"


def main():
    parser = argparse.ArgumentParser(
        description="Edit Ranking Diagnostic v2: model vs oracle edit ranking")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--vocab_file", type=str,
                        default="/data6/duanbh/desktop/retrosynthesis/dataset/"
                                "USPTO_50K_PtoR_aug20_#global#/example.vocab.src")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--max_edits", type=int, default=20)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--k_ins_token", type=int, default=4)
    parser.add_argument("--k_sub_token", type=int, default=4)
    parser.add_argument("--k_edit_expand", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
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

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Rate reparam: {use_rate_reparam}")
    print(f"Origin mask: {use_origin_mask}")
    print(f"Train scheduler: {train_scheduler_name}, Sample scheduler: {sample_scheduler_name}")
    print(f"Time input: {time_input}")
    print(f"Data: {args.data_dir}, n_samples={args.n_samples}")
    print(f"max_edits={args.max_edits}")

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

    # ---- Statistics accumulators ----
    total_steps = 0
    # Overall rank counts.
    rank_counts = {1: 0, 5: 0, 16: 0, -1: 0}
    # Per-step-bin rank counts: bin_label -> {1: count, 5: count, 16: count, -1: count}
    step_rank_counts: dict[str, dict[int, int]] = defaultdict(lambda: {1: 0, 5: 0, 16: 0, -1: 0})
    step_total: dict[str, int] = defaultdict(int)

    score_gaps = []
    step_score_gaps: dict[str, list[float]] = defaultdict(list)

    detail_lines = []
    per_sample_steps: list[int] = []  # how many oracle steps each sample took

    for sample_idx in tqdm(range(n_eval), desc="Diagnostic"):
        x_t = make_sequence(product_ids[sample_idx]).to(device)
        x_1 = make_sequence(target_ids[sample_idx]).to(device)
        n_steps = 0

        for step in range(args.max_edits):
            t_val = _depth_time_value(step, args.max_edits)

            oracle_cands = _oracle_candidates_for_state(
                x_t, x_1, t_val, scheduler, model_vocab, device,
                args.k_ins_token, args.k_sub_token, args.k_edit_expand,
                forbidden_mask,
            )

            if not oracle_cands:
                break

            # Stop when oracle has no meaningful edits left.
            # Real edits: score > -2.3 (typically > -1.6).
            # Noise (all rates at SMALL_RATE): score < -4.0 (typically -5.0).
            # Threshold -3.0 gives ≥ 1 nat safety margin.
            # Exclude step 0: rare extreme samples may have n_active ≥ 20.
            if step > 0 and oracle_cands[0].score < -3.0:
                break

            n_steps += 1

            oracle_top = oracle_cands[0]
            oracle_best_score = oracle_top.score
            oracle_targets = {
                (c.pos, c.op, c.token)
                for c in oracle_cands
                if abs(c.score - oracle_best_score) <= 1e-6
            }

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
            bin_label = step_bin(step)
            step_total[bin_label] += 1

            if rank == 0:
                rank_counts[1] += 1
                step_rank_counts[bin_label][1] += 1
            elif 0 < rank < 5:
                rank_counts[5] += 1
                step_rank_counts[bin_label][5] += 1
            elif 5 <= rank < 16:
                rank_counts[16] += 1
                step_rank_counts[bin_label][16] += 1
            else:
                rank_counts[-1] += 1
                step_rank_counts[bin_label][-1] += 1

            if rank >= 0:
                gap = model_cands[0].score - model_cands[rank].score
            else:
                gap = 999.0
            score_gaps.append(gap)
            step_score_gaps[bin_label].append(gap)

            if sample_idx < 10:
                oracle_token_str = id2token.get(oracle_top.token, f"<{oracle_top.token}>")
                model_top1_str = id2token.get(model_cands[0].token, f"<{model_cands[0].token}>") \
                    if model_cands[0].token is not None else "-"
                detail_lines.append(
                    f"sample={sample_idx:>3} step={step:>2} t={t_val:.4f} "
                    f"oracle=({oracle_top.op},{oracle_top.pos},{oracle_token_str}) "
                    f"model_top1=({model_cands[0].op},{model_cands[0].pos},{model_top1_str}) "
                    f"oracle_rank={rank} gap={gap:.4f}"
                )

            from edit_flows.sampling.beam import _apply_single_edit_to_sequence
            x_t, _ = _apply_single_edit_to_sequence(
                x_t, None, oracle_top, args.max_seq_len, PAD_TOKEN,
            )
            x_t = x_t.to(device)

        per_sample_steps.append(n_steps)

    # ---- Report ----
    report_path = os.path.join(args.output_dir, "ranking_report.txt")
    with open(report_path, "w") as f:
        f.write("=== Edit Ranking Diagnostic v2 Report ===\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Data: {args.data_dir}\n")
        f.write(f"Samples: {n_eval}\n")
        f.write(f"Total oracle steps: {total_steps}\n")
        f.write(f"Mean steps/sample: {total_steps / n_eval:.2f}\n")
        f.write(f"k_ins_token={args.k_ins_token}, k_sub_token={args.k_sub_token}, "
                f"k_edit_expand={args.k_edit_expand}\n")
        f.write(f"max_edits={args.max_edits}\n\n")

        # --- Overall ---
        f.write("=" * 60 + "\n")
        f.write("OVERALL: Oracle Top-1 Edit Rank in Model Candidates\n")
        f.write("=" * 60 + "\n")
        for label, count in [("top-1", rank_counts[1]), ("top-5", rank_counts[5]),
                              ("top-16", rank_counts[16]), ("not in top-16", rank_counts[-1])]:
            pct = count / total_steps * 100 if total_steps > 0 else 0
            f.write(f"  {label:>20s}: {count:>6}  ({pct:5.1f}%)\n")

        # Cumulative.
        cum_top1 = rank_counts[1]
        cum_top5 = rank_counts[1] + rank_counts[5]
        cum_top16 = rank_counts[1] + rank_counts[5] + rank_counts[16]
        f.write(f"\n  Cumulative:\n")
        f.write(f"  {'Top-1':>20s}: {cum_top1:>6}  ({cum_top1/total_steps*100:5.1f}%)\n")
        f.write(f"  {'Top-5':>20s}: {cum_top5:>6}  ({cum_top5/total_steps*100:5.1f}%)\n")
        f.write(f"  {'Top-16':>20s}: {cum_top16:>6}  ({cum_top16/total_steps*100:5.1f}%)\n")

        # Score gap stats.
        finite_gaps = [g for g in score_gaps if g < 900]
        if finite_gaps:
            avg_gap = sum(finite_gaps) / len(finite_gaps)
            f.write(f"\n--- Score Gap (model_top1.score - oracle_edit.score in model list) ---\n")
            f.write(f"  Mean gap: {avg_gap:.4f} nats\n")
            f.write(f"  Min gap:  {min(finite_gaps):.4f} nats\n")
            f.write(f"  Max gap:  {max(finite_gaps):.4f} nats\n")
            f.write(f"  Steps where oracle edit not in model candidates: "
                    f"{len(score_gaps) - len(finite_gaps)} ({ (len(score_gaps)-len(finite_gaps))/len(score_gaps)*100:.1f}%)\n"
                    if len(score_gaps) > 0 else "\n")

        # --- Per-step-bin breakdown ---
        f.write("\n" + "=" * 60 + "\n")
        f.write("PER-STEP-BIN BREAKDOWN\n")
        f.write("=" * 60 + "\n")
        bin_order = ["step=0", "step=1-3", "step=4-7", "step>=8"]
        for bin_label in bin_order:
            if bin_label not in step_total or step_total[bin_label] == 0:
                continue
            n = step_total[bin_label]
            s_ranks = step_rank_counts[bin_label]
            f.write(f"\n--- {bin_label} (n={n} steps) ---\n")
            for label, key in [("top-1", 1), ("top-5", 5), ("top-16", 16), ("not in top-16", -1)]:
                cnt = s_ranks.get(key, 0)
                pct = cnt / n * 100 if n > 0 else 0
                f.write(f"  {label:>20s}: {cnt:>6}  ({pct:5.1f}%)\n")
            cum_1 = s_ranks.get(1, 0)
            cum_5 = cum_1 + s_ranks.get(5, 0)
            cum_16 = cum_5 + s_ranks.get(16, 0)
            f.write(f"  {'Cum Top-5':>20s}: {cum_5:>6}  ({cum_5/n*100:5.1f}%)\n")
            f.write(f"  {'Cum Top-16':>20s}: {cum_16:>6}  ({cum_16/n*100:5.1f}%)\n")

            # Per-bin score gap.
            bin_gaps = [g for g in step_score_gaps.get(bin_label, []) if g < 900]
            if bin_gaps:
                avg_bin_gap = sum(bin_gaps) / len(bin_gaps)
                f.write(f"  Mean score gap: {avg_bin_gap:.4f} nats\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("PER-SAMPLE STEP COUNT DISTRIBUTION\n")
        f.write("=" * 60 + "\n")
        f.write(f"  Mean: {sum(per_sample_steps)/len(per_sample_steps):.2f}\n")
        f.write(f"  Median: {sorted(per_sample_steps)[len(per_sample_steps)//2]:.1f}\n")
        f.write(f"  Min: {min(per_sample_steps)}, Max: {max(per_sample_steps)}\n")
        # Histogram.
        hist: dict[str, int] = defaultdict(int)
        for s in per_sample_steps:
            if s <= 2:
                hist["0-2"] += 1
            elif s <= 5:
                hist["3-5"] += 1
            elif s <= 10:
                hist["6-10"] += 1
            elif s <= 15:
                hist["11-15"] += 1
            else:
                hist["16+"] += 1
        f.write("  Distribution:\n")
        for k in ["0-2", "3-5", "6-10", "11-15", "16+"]:
            f.write(f"    {k:>8s}: {hist.get(k, 0):>5}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("FIRST 10 SAMPLES DETAIL\n")
        f.write("=" * 60 + "\n")
        for line in detail_lines:
            f.write(line + "\n")

    # ---- JSON summary ----
    summary = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "n_samples": n_eval,
        "total_steps": total_steps,
        "mean_steps_per_sample": total_steps / n_eval if n_eval > 0 else 0,
        "max_edits": args.max_edits,
        "overall": {
            "top1": rank_counts[1],
            "top5": rank_counts[5],
            "top16": rank_counts[16],
            "not_in_top16": rank_counts[-1],
            "cum_top1": rank_counts[1],
            "cum_top5": rank_counts[1] + rank_counts[5],
            "cum_top16": rank_counts[1] + rank_counts[5] + rank_counts[16],
            "top1_pct": rank_counts[1] / total_steps * 100 if total_steps > 0 else 0,
            "top5_pct": (rank_counts[1] + rank_counts[5]) / total_steps * 100 if total_steps > 0 else 0,
            "top16_pct": (rank_counts[1] + rank_counts[5] + rank_counts[16]) / total_steps * 100 if total_steps > 0 else 0,
            "not_in_top16_pct": rank_counts[-1] / total_steps * 100 if total_steps > 0 else 0,
            "mean_score_gap": sum(g for g in score_gaps if g < 900) / max(len([g for g in score_gaps if g < 900]), 1),
            "n_not_in_candidates": len([g for g in score_gaps if g >= 900]),
        },
        "per_step_bin": {},
        "per_sample_step_distribution": {
            "mean": sum(per_sample_steps) / len(per_sample_steps),
            "median": sorted(per_sample_steps)[len(per_sample_steps)//2],
            "min": min(per_sample_steps),
            "max": max(per_sample_steps),
        },
    }
    for bin_label in ["step=0", "step=1-3", "step=4-7", "step>=8"]:
        if bin_label in step_total and step_total[bin_label] > 0:
            n = step_total[bin_label]
            sr = step_rank_counts[bin_label]
            bgaps = [g for g in step_score_gaps.get(bin_label, []) if g < 900]
            summary["per_step_bin"][bin_label] = {
                "n_steps": n,
                "top1": sr.get(1, 0),
                "top5": sr.get(5, 0),
                "top16": sr.get(16, 0),
                "not_in_top16": sr.get(-1, 0),
                "cum_top1_pct": sr.get(1, 0) / n * 100,
                "cum_top5_pct": (sr.get(1, 0) + sr.get(5, 0)) / n * 100,
                "cum_top16_pct": (sr.get(1, 0) + sr.get(5, 0) + sr.get(16, 0)) / n * 100,
                "not_in_top16_pct": sr.get(-1, 0) / n * 100,
                "mean_score_gap": sum(bgaps) / max(len(bgaps), 1),
            }

    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary to stdout.
    print(f"\n=== Results ===")
    print(f"Total oracle steps: {total_steps} ({total_steps/n_eval:.1f} per sample)")
    print(f"Overall:")
    print(f"  Top-1:              {rank_counts[1]:>6} ({rank_counts[1]/total_steps*100:5.1f}%)" if total_steps > 0 else "")
    print(f"  Top-5 (cum):        {rank_counts[1]+rank_counts[5]:>6} ({(rank_counts[1]+rank_counts[5])/total_steps*100:5.1f}%)" if total_steps > 0 else "")
    print(f"  Top-16 (cum):       {rank_counts[1]+rank_counts[5]+rank_counts[16]:>6} ({(rank_counts[1]+rank_counts[5]+rank_counts[16])/total_steps*100:5.1f}%)" if total_steps > 0 else "")
    print(f"  Not in top-16:      {rank_counts[-1]:>6} ({rank_counts[-1]/total_steps*100:5.1f}%)" if total_steps > 0 else "")
    if finite_gaps:
        print(f"Mean score gap: {sum(finite_gaps)/len(finite_gaps):.4f} nats")
    print(f"\nPer-step-bin (cumulative top-1 / top-5 / top-16):")
    for bin_label in ["step=0", "step=1-3", "step=4-7", "step>=8"]:
        if bin_label in step_total and step_total[bin_label] > 0:
            n = step_total[bin_label]
            sr = step_rank_counts[bin_label]
            t1 = sr.get(1, 0) / n * 100
            t5 = (sr.get(1, 0) + sr.get(5, 0)) / n * 100
            t16 = (sr.get(1, 0) + sr.get(5, 0) + sr.get(16, 0)) / n * 100
            print(f"  {bin_label:>12s} (n={n:>5}): top-1={t1:5.1f}%  top-5={t5:5.1f}%  top-16={t16:5.1f}%")
    print(f"\nFull report: {report_path}")
    print(f"JSON summary: {json_path}")


if __name__ == "__main__":
    main()
