#!/usr/bin/env python
"""Static first-step forward analysis for Edit Flows retrosynthesis."""

import argparse
import os

import torch
import torch.nn.functional as F

from edit_flows.analysis.first_step import (
    build_model_batch,
    compute_average_precision,
    dump_json,
    extract_oracle_event_set,
    extract_position_labels,
    load_parallel_texts,
    parse_time_grid,
    tokenize_smiles,
)
from edit_flows.core.rate_scale import apply_rate_parameterization
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.data.dataset import load_vocab
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.sampling.euler import _compute_model_time
from edit_flows.sampling.oracle import compute_oracle_model_output


def _init_metrics() -> dict:
    return {
        "n": 0,
        "center_hit_at_1": 0,
        "center_hit_at_3": 0,
        "center_hit_at_5": 0,
        "center_mrr_sum": 0.0,
        "position_ap_sum": 0.0,
        "type_correct": 0,
        "type_total": 0,
        "ins_top1_correct": 0,
        "ins_top5_correct": 0,
        "ins_total": 0,
        "sub_top1_correct": 0,
        "sub_top5_correct": 0,
        "sub_total": 0,
        "full_first_edit_correct": 0,
        "ins_kl_anchor_sum": 0.0,
        "ins_kl_anchor_n": 0,
        "sub_kl_anchor_sum": 0.0,
        "sub_kl_anchor_n": 0,
        "ins_kl_oracle_pos_sum": 0.0,
        "ins_kl_oracle_pos_n": 0,
        "sub_kl_oracle_pos_sum": 0.0,
        "sub_kl_oracle_pos_n": 0,
    }


def _finalize_metrics(metrics: dict) -> dict:
    n = max(metrics["n"], 1)
    type_total = max(metrics["type_total"], 1)
    ins_total = max(metrics["ins_total"], 1)
    sub_total = max(metrics["sub_total"], 1)
    ins_kl_anchor_n = max(metrics["ins_kl_anchor_n"], 1)
    sub_kl_anchor_n = max(metrics["sub_kl_anchor_n"], 1)
    ins_kl_oracle_n = max(metrics["ins_kl_oracle_pos_n"], 1)
    sub_kl_oracle_n = max(metrics["sub_kl_oracle_pos_n"], 1)
    return {
        "n": metrics["n"],
        "Center Hit@1": metrics["center_hit_at_1"] / n,
        "Center Hit@3": metrics["center_hit_at_3"] / n,
        "Center Hit@5": metrics["center_hit_at_5"] / n,
        "Center MRR": metrics["center_mrr_sum"] / n,
        "Position AP": metrics["position_ap_sum"] / n,
        "Type Acc@oracle-pos": metrics["type_correct"] / type_total,
        "Ins Token Acc@1": metrics["ins_top1_correct"] / ins_total,
        "Ins Token Acc@5": metrics["ins_top5_correct"] / ins_total,
        "Sub Token Acc@1": metrics["sub_top1_correct"] / sub_total,
        "Sub Token Acc@5": metrics["sub_top5_correct"] / sub_total,
        "Full First-Edit Acc": metrics["full_first_edit_correct"] / n,
        "Ins KL@anchor": metrics["ins_kl_anchor_sum"] / ins_kl_anchor_n,
        "Sub KL@anchor": metrics["sub_kl_anchor_sum"] / sub_kl_anchor_n,
        "Ins KL@oracle-pos": metrics["ins_kl_oracle_pos_sum"] / ins_kl_oracle_n,
        "Sub KL@oracle-pos": metrics["sub_kl_oracle_pos_sum"] / sub_kl_oracle_n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Static first-step forward analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--products_file", type=str, required=True)
    parser.add_argument("--targets_file", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scheduler", type=str, default=None, choices=["cubic", "linear"])
    parser.add_argument("--time_grid", type=str, default="0,1e-3,1e-2,5e-2,0.1,0.2,0.3")
    parser.add_argument("--deduplicate", type=int, default=0)
    parser.add_argument("--max_lines", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    scheduler_name = args.scheduler or cfg.get("sample_scheduler", cfg.get("scheduler", "cubic"))
    scheduler = LinearScheduler() if scheduler_name == "linear" else CubicScheduler()
    train_scheduler_name = cfg.get("scheduler", "cubic")
    train_scheduler = LinearScheduler() if train_scheduler_name == "linear" else CubicScheduler()

    vocab_path = args.vocab_file or os.path.join(cfg["data_dir"], cfg.get("vocab_file", "example.vocab.src"))
    token2id, model_vocab = load_vocab(vocab_path)

    model = EditFlowsTransformer(
        vocab_size=ckpt.get("model_vocab", model_vocab),
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    products, targets = load_parallel_texts(
        args.products_file, args.targets_file,
        deduplicate=args.deduplicate, max_lines=args.max_lines,
    )
    product_ids = [tokenize_smiles(line, token2id) for line in products]
    target_ids = [tokenize_smiles(line, token2id) for line in targets]

    summary = {
        "checkpoint": args.checkpoint,
        "scheduler": scheduler_name,
        "time_grid": parse_time_grid(args.time_grid),
        "n_examples": len(products),
        "metrics": {},
    }
    per_example = []

    for t_value in summary["time_grid"]:
        metrics_base = _init_metrics()
        metrics_eff = _init_metrics()

        for start in range(0, len(products), args.batch_size):
            end = min(start + args.batch_size, len(products))
            x_0, x_1 = build_model_batch(product_ids[start:end], target_ids[start:end])
            x_0 = x_0.to(device)
            x_1 = x_1.to(device)
            x_pad_mask = x_0 == 0
            t = torch.full((x_0.shape[0], 1), t_value, dtype=torch.float, device=device)
            t_model = _compute_model_time(
                t, scheduler, cfg.get("time_input", "t"), train_scheduler,
            )

            log_rates, log_ins_probs, log_sub_probs = model(x_0, t_model, x_pad_mask)
            if not cfg.get("use_rate_reparam", False) and scheduler.name != train_scheduler.name:
                k_sample = __import__(
                    "edit_flows.core.rate_scale",
                    fromlist=["get_rate_scale"],
                ).get_rate_scale(
                    t, scheduler,
                    clamp_kappa=cfg.get("clamp_kappa", False),
                    clamp_max=cfg.get("clamp_max", 50.0),
                )
                k_train = __import__(
                    "edit_flows.core.rate_scale",
                    fromlist=["get_rate_scale"],
                ).get_rate_scale(
                    t_model, train_scheduler,
                    clamp_kappa=cfg.get("clamp_kappa", False),
                    clamp_max=cfg.get("clamp_max", 50.0),
                )
                log_rates = log_rates + torch.log(
                    k_sample / k_train.clamp_min(1e-12)
                ).unsqueeze(1)
            log_rates_eff = apply_rate_parameterization(
                log_rates, t, scheduler,
                use_rate_reparam=cfg.get("use_rate_reparam", False),
                clamp_kappa=cfg.get("clamp_kappa", False),
                clamp_max=cfg.get("clamp_max", 50.0),
            )

            oracle_out = compute_oracle_model_output(
                x_0, x_1, t, scheduler, model.vocab_size,
            )
            oracle = extract_oracle_event_set(
                oracle_out[0], oracle_out[1], oracle_out[2], x_0,
            )

            for i in range(x_0.shape[0]):
                pos_scores_base, pos_labels = extract_position_labels(oracle_out[0][i], x_0[i])
                oracle_positions = [idx for idx, flag in enumerate(pos_labels) if flag]
                if not oracle_positions:
                    continue

                def update(metrics: dict, rates_tensor: torch.Tensor, prefix: str) -> None:
                    pos_scores = torch.exp(rates_tensor[i]).sum(dim=-1).tolist()
                    pos_scores = [
                        score if (j != 0 and x_0[i, j].item() != 0) else float("-inf")
                        for j, score in enumerate(pos_scores)
                    ]
                    ranked_positions = sorted(
                        range(len(pos_scores)),
                        key=lambda idx: pos_scores[idx],
                        reverse=True,
                    )
                    top1 = ranked_positions[:1]
                    top3 = ranked_positions[:3]
                    top5 = ranked_positions[:5]
                    metrics["n"] += 1
                    metrics["center_hit_at_1"] += int(any(pos in oracle_positions for pos in top1))
                    metrics["center_hit_at_3"] += int(any(pos in oracle_positions for pos in top3))
                    metrics["center_hit_at_5"] += int(any(pos in oracle_positions for pos in top5))
                    first_rank = next(
                        (rank for rank, pos in enumerate(ranked_positions, start=1) if pos in oracle_positions),
                        None,
                    )
                    metrics["center_mrr_sum"] += 0.0 if first_rank is None else 1.0 / first_rank
                    metrics["position_ap_sum"] += compute_average_precision(pos_scores, pos_labels)

                    anchor_pos = ranked_positions[0]
                    pred_type = int(torch.argmax(torch.exp(rates_tensor[i, anchor_pos])).item())
                    oracle_type = int(oracle["type_argmax"][i, anchor_pos].item())
                    type_correct = pred_type == oracle_type and bool(oracle["pos_mask"][i, anchor_pos].item())
                    metrics["type_total"] += 1
                    metrics["type_correct"] += int(type_correct)

                    full_correct = False
                    if pred_type == 0 and bool(oracle["ins_mask"][i, anchor_pos].item()):
                        token_logits = log_ins_probs[i, anchor_pos]
                        top5_tokens = torch.topk(token_logits, k=min(5, token_logits.shape[-1])).indices.tolist()
                        ora_ins_probs = torch.exp(oracle_out[1][i, anchor_pos])
                        ora_max_prob = ora_ins_probs.max()
                        ora_valid_tokens = set(
                            (ora_ins_probs >= ora_max_prob - 1e-6).nonzero(as_tuple=True)[0].tolist()
                        )
                        metrics["ins_total"] += 1
                        metrics["ins_top1_correct"] += int(top5_tokens[0] in ora_valid_tokens)
                        metrics["ins_top5_correct"] += int(
                            any(t in ora_valid_tokens for t in top5_tokens)
                        )
                        full_correct = type_correct and top5_tokens[0] in ora_valid_tokens
                        if prefix == "base":
                            kl = F.kl_div(
                                log_ins_probs[i, anchor_pos].unsqueeze(0),
                                oracle_out[1][i, anchor_pos].clamp(min=-1e9).unsqueeze(0),
                                log_target=True, reduction='sum',
                            ).item()
                            metrics["ins_kl_anchor_sum"] += kl
                            metrics["ins_kl_anchor_n"] += 1
                    elif pred_type == 1 and bool(oracle["sub_mask"][i, anchor_pos].item()):
                        token_logits = log_sub_probs[i, anchor_pos]
                        top5_tokens = torch.topk(token_logits, k=min(5, token_logits.shape[-1])).indices.tolist()
                        ora_sub_probs = torch.exp(oracle_out[2][i, anchor_pos])
                        ora_max_prob = ora_sub_probs.max()
                        ora_valid_tokens = set(
                            (ora_sub_probs >= ora_max_prob - 1e-6).nonzero(as_tuple=True)[0].tolist()
                        )
                        metrics["sub_total"] += 1
                        metrics["sub_top1_correct"] += int(top5_tokens[0] in ora_valid_tokens)
                        metrics["sub_top5_correct"] += int(
                            any(t in ora_valid_tokens for t in top5_tokens)
                        )
                        full_correct = type_correct and top5_tokens[0] in ora_valid_tokens
                        if prefix == "base":
                            kl = F.kl_div(
                                log_sub_probs[i, anchor_pos].unsqueeze(0),
                                oracle_out[2][i, anchor_pos].clamp(min=-1e9).unsqueeze(0),
                                log_target=True, reduction='sum',
                            ).item()
                            metrics["sub_kl_anchor_sum"] += kl
                            metrics["sub_kl_anchor_n"] += 1
                    elif pred_type == 2 and bool(oracle["del_mask"][i, anchor_pos].item()):
                        full_correct = type_correct

                    metrics["full_first_edit_correct"] += int(full_correct)

                    if prefix == "effective":
                        per_example.append({
                            "example_idx": start + i,
                            "t": t_value,
                            "anchor_pos": anchor_pos,
                            "oracle_positions": oracle_positions,
                            "pred_type": pred_type,
                            "oracle_type": oracle_type,
                            "full_correct": full_correct,
                        })

                update(metrics_base, log_rates, "base")
                update(metrics_eff, log_rates_eff, "effective")

                # KL@oracle-pos: all oracle-positive positions (once per sample)
                _oracle_ins_positions = torch.where(oracle["ins_mask"][i])[0]
                for _pos in _oracle_ins_positions:
                    _kl = F.kl_div(
                        log_ins_probs[i, _pos].unsqueeze(0),
                        oracle_out[1][i, _pos].clamp(min=-1e9).unsqueeze(0),
                        log_target=True, reduction='sum',
                    ).item()
                    metrics_base["ins_kl_oracle_pos_sum"] += _kl
                    metrics_base["ins_kl_oracle_pos_n"] += 1
                _oracle_sub_positions = torch.where(oracle["sub_mask"][i])[0]
                for _pos in _oracle_sub_positions:
                    _kl = F.kl_div(
                        log_sub_probs[i, _pos].unsqueeze(0),
                        oracle_out[2][i, _pos].clamp(min=-1e9).unsqueeze(0),
                        log_target=True, reduction='sum',
                    ).item()
                    metrics_base["sub_kl_oracle_pos_sum"] += _kl
                    metrics_base["sub_kl_oracle_pos_n"] += 1

        base_metrics = _finalize_metrics(metrics_base)
        eff_metrics = _finalize_metrics(metrics_eff)
        for kl_key in ["Ins KL@anchor", "Sub KL@anchor", "Ins KL@oracle-pos", "Sub KL@oracle-pos"]:
            eff_metrics[kl_key] = base_metrics[kl_key]
        summary["metrics"][str(t_value)] = {
            "base": base_metrics,
            "effective": eff_metrics,
        }

    dump_json(summary, os.path.join(args.output_dir, "summary.json"))
    torch.save(per_example, os.path.join(args.output_dir, "per_example.pt"))
    with open(os.path.join(args.output_dir, "report.md"), "w") as f:
        f.write("# First-Step Forward Analysis\n\n")
        for t_value in summary["time_grid"]:
            metrics = summary["metrics"][str(t_value)]
            f.write(f"## t={t_value}\n")
            f.write(f"- base Center Hit@1: {metrics['base']['Center Hit@1']:.4f}\n")
            f.write(f"- effective Center Hit@1: {metrics['effective']['Center Hit@1']:.4f}\n")
            f.write(f"- base Full First-Edit Acc: {metrics['base']['Full First-Edit Acc']:.4f}\n")
            f.write(f"- effective Full First-Edit Acc: {metrics['effective']['Full First-Edit Acc']:.4f}\n")
            f.write(f"- Ins Token Acc@1: {metrics['base']['Ins Token Acc@1']:.4f}\n")
            f.write(f"- Ins KL@anchor: {metrics['base']['Ins KL@anchor']:.4f}\n")
            f.write(f"- Ins KL@oracle-pos: {metrics['base']['Ins KL@oracle-pos']:.4f}\n")
            f.write(f"- Sub Token Acc@1: {metrics['base']['Sub Token Acc@1']:.4f}\n")
            f.write(f"- Sub KL@anchor: {metrics['base']['Sub KL@anchor']:.4f}\n")
            f.write(f"- Sub KL@oracle-pos: {metrics['base']['Sub KL@oracle-pos']:.4f}\n")


if __name__ == "__main__":
    main()
