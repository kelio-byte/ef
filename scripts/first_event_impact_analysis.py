#!/usr/bin/env python
"""First-event impact analysis for Edit Flows retrosynthesis."""

import argparse
import os

import torch

from edit_flows.analysis.first_step import (
    build_model_batch,
    compute_reaction_edit_distance,
    decode_sequence,
    dump_json,
    load_parallel_texts,
    tokenize_smiles,
)
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.data.dataset import load_vocab
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.sampling.euler import sample_euler, sample_euler_with_first_step_intervention


def _aggregate_correlation(first_events, predictions, targets, id2token):
    exact_matches = 0
    first_correct = 0
    first_wrong = 0
    final_correct_given_first_correct = 0
    final_correct_given_first_wrong = 0
    per_sample = []

    for idx, (event, pred_row, tgt_row) in enumerate(zip(first_events, predictions, targets)):
        pred_text = decode_sequence(pred_row.tolist(), id2token)
        tgt_text = decode_sequence(tgt_row.tolist(), id2token)
        final_correct = pred_text == tgt_text
        exact_matches += int(final_correct)
        event_correct = bool(event and event.get("event_set_correct", False))
        first_correct += int(event_correct)
        first_wrong += int(not event_correct)
        final_correct_given_first_correct += int(event_correct and final_correct)
        final_correct_given_first_wrong += int((not event_correct) and final_correct)
        per_sample.append({
            "example_idx": idx,
            "first_event": event,
            "prediction": pred_text,
            "target": tgt_text,
            "final_correct": final_correct,
            "final_edit_distance": compute_reaction_edit_distance(pred_row.tolist(), tgt_row.tolist()),
        })

    return {
        "n": len(predictions),
        "top1_acc": exact_matches / max(len(predictions), 1),
        "P(final correct | first event set correct)": (
            final_correct_given_first_correct / max(first_correct, 1)
        ),
        "P(final correct | first event set wrong)": (
            final_correct_given_first_wrong / max(first_wrong, 1)
        ),
        "n_first_event_correct": first_correct,
        "n_first_event_wrong": first_wrong,
    }, per_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="First-event impact analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--products_file", type=str, required=True)
    parser.add_argument("--targets_file", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scheduler", type=str, default=None, choices=["cubic", "linear"])
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--deduplicate", type=int, default=0)
    parser.add_argument("--max_lines", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=str, default="correlation", choices=["correlation", "intervention"])
    parser.add_argument("--event_prob_mode", type=str, default="poisson", choices=["poisson", "linear"])
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
    id2token = {v: k for k, v in token2id.items()}

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

    if args.mode == "correlation":
        all_predictions = []
        all_targets = []
        all_events = []
        for start in range(0, len(products), args.batch_size):
            end = min(start + args.batch_size, len(products))
            x_0, x_1 = build_model_batch(product_ids[start:end], target_ids[start:end])
            x_0 = x_0.repeat_interleave(args.n_samples, dim=0).to(device)
            x_1 = x_1.repeat_interleave(args.n_samples, dim=0).to(device)
            results, _, first_events = sample_euler(
                model,
                x_0,
                scheduler,
                n_steps=args.n_steps,
                max_seq_len=cfg["max_seq_len"],
                use_rate_reparam=cfg.get("use_rate_reparam", False),
                clamp_kappa=cfg.get("clamp_kappa", False),
                clamp_max=cfg.get("clamp_max", 50.0),
                time_input=cfg.get("time_input", "t"),
                train_scheduler=train_scheduler,
                event_prob_mode=args.event_prob_mode,
                record_first_events=True,
                x_1=x_1,
                vocab_size=model.vocab_size,
            )
            all_predictions.extend(results.cpu())
            all_targets.extend(x_1.cpu())
            all_events.extend(first_events)

        summary, per_sample = _aggregate_correlation(all_events, all_predictions, all_targets, id2token)
        dump_json(summary, os.path.join(args.output_dir, "correlation_summary.json"))
        torch.save(per_sample, os.path.join(args.output_dir, "per_sample_events.pt"))
        with open(os.path.join(args.output_dir, "report.md"), "w") as f:
            f.write("# First Event Correlation Analysis\n\n")
            for key, value in summary.items():
                f.write(f"- {key}: {value}\n")
        return

    intervention_summary = {}
    all_outputs = {}
    for mode in ["normal", "force_correct_first", "force_wrong_first"]:
        preds = []
        tgts = []
        events = []
        for start in range(0, len(products), args.batch_size):
            end = min(start + args.batch_size, len(products))
            x_0, x_1 = build_model_batch(product_ids[start:end], target_ids[start:end])
            x_0 = x_0.repeat_interleave(args.n_samples, dim=0).to(device)
            x_1 = x_1.repeat_interleave(args.n_samples, dim=0).to(device)
            results, first_events = sample_euler_with_first_step_intervention(
                model,
                x_0,
                x_1,
                scheduler,
                vocab_size=model.vocab_size,
                mode=mode,
                n_steps=args.n_steps,
                max_seq_len=cfg["max_seq_len"],
                use_rate_reparam=cfg.get("use_rate_reparam", False),
                clamp_kappa=cfg.get("clamp_kappa", False),
                clamp_max=cfg.get("clamp_max", 50.0),
                time_input=cfg.get("time_input", "t"),
                train_scheduler=train_scheduler,
                event_prob_mode=args.event_prob_mode,
                record_first_events=True,
            )
            preds.extend(results.cpu())
            tgts.extend(x_1.cpu())
            events.extend(first_events)

        summary, per_sample = _aggregate_correlation(events, preds, tgts, id2token)
        intervention_summary[mode] = summary
        all_outputs[mode] = per_sample

    dump_json(intervention_summary, os.path.join(args.output_dir, "intervention_summary.json"))
    torch.save(all_outputs, os.path.join(args.output_dir, "per_sample_events.pt"))
    with open(os.path.join(args.output_dir, "report.md"), "w") as f:
        f.write("# First Event Intervention Analysis\n\n")
        for mode, metrics in intervention_summary.items():
            f.write(f"## {mode}\n")
            for key, value in metrics.items():
                f.write(f"- {key}: {value}\n")


if __name__ == "__main__":
    main()
