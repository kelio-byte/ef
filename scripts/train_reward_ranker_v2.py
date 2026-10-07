#!/usr/bin/env python3
"""Train/evaluate pre-registered endpoint reward rankers without target leakage.

The script has two modes.  ``train`` reads only the new ranker-train and
ranker-validation candidate pools.  It freezes two small, controlled models:

* ``residual``: raw forward reciprocal-rank plus a bounded learned residual;
* ``listwise``: an unconstrained listwise/hard-negative ranker.

``evaluate`` reads the fresh holdout once and writes one report containing raw
forward, residual and listwise endpoint rankings.  All labels are offline and
all score features are decoded from product/candidate records; the dataset
target is never passed into a feature.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch
from torch import nn

try:
    from scripts.train_correctness_reward import (
        INVALID_SCORE,
        _auc_from_pairs,
        _canonical_target_cache,
        _deduplicate_records,
        _group_auc,
        _load_records,
        _rank_metrics,
        _read_original_targets,
    )
    from edit_flows.data.dataset import load_vocab
except ModuleNotFoundError:  # pragma: no cover - direct script imports
    from train_correctness_reward import (
        INVALID_SCORE,
        _auc_from_pairs,
        _canonical_target_cache,
        _deduplicate_records,
        _group_auc,
        _load_records,
        _rank_metrics,
        _read_original_targets,
    )
    from edit_flows.data.dataset import load_vocab


SCHEMA_VERSION = 1
FEATURE_VERSION = "correctness_features_noleak_v2"
DEFAULT_STEPS = 2000
DEFAULT_LR = 1e-3
DEFAULT_TEMPERATURE = 0.25
DEFAULT_MARGIN = 0.05
DEFAULT_PAIR_WEIGHT = 0.5
DEFAULT_RESIDUAL_CAP = 0.25
DEFAULT_RESIDUAL_REG = 0.01
DEFAULT_HARD_NEGATIVE_K = 3


class RewardRanker(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_examples(
    data_path: str | Path,
    targets_file: str | Path,
    vocab_file: str | Path,
    augmentation: int,
) -> tuple[list[dict], dict, dict]:
    records, metadata = _load_records(data_path)
    token2id, _ = load_vocab(vocab_file)
    id2token = {value: key for key, value in token2id.items()}
    targets = _read_original_targets(targets_file, augmentation)
    canonical_targets = _canonical_target_cache(targets)
    examples, summary = _deduplicate_records(
        records,
        id2token=id2token,
        canonical_targets=canonical_targets,
        vocab_size=len(token2id),
    )
    summary = dict(summary)
    summary["data_path"] = str(Path(data_path).resolve())
    summary["data_sha256"] = _sha256(data_path)
    summary["metadata"] = metadata
    return examples, summary, {"vocab_size": len(token2id)}


def _features_and_labels(
    examples: Sequence[Mapping[str, Any]],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [row["features"] for row in examples], dtype=torch.float32,
        device=device,
    )
    labels = torch.tensor(
        [float(row["label"]) for row in examples], dtype=torch.float32,
        device=device,
    )
    raw = torch.tensor(
        [float(row["raw_forward_reward"]) for row in examples],
        dtype=torch.float32, device=device,
    )
    return (features - mean.to(device)) / std.to(device), labels, raw


def _group_indices(examples: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(examples):
        groups[int(row["product_index"])].append(index)
    return list(groups.values())


def _prepare_training_groups(
    examples: Sequence[Mapping[str, Any]],
    groups: Sequence[Sequence[int]],
    raw_scores: torch.Tensor,
    *,
    hard_negative_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute reaction groups and raw-score hard negatives once.

    The contents are fixed for the whole optimization run.  Keeping this
    preparation outside the step loop preserves the loss exactly while
    avoiding millions of repeated Python and tensor-construction operations.
    """

    active: list[list[int]] = []
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    for group in groups:
        positive = [index for index in group if examples[index]["label"]]
        negative = [index for index in group if not examples[index]["label"]]
        if not positive or not negative:
            continue
        active.append(list(group))
        hard_negative = sorted(
            negative,
            key=lambda index: (-float(raw_scores[index].item()), index),
        )[:hard_negative_k]
        for pos_index in positive:
            for neg_index in hard_negative:
                positive_indices.append(pos_index)
                negative_indices.append(neg_index)
    if not active:
        raise ValueError("training split has no reaction with both positive and negative candidates")
    max_group_size = max(len(group) for group in active)
    group_indices = torch.zeros(
        (len(active), max_group_size), dtype=torch.long, device=device,
    )
    group_mask = torch.zeros(
        (len(active), max_group_size), dtype=torch.bool, device=device,
    )
    for row_index, group in enumerate(active):
        group_indices[row_index, :len(group)] = torch.tensor(
            group, dtype=torch.long, device=device,
        )
        group_mask[row_index, :len(group)] = True
    return (
        group_indices,
        group_mask,
        torch.tensor(positive_indices, dtype=torch.long, device=device),
        torch.tensor(negative_indices, dtype=torch.long, device=device),
    )


def _final_scores(
    model: RewardRanker,
    features: torch.Tensor,
    raw_scores: torch.Tensor,
    mode: str,
    residual_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(features)
    if mode == "residual":
        residual = residual_cap * torch.tanh(output)
        return raw_scores + residual, residual
    if mode == "listwise":
        return output, output
    raise ValueError(f"unknown ranker mode: {mode}")


def _ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_indices: torch.Tensor,
    group_mask: torch.Tensor,
    *,
    mode: str,
    temperature: float,
    margin: float,
    pair_weight: float,
    residual: torch.Tensor,
    residual_reg: float,
    pair_positive: torch.Tensor,
    pair_negative: torch.Tensor,
) -> torch.Tensor:
    if group_indices.numel() == 0:
        raise ValueError("training split has no reaction with both positive and negative candidates")
    safe_indices = group_indices.reshape(-1)
    group_scores = scores.index_select(0, safe_indices).reshape(group_indices.shape)
    group_labels = labels.index_select(0, safe_indices).reshape(group_indices.shape)
    group_scores = group_scores.masked_fill(~group_mask, -1.0e9)
    group_labels = group_labels * group_mask
    target = group_labels / group_labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    loss = -(
        target * torch.log_softmax(group_scores / temperature, dim=1)
    ).sum(dim=1).mean()
    if pair_positive.numel():
        loss = loss + pair_weight * torch.nn.functional.softplus(
            margin - (scores.index_select(0, pair_positive)
                      - scores.index_select(0, pair_negative))
        ).mean()
    if mode == "residual":
        loss = loss + residual_reg * residual.square().mean()
    return loss


def _fit_model(
    examples: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    args: argparse.Namespace,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[RewardRanker, list[dict[str, float]]]:
    train_examples = [row for row in examples if row["valid"]]
    features, labels, raw_scores = _features_and_labels(
        train_examples, mean, std, device,
    )
    model = RewardRanker(int(features.shape[1])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    groups = _group_indices(train_examples)
    group_indices, group_mask, pair_positive, pair_negative = _prepare_training_groups(
        train_examples, groups, raw_scores,
        hard_negative_k=args.hard_negative_k, device=device,
    )
    history: list[dict[str, float]] = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        scores, residual = _final_scores(
            model, features, raw_scores, mode, args.residual_cap,
        )
        loss = _ranking_loss(
            scores, labels, group_indices, group_mask,
            mode=mode,
            temperature=args.temperature,
            margin=args.margin,
            pair_weight=args.pair_weight,
            residual=residual,
            residual_reg=args.residual_reg,
            pair_positive=pair_positive,
            pair_negative=pair_negative,
        )
        loss.backward()
        optimizer.step()
        if step == 1 or step == args.steps or step % max(args.steps // 10, 1) == 0:
            history.append({"step": float(step), "loss": float(loss.item())})
    return model, history


def _score_examples(
    model: RewardRanker,
    examples: Sequence[Mapping[str, Any]],
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    mode: str,
    residual_cap: float,
    device: torch.device,
) -> list[float]:
    valid = [row["valid"] for row in examples]
    features, _, raw = _features_and_labels(examples, mean, std, device)
    model.eval()
    with torch.no_grad():
        scores, _ = _final_scores(model, features, raw, mode, residual_cap)
    output = scores.detach().cpu().tolist()
    return [float(value) if is_valid else INVALID_SCORE for value, is_valid in zip(output, valid)]


def _load_checkpoint(path: str | Path, device: torch.device) -> tuple[RewardRanker, dict, torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError(f"incompatible ranker checkpoint: {path}")
    model = RewardRanker(int(payload["feature_dim"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload["config"], payload["feature_mean"].float(), payload["feature_std"].float()


def _save_checkpoint(
    path: Path,
    model: RewardRanker,
    *,
    mode: str,
    args: argparse.Namespace,
    mean: torch.Tensor,
    std: torch.Tensor,
    feature_dim: int,
    train_summary: dict,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "model_type": "reaction_listwise_reward_ranker",
        "mode": mode,
        "model_state_dict": model.state_dict(),
        "feature_dim": feature_dim,
        "feature_mean": mean.cpu(),
        "feature_std": std.cpu(),
        "config": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "margin": args.margin,
            "pair_weight": args.pair_weight,
            "residual_cap": args.residual_cap,
            "residual_reg": args.residual_reg,
            "hard_negative_k": args.hard_negative_k,
            "seed": args.seed,
            "augmentation": args.augmentation,
            "invalid_policy": "excluded_from_training; fixed lowest score at evaluation",
        },
        "train_summary": train_summary,
    }
    torch.save(payload, path)


def _bootstrap_rank_deltas(
    baseline: dict,
    candidate: dict,
    *,
    metrics: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    products = sorted(int(key) for key in baseline["per_reaction"])
    rng = random.Random(seed)
    values = {metric: [] for metric in metrics}
    for _ in range(samples):
        chosen = [products[rng.randrange(len(products))] for _ in products]
        for metric in metrics:
            delta = sum(
                candidate["per_reaction"][str(product)][metric]
                - baseline["per_reaction"][str(product)][metric]
                for product in chosen
            )
            values[metric].append(100.0 * delta / len(chosen))
    output = {}
    for metric, metric_values in values.items():
        metric_values.sort()
        output[metric] = [
            metric_values[int(0.025 * (len(metric_values) - 1))],
            metric_values[int(0.975 * (len(metric_values) - 1))],
        ]
    return output


def _transition_counts(
    baseline: dict,
    candidate: dict,
) -> dict[str, int]:
    counts = defaultdict(int)
    for product in baseline["per_reaction"]:
        counts[
            f"{bool(baseline['per_reaction'][product]['1'])}->"
            f"{bool(candidate['per_reaction'][product]['1'])}"
        ] += 1
    return dict(sorted(counts.items()))


def _metrics_for_scores(
    examples: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    raw_scores: Sequence[float],
) -> dict:
    valid_indices = [index for index, row in enumerate(examples) if row["valid"]]
    labels = [bool(examples[index]["label"]) for index in valid_indices]
    values = [scores[index] for index in valid_indices]
    raw_values = [raw_scores[index] for index in valid_indices]
    global_auc, _, _, _ = _auc_from_pairs(values, labels)
    raw_auc, _, _, _ = _auc_from_pairs(raw_values, labels)
    group_auc, group_pairs = _group_auc(examples, scores, group_key="source_index")
    raw_group_auc, raw_group_pairs = _group_auc(examples, raw_scores, group_key="source_index")
    endpoint = _rank_metrics(examples, scores, order="correctness_reward")
    raw_endpoint = _rank_metrics(examples, raw_scores, order="raw_forward_reward")
    return {
        "global_valid_auc": global_auc,
        "raw_global_valid_auc": raw_auc,
        "global_auc_delta": global_auc - raw_auc if global_auc is not None and raw_auc is not None else None,
        "within_anchor_auc": group_auc,
        "raw_within_anchor_auc": raw_group_auc,
        "within_anchor_auc_delta": group_auc - raw_group_auc if group_auc is not None and raw_group_auc is not None else None,
        "within_anchor_pair_count": group_pairs,
        "raw_within_anchor_pair_count": raw_group_pairs,
        "endpoint": endpoint,
        "raw_endpoint": raw_endpoint,
        "transition_counts": _transition_counts(raw_endpoint, endpoint),
    }


def _train(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    train_examples, train_summary, vocab_summary = _prepare_examples(
        args.train_data, args.targets_file, args.vocab_file, args.augmentation,
    )
    validation_examples, validation_summary, _ = _prepare_examples(
        args.validation_data, args.targets_file, args.vocab_file, args.augmentation,
    )
    train_products = {int(row["product_index"]) for row in train_examples}
    validation_products = {int(row["product_index"]) for row in validation_examples}
    if train_products & validation_products:
        raise ValueError("train and validation reaction sets overlap")
    train_valid = [row for row in train_examples if row["valid"]]
    feature_tensor = torch.tensor([row["features"] for row in train_valid], dtype=torch.float32)
    mean = feature_tensor.mean(dim=0)
    std = feature_tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation_reports = {}
    histories = {}
    for mode in ("residual", "listwise"):
        model, history = _fit_model(
            train_examples,
            mode=mode,
            args=args,
            device=device,
            mean=mean,
            std=std,
        )
        checkpoint_path = output / f"ranker_{mode}.pt"
        _save_checkpoint(
            checkpoint_path, model, mode=mode, args=args, mean=mean, std=std,
            feature_dim=int(feature_tensor.shape[1]), train_summary=train_summary,
        )
        scores = _score_examples(
            model, validation_examples, mean, std,
            mode=mode, residual_cap=args.residual_cap, device=device,
        )
        raw_scores = [float(row["raw_forward_reward"]) for row in validation_examples]
        validation_reports[mode] = _metrics_for_scores(validation_examples, scores, raw_scores)
        histories[mode] = history
    report = {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "mode": "train_without_holdout",
        "config": vars(args),
        "train_summary": train_summary,
        "validation_summary": validation_summary,
        "vocab_summary": vocab_summary,
        "validation_reports": validation_reports,
        "histories": histories,
        "holdout_read": False,
    }
    (output / "train_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str)
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    holdout_examples, holdout_summary, _ = _prepare_examples(
        args.holdout_data, args.targets_file, args.vocab_file, args.augmentation,
    )
    raw_scores = [float(row["raw_forward_reward"]) for row in holdout_examples]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_reports = {}
    score_rows = []
    for mode in ("residual", "listwise"):
        checkpoint_path = Path(args.reward_dir) / f"ranker_{mode}.pt"
        model, config, mean, std = _load_checkpoint(checkpoint_path, device)
        scores = _score_examples(
            model, holdout_examples, mean, std,
            mode=mode, residual_cap=float(config["residual_cap"]), device=device,
        )
        model_reports[mode] = _metrics_for_scores(holdout_examples, scores, raw_scores)
        for row, score in zip(holdout_examples, scores):
            score_rows.append({
                "mode": mode,
                "record_index": row["record_index"],
                "product_index": row["product_index"],
                "source_index": row["source_index"],
                "candidate_canonical": row["candidate_canonical"],
                "label": row["label"],
                "valid": row["valid"],
                "raw_forward_reward": row["raw_forward_reward"],
                "ranker_score": score,
            })
    baseline_endpoint = _rank_metrics(
        holdout_examples, raw_scores, order="raw_forward_reward",
    )
    bootstrap = {}
    metrics = ("1", "3", "10", "oracle")
    for mode in model_reports:
        bootstrap[mode] = _bootstrap_rank_deltas(
            baseline_endpoint, model_reports[mode]["endpoint"],
            metrics=metrics, samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + (0 if mode == "residual" else 1),
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "mode": "single_frozen_holdout_report",
        "holdout_read": True,
        "holdout_summary": holdout_summary,
        "reward_dir": str(Path(args.reward_dir).resolve()),
        "models": {
            mode: {
                **metrics_report,
                "checkpoint": str((Path(args.reward_dir) / f"ranker_{mode}.pt").resolve()),
                "checkpoint_sha256": _sha256(Path(args.reward_dir) / f"ranker_{mode}.pt"),
                "bootstrap_delta_percentage_points_95ci": bootstrap[mode],
            }
            for mode, metrics_report in model_reports.items()
        },
        "raw_forward": {
            "endpoint": baseline_endpoint,
            "candidate_pool": {
                "candidate_count": len(holdout_examples),
                "valid_count": sum(row["valid"] for row in holdout_examples),
                "invalid_count": sum(not row["valid"] for row in holdout_examples),
            },
        },
        "bootstrap_config": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "statistical_unit": "original reaction",
        },
    }
    (output / "holdout_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str)
    )
    (output / "holdout_scores.json").write_text(
        json.dumps(score_rows, indent=2, sort_keys=True, default=str)
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate"), required=True)
    parser.add_argument("--train_data", type=Path)
    parser.add_argument("--validation_data", type=Path)
    parser.add_argument("--holdout_data", type=Path)
    parser.add_argument("--reward_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--targets_file", type=Path, required=True)
    parser.add_argument("--vocab_file", type=Path, required=True)
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--pair_weight", type=float, default=DEFAULT_PAIR_WEIGHT)
    parser.add_argument("--residual_cap", type=float, default=DEFAULT_RESIDUAL_CAP)
    parser.add_argument("--residual_reg", type=float, default=DEFAULT_RESIDUAL_REG)
    parser.add_argument("--hard_negative_k", type=int, default=DEFAULT_HARD_NEGATIVE_K)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260813)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "train":
        if args.train_data is None or args.validation_data is None:
            raise SystemExit("train mode requires --train_data and --validation_data")
        return _train(args)
    if args.holdout_data is None or args.reward_dir is None:
        raise SystemExit("evaluate mode requires --holdout_data and --reward_dir")
    return _evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
