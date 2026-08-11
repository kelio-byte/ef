#!/usr/bin/env python3
"""Train and freeze a low-capacity candidate-correctness reward.

This script deliberately separates training from the independent reward
holdout evaluation.  ``--mode train`` never opens the holdout guidance file;
``--mode evaluate`` loads the already-frozen checkpoint and produces one
combined AUC + endpoint-rerank report.  Labels are used only offline:
canonical(candidate) == canonical(dataset target) is the positive class.

The model is a logistic head over label-free, inference-available features:
product/candidate token histograms, their difference, lengths, validity and
the raw forward-beam reciprocal-rank score.  It is intentionally small and
auditable rather than a second sequence model.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

try:
    from audit_guidance_reward_quality import (
        _canonical_global,
        _decode_terminal,
        _read_original_targets,
    )
except ModuleNotFoundError:  # package-style imports in tests/tools
    from scripts.audit_guidance_reward_quality import (
        _canonical_global,
        _decode_terminal,
        _read_original_targets,
    )
from edit_flows.data.dataset import load_vocab
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


SCHEMA_VERSION = 1
FEATURE_VERSION = "correctness_reward_features_v1"
DEFAULT_STEPS = 2000
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
INVALID_SCORE = -1.0e9


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_records(path: str | Path) -> tuple[list[dict], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError(f"{path} is not a guidance-record payload")
    records = payload["records"]
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} has no records")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} metadata is not a mapping")
    return records, metadata


def _token_sequence(value: Sequence[int]) -> list[int]:
    tokens: list[int] = []
    for raw in value:
        token = int(raw)
        if token == PAD_TOKEN:
            break
        if token != BOS_TOKEN:
            tokens.append(token)
    return tokens


def _raw_forward_reward(record: Mapping[str, Any]) -> float:
    rank = int(record.get("forward_beam_rank", 0))
    if rank < 0:
        raise ValueError("forward_beam_rank must be non-negative")
    return 1.0 / rank if rank else 0.0


def _canonical_target_cache(
    targets: Sequence[str],
) -> list[str]:
    return [_canonical_global(value) for value in targets]


def _feature_vector(
    product_tokens: Sequence[int],
    candidate_tokens: Sequence[int],
    *,
    product_canonical: str,
    candidate_canonical: str,
    raw_forward_reward: float,
    vocab_size: int,
) -> list[float]:
    """Build inference-available features; no target or product index enters."""

    product = _token_sequence(product_tokens)
    candidate = _token_sequence(candidate_tokens)
    p_counts = torch.bincount(
        torch.tensor(product, dtype=torch.long), minlength=vocab_size,
    ).float()
    c_counts = torch.bincount(
        torch.tensor(candidate, dtype=torch.long), minlength=vocab_size,
    ).float()
    p_norm = p_counts / max(float(len(product)), 1.0)
    c_norm = c_counts / max(float(len(candidate)), 1.0)
    diff = c_norm - p_norm
    p_components = float(product_canonical.count(".") + 1) if product_canonical else 0.0
    c_components = float(candidate_canonical.count(".") + 1) if candidate_canonical else 0.0
    valid = float(bool(candidate_canonical))
    p_len = float(len(product))
    c_len = float(len(candidate))
    scalar = torch.tensor(
        [
            float(raw_forward_reward),
            valid,
            math.log1p(p_len),
            math.log1p(c_len),
            math.log1p(abs(c_len - p_len)),
            c_len / max(p_len, 1.0),
            p_components,
            c_components,
            c_components - p_components,
        ],
        dtype=torch.float32,
    )
    return torch.cat((scalar, p_norm, c_norm, diff)).tolist()


def _candidate_label(
    record: Mapping[str, Any],
    id2token: Mapping[int, str],
    canonical_targets: Sequence[str],
) -> tuple[str, bool, str]:
    product_index = int(record["product_index"])
    terminal = _canonical_global(_decode_terminal(record, id2token))
    if product_index < 0 or product_index >= len(canonical_targets):
        raise ValueError(
            f"product_index={product_index} is outside {len(canonical_targets)} targets"
        )
    return terminal, bool(terminal and terminal == canonical_targets[product_index]), terminal


def _deduplicate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    id2token: Mapping[int, str],
    canonical_targets: Sequence[str],
    vocab_size: int,
) -> tuple[list[dict], dict]:
    """Deduplicate one candidate per product, keeping strongest raw score."""

    chosen: dict[tuple[int, str], dict] = {}
    raw_invalid = 0
    raw_positive = 0
    raw_products: set[int] = set()
    raw_source_groups: set[int] = set()
    for record_index, record in enumerate(records):
        product_index = int(record["product_index"])
        source_index = int(record.get("source_index", product_index))
        raw_products.add(product_index)
        raw_source_groups.add(source_index)
        candidate_canonical, label, _ = _candidate_label(
            record, id2token, canonical_targets,
        )
        raw_invalid += int(not candidate_canonical)
        raw_positive += int(label)
        # Invalid candidates are intentionally collapsed to one lowest-score
        # representative per product.  They are not used to train the model.
        key = (product_index, candidate_canonical)
        raw_score = _raw_forward_reward(record)
        candidate = {
            "record_index": record_index,
            "product_index": product_index,
            "source_index": source_index,
            "sample_index": int(record.get("sample_index", -1)),
            "time_index": int(record.get("time_index", -1)),
            "anchor_ordinal": int(record.get("anchor_ordinal", -1)),
            "candidate_canonical": candidate_canonical,
            "label": bool(label),
            "valid": bool(candidate_canonical),
            "raw_forward_reward": raw_score,
            "forward_beam_rank": int(record.get("forward_beam_rank", 0)),
            "features": _feature_vector(
                record["product_tokens"], record["terminal_tokens"],
                product_canonical=canonical_targets[product_index]
                if product_index < len(canonical_targets) else "",
                candidate_canonical=candidate_canonical,
                raw_forward_reward=raw_score,
                vocab_size=vocab_size,
            ),
        }
        if key not in chosen:
            chosen[key] = candidate
            continue
        previous = chosen[key]
        if previous["label"] != candidate["label"]:
            raise AssertionError("duplicate candidate received inconsistent labels")
        # Prefer a representative with the strongest frozen raw reward, then
        # first-seen order.  This is fixed before any holdout score is read.
        if candidate["raw_forward_reward"] > previous["raw_forward_reward"]:
            chosen[key] = candidate

    examples = list(chosen.values())
    examples.sort(key=lambda row: (row["product_index"], row["record_index"]))
    products_with_positive = len({row["product_index"] for row in examples if row["label"]})
    summary = {
        "raw_record_count": len(records),
        "deduplicated_candidate_count": len(examples),
        "raw_duplicate_count": len(records) - len(examples),
        "raw_invalid_count": raw_invalid,
        "raw_invalid_fraction": raw_invalid / len(records),
        "raw_positive_count": raw_positive,
        "raw_positive_fraction": raw_positive / len(records),
        "deduplicated_invalid_count": sum(not row["valid"] for row in examples),
        "deduplicated_positive_count": sum(row["label"] for row in examples),
        "deduplicated_positive_fraction": (
            sum(row["label"] for row in examples) / len(examples)
            if examples else None
        ),
        "product_count": len(raw_products),
        "products_with_positive_candidate": products_with_positive,
        "source_group_count": len(raw_source_groups),
    }
    return examples, summary


def _split_product_ids(examples: Sequence[Mapping[str, Any]], split_end: int) -> tuple[list[int], list[int]]:
    product_ids = sorted({int(row["product_index"]) for row in examples})
    train_ids = [index for index in product_ids if index < split_end]
    validation_ids = [index for index in product_ids if index >= split_end]
    if not train_ids or not validation_ids:
        raise ValueError("internal reaction split must contain both sides")
    return train_ids, validation_ids


def _normalization(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


class CorrectnessReward(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


def _prepare_tensors(
    examples: Sequence[Mapping[str, Any]],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor([row["features"] for row in examples], dtype=torch.float32)
    labels = torch.tensor([float(row["label"]) for row in examples], dtype=torch.float32)
    return (features - mean) / std, labels


def _group_indices(examples: Sequence[Mapping[str, Any]]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(examples):
        groups[int(row["product_index"])].append(index)
    return dict(groups)


def _train(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    records, source_metadata = _load_records(args.train_data)
    token2id, _ = load_vocab(args.vocab_file)
    id2token = {value: key for key, value in token2id.items()}
    targets = _read_original_targets(args.targets_file, args.augmentation)
    canonical_targets = _canonical_target_cache(targets)
    examples, data_summary = _deduplicate_records(
        records, id2token=id2token, canonical_targets=canonical_targets,
        vocab_size=len(token2id),
    )
    if any(int(row["product_index"]) >= args.train_product_end for row in examples):
        raise ValueError("training guidance file contains products outside the declared train split")
    train_ids, validation_ids = _split_product_ids(examples, args.internal_validation_start)
    train_set = [row for row in examples if int(row["product_index"]) in set(train_ids) and row["valid"]]
    validation_set = [row for row in examples if int(row["product_index"]) in set(validation_ids) and row["valid"]]
    if not train_set or not validation_set:
        raise ValueError("valid training and internal-validation examples are required")
    all_train_features = torch.tensor([row["features"] for row in train_set], dtype=torch.float32)
    mean, std = _normalization(all_train_features)
    train_x, train_y = _prepare_tensors(train_set, mean, std)
    validation_x, validation_y = _prepare_tensors(validation_set, mean, std)
    feature_dim = int(train_x.shape[1])
    model = CorrectnessReward(feature_dim)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    by_product = _group_indices(train_set)
    product_ids = sorted(by_product)
    generator = torch.Generator().manual_seed(args.seed)
    history: list[dict[str, float]] = []
    for step in range(1, args.steps + 1):
        chosen_products = [
            product_ids[int(torch.randint(len(product_ids), (1,), generator=generator))]
            for _ in range(args.batch_size)
        ]
        selected_indices: list[int] = []
        for product_id in chosen_products:
            candidates = by_product[product_id]
            positives = [index for index in candidates if train_set[index]["label"]]
            negatives = [index for index in candidates if not train_set[index]["label"]]
            if positives and negatives:
                wanted = positives if bool(torch.randint(2, (1,), generator=generator).item()) else negatives
            else:
                wanted = positives or negatives
            selected_indices.append(wanted[int(torch.randint(len(wanted), (1,), generator=generator))])
        batch_x = train_x[selected_indices]
        batch_y = train_y[selected_indices]
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()
        if step == 1 or step == args.steps or step % max(args.steps // 10, 1) == 0:
            with torch.no_grad():
                train_loss = float(loss_fn(model(train_x), train_y).item())
                val_loss = float(loss_fn(model(validation_x), validation_y).item())
            history.append({"step": float(step), "train_loss": train_loss, "validation_loss": val_loss})

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "model_type": "linear_logistic_correctness_reward",
        "model_state_dict": model.state_dict(),
        "feature_dim": feature_dim,
        "feature_description": (
            "raw_forward_reward, validity, log lengths, length ratio, component counts, "
            "normalized product token histogram, normalized candidate token histogram, difference"
        ),
        "feature_mean": mean,
        "feature_std": std,
        "config": {
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "augmentation": args.augmentation,
            "train_product_end": args.train_product_end,
            "internal_validation_start": args.internal_validation_start,
            "vocab_size": len(token2id),
            "invalid_policy": "excluded_from_training; fixed lowest score at evaluation",
        },
        "provenance": {
            "train_data": str(Path(args.train_data).resolve()),
            "train_data_sha256": _sha256(args.train_data),
            "targets_file": str(Path(args.targets_file).resolve()),
            "targets_file_sha256": _sha256(args.targets_file),
            "vocab_file": str(Path(args.vocab_file).resolve()),
            "vocab_file_sha256": _sha256(args.vocab_file),
            "source_metadata": source_metadata,
        },
    }
    torch.save(checkpoint, output / "reward_model.pt")
    with (output / "train_dataset.json").open("w") as handle:
        json.dump({"schema_version": SCHEMA_VERSION, "feature_version": FEATURE_VERSION,
                   "data_summary": data_summary, "train_examples": len(train_set),
                   "validation_examples": len(validation_set),
                   "train_product_ids": train_ids, "validation_product_ids": validation_ids},
                  handle, indent=2, sort_keys=True)
    with (output / "train_history.json").open("w") as handle:
        json.dump(history, handle, indent=2, sort_keys=True)
    print(json.dumps({"mode": "train", "output_dir": str(output),
                      "data_summary": data_summary, "train_examples": len(train_set),
                      "validation_examples": len(validation_set), "history": history},
                     indent=2, sort_keys=True))
    return 0


def _auc_from_pairs(scores: Sequence[float], labels: Sequence[bool]) -> tuple[float | None, int, int, int]:
    buckets: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for score, label in zip(scores, labels):
        buckets[float(score)][0 if label else 1] += 1
    positives = sum(value[0] for value in buckets.values())
    negatives = sum(value[1] for value in buckets.values())
    if not positives or not negatives:
        return None, 0, 0, 0
    below = 0
    wins = 0
    ties = 0
    for score in sorted(buckets):
        pos, neg = buckets[score]
        wins += pos * below
        ties += pos * neg
        below += neg
    pairs = positives * negatives
    return (wins + 0.5 * ties) / pairs, wins, ties, pairs


def _group_auc(
    examples: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    group_key: str,
    valid_only: bool = True,
) -> tuple[float | None, int]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, row in enumerate(examples):
        if valid_only and not row["valid"]:
            continue
        groups[row[group_key]].append(index)
    wins = 0.0
    pairs = 0
    for indices in groups.values():
        auc, group_wins, ties, group_pairs = _auc_from_pairs(
            [scores[index] for index in indices],
            [bool(examples[index]["label"]) for index in indices],
        )
        if auc is None:
            continue
        wins += group_wins + 0.5 * ties
        pairs += group_pairs
    return (wins / pairs if pairs else None), pairs


def _bootstrap_auc_delta(
    examples: Sequence[Mapping[str, Any]],
    model_scores: Sequence[float],
    raw_scores: Sequence[float],
    *,
    group_key: str | None,
    seed: int,
    samples: int,
) -> list[float] | None:
    products = sorted({int(row["product_index"]) for row in examples})
    by_product: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(examples):
        by_product[int(row["product_index"])].append(index)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        chosen = [products[rng.randrange(len(products))] for _ in products]
        selected = [index for product in chosen for index in by_product[product]]
        subset = [examples[index] for index in selected]
        model_subset = [model_scores[index] for index in selected]
        raw_subset = [raw_scores[index] for index in selected]
        if group_key is None:
            valid_subset = [index for index, row in enumerate(subset) if row["valid"]]
            model_auc, _, _, _ = _auc_from_pairs(
                [model_subset[index] for index in valid_subset],
                [bool(subset[index]["label"]) for index in valid_subset],
            )
            raw_auc, _, _, _ = _auc_from_pairs(
                [raw_subset[index] for index in valid_subset],
                [bool(subset[index]["label"]) for index in valid_subset],
            )
        else:
            model_auc, _ = _group_auc(subset, model_subset, group_key=group_key)
            raw_auc, _ = _group_auc(subset, raw_subset, group_key=group_key)
        if model_auc is not None and raw_auc is not None:
            deltas.append(100.0 * (model_auc - raw_auc))
    if not deltas:
        return None
    deltas.sort()
    return [
        deltas[int(0.025 * (len(deltas) - 1))],
        deltas[int(0.975 * (len(deltas) - 1))],
    ]


def _rank_metrics(
    examples: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    order: str,
    top_ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    by_product: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(examples):
        by_product[int(row["product_index"])].append(index)
    hits = {k: [] for k in top_ks}
    oracle: list[int] = []
    valid_counts: list[int] = []
    total_counts: list[int] = []
    invalid_counts: list[int] = []
    for product_index in sorted(by_product):
        indices = by_product[product_index]
        total_counts.append(len(indices))
        invalid_counts.append(sum(not examples[index]["valid"] for index in indices))
        valid = [index for index in indices if examples[index]["valid"]]
        valid_counts.append(len(valid))
        if order == "first_seen":
            ranked = valid
        elif order in {"raw_forward_reward", "correctness_reward"}:
            ranked = sorted(
                valid,
                key=lambda index: (-float(scores[index]), int(examples[index]["record_index"])),
            )
        else:
            raise ValueError(f"unknown ranking order {order}")
        labels = [bool(examples[index]["label"]) for index in ranked]
        oracle.append(int(any(labels)))
        for k in top_ks:
            hits[k].append(int(any(labels[:k])))
    result = {
        "reaction_count": len(by_product),
        "oracle_percent": 100.0 * sum(oracle) / len(oracle),
        "mean_total_candidates": sum(total_counts) / len(total_counts),
        "mean_valid_candidates": sum(valid_counts) / len(valid_counts),
        "mean_invalid_candidates": sum(invalid_counts) / len(invalid_counts),
        "invalid_candidate_percent": 100.0 * sum(invalid_counts) / sum(total_counts),
        "top_k": {
            str(k): {"percent": 100.0 * sum(values) / len(values),
                    "count": sum(values)}
            for k, values in hits.items()
        },
        "per_reaction": {
            str(product): {"oracle": oracle[i], **{str(k): hits[k][i] for k in top_ks}}
            for i, product in enumerate(sorted(by_product))
        },
        "order_definition": order,
    }
    return result


def _evaluate(args: argparse.Namespace) -> int:
    checkpoint = torch.load(args.reward_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("feature_version") != FEATURE_VERSION:
        raise ValueError("reward checkpoint feature version is incompatible")
    records, source_metadata = _load_records(args.holdout_data)
    token2id, _ = load_vocab(args.vocab_file)
    id2token = {value: key for key, value in token2id.items()}
    targets = _read_original_targets(args.targets_file, args.augmentation)
    canonical_targets = _canonical_target_cache(targets)
    examples, data_summary = _deduplicate_records(
        records, id2token=id2token, canonical_targets=canonical_targets,
        vocab_size=len(token2id),
    )
    if not examples:
        raise ValueError("holdout has no deduplicated candidates")
    feature_mean = checkpoint["feature_mean"].float()
    feature_std = checkpoint["feature_std"].float()
    model = CorrectnessReward(int(checkpoint["feature_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    features, labels = _prepare_tensors(examples, feature_mean, feature_std)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(features)).tolist()
    raw_scores = [float(row["raw_forward_reward"]) for row in examples]
    model_scores = [
        float(probability) if row["valid"] else INVALID_SCORE
        for probability, row in zip(probabilities, examples)
    ]
    valid_indices = [index for index, row in enumerate(examples) if row["valid"]]
    valid_labels = [bool(examples[index]["label"]) for index in valid_indices]
    valid_raw = [raw_scores[index] for index in valid_indices]
    valid_model = [model_scores[index] for index in valid_indices]
    raw_auc, _, _, _ = _auc_from_pairs(valid_raw, valid_labels)
    model_auc, _, _, _ = _auc_from_pairs(valid_model, valid_labels)
    raw_group_auc, raw_group_pairs = _group_auc(examples, raw_scores, group_key="source_index")
    model_group_auc, model_group_pairs = _group_auc(examples, model_scores, group_key="source_index")
    report = {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "mode": "single_frozen_holdout_report",
        "label_definition": "canonical_terminal_equals_dataset_target",
        "invalid_policy": "fixed lowest score; excluded from valid-only AUC",
        "holdout_data": str(Path(args.holdout_data).resolve()),
        "holdout_data_sha256": _sha256(args.holdout_data),
        "reward_checkpoint": str(Path(args.reward_checkpoint).resolve()),
        "reward_checkpoint_sha256": _sha256(args.reward_checkpoint),
        "targets_file": str(Path(args.targets_file).resolve()),
        "vocab_file": str(Path(args.vocab_file).resolve()),
        "source_metadata": source_metadata,
        "data_summary": data_summary,
        "global_valid_auc": {"raw_forward_reward": raw_auc, "correctness_reward": model_auc,
                             "delta": (model_auc - raw_auc if model_auc is not None and raw_auc is not None else None)},
        "shared_anchor_within_group_auc": {
            "group_key": "source_index",
            "raw_forward_reward": raw_group_auc,
            "correctness_reward": model_group_auc,
            "delta": (model_group_auc - raw_group_auc
                      if model_group_auc is not None and raw_group_auc is not None else None),
            "raw_pair_count": raw_group_pairs,
            "correctness_pair_count": model_group_pairs,
        },
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "global_auc_delta_percentage_points_95ci": _bootstrap_auc_delta(
                examples, model_scores, raw_scores, group_key=None,
                seed=args.bootstrap_seed, samples=args.bootstrap_samples,
            ),
            "within_anchor_auc_delta_percentage_points_95ci": _bootstrap_auc_delta(
                examples, model_scores, raw_scores, group_key="source_index",
                seed=args.bootstrap_seed + 1, samples=args.bootstrap_samples,
            ),
        },
        "endpoint_rerank": {
            "first_seen_order": _rank_metrics(examples, raw_scores, order="first_seen"),
            "raw_forward_reward": _rank_metrics(examples, raw_scores, order="raw_forward_reward"),
            "correctness_reward": _rank_metrics(examples, model_scores, order="correctness_reward"),
        },
        "score_audit": {
            "valid_candidate_count": len(valid_indices),
            "invalid_candidate_count": len(examples) - len(valid_indices),
            "invalid_high_score_count": sum(
                (not row["valid"]) and score > INVALID_SCORE
                for row, score in zip(examples, model_scores)
            ),
            "correct_mean_score": (
                sum(model_scores[index] for index in valid_indices if labels[index]) /
                max(sum(bool(labels[index]) for index in valid_indices), 1)
            ),
            "incorrect_mean_score": (
                sum(model_scores[index] for index in valid_indices if not labels[index]) /
                max(sum(not bool(labels[index]) for index in valid_indices), 1)
            ),
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "holdout_report.json").open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    score_rows = []
    for row, score, raw in zip(examples, model_scores, raw_scores):
        score_rows.append({**{key: row[key] for key in (
            "record_index", "product_index", "source_index", "sample_index", "time_index",
            "anchor_ordinal", "candidate_canonical", "label", "valid", "forward_beam_rank",
        )}, "raw_forward_reward": raw, "correctness_score": score})
    with (output / "holdout_scores.json").open("w") as handle:
        json.dump(score_rows, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate"), required=True)
    parser.add_argument("--train_data", type=Path)
    parser.add_argument("--holdout_data", type=Path)
    parser.add_argument("--reward_checkpoint", type=Path)
    parser.add_argument("--targets_file", type=Path, required=True)
    parser.add_argument("--vocab_file", type=Path, required=True)
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_product_end", type=int, default=1000)
    parser.add_argument("--internal_validation_start", type=int, default=800)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260812)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.mode == "train":
        if args.train_data is None:
            raise SystemExit("--train_data is required in train mode")
        return _train(args)
    if args.holdout_data is None or args.reward_checkpoint is None:
        raise SystemExit("--holdout_data and --reward_checkpoint are required in evaluate mode")
    return _evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
