#!/usr/bin/env python3
"""Audit whether an offline guidance reward separates correct terminals.

This is a read-only diagnostic.  It uses dataset targets only *after* a
guidance-data file has been generated, in order to quantify how well a stored
reward correlates with exact retrosynthesis correctness.  It never writes
labels back into the guidance data and must not be used on test targets to
select a reward, guidance checkpoint, or sampling hyperparameter.

The most relevant statistic for shared-anchor guidance is the within-group
pairwise AUC: when terminals start from the same product, intermediate state,
and time, does a higher stored reward tend to belong to the terminal matching
the dataset reactants?
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rdkit import Chem, RDLogger

from edit_flows.data.dataset import load_vocab
from edit_flows.forward.molecular_transformer import retro_global_to_smiles
from edit_flows.guidance.data import load_guidance_dataset
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


RDLogger.DisableLog("rdApp.*")


def _canonical_global(value: str) -> str:
    """Canonicalize one tokenized/global-aligned reaction side or return ``""``."""

    try:
        molecule = Chem.MolFromSmiles(retro_global_to_smiles(value))
    except (TypeError, ValueError):
        return ""
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def _read_original_targets(path: str | Path, augmentation: int) -> list[str]:
    if augmentation < 1:
        raise ValueError("augmentation must be positive")
    lines = Path(path).read_text().splitlines()
    if len(lines) % augmentation:
        raise ValueError(
            f"{path} has {len(lines)} lines, not divisible by augmentation={augmentation}"
        )
    return [line.strip() for line in lines[::augmentation]]


def _decode_terminal(record: Mapping[str, Any], id2token: Mapping[int, str]) -> str:
    tokens: list[str] = []
    for raw_token in record["terminal_tokens"]:
        token = int(raw_token)
        if token == PAD_TOKEN:
            break
        if token == BOS_TOKEN:
            continue
        if token not in id2token:
            raise ValueError(f"terminal token id {token} is absent from the vocabulary")
        tokens.append(id2token[token])
    return " ".join(tokens)


def _score_from_record(record: Mapping[str, Any], score_field: str) -> float:
    """Return a larger-is-better score without mutating the stored record."""

    if score_field == "forward_beam_rank":
        rank = int(record.get(score_field, 0))
        if rank < 0:
            raise ValueError("forward_beam_rank must be non-negative")
        return 1.0 / rank if rank else 0.0
    if score_field not in record:
        raise KeyError(f"record lacks score field: {score_field}")
    score = float(record[score_field])
    if not math.isfinite(score):
        raise ValueError(f"score field {score_field} contains a non-finite value")
    return score


def pairwise_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Exact AUC with half credit for ties, computed without quadratic tensors."""

    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    buckets: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for score, label in zip(scores, labels):
        if not math.isfinite(float(score)):
            raise ValueError("scores must be finite")
        buckets[float(score)][0 if label else 1] += 1
    positive_count = sum(counts[0] for counts in buckets.values())
    negative_count = sum(counts[1] for counts in buckets.values())
    if not positive_count or not negative_count:
        return None
    negative_below = 0
    wins = 0
    ties = 0
    for score in sorted(buckets):
        positives, negatives = buckets[score]
        wins += positives * negative_below
        ties += positives * negatives
        negative_below += negatives
    return (wins + 0.5 * ties) / (positive_count * negative_count)


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _score_summary(scores: Sequence[float], labels: Sequence[bool]) -> dict[str, Any]:
    positive_scores = [score for score, label in zip(scores, labels) if label]
    negative_scores = [score for score, label in zip(scores, labels) if not label]
    histogram: dict[str, dict[str, int]] = {}
    for score, label in zip(scores, labels):
        key = f"{score:.12g}"
        if key not in histogram:
            histogram[key] = {"correct": 0, "incorrect": 0}
        histogram[key]["correct" if label else "incorrect"] += 1
    return {
        "correctness_auc": pairwise_auc(scores, labels),
        "correct_mean_score": _mean(positive_scores),
        "incorrect_mean_score": _mean(negative_scores),
        "correct_positive_score_fraction": (
            sum(score > 0 for score in positive_scores) / len(positive_scores)
            if positive_scores else None
        ),
        "incorrect_positive_score_fraction": (
            sum(score > 0 for score in negative_scores) / len(negative_scores)
            if negative_scores else None
        ),
        "value_counts": {
            key: histogram[key] for key in sorted(histogram, key=float)
        },
    }


def summarize_reward_correctness(
    records: Sequence[Mapping[str, Any]],
    targets: Sequence[str],
    id2token: Mapping[int, str],
    *,
    target_start_product: int = 0,
    score_field: str = "forward_beam_rank",
) -> dict[str, Any]:
    """Compare stored reward scores to canonical target equality.

    ``product_index`` is preferred because multi-time shared-anchor records
    use ``source_index`` for a ``(product, anchor)`` pair.  Older records that
    omit ``product_index`` retain the historical one-product-per-source-index
    interpretation.
    """

    if target_start_product < 0:
        raise ValueError("target_start_product must be non-negative")
    canonical_targets = [_canonical_global(value) for value in targets]
    target_invalid_count = sum(not value for value in canonical_targets)
    scores: list[float] = []
    labels: list[bool] = []
    time_groups: dict[int, list[int]] = defaultdict(list)
    source_groups: dict[int, list[int]] = defaultdict(list)
    invalid_terminal_count = 0

    for record_index, record in enumerate(records):
        product_index = int(record.get("product_index", record["source_index"]))
        target_index = target_start_product + product_index
        if target_index >= len(canonical_targets):
            raise ValueError(
                f"record {record_index} maps to target {target_index}, but only "
                f"{len(canonical_targets)} original targets are available"
            )
        terminal = _canonical_global(_decode_terminal(record, id2token))
        invalid_terminal_count += int(not terminal)
        label = bool(terminal and terminal == canonical_targets[target_index])
        scores.append(_score_from_record(record, score_field))
        labels.append(label)
        time_groups[int(record.get("time_index", -1))].append(record_index)
        source_groups[int(record["source_index"])].append(record_index)

    correct_count = sum(labels)
    grouped_wins = 0.0
    grouped_weight = 0
    positive_group_count = 0
    mixed_group_count = 0
    for indices in source_groups.values():
        group_labels = [labels[index] for index in indices]
        positive_group_count += int(any(group_labels))
        mixed_group_count += int(any(group_labels) and not all(group_labels))
        positive_count = sum(group_labels)
        negative_count = len(group_labels) - positive_count
        if positive_count and negative_count:
            auc = pairwise_auc(
                [scores[index] for index in indices], group_labels,
            )
            assert auc is not None
            weight = positive_count * negative_count
            grouped_wins += auc * weight
            grouped_weight += weight

    by_time_index = {}
    for time_index, indices in sorted(time_groups.items()):
        local_scores = [scores[index] for index in indices]
        local_labels = [labels[index] for index in indices]
        by_time_index[str(time_index)] = {
            "record_count": len(indices),
            "correct_candidate_count": sum(local_labels),
            "correct_candidate_fraction": sum(local_labels) / len(local_labels),
            **_score_summary(local_scores, local_labels),
        }

    return {
        "record_count": len(records),
        "target_count": len(targets),
        "target_start_product": target_start_product,
        "score_field": score_field,
        "label_definition": "canonical_terminal_equals_dataset_target",
        "invalid_target_count": target_invalid_count,
        "invalid_terminal_count": invalid_terminal_count,
        "correct_candidate_count": correct_count,
        "incorrect_candidate_count": len(records) - correct_count,
        "correct_candidate_fraction": correct_count / len(records) if records else None,
        "score": _score_summary(scores, labels),
        "shared_anchor_groups": {
            "group_count": len(source_groups),
            "groups_with_any_correct_terminal": positive_group_count,
            "groups_with_any_correct_terminal_fraction": (
                positive_group_count / len(source_groups) if source_groups else None
            ),
            "groups_with_mixed_correctness": mixed_group_count,
            "groups_with_mixed_correctness_fraction": (
                mixed_group_count / len(source_groups) if source_groups else None
            ),
            "within_group_correctness_auc": (
                grouped_wins / grouped_weight if grouped_weight else None
            ),
            "within_group_correct_vs_incorrect_pair_count": grouped_weight,
        },
        "by_time_index": by_time_index,
    }


def audit(
    data_path: str | Path,
    targets_file: str | Path,
    vocab_file: str | Path,
    *,
    augmentation: int,
    target_start_product: int,
    score_field: str,
) -> dict[str, Any]:
    records, metadata = load_guidance_dataset(data_path)
    token2id, _ = load_vocab(vocab_file)
    id2token = {index: token for token, index in token2id.items()}
    summary = summarize_reward_correctness(
        records,
        _read_original_targets(targets_file, augmentation),
        id2token,
        target_start_product=target_start_product,
        score_field=score_field,
    )
    summary.update({
        "schema_version": 1,
        "data_path": str(Path(data_path).resolve()),
        "targets_file": str(Path(targets_file).resolve()),
        "vocab_file": str(Path(vocab_file).resolve()),
        "augmentation": augmentation,
        "guidance_metadata": metadata,
    })
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--targets_file", required=True)
    parser.add_argument("--vocab_file", required=True)
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--target_start_product", type=int, default=0)
    parser.add_argument("--score_field", default="forward_beam_rank")
    parser.add_argument("--output_json", default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary = audit(
        args.data,
        args.targets_file,
        args.vocab_file,
        augmentation=args.augmentation,
        target_start_product=args.target_start_product,
        score_field=args.score_field,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n")
    return summary


if __name__ == "__main__":
    run(build_parser().parse_args())
