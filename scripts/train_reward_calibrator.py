#!/usr/bin/env python3
"""Fit one leakage-controlled linear reward-calibration pilot.

The frozen forward-beam reward is useful but imperfect: an incorrect
retrosynthesis candidate can still reconstruct the input product.  This tool
fits a small logistic calibrator using *only* features that exist after a
candidate and its forward-beam score have been produced.  Dataset targets are
read solely to create correctness labels for the calibration-training split
and the isolated holdout evaluation; no per-record label is written into the
saved calibrated guidance data.

The default P1 feature set and split discipline are documented in
``new_docs/dgm_reward_quality_protocol.md``.  It deliberately does not alter
the stored ``reward`` field, so a successful pilot must be explicitly reviewed
before any guidance training can consume a calibrated reward.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F
from rdkit import Chem, RDLogger

from edit_flows.data.dataset import load_vocab
from edit_flows.guidance.data import load_guidance_dataset, save_guidance_dataset
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN
try:  # ``python scripts/...`` and import-based unit tests need both paths.
    from scripts.audit_guidance_reward_quality import (
        _canonical_global,
        _read_original_targets,
        pairwise_auc,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI invocation.
    from audit_guidance_reward_quality import (
        _canonical_global,
        _read_original_targets,
        pairwise_auc,
    )


RDLogger.DisableLog("rdApp.*")


# P1 is deliberately small and frozen before inspecting the isolated holdout.
FEATURE_NAMES = (
    "forward_beam_reciprocal_rank",
    "forward_beam_hit",
    "terminal_rdkit_valid",
    "relative_token_length_difference",
    "relative_atom_count_difference",
    "terminal_fragment_count",
    "euler_time",
)


@dataclass
class CandidateTable:
    """Features plus offline labels/provenance used only during calibration."""

    features: torch.Tensor
    labels: torch.Tensor
    product_indices: list[int]
    source_indices: list[int]
    canonical_terminals: list[str]
    canonical_targets: dict[int, str]

    def __post_init__(self) -> None:
        count = self.features.shape[0]
        if self.features.ndim != 2 or self.features.shape[1] != len(FEATURE_NAMES):
            raise ValueError("candidate features have an unexpected shape")
        if self.labels.shape != (count,):
            raise ValueError("candidate labels have an unexpected shape")
        if not all(len(values) == count for values in (
            self.product_indices,
            self.source_indices,
            self.canonical_terminals,
        )):
            raise ValueError("candidate provenance has an unexpected length")


def _decode_tokens(
    record: Mapping[str, Any],
    field: str,
    id2token: Mapping[int, str],
) -> tuple[str, int]:
    """Decode one saved Edit Flows sequence and return text plus token count."""

    if field not in record:
        raise KeyError(f"record lacks token field: {field}")
    tokens: list[str] = []
    for raw_token in record[field]:
        token = int(raw_token)
        if token == PAD_TOKEN:
            break
        if token == BOS_TOKEN:
            continue
        if token not in id2token:
            raise ValueError(f"token id {token} is absent from the vocabulary")
        tokens.append(id2token[token])
    return " ".join(tokens), len(tokens)


def _molecule_properties(canonical_smiles: str) -> tuple[int, int]:
    """Return atom and dot-fragment counts, or zeros for an invalid candidate."""

    if not canonical_smiles:
        return 0, 0
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        return 0, 0
    return molecule.GetNumAtoms(), len(Chem.GetMolFrags(molecule))


def _record_product_index(record: Mapping[str, Any]) -> int:
    """Respect multi-anchor provenance while retaining old single-anchor data."""

    try:
        return int(record.get("product_index", record["source_index"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("record lacks a valid product/source index") from exc


def build_candidate_table(
    records: Sequence[Mapping[str, Any]],
    targets: Sequence[str],
    id2token: Mapping[int, str],
    *,
    target_start_product: int = 0,
) -> CandidateTable:
    """Create P1 features and exact labels without mutating saved records.

    ``targets`` are used only here for calibration labels.  The feature vector
    itself comes exclusively from saved product/terminal tokens, the stored
    forward-beam rank, and the saved Euler time.
    """

    if target_start_product < 0:
        raise ValueError("target_start_product must be non-negative")
    if not records:
        raise ValueError("cannot calibrate an empty record collection")
    canonical_targets = [_canonical_global(value) for value in targets]
    if any(not value for value in canonical_targets):
        raise ValueError("targets contain an invalid SMILES after canonicalization")

    feature_rows: list[list[float]] = []
    labels: list[bool] = []
    product_indices: list[int] = []
    source_indices: list[int] = []
    canonical_terminals: list[str] = []
    targets_by_product: dict[int, str] = {}

    for record_index, record in enumerate(records):
        product_index = _record_product_index(record)
        target_index = target_start_product + product_index
        if target_index >= len(canonical_targets):
            raise ValueError(
                f"record {record_index} maps to target {target_index}, but only "
                f"{len(canonical_targets)} targets are available"
            )
        try:
            rank = int(record["forward_beam_rank"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "P1 requires forward_beam_rank on every record"
            ) from exc
        if rank < 0:
            raise ValueError("forward_beam_rank must be non-negative")
        try:
            time_value = float(record["time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("record lacks a finite Euler time") from exc
        if not math.isfinite(time_value) or not 0.0 <= time_value <= 1.0:
            raise ValueError("record time must be finite and within [0, 1]")

        product_text, product_token_count = _decode_tokens(
            record, "product_tokens", id2token,
        )
        terminal_text, terminal_token_count = _decode_tokens(
            record, "terminal_tokens", id2token,
        )
        canonical_product = _canonical_global(product_text)
        canonical_terminal = _canonical_global(terminal_text)
        product_atoms, _ = _molecule_properties(canonical_product)
        terminal_atoms, terminal_fragments = _molecule_properties(canonical_terminal)
        target = canonical_targets[target_index]
        previous_target = targets_by_product.setdefault(product_index, target)
        if previous_target != target:
            raise ValueError("one product index maps to inconsistent targets")

        feature_rows.append([
            1.0 / rank if rank else 0.0,
            float(rank > 0),
            float(bool(canonical_terminal)),
            abs(terminal_token_count - product_token_count)
            / max(product_token_count, 1),
            abs(terminal_atoms - product_atoms) / max(product_atoms, 1),
            float(terminal_fragments),
            time_value,
        ])
        labels.append(bool(canonical_terminal and canonical_terminal == target))
        product_indices.append(product_index)
        try:
            source_indices.append(int(record["source_index"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("record lacks a valid source_index") from exc
        canonical_terminals.append(canonical_terminal)

    return CandidateTable(
        features=torch.tensor(feature_rows, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.bool),
        product_indices=product_indices,
        source_indices=source_indices,
        canonical_terminals=canonical_terminals,
        canonical_targets=targets_by_product,
    )


def fit_logistic_calibrator(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    l2: float,
    max_steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Fit a standardized, L2-regularized logistic regression on CPU."""

    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("features must have shape [N, len(FEATURE_NAMES)]")
    if labels.shape != (features.shape[0],):
        raise ValueError("labels must have shape [N]")
    if features.shape[0] < 2:
        raise ValueError("at least two records are required")
    if not torch.isfinite(features).all():
        raise ValueError("features must be finite")
    if not labels.any() or labels.all():
        raise ValueError("both correctness classes are required")
    if l2 < 0 or max_steps < 1 or learning_rate <= 0:
        raise ValueError("l2/max_steps/learning_rate are invalid")

    x = features.detach().to(dtype=torch.float32, device="cpu")
    y = labels.detach().to(dtype=torch.float32, device="cpu")
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    normalized = (x - mean) / std

    prevalence = float(y.mean())
    prevalence = min(max(prevalence, 1e-4), 1.0 - 1e-4)
    weight = torch.zeros(normalized.shape[1], requires_grad=True)
    bias = torch.tensor(
        math.log(prevalence / (1.0 - prevalence)), requires_grad=True,
    )
    optimizer = torch.optim.Adam((weight, bias), lr=learning_rate)
    history: list[dict[str, float | int]] = []
    checkpoints = {1, max_steps}
    checkpoints.update(range(100, max_steps + 1, 100))
    for step in range(1, max_steps + 1):
        logits = normalized @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logits, y)
        penalty = 0.5 * float(l2) * weight.square().sum()
        objective = loss + penalty
        if not torch.isfinite(objective):
            raise RuntimeError("non-finite calibration loss")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        if step in checkpoints:
            history.append({
                "step": step,
                "binary_cross_entropy": float(loss.detach()),
                "objective": float(objective.detach()),
            })

    return {
        "schema_version": 1,
        "method": "p1_structural_linear_logistic",
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean.detach().cpu(),
        "feature_std": std.detach().cpu(),
        "weight": weight.detach().cpu(),
        "bias": float(bias.detach()),
        "l2": float(l2),
        "max_steps": int(max_steps),
        "learning_rate": float(learning_rate),
        "training_record_count": int(features.shape[0]),
        "training_positive_fraction": float(labels.float().mean()),
        "loss_history": history,
    }


def predict_probability(features: torch.Tensor, state: Mapping[str, Any]) -> torch.Tensor:
    """Return calibrated correctness probabilities using a saved state."""

    if list(state.get("feature_names", ())) != list(FEATURE_NAMES):
        raise ValueError("calibrator feature schema does not match P1")
    mean = torch.as_tensor(state["feature_mean"], dtype=torch.float32)
    std = torch.as_tensor(state["feature_std"], dtype=torch.float32)
    weight = torch.as_tensor(state["weight"], dtype=torch.float32)
    bias = float(state["bias"])
    if mean.shape != weight.shape or std.shape != weight.shape:
        raise ValueError("calibrator parameter shapes are inconsistent")
    if features.ndim != 2 or features.shape[1] != weight.shape[0]:
        raise ValueError("features do not match calibrator dimensionality")
    if (std <= 0).any():
        raise ValueError("calibrator feature standard deviations must be positive")
    logits = (features.to(dtype=torch.float32, device="cpu") - mean) / std
    logits = logits @ weight + bias
    return torch.sigmoid(logits)


def _within_group_auc(
    scores: Sequence[float],
    labels: Sequence[bool],
    source_indices: Sequence[int],
) -> tuple[float | None, int]:
    """Weighted AUC over only groups that contain both correctness classes."""

    groups: dict[int, list[int]] = defaultdict(list)
    for index, source_index in enumerate(source_indices):
        groups[int(source_index)].append(index)
    wins = 0.0
    weight = 0
    for indices in groups.values():
        group_labels = [bool(labels[index]) for index in indices]
        positives = sum(group_labels)
        negatives = len(group_labels) - positives
        if not positives or not negatives:
            continue
        auc = pairwise_auc([scores[index] for index in indices], group_labels)
        assert auc is not None
        pair_count = positives * negatives
        wins += auc * pair_count
        weight += pair_count
    return (wins / weight if weight else None), weight


def score_quality_summary(table: CandidateTable, scores: torch.Tensor) -> dict[str, Any]:
    """Summarize global and shared-state correctness discrimination."""

    values = [float(value) for value in scores.tolist()]
    labels = [bool(value) for value in table.labels.tolist()]
    within_auc, pair_count = _within_group_auc(
        values, labels, table.source_indices,
    )
    return {
        "global_correctness_auc": pairwise_auc(values, labels),
        "within_group_correctness_auc": within_auc,
        "within_group_correct_vs_incorrect_pair_count": pair_count,
        "correct_mean_score": _mean(
            value for value, label in zip(values, labels) if label
        ),
        "incorrect_mean_score": _mean(
            value for value, label in zip(values, labels) if not label
        ),
    }


def _mean(values: Sequence[float] | Any) -> float | None:
    values = list(values)
    return sum(float(value) for value in values) / len(values) if values else None


def choose_threshold_at_reference_rate(
    scores: torch.Tensor,
    reference_fraction: float,
) -> tuple[float, float]:
    """Freeze a conservative score threshold near a training-set selection rate."""

    if not 0.0 <= reference_fraction <= 1.0:
        raise ValueError("reference_fraction must lie in [0, 1]")
    values = [float(value) for value in scores.tolist()]
    if not values:
        raise ValueError("cannot choose a threshold from no scores")
    options = [math.inf, *sorted(set(values), reverse=True)]
    candidates = []
    for threshold in options:
        selected_fraction = sum(value >= threshold for value in values) / len(values)
        # On an equally close match, choose the higher threshold.  This avoids
        # silently selecting more candidates just because a score is tied.
        candidates.append((
            abs(selected_fraction - reference_fraction),
            -threshold,
            threshold,
            selected_fraction,
        ))
    _, _, threshold, selected_fraction = min(candidates)
    return float(threshold), float(selected_fraction)


def selection_summary(
    labels: torch.Tensor,
    selected: torch.Tensor,
) -> dict[str, float | int | None]:
    """Report a fixed-operating-point selection quality without score semantics."""

    if labels.shape != selected.shape:
        raise ValueError("labels and selected masks must have equal shape")
    labels = labels.to(dtype=torch.bool)
    selected = selected.to(dtype=torch.bool)
    correct = labels.sum().item()
    incorrect = (~labels).sum().item()
    selected_count = selected.sum().item()
    selected_correct = (selected & labels).sum().item()
    selected_incorrect = (selected & ~labels).sum().item()
    return {
        "selected_count": int(selected_count),
        "selected_fraction": float(selected.float().mean()),
        "correct_selected_fraction": (
            selected_correct / correct if correct else None
        ),
        "incorrect_selected_fraction": (
            selected_incorrect / incorrect if incorrect else None
        ),
        "selected_precision": (
            selected_correct / selected_count if selected_count else None
        ),
    }


def rerank_summary(
    table: CandidateTable,
    scores: torch.Tensor,
    *,
    top_ks: Sequence[int] = (1, 3, 10),
) -> tuple[dict[str, Any], dict[int, dict[str, float]]]:
    """Rerank the same canonical terminal pool for each original product."""

    if any(k < 1 for k in top_ks):
        raise ValueError("top_ks must contain positive values")
    by_product: dict[int, list[int]] = defaultdict(list)
    for index, product_index in enumerate(table.product_indices):
        by_product[product_index].append(index)
    per_product: dict[int, dict[str, float]] = {}
    for product_index in sorted(by_product):
        best_by_terminal: dict[str, float] = {}
        for index in by_product[product_index]:
            terminal = table.canonical_terminals[index]
            if not terminal:
                continue
            score = float(scores[index])
            previous = best_by_terminal.get(terminal)
            if previous is None or score > previous:
                best_by_terminal[terminal] = score
        ranked = sorted(best_by_terminal.items(), key=lambda item: (-item[1], item[0]))
        target = table.canonical_targets[product_index]
        result = {
            f"top_{k}": float(any(terminal == target for terminal, _ in ranked[:k]))
            for k in top_ks
        }
        result["oracle"] = float(any(terminal == target for terminal, _ in ranked))
        result["valid_unique_candidates"] = float(len(ranked))
        per_product[product_index] = result

    summary = {
        "reaction_count": len(per_product),
        **{
            f"top_{k}": _mean(
                values[f"top_{k}"] for values in per_product.values()
            )
            for k in top_ks
        },
        "oracle": _mean(values["oracle"] for values in per_product.values()),
        "mean_valid_unique_candidates": _mean(
            values["valid_unique_candidates"] for values in per_product.values()
        ),
    }
    return summary, per_product


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _product_rows(table: CandidateTable) -> dict[int, list[int]]:
    rows: dict[int, list[int]] = defaultdict(list)
    for index, product_index in enumerate(table.product_indices):
        rows[product_index].append(index)
    return dict(rows)


def _product_within_components(
    table: CandidateTable,
    scores: torch.Tensor,
) -> dict[int, tuple[float, int]]:
    """Precompute numerator/denominator for block-bootstrap group AUC."""

    group_rows: dict[int, list[int]] = defaultdict(list)
    for index, source_index in enumerate(table.source_indices):
        group_rows[source_index].append(index)
    output: dict[int, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    labels = [bool(value) for value in table.labels.tolist()]
    values = [float(value) for value in scores.tolist()]
    for rows in group_rows.values():
        product_ids = {table.product_indices[index] for index in rows}
        if len(product_ids) != 1:
            raise ValueError("one shared-anchor group spans multiple products")
        product_index = next(iter(product_ids))
        group_labels = [labels[index] for index in rows]
        positives = sum(group_labels)
        negatives = len(group_labels) - positives
        numerator, denominator = output[product_index]
        if positives and negatives:
            auc = pairwise_auc([values[index] for index in rows], group_labels)
            assert auc is not None
            pair_count = positives * negatives
            numerator += auc * pair_count
            denominator += pair_count
        output[product_index] = (numerator, denominator)
    return dict(output)


def bootstrap_comparison(
    table: CandidateTable,
    baseline_scores: torch.Tensor,
    candidate_scores: torch.Tensor,
    baseline_selected: torch.Tensor,
    candidate_selected: torch.Tensor,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Block-bootstrap calibrated-minus-raw deltas by original reaction."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    product_rows = _product_rows(table)
    product_ids = sorted(product_rows)
    if len(product_ids) < 2:
        raise ValueError("at least two original reactions are required for bootstrap")
    raw_per_product = rerank_summary(table, baseline_scores)[1]
    calibrated_per_product = rerank_summary(table, candidate_scores)[1]
    raw_group_components = _product_within_components(table, baseline_scores)
    calibrated_group_components = _product_within_components(table, candidate_scores)
    labels = [bool(value) for value in table.labels.tolist()]
    raw_values = [float(value) for value in baseline_scores.tolist()]
    calibrated_values = [float(value) for value in candidate_scores.tolist()]
    selected_raw = [bool(value) for value in baseline_selected.tolist()]
    selected_calibrated = [bool(value) for value in candidate_selected.tolist()]
    rng = random.Random(seed)
    deltas: dict[str, list[float]] = defaultdict(list)

    for _ in range(bootstrap_samples):
        sampled_products = [rng.choice(product_ids) for _ in product_ids]
        indices = [
            index
            for product_index in sampled_products
            for index in product_rows[product_index]
        ]
        sampled_labels = [labels[index] for index in indices]
        raw_auc = pairwise_auc(
            [raw_values[index] for index in indices], sampled_labels,
        )
        calibrated_auc = pairwise_auc(
            [calibrated_values[index] for index in indices], sampled_labels,
        )
        if raw_auc is not None and calibrated_auc is not None:
            deltas["global_correctness_auc"].append(calibrated_auc - raw_auc)

        raw_numerator = raw_denominator = 0.0
        calibrated_numerator = calibrated_denominator = 0.0
        for product_index in sampled_products:
            numerator, denominator = raw_group_components[product_index]
            raw_numerator += numerator
            raw_denominator += denominator
            numerator, denominator = calibrated_group_components[product_index]
            calibrated_numerator += numerator
            calibrated_denominator += denominator
        if raw_denominator and calibrated_denominator:
            deltas["within_group_correctness_auc"].append(
                calibrated_numerator / calibrated_denominator
                - raw_numerator / raw_denominator
            )

        raw_selected_incorrect = raw_incorrect = 0
        calibrated_selected_incorrect = calibrated_incorrect = 0
        for index in indices:
            if not labels[index]:
                raw_incorrect += 1
                calibrated_incorrect += 1
                raw_selected_incorrect += int(selected_raw[index])
                calibrated_selected_incorrect += int(selected_calibrated[index])
        if raw_incorrect and calibrated_incorrect:
            deltas["incorrect_selected_fraction"].append(
                calibrated_selected_incorrect / calibrated_incorrect
                - raw_selected_incorrect / raw_incorrect
            )

        for key in ("top_1", "top_3", "top_10", "oracle"):
            deltas[key].append(_mean(
                calibrated_per_product[product_index][key]
                - raw_per_product[product_index][key]
                for product_index in sampled_products
            ))

    return {
        key: {
            "bootstrap_samples": len(values),
            "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
        }
        for key, values in sorted(deltas.items())
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_output_paths(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    paths = {
        "calibrator": output_dir / "calibrator.pt",
        "holdout_data": output_dir / "holdout_calibrated.pt",
        "summary": output_dir / "summary.json",
    }
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions and not overwrite:
        raise FileExistsError(
            "calibration output already exists; pass --overwrite to replace: "
            + ", ".join(collisions)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return paths


def _add_prediction_field(
    records: Sequence[Mapping[str, Any]],
    scores: torch.Tensor,
    field: str,
) -> list[dict[str, Any]]:
    if len(records) != scores.numel():
        raise ValueError("record count and prediction count differ")
    if field in {"reward", "validity_reward", "forward_beam_rank"}:
        raise ValueError("prediction field would overwrite an existing reward source")
    result = []
    for record, score in zip(records, scores.tolist()):
        updated = dict(record)
        updated[field] = float(score)
        result.append(updated)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Fit P1 on one training block and evaluate once on an isolated block."""

    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    output_paths = _prepare_output_paths(Path(args.output_dir), args.overwrite)
    token2id, _ = load_vocab(args.vocab_file)
    id2token = {index: token for token, index in token2id.items()}
    train_records, train_metadata = load_guidance_dataset(args.train_data)
    holdout_records, holdout_metadata = load_guidance_dataset(args.holdout_data)
    train_table = build_candidate_table(
        train_records,
        _read_original_targets(args.train_targets_file, args.augmentation),
        id2token,
        target_start_product=args.train_target_start_product,
    )
    holdout_table = build_candidate_table(
        holdout_records,
        _read_original_targets(args.holdout_targets_file, args.augmentation),
        id2token,
        target_start_product=args.holdout_target_start_product,
    )
    overlap = set(train_table.product_indices).intersection(holdout_table.product_indices)
    if overlap:
        preview = sorted(overlap)[:10]
        raise ValueError(
            "train and holdout share original product indices; first overlap="
            f"{preview}"
        )

    started = time.perf_counter()
    state = fit_logistic_calibrator(
        train_table.features,
        train_table.labels,
        l2=args.l2,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
    )
    train_scores = predict_probability(train_table.features, state)
    holdout_scores = predict_probability(holdout_table.features, state)
    raw_train_scores = train_table.features[:, 0]
    raw_holdout_scores = holdout_table.features[:, 0]
    raw_train_selected = raw_train_scores > 0
    threshold, train_selected_fraction = choose_threshold_at_reference_rate(
        train_scores, float(raw_train_selected.float().mean()),
    )
    calibrated_train_selected = train_scores >= threshold
    raw_holdout_selected = raw_holdout_scores > 0
    calibrated_holdout_selected = holdout_scores >= threshold

    train_quality = {
        "raw_forward_beam": score_quality_summary(train_table, raw_train_scores),
        "calibrated": score_quality_summary(train_table, train_scores),
    }
    holdout_quality = {
        "raw_forward_beam": score_quality_summary(
            holdout_table, raw_holdout_scores,
        ),
        "calibrated": score_quality_summary(holdout_table, holdout_scores),
    }
    train_selection = {
        "raw_forward_beam": selection_summary(
            train_table.labels, raw_train_selected,
        ),
        "calibrated": selection_summary(
            train_table.labels, calibrated_train_selected,
        ),
    }
    holdout_selection = {
        "raw_forward_beam": selection_summary(
            holdout_table.labels, raw_holdout_selected,
        ),
        "calibrated": selection_summary(
            holdout_table.labels, calibrated_holdout_selected,
        ),
    }
    raw_rerank, _ = rerank_summary(holdout_table, raw_holdout_scores)
    calibrated_rerank, _ = rerank_summary(holdout_table, holdout_scores)
    bootstrap = bootstrap_comparison(
        holdout_table,
        raw_holdout_scores,
        holdout_scores,
        raw_holdout_selected,
        calibrated_holdout_selected,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    wall_seconds = time.perf_counter() - started

    state.update({
        "prediction_field": args.prediction_field,
        "operating_threshold": threshold,
        "reference_raw_positive_fraction": float(raw_train_selected.float().mean()),
        "train_calibrated_selected_fraction": train_selected_fraction,
        "train_data_sha256": _sha256(args.train_data),
        "holdout_data_sha256": _sha256(args.holdout_data),
        "vocab_file": str(Path(args.vocab_file).resolve()),
    })
    torch.save(state, output_paths["calibrator"])

    calibrated_records = _add_prediction_field(
        holdout_records, holdout_scores, args.prediction_field,
    )
    calibrated_metadata = dict(holdout_metadata)
    calibrated_metadata["reward_calibration"] = {
        "method": state["method"],
        "prediction_field": args.prediction_field,
        "feature_names": list(FEATURE_NAMES),
        "calibrator_path": str(output_paths["calibrator"].resolve()),
        "operating_threshold": threshold,
        "training_record_count": state["training_record_count"],
        "training_product_count": len(set(train_table.product_indices)),
        "target_labels_saved": False,
    }
    save_guidance_dataset(
        output_paths["holdout_data"], calibrated_records,
        metadata=calibrated_metadata,
    )

    report = {
        "schema_version": 1,
        "method": state["method"],
        "feature_names": list(FEATURE_NAMES),
        "prediction_field": args.prediction_field,
        "inputs": {
            "train_data": str(Path(args.train_data).resolve()),
            "train_data_sha256": state["train_data_sha256"],
            "holdout_data": str(Path(args.holdout_data).resolve()),
            "holdout_data_sha256": state["holdout_data_sha256"],
            "train_target_file": str(Path(args.train_targets_file).resolve()),
            "holdout_target_file": str(Path(args.holdout_targets_file).resolve()),
            "augmentation": args.augmentation,
            "train_target_start_product": args.train_target_start_product,
            "holdout_target_start_product": args.holdout_target_start_product,
            "train_product_count": len(set(train_table.product_indices)),
            "holdout_product_count": len(set(holdout_table.product_indices)),
            "train_record_count": len(train_records),
            "holdout_record_count": len(holdout_records),
        },
        "training": {
            key: value
            for key, value in state.items()
            if key not in {
                "feature_mean", "feature_std", "weight", "bias",
                "train_data_sha256", "holdout_data_sha256", "vocab_file",
                "prediction_field", "operating_threshold",
                "reference_raw_positive_fraction",
                "train_calibrated_selected_fraction",
            }
        },
        "operating_point": {
            "threshold": threshold,
            "reference_raw_positive_fraction": float(raw_train_selected.float().mean()),
            "train_calibrated_selected_fraction": train_selected_fraction,
            "train": train_selection,
            "holdout": holdout_selection,
        },
        "quality": {
            "train": train_quality,
            "holdout": holdout_quality,
        },
        "rerank_same_holdout_pool": {
            "raw_forward_beam": raw_rerank,
            "calibrated": calibrated_rerank,
        },
        "bootstrap_delta_calibrated_minus_raw": bootstrap,
        "wall_seconds": wall_seconds,
        "outputs": {
            "calibrator": str(output_paths["calibrator"].resolve()),
            "holdout_calibrated_data": str(output_paths["holdout_data"].resolve()),
            "summary": str(output_paths["summary"].resolve()),
        },
        "label_safety": (
            "dataset targets were read only to fit/evaluate the calibrator; "
            "no per-record correctness label was saved into calibrated data"
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    output_paths["summary"].write_text(text + "\n")
    print(text)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--train_targets_file", required=True)
    parser.add_argument("--holdout_data", required=True)
    parser.add_argument("--holdout_targets_file", required=True)
    parser.add_argument("--vocab_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--train_target_start_product", type=int, default=0)
    parser.add_argument("--holdout_target_start_product", type=int, default=0)
    parser.add_argument("--prediction_field", default="calibrated_reward")
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
