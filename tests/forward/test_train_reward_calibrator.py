import importlib.util
from pathlib import Path
import sys

import pytest
import torch


SCRIPT_PATH = (
    Path(__file__).parents[2] / "scripts" / "train_reward_calibrator.py"
)
SPEC = importlib.util.spec_from_file_location(
    "train_reward_calibrator", SCRIPT_PATH,
)
calibrator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = calibrator
SPEC.loader.exec_module(calibrator)


def _record(product_index, source_index, product_tokens, terminal_tokens, rank, time):
    return {
        "product_index": product_index,
        "source_index": source_index,
        "product_tokens": product_tokens,
        "terminal_tokens": terminal_tokens,
        "forward_beam_rank": rank,
        "time": time,
    }


def _table():
    id2token = {0: "<pad>", 1: "<bos>", 2: "C", 3: "O", 4: "N"}
    records = [
        _record(0, 0, [1, 2, 3], [1, 2, 3], 1, 0.1),  # CO, correct
        _record(0, 0, [1, 2, 3], [1, 2], 0, 0.1),     # C, wrong
        _record(1, 1, [1, 4], [1, 4], 2, 0.5),         # N, correct
        _record(1, 1, [1, 4], [1, 2], 1, 0.5),         # C, wrong
    ]
    return calibrator.build_candidate_table(records, ["C O", "N"], id2token)


def test_candidate_table_uses_only_record_features_and_target_labels():
    table = _table()
    assert table.features.shape == (4, len(calibrator.FEATURE_NAMES))
    assert table.labels.tolist() == [True, False, True, False]
    assert table.features[:, 0].tolist() == [1.0, 0.0, 0.5, 1.0]
    assert table.features[:, 1].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert table.canonical_targets == {0: "CO", 1: "N"}


def test_linear_calibrator_scores_a_simple_separable_signal():
    features = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.9, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    ])
    labels = torch.tensor([False, False, True, True])
    state = calibrator.fit_logistic_calibrator(
        features, labels, l2=0.01, max_steps=300, learning_rate=0.05,
    )
    scores = calibrator.predict_probability(features, state)
    assert scores[2:].min() > scores[:2].max()
    assert state["loss_history"][0]["objective"] > state["loss_history"][-1]["objective"]


def test_rerank_and_bootstrap_compare_the_same_product_pool():
    table = _table()
    raw = table.features[:, 0]
    calibrated = torch.tensor([0.95, 0.05, 0.8, 0.1])
    raw_summary, _ = calibrator.rerank_summary(table, raw)
    calibrated_summary, _ = calibrator.rerank_summary(table, calibrated)
    assert raw_summary["top_1"] == 0.5
    assert calibrated_summary["top_1"] == 1.0
    threshold, fraction = calibrator.choose_threshold_at_reference_rate(
        calibrated, 0.5,
    )
    assert threshold == pytest.approx(0.8)
    assert fraction == 0.5
    result = calibrator.bootstrap_comparison(
        table,
        raw,
        calibrated,
        raw > 0,
        calibrated >= threshold,
        bootstrap_samples=20,
        seed=42,
    )
    assert set(result) >= {
        "global_correctness_auc", "within_group_correctness_auc", "top_1",
    }
    assert result["top_1"]["bootstrap_samples"] == 20


def test_parser_defaults_are_frozen_for_p1():
    args = calibrator.build_parser().parse_args([
        "--train_data", "train.pt",
        "--train_targets_file", "train.txt",
        "--holdout_data", "holdout.pt",
        "--holdout_targets_file", "holdout.txt",
        "--vocab_file", "vocab.txt",
        "--output_dir", "out",
    ])
    assert args.prediction_field == "calibrated_reward"
    assert args.l2 == 0.01
    assert args.max_steps == 2000
    assert args.bootstrap_samples == 2000
