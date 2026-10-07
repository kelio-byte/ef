import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SPEC = importlib.util.spec_from_file_location(
    "score_global",
    SCRIPTS_DIR / "score_#global#.py",
)
score_global = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score_global)


def candidate(smiles):
    return (smiles, smiles)


def test_resolve_input_layout_requires_exact_complete_default_layout():
    assert score_global.resolve_input_layout(
        prediction_count=3000,
        target_count=1000,
        augmentation=20,
        beam_size=3,
    ) == (50, 3000, 1000)

    with pytest.raises(ValueError, match="not divisible"):
        score_global.resolve_input_layout(
            prediction_count=2999,
            target_count=1000,
            augmentation=20,
            beam_size=3,
        )

    with pytest.raises(ValueError, match="target line count"):
        score_global.resolve_input_layout(
            prediction_count=3000,
            target_count=980,
            augmentation=20,
            beam_size=3,
        )


def test_resolve_input_layout_allows_only_complete_explicit_prefix():
    assert score_global.resolve_input_layout(
        prediction_count=3000,
        target_count=1000,
        augmentation=20,
        beam_size=3,
        length=10,
    ) == (10, 600, 200)

    with pytest.raises(ValueError, match="requires 600 prediction"):
        score_global.resolve_input_layout(
            prediction_count=599,
            target_count=1000,
            augmentation=20,
            beam_size=3,
            length=10,
        )

    with pytest.raises(ValueError, match="requires 200 target"):
        score_global.resolve_input_layout(
            prediction_count=3000,
            target_count=199,
            augmentation=20,
            beam_size=3,
            length=10,
        )


def test_prediction_metadata_validates_sampling_layout_and_hash(tmp_path):
    prediction_path = tmp_path / "predictions.txt"
    prediction_bytes = b"A\nB\nC\nD\nE\nF\n"
    prediction_path.write_bytes(prediction_bytes)
    metadata = {
        "augmentation": 2,
        "product_count": 2,
        "output_beam_size": 3,
        "output_line_count": 6,
        "output_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
    }

    score_global.validate_prediction_metadata(
        metadata,
        metadata_path=str(tmp_path / "sampling_metadata.json"),
        prediction_path=str(prediction_path),
        prediction_count=6,
        augmentation=2,
        beam_size=3,
    )


def test_prediction_metadata_cross_checks_target_offset(tmp_path):
    prediction_path = tmp_path / "predictions.txt"
    prediction_bytes = b"A\nB\nC\nD\nE\nF\n"
    prediction_path.write_bytes(prediction_bytes)
    metadata = {
        "augmentation": 20,
        "product_count": 2,
        "output_beam_size": 3,
        "output_line_count": 6,
        "output_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "input": {"selection_start_product": 1000},
    }

    score_global.validate_prediction_metadata(
        metadata,
        metadata_path=str(tmp_path / "sampling_metadata.json"),
        prediction_path=str(prediction_path),
        prediction_count=6,
        augmentation=20,
        beam_size=3,
        target_offset=50,
    )
    with pytest.raises(ValueError, match="target_offset"):
        score_global.validate_prediction_metadata(
            metadata,
            metadata_path=str(tmp_path / "sampling_metadata.json"),
            prediction_path=str(prediction_path),
            prediction_count=6,
            augmentation=20,
            beam_size=3,
            target_offset=0,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("output_beam_size", 2, "beam_size"),
        ("augmentation", 10, "augmentation"),
        ("output_line_count", 5, "line count"),
        ("output_sha256", "stale", "SHA-256"),
    ],
)
def test_prediction_metadata_rejects_mismatched_score_inputs(
    tmp_path,
    field,
    value,
    message,
):
    prediction_path = tmp_path / "predictions.txt"
    prediction_bytes = b"A\nB\nC\nD\nE\nF\n"
    prediction_path.write_bytes(prediction_bytes)
    metadata = {
        "augmentation": 2,
        "product_count": 2,
        "output_beam_size": 3,
        "output_line_count": 6,
        "output_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
    }
    metadata[field] = value

    with pytest.raises(ValueError, match=message):
        score_global.validate_prediction_metadata(
            metadata,
            metadata_path=str(tmp_path / "sampling_metadata.json"),
            prediction_path=str(prediction_path),
            prediction_count=6,
            augmentation=2,
            beam_size=3,
        )


def test_metadata_discovery_supports_legacy_inputs_and_rejects_ambiguity(
    tmp_path,
):
    prediction_path = tmp_path / "predictions.txt"
    prediction_path.write_text("A\n")
    assert score_global.load_and_validate_prediction_metadata(
        str(prediction_path), 1, augmentation=1, beam_size=1,
    ) is None

    (tmp_path / "sampling_metadata.json").write_text("{}")
    (tmp_path / "mixing_metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="multiple prediction metadata"):
        score_global.load_and_validate_prediction_metadata(
            str(prediction_path), 1, augmentation=1, beam_size=1,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("augmentation", 0, "augmentation"),
        ("beam_size", 0, "beam_size"),
        ("n_best", 0, "n_best"),
        ("process_number", 0, "process_number"),
        ("length", 0, "length"),
    ],
)
def test_validate_scoring_options_rejects_invalid_shape_arguments(
    field,
    value,
    message,
):
    options = SimpleNamespace(
        augmentation=20,
        beam_size=3,
        n_best=5,
        process_number=2,
        length=-1,
        raw=False,
    )
    setattr(options, field, value)
    with pytest.raises(ValueError, match=message):
        score_global.validate_scoring_options(options)


def test_compute_rank_preserves_legacy_best_local_rank_semantics_and_input():
    prediction = [
        [candidate("A"), candidate(""), candidate("B")],
        [candidate("B"), candidate("A"), candidate("A")],
    ]
    original = copy.deepcopy(prediction)

    rank, invalid = score_global.compute_rank(
        prediction,
        alpha=1.0,
        beam_size=3,
    )

    assert prediction == original
    assert invalid == [0, 1, 0]
    assert rank[candidate("A")] == pytest.approx(1.5)
    assert rank[candidate("B")] == pytest.approx(1.5)


def test_aggregation_modes_are_opt_in_and_have_distinct_priorities():
    prediction = [
        [candidate("A"), candidate("B")],
        [candidate("C"), candidate("B")],
        [candidate("D"), candidate("B")],
        [candidate("E"), candidate("B")],
    ]

    legacy, _ = score_global.compute_rank(
        prediction,
        beam_size=2,
        aggregation_mode="legacy_best_rank",
    )
    rrf, _ = score_global.compute_rank(
        prediction,
        beam_size=2,
        aggregation_mode="rrf",
    )
    frequency, _ = score_global.compute_rank(
        prediction,
        beam_size=2,
        aggregation_mode="frequency_first",
    )
    hybrid, _ = score_global.compute_rank(
        prediction,
        beam_size=2,
        aggregation_mode="hybrid",
    )

    assert legacy[candidate("A")] > legacy[candidate("B")]
    assert rrf[candidate("B")] > rrf[candidate("A")]
    assert frequency[candidate("B")] > frequency[candidate("A")]
    assert hybrid[candidate("B")] > hybrid[candidate("A")]


def test_sampling_diagnostics_separates_coverage_runs_and_duplicates():
    predictions = [
        [
            [candidate("A"), candidate("B")],
            [candidate("C"), candidate("A")],
        ],
        [
            [candidate(""), candidate("D")],
            [candidate("D"), candidate("D")],
        ],
    ]
    targets = [candidate("A"), candidate("Z")]

    diagnostics = score_global.compute_sampling_diagnostics(
        predictions,
        targets,
        beam_size=2,
        top_k=2,
        report_n_best=3,
    )
    summary = diagnostics["summary"]

    assert summary["oracle_any_count"] == 1
    assert summary["oracle_any_percent"] == pytest.approx(50.0)
    assert summary["covered_outside_top_k"] == 0
    assert summary["mean_target_augmentation_count"] == pytest.approx(1.0)
    assert summary["best_local_rank_counts"] == {"1": 1, "2": 0}
    assert summary["mean_valid_candidates_per_reaction"] == pytest.approx(3.5)
    assert summary["mean_true_unique_candidates_per_reaction"] == pytest.approx(2.0)
    assert summary["aggregated_rank_availability_percent"] == {
        "1": pytest.approx(100.0),
        "2": pytest.approx(50.0),
        "3": pytest.approx(50.0),
    }

    run_1, run_2 = summary["run_metrics"]
    assert run_1["target_hit_rate_percent"] == pytest.approx(50.0)
    assert run_1["invalid_rate_percent"] == pytest.approx(25.0)
    assert run_2["target_hit_rate_percent"] == pytest.approx(50.0)
    assert run_2["duplicate_rate_among_valid_percent"] == pytest.approx(25.0)

    assert diagnostics["per_reaction"][0]["target_final_rank"] == 1
    assert diagnostics["per_reaction"][1]["target_final_rank"] is None
    json.dumps(diagnostics)


def test_sampling_diagnostics_prints_generic_input_rank_labels(capsys):
    diagnostics = score_global.compute_sampling_diagnostics(
        [[[candidate("A"), candidate("B")]]],
        [candidate("A")],
        beam_size=2,
        top_k=2,
    )

    score_global.print_sampling_diagnostics(diagnostics)
    output = capsys.readouterr().out

    assert "Input rank 1: target-hit" in output
    assert "Overlap input rank_1_vs_2:" in output
    assert "Run 1:" not in output
