import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "compare_retro_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("compare_retro_diagnostics", SCRIPT_PATH)
comparison = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(comparison)


def _write_diagnostics(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"per_reaction": rows, "summary": {}}))
    return path


def test_compares_aggregated_reactions_with_paired_bootstrap(tmp_path):
    baseline = _write_diagnostics(tmp_path / "baseline.json", [
        {"reaction_index": 0, "target_final_rank": 1, "oracle_any": True},
        {"reaction_index": 1, "target_final_rank": 4, "oracle_any": True},
        {"reaction_index": 2, "target_final_rank": None, "oracle_any": False},
        {"reaction_index": 3, "target_final_rank": 2, "oracle_any": True},
    ])
    candidate = _write_diagnostics(tmp_path / "candidate.json", [
        {"reaction_index": 0, "target_final_rank": 2, "oracle_any": True},
        {"reaction_index": 1, "target_final_rank": 1, "oracle_any": True},
        {"reaction_index": 2, "target_final_rank": 3, "oracle_any": True},
        {"reaction_index": 3, "target_final_rank": None, "oracle_any": False},
    ])

    result = comparison.compare_diagnostics(
        baseline, candidate, bootstrap_samples=100, seed=3,
    )

    assert result["reaction_count"] == 4
    assert result["metrics"]["top_1"]["baseline_percent"] == 25.0
    assert result["metrics"]["top_1"]["candidate_percent"] == 25.0
    assert result["metrics"]["top_1"]["candidate_only_count"] == 1
    assert result["metrics"]["top_1"]["baseline_only_count"] == 1
    assert result["metrics"]["top_3"]["delta_percentage_points"] == 25.0
    assert "paired_bootstrap_95ci_delta_percentage_points" in result["metrics"]["oracle"]


def test_rejects_different_reaction_sets(tmp_path):
    baseline = _write_diagnostics(tmp_path / "baseline.json", [
        {"reaction_index": 0, "target_final_rank": 1, "oracle_any": True},
    ])
    candidate = _write_diagnostics(tmp_path / "candidate.json", [
        {"reaction_index": 1, "target_final_rank": 1, "oracle_any": True},
    ])

    with pytest.raises(ValueError, match="different reactions"):
        comparison.compare_diagnostics(
            baseline, candidate, bootstrap_samples=2,
        )
