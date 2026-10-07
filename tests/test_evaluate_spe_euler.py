import json

from scripts.evaluate_spe_euler import evaluate


def test_evaluate_spe_euler_reports_topk_invalid_and_unique(tmp_path):
    predictions = tmp_path / "predictions.txt"
    targets = tmp_path / "targets.txt"
    # Two reactions, two augmentations, two independent samples.
    predictions.write_text(
        "C C\n"
        "C1CC1C1\n"
        "O\n"
        "C C\n"
        "N\n"
        "N\n"
        "C\n"
        "O\n"
    )
    targets.write_text("CC\nCC\nO\nO\n")

    result = evaluate(
        predictions,
        targets,
        augmentation=2,
        beam_size=2,
    )
    metrics = result["metrics"]
    assert metrics["oracle_count"] == 2
    assert metrics["top_k_percent"] == {"1": 50.0, "3": 100.0, "5": 100.0, "10": 100.0}
    assert metrics["invalid_count"] == 1
    assert metrics["unique_candidate_count"] == 5
    assert metrics["duplicate_count_among_valid"] == 2
    json.dumps(result)


def test_evaluate_spe_euler_supports_prefix(tmp_path):
    predictions = tmp_path / "predictions.txt"
    targets = tmp_path / "targets.txt"
    predictions.write_text("CC\nCC\nO\nN\n")
    targets.write_text("CC\nO\n")
    result = evaluate(predictions, targets, augmentation=1, beam_size=2, length=1)
    assert result["protocol"]["reaction_count"] == 1
    assert result["metrics"]["oracle_percent"] == 100.0


def test_evaluate_spe_euler_treats_empty_rdkit_molecule_as_invalid():
    from scripts.evaluate_spe_euler import canonicalize_smiles_line

    assert canonicalize_smiles_line("") is None
    assert canonicalize_smiles_line("(") is None


def test_evaluate_spe_euler_applies_global_inverse_after_joining_tokens():
    from scripts.evaluate_spe_euler import canonicalize_smiles_line

    assert canonicalize_smiles_line("C ( . O )") == "C.O"
