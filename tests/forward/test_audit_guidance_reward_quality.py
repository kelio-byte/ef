import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[2] / "scripts" / "audit_guidance_reward_quality.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_guidance_reward_quality", SCRIPT_PATH,
)
audit_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit_script)


def _record(source_index, product_index, terminal_tokens, forward_beam_rank, time_index):
    return {
        "source_index": source_index,
        "product_index": product_index,
        "terminal_tokens": terminal_tokens,
        "forward_beam_rank": forward_beam_rank,
        "time_index": time_index,
    }


def test_pairwise_auc_handles_ties_exactly() -> None:
    assert audit_script.pairwise_auc([1.0, 0.0, 0.5, 1.0], [True, False, True, False]) == 0.625
    assert audit_script.pairwise_auc([1.0, 2.0], [True, True]) is None


def test_summary_measures_global_and_shared_anchor_correctness() -> None:
    id2token = {0: "<pad>", 1: "<bos>", 2: "C", 3: "O", 4: "N"}
    records = [
        _record(0, 0, [1, 2, 3], 1, 10),  # CO, correct, score 1
        _record(0, 0, [1, 2], 0, 10),     # C, incorrect, score 0
        _record(1, 1, [1, 4], 2, 30),     # N, correct, score .5
        _record(1, 1, [1, 2], 1, 30),     # C, incorrect, score 1
    ]

    summary = audit_script.summarize_reward_correctness(
        records,
        ["C O", "N"],
        id2token,
    )

    assert summary["correct_candidate_count"] == 2
    assert summary["incorrect_candidate_count"] == 2
    assert summary["correct_candidate_fraction"] == 0.5
    assert summary["score"]["correctness_auc"] == 0.625
    assert summary["shared_anchor_groups"]["group_count"] == 2
    assert summary["shared_anchor_groups"]["groups_with_any_correct_terminal"] == 2
    assert summary["shared_anchor_groups"]["groups_with_mixed_correctness"] == 2
    assert summary["shared_anchor_groups"]["within_group_correctness_auc"] == 0.5
    assert summary["by_time_index"]["10"]["correctness_auc"] == 1.0
    assert summary["by_time_index"]["30"]["correctness_auc"] == 0.0


def test_parser_defaults_to_forward_beam_rank() -> None:
    args = audit_script.build_parser().parse_args([
        "--data", "records.pt",
        "--targets_file", "tgt.txt",
        "--vocab_file", "vocab.txt",
    ])
    assert args.augmentation == 20
    assert args.target_start_product == 0
    assert args.score_field == "forward_beam_rank"
