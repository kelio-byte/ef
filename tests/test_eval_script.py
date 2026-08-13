import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "eval.py"
SPEC = importlib.util.spec_from_file_location("evaluation_entrypoint", SCRIPT_PATH)
eval_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(eval_script)


def _args(*extra):
    return eval_script.build_parser().parse_args([
        "--checkpoint", "checkpoint.pt",
        "--products_file", "src-aug20.txt",
        "--targets", "tgt.txt",
        "--output_dir", "results/eval",
        *extra,
    ])


def test_derive_score_layout_uses_sampling_metadata():
    metadata = {
        "augmentation": 20,
        "output_beam_size": 3,
        "product_count": 2000,
        "input": {"selection_start_product": 400},
    }
    assert eval_script.derive_score_layout(metadata, 20) == (3, 100, 20)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"augmentation": 10, "output_beam_size": 3,
          "product_count": 20}, "augmentation does not match"),
        ({"augmentation": 20, "output_beam_size": 3,
          "product_count": 21}, "complete augmentation blocks"),
        ({"augmentation": 20, "output_beam_size": 3,
          "product_count": 20,
          "input": {"selection_start_product": 1}},
         "augmentation block"),
    ],
)
def test_derive_score_layout_rejects_misalignment(metadata, message):
    with pytest.raises(ValueError, match=message):
        eval_script.derive_score_layout(metadata, 20)


def test_sample_command_carries_current_euler_beam_settings():
    args = _args(
        "--n_branches", "5", "--n_children", "3", "--n_runs", "2",
        "--max_products", "200", "--euler_beam_share_identical_forwards",
        "--euler_beam_first_edit_diversity",
    )
    command = eval_script.build_sample_command(args)
    rendered = " ".join(command)
    assert "--n_branches 5" in rendered
    assert "--n_children 3" in rendered
    assert "--n_runs 2" in rendered
    assert "--euler_beam_q_temperature 1.0" in rendered
    assert "--euler_beam_n_return" not in rendered
    assert "--max_products 200" in rendered
    assert "--euler_beam_share_identical_forwards" in rendered
    assert "--euler_beam_first_edit_diversity" in rendered


def test_sample_command_carries_optional_guidance_settings():
    args = _args(
        "--sampler", "euler",
        "--guidance_checkpoint", "guidance.pt",
        "--guidance_beta", "0.1",
        "--guidance_rate_normalization", "per_sample",
    )
    command = eval_script.build_sample_command(args)
    rendered = " ".join(command)
    assert "--guidance_checkpoint guidance.pt" in rendered
    assert "--guidance_beta 0.1" in rendered
    assert "--guidance_rate_normalization per_sample" in rendered


def test_sample_command_carries_structured_sampler_settings():
    args = _args(
        "--sampler", "structured_diversification",
        "--structured_n_trajectories", "9",
        "--structured_token_selection", "argmax",
    )
    command = eval_script.build_sample_command(args)
    rendered = " ".join(command)
    assert "--sampler structured_diversification" in rendered
    assert "--structured_n_trajectories 9" in rendered
    assert "--structured_token_selection argmax" in rendered


def test_sample_command_carries_delayed_structured_v2_settings():
    args = _args(
        "--sampler", "structured_diversification_v2",
        "--structured_v2_k_mode", "3",
        "--structured_v2_k_completion", "3",
        "--structured_v2_mode_pool_size", "9",
    )
    command = eval_script.build_sample_command(args)
    rendered = " ".join(command)
    assert "--sampler structured_diversification_v2" in rendered
    assert "--structured_v2_k_mode 3" in rendered
    assert "--structured_v2_k_completion 3" in rendered
    assert "--structured_v2_mode_pool_size 9" in rendered


def test_score_command_defaults_to_top10_diagnostics():
    args = _args()
    command = eval_script.build_score_command(
        args, beam_size=3, reaction_count=50, target_offset=10,
    )
    rendered = " ".join(command)
    assert "--beam_size 3" in rendered
    assert "--n_best 10" in rendered
    assert "--length 50" in rendered
    assert "--target_offset 10" in rendered
    assert "--diagnostics" in command
    assert "results/eval/diagnostics.json" in command


def test_main_refuses_to_overwrite_existing_predictions(tmp_path):
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    (output_dir / "predictions.txt").write_text("existing\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        eval_script.main([
            "--checkpoint", "checkpoint.pt",
            "--products_file", "src-aug20.txt",
            "--targets", "tgt.txt",
            "--output_dir", str(output_dir),
            "--max_products", "20",
            "--dry_run",
        ])


def test_dry_run_derives_score_layout_without_running_subprocess(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.setattr(
        eval_script.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("dry run executed a subprocess"),
    )
    result = eval_script.main([
        "--checkpoint", "checkpoint.pt",
        "--products_file", "src-aug20.txt",
        "--targets", "tgt.txt",
        "--output_dir", str(tmp_path / "new"),
        "--max_products", "200",
        "--start_product", "400",
        "--n_runs", "3",
        "--n_branches", "3",
        "--dry_run",
    ])
    output = capsys.readouterr().out
    assert result == 0
    assert "--beam_size 9" in output
    assert "--length 10" in output
    assert "--target_offset 20" in output
