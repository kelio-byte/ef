import hashlib
from types import SimpleNamespace

import pytest

from scripts.sample_retro import (
    _apply_sampling_seed,
    _build_sampling_metadata,
    _is_frozen_r9k1m2,
    _infer_augmentation,
    _make_center_bias_batch,
    _outputs_per_product,
    _select_products,
)
import torch


def _euler_beam_args(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    dataset_dir = tmp_path / "USPTO_aug20_global"
    dataset_dir.mkdir()
    products = dataset_dir / "src-test.txt"
    products.write_text("C C\nN N\n")
    return SimpleNamespace(
        sampler="euler_beam",
        checkpoint=str(checkpoint),
        products_file=str(products),
        data_dir=None,
        n_steps=100,
        n_samples=99,
        n_branches=3,
        n_children=2,
        n_runs=3,
        structured_n_trajectories=9,
        structured_token_selection="argmax",
        euler_beam_initial_seed_groups=None,
        seed=42,
        euler_beam_score_mode="full_probability",
        euler_beam_changed_state_bonus=0.5,
        euler_beam_q_temperature=1.0,
        euler_beam_matmul_precision="high",
        euler_beam_child_policy="stochastic_noop",
        euler_beam_first_edit_diversity=False,
        euler_beam_share_identical_forwards=False,
        batch_size=64,
        device="cuda",
    )


def test_euler_beam_output_count_uses_runs_times_branches(tmp_path):
    args = _euler_beam_args(tmp_path)
    assert _outputs_per_product(args) == 9
    args.sampler = "euler"
    assert _outputs_per_product(args) == 99


def test_center_bias_beam_guard_accepts_only_frozen_r9k1m2(tmp_path):
    args = _euler_beam_args(tmp_path)
    args.n_runs = 9
    args.n_branches = 1
    args.n_children = 2
    assert _is_frozen_r9k1m2(args)
    args.euler_beam_share_identical_forwards = True
    assert not _is_frozen_r9k1m2(args)


def test_center_bias_batch_assigns_components_cyclically():
    product_ids = [[4, 5]]
    component_a = {
        "component_id": 0,
        "atom_maps": [1],
        "position_scores": [[0.0, 0.0, 0.0]] * 3,
    }
    component_b = {
        "component_id": 1,
        "atom_maps": [2],
        "position_scores": [[1.0, 1.0, 1.0]] * 3,
    }
    records = [{
        "input_row_index": 7,
        "reaction_position": 2,
        "augmentation_index": 3,
        "oracle_components": [component_a, component_b],
    }]
    scores, metadata, enabled = _make_center_bias_batch(
        product_ids,
        records,
        n_samples=5,
        source="oracle",
        global_start=7,
    )
    assert scores.shape == (5, 3, 3)
    assert torch.equal(scores[:, 0, 0], torch.tensor([0., 1., 0., 1., 0.]))
    assert [item["component_id"] for item in metadata] == [0, 1, 0, 1, 0]
    assert torch.equal(enabled, torch.ones(5, dtype=torch.bool))
    assert {item["trajectory_role"] for item in metadata} == {"center_guided"}


def test_center_bias_batch_can_mix_guided_and_ordinary_trajectories():
    product_ids = [[4, 5]]
    component = {
        "component_id": 0,
        "atom_maps": [1],
        "position_scores": [[1.0, 1.0, 1.0]] * 3,
    }
    records = [{
        "input_row_index": 7,
        "reaction_position": 2,
        "augmentation_index": 3,
        "oracle_components": [component],
    }]
    scores, metadata, enabled = _make_center_bias_batch(
        product_ids,
        records,
        n_samples=5,
        source="oracle",
        global_start=7,
        guided_trajectories=2,
    )
    assert scores.shape == (5, 3, 3)
    assert torch.equal(
        enabled, torch.tensor([True, True, False, False, False])
    )
    assert [item["trajectory_role"] for item in metadata] == [
        "center_guided", "center_guided", "ordinary_euler",
        "ordinary_euler", "ordinary_euler",
    ]


def test_structured_output_count_uses_trajectory_budget(tmp_path):
    args = _euler_beam_args(tmp_path)
    args.sampler = "structured_diversification"
    args.structured_n_trajectories = 9
    assert _outputs_per_product(args) == 9


def test_structured_v2_output_count_uses_mode_completion_budget(tmp_path):
    args = _euler_beam_args(tmp_path)
    args.sampler = "structured_diversification_v2"
    args.structured_v2_k_mode = 3
    args.structured_v2_k_completion = 3
    assert _outputs_per_product(args) == 9


def test_augmentation_inference_requires_unambiguous_aug_path():
    assert _infer_augmentation("datasets/example_aug20_global/src.txt") == (
        20,
        "datasets/example_aug20_global/src.txt",
    )
    assert _infer_augmentation("datasets/plain/src.txt") == (None, None)
    assert _infer_augmentation("a_aug10/x", "b_aug20/y") == (None, None)


def test_product_selection_preserves_complete_augmentation_blocks():
    products = [str(index) for index in range(100)]
    selected, end = _select_products(
        products, start_product=20, max_products=40, augmentation=20,
    )
    assert selected == products[20:60]
    assert end == 60

    for start, count in ((1, 20), (20, 21)):
        with pytest.raises(ValueError, match="augmentation blocks"):
            _select_products(
                products,
                start_product=start,
                max_products=count,
                augmentation=20,
            )


def test_sampling_metadata_records_effective_euler_beam_configuration(
    tmp_path,
):
    args = _euler_beam_args(tmp_path)
    prediction_path = tmp_path / "predictions.txt"
    prediction_bytes = b"A\n" * 18
    prediction_path.write_bytes(prediction_bytes)

    metadata = _build_sampling_metadata(
        args,
        {"data_dir": "datasets/USPTO_aug20_global", "use_origin_mask": False},
        prediction_path=str(prediction_path),
        source_product_count=100,
        selection_start_product=20,
        product_count=2,
        output_line_count=18,
        n_sampling_steps=80,
        sample_scheduler_name="cubic",
        train_scheduler_name="linear",
        use_origin_mask=False,
        elapsed_seconds=1.25,
        peak_cuda_allocated_bytes=1024,
        peak_cuda_reserved_bytes=2048,
    )

    assert metadata["sampler"] == "euler_beam"
    assert metadata["layout"] == (
        "input-product-major, branch-rank-major, run-minor"
    )
    assert metadata["augmentation"] == 20
    assert metadata["output_beam_size"] == 9
    assert metadata["output_line_count"] == 18
    assert metadata["output_sha256"] == hashlib.sha256(
        prediction_bytes,
    ).hexdigest()
    assert metadata["sampling"] == {
        "n_steps": 80,
        "sample_scheduler": "cubic",
        "train_scheduler": "linear",
        "seed": 42,
        "seed_applied_to_sampler": True,
        "n_branches": 3,
        "n_children": 2,
        "n_runs": 3,
        "final_branches_per_run": 3,
        "output_order": "branch-rank-major, run-minor",
        "initial_seed_groups": None,
        "score_mode": "full_probability",
        "changed_state_bonus": 0.5,
        "q_temperature": 1.0,
        "matmul_precision": "high",
        "child_policy": "stochastic_noop",
        "share_identical_forwards": False,
        "first_edit_diversity": False,
        "seed_scope": "stable product/run streams",
    }
    assert metadata["input"]["product_count"] == 2
    assert metadata["input"]["source_product_count"] == 100
    assert metadata["input"]["selection_start_product"] == 20
    assert metadata["input"]["selection_end_product_exclusive"] == 22
    assert metadata["input"]["sha256"] == hashlib.sha256(
        b"C C\nN N\n",
    ).hexdigest()
    assert metadata["runtime"]["peak_cuda_allocated_bytes"] == 1024
    assert metadata["runtime"]["peak_cuda_reserved_bytes"] == 2048


def test_euler_beam_output_count_includes_all_final_branches(tmp_path):
    args = _euler_beam_args(tmp_path)
    args.n_runs = 1
    assert _outputs_per_product(args) == 3


def test_ordinary_euler_seed_is_applied_and_recorded(tmp_path):
    args = _euler_beam_args(tmp_path)
    args.sampler = "euler"
    args.n_samples = 2
    prediction_path = tmp_path / "predictions.txt"
    prediction_path.write_bytes(b"A\n" * 4)
    _apply_sampling_seed(42, torch.device("cpu"))
    first = torch.rand(4)
    _apply_sampling_seed(42, torch.device("cpu"))
    second = torch.rand(4)
    assert torch.equal(first, second)
    metadata = _build_sampling_metadata(
        args,
        {"data_dir": "datasets/USPTO_aug20_global", "use_origin_mask": False},
        prediction_path=str(prediction_path),
        source_product_count=2,
        selection_start_product=0,
        product_count=2,
        output_line_count=4,
        n_sampling_steps=100,
        sample_scheduler_name="cubic",
        train_scheduler_name="cubic",
        use_origin_mask=False,
        elapsed_seconds=1.0,
    )
    assert metadata["sampling"]["seed_applied_to_sampler"] is True
    assert metadata["sampling"]["seed_scope"] == "global torch RNG"
