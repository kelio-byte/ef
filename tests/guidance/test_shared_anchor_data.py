from types import SimpleNamespace

import pytest

from scripts.generate_shared_anchor_guidance import (
    resolve_anchor_steps,
    shared_anchor_group_index,
    validate_shared_anchor_config,
    validate_shared_anchor_steps,
)
from scripts.generate_guidance_data import _read_original_products


def test_shared_anchor_config_maps_interior_time_to_step():
    assert validate_shared_anchor_config(100, 0.5, 4) == 50


def test_multiple_anchor_steps_are_sorted_and_remain_interior():
    assert validate_shared_anchor_steps(100, [90, 10, 50, 30, 70], 4) == (
        10, 30, 50, 70, 90,
    )


def test_anchor_step_cli_takes_precedence_only_when_time_is_absent():
    args = SimpleNamespace(
        n_steps=100, n_children=4,
        anchor_steps=[70, 10], anchor_time=None,
    )
    assert resolve_anchor_steps(args) == (10, 70)
    args.anchor_time = 0.5
    with pytest.raises(ValueError):
        resolve_anchor_steps(args)


@pytest.mark.parametrize(
    "anchor_steps",
    [[], [0], [100], [10, 10]],
)
def test_multiple_anchor_steps_reject_invalid_or_duplicate_values(anchor_steps):
    with pytest.raises(ValueError):
        validate_shared_anchor_steps(100, anchor_steps, 4)


def test_anchor_group_indices_separate_times_for_one_product():
    groups = {
        shared_anchor_group_index(product_index, anchor_ordinal, 5)
        for product_index in range(3)
        for anchor_ordinal in range(5)
    }
    assert groups == set(range(15))


def test_original_product_reader_keeps_one_row_per_augmentation_block(tmp_path):
    """``--max_products`` is applied after collapsing augmentation blocks."""
    products_file = tmp_path / "products.txt"
    products_file.write_text(
        "reaction0_view0\nreaction0_view1\n"
        "reaction1_view0\nreaction1_view1\n"
        "reaction2_view0\nreaction2_view1\n",
    )

    assert _read_original_products(str(products_file), augmentation=2) == [
        "reaction0_view0",
        "reaction1_view0",
        "reaction2_view0",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_steps": 1, "anchor_time": 0.5, "n_children": 4},
        {"n_steps": 100, "anchor_time": 0.0, "n_children": 4},
        {"n_steps": 100, "anchor_time": 1.0, "n_children": 4},
        {"n_steps": 100, "anchor_time": 0.5, "n_children": 1},
    ],
)
def test_shared_anchor_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        validate_shared_anchor_config(**kwargs)
