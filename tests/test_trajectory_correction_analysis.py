from __future__ import annotations

import pytest

from scripts.trajectory_correction_analysis import _select_augmentation_view


def test_select_augmentation_view_keeps_source_row_provenance() -> None:
    products = [f"p{reaction}_{view}" for reaction in range(2) for view in range(3)]
    targets = [f"t{reaction}_{view}" for reaction in range(2) for view in range(3)]
    selected_products, selected_targets, reactions, source_rows = _select_augmentation_view(
        products,
        targets,
        augmentation=3,
        max_reactions=2,
        augmentation_index=2,
    )
    assert selected_products == ["p0_2", "p1_2"]
    assert selected_targets == ["t0_2", "t1_2"]
    assert reactions == [0, 1]
    assert source_rows == [2, 5]


def test_select_augmentation_view_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="augmentation_index"):
        _select_augmentation_view(
            ["p0", "p1"], ["t0", "t1"],
            augmentation=2, max_reactions=1, augmentation_index=2,
        )
