import pytest

from scripts.generate_shared_anchor_guidance import validate_shared_anchor_config


def test_shared_anchor_config_maps_interior_time_to_step():
    assert validate_shared_anchor_config(100, 0.5, 4) == 50


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
