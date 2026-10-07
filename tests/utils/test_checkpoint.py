import pytest
import torch

from edit_flows.utils.checkpoint import atomic_torch_save, average_model_checkpoints
from scripts.average_checkpoints import compute_weights, gather_checkpoints
from scripts.train_retro import prune_checkpoints


def test_atomic_save_roundtrip(tmp_path):
    destination = tmp_path / "best.pt"
    atomic_torch_save({"weights": torch.arange(3), "step": 2}, destination)
    restored = torch.load(destination, weights_only=True)
    assert restored["step"] == 2
    assert torch.equal(restored["weights"], torch.arange(3))
    assert list(tmp_path.iterdir()) == [destination]


def test_failed_serialization_preserves_existing_checkpoint(tmp_path, monkeypatch):
    destination = tmp_path / "best.pt"
    destination.write_bytes(b"existing checkpoint")

    def fail(state, stream):
        stream.write(b"partial new checkpoint")
        raise OSError("simulated disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        atomic_torch_save({}, destination)
    assert destination.read_bytes() == b"existing checkpoint"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("keep", [0, -1])
def test_pruning_rejects_settings_that_lose_all_checkpoints(tmp_path, keep):
    path = tmp_path / "checkpoint_step1.pt"
    path.write_bytes(b"keep")
    with pytest.raises(ValueError, match="keep_checkpoints"):
        prune_checkpoints(str(tmp_path), keep)
    assert path.read_bytes() == b"keep"


def _checkpoint(tmp_path, name, value, config=None, **extra):
    path = tmp_path / name
    torch.save({"config": config or {"data_dir": "global", "hidden_dim": 2},
                "model_vocab": 6,
                "model_state_dict": {"w": torch.tensor([value], dtype=torch.float32),
                                     "counter": torch.tensor(3)}, **extra}, path)
    return str(path)


def test_average_preserves_explicit_weight_order_and_loads_once(tmp_path, monkeypatch):
    later = _checkpoint(tmp_path, "checkpoint_step10.pt", 10)
    earlier = _checkpoint(tmp_path, "checkpoint_step2.pt", 2)
    paths = gather_checkpoints(None, [later, earlier], None, None)
    assert paths == [later, earlier]
    weights = compute_weights(paths, "uniform", [3, 1], 0.5)
    calls = []
    original_load = torch.load

    def load(path, **kwargs):
        calls.append(path)
        return original_load(path, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    result = average_model_checkpoints(paths, weights)
    assert result["model_state_dict"]["w"].item() == 8
    assert result["model_state_dict"]["counter"].dtype == torch.int64
    assert calls == paths


def test_average_rejects_mixed_experiments(tmp_path):
    first = _checkpoint(tmp_path, "first.pt", 1)
    second = _checkpoint(tmp_path, "second.pt", 3, config={"data_dir": "other"})
    with pytest.raises(ValueError, match="configuration/vocabulary"):
        average_model_checkpoints([first, second])


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [float("nan"), 1], [float("inf"), 1]])
def test_average_rejects_invalid_weights_before_loading(weights):
    with pytest.raises(ValueError, match="Weights"):
        average_model_checkpoints(["missing1.pt", "missing2.pt"], weights)


@pytest.mark.parametrize("last_n", [0, -1])
def test_average_selection_rejects_nonpositive_count(last_n):
    with pytest.raises(ValueError, match="last_n"):
        gather_checkpoints("missing", None, last_n, None)


@pytest.mark.parametrize("state", [
    {"other": torch.ones(1), "counter": torch.tensor(3)},
    {"w": torch.ones(2), "counter": torch.tensor(3)},
    {"w": torch.ones(1, dtype=torch.float64), "counter": torch.tensor(3)},
    {"w": torch.ones(1), "counter": torch.tensor(4)},
])
def test_average_rejects_incompatible_tensors(tmp_path, state):
    first = _checkpoint(tmp_path, "first.pt", 1)
    second = _checkpoint(tmp_path, "second.pt", 3, model_state_dict=state)
    with pytest.raises(ValueError, match="mismatch"):
        average_model_checkpoints([first, second])


def test_legacy_eval_average_uses_shared_validated_implementation(tmp_path):
    from scripts.eval_retro import average_checkpoints_in_dir
    _checkpoint(tmp_path, "checkpoint_step1.pt", 2)
    _checkpoint(tmp_path, "checkpoint_step2.pt", 6)
    output = tmp_path / "averaged.pt"
    assert average_checkpoints_in_dir(str(tmp_path), str(output)) == str(output)
    restored = torch.load(output, weights_only=True)
    assert restored["model_state_dict"]["w"].item() == 4
    assert restored["averaging"]["weights"] == [0.5, 0.5]
