import re

import torch

from scripts.train_retro import (
    DistributedEpochRandomSampler,
    EpochRandomSampler,
    Tee,
    _metrics_are_finite,
    gradient_diagnostics,
    validation_due,
)


def test_tee_adds_minute_timestamp_once_per_logical_line(tmp_path):
    log_path = tmp_path / "train.log"
    with Tee(str(log_path)) as tee:
        tee.write("first")
        tee.write("\nsecond\n")

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    timestamp = r"\[\d{2}/\d{2}/\d{2}/\d{2}\]"
    assert re.fullmatch(timestamp + r" first", lines[0])
    assert re.fullmatch(timestamp + r" second", lines[1])


def test_validation_due_respects_start_step_and_interval():
    assert not validation_due(99_999, 100_000, 20_000)
    assert validation_due(100_000, 100_000, 20_000)
    assert not validation_due(110_000, 100_000, 20_000)
    assert validation_due(120_000, 100_000, 20_000)
    assert not validation_due(120_000, 100_000, 0)


def test_epoch_random_sampler_reconstructs_remaining_permutation():
    data = list(range(7))
    sampler = EpochRandomSampler(data, seed=42)
    full = list(iter(sampler))

    sampler.set_position(epoch=0, start_index=4)
    remaining = list(iter(sampler))

    assert remaining == full[4:]
    assert sorted(full) == data


def test_epoch_random_sampler_changes_permutation_by_epoch():
    data = list(range(8))
    sampler = EpochRandomSampler(data, seed=42)
    first = list(iter(sampler))
    sampler.set_position(epoch=1)
    second = list(iter(sampler))

    assert first != second
    assert sorted(second) == data


def test_distributed_epoch_sampler_shards_one_shared_permutation_and_resumes():
    data = list(range(11))
    expected = list(EpochRandomSampler(data, seed=42))
    rank0 = DistributedEpochRandomSampler(
        data, num_replicas=2, rank=0, seed=42, shuffle=True, drop_last=True,
    )
    rank1 = DistributedEpochRandomSampler(
        data, num_replicas=2, rank=1, seed=42, shuffle=True, drop_last=True,
    )

    rank0_indices = list(rank0)
    rank1_indices = list(rank1)
    assert rank0_indices == expected[:10:2]
    assert rank1_indices == expected[1:10:2]
    assert not set(rank0_indices).intersection(rank1_indices)
    assert sorted(rank0_indices + rank1_indices) == sorted(expected[:10])

    rank1.set_position(epoch=0, start_index=3)
    assert list(rank1) == rank1_indices[3:]


def test_distributed_validation_sampler_never_pads_examples():
    data = list(range(5))
    rank0 = DistributedEpochRandomSampler(
        data, num_replicas=2, rank=0, seed=42, shuffle=False, drop_last=False,
    )
    rank1 = DistributedEpochRandomSampler(
        data, num_replicas=2, rank=1, seed=42, shuffle=False, drop_last=False,
    )

    assert list(rank0) == [0, 2, 4]
    assert list(rank1) == [1, 3]
    assert sorted(list(rank0) + list(rank1)) == data


def test_gradient_diagnostics_and_metric_finiteness():
    import torch

    model = torch.nn.Linear(3, 2)
    output = model(torch.ones(1, 3)).sum()
    output.backward()
    diagnostics = gradient_diagnostics(model)
    assert diagnostics["gradient_tensors"] == 2
    assert diagnostics["nonfinite_grad_values"] == 0
    assert diagnostics["nonfinite_parameter_values"] == 0
    assert diagnostics["grad_norm"] > 0.0
    assert _metrics_are_finite({"loss": 1.0, "u_tot": 2})
    assert not _metrics_are_finite({"loss": float("nan")})
