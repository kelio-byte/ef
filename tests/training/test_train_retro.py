import re

import torch

from scripts.train_retro import EpochRandomSampler, Tee, validation_due


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
