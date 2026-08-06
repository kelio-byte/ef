import torch

from scripts.train_retro import EpochRandomSampler, validation_due


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
