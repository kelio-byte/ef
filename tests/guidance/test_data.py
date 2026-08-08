from pathlib import Path

import torch
import pytest

from edit_flows.core.scheduler import LinearScheduler
from edit_flows.guidance.data import (
    GuidanceDataset,
    ProductGroupBatchSampler,
    collate_guidance_records,
    make_guidance_record,
    load_guidance_dataset,
    sample_intermediate_states,
    save_guidance_dataset,
)
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


def _states():
    product = torch.tensor([
        [BOS_TOKEN, 4, 5, PAD_TOKEN],
        [BOS_TOKEN, 7, PAD_TOKEN, PAD_TOKEN],
    ])
    terminal = torch.tensor([
        [BOS_TOKEN, 4, 6, PAD_TOKEN],
        [BOS_TOKEN, 7, 8, 9],
    ])
    return product, terminal


def test_sample_intermediate_states_respects_time_endpoints():
    product, terminal = _states()
    torch.manual_seed(10)
    states = sample_intermediate_states(
        product, terminal, [0.0, 1.0], vocab_size=32,
        scheduler=LinearScheduler(),
    )
    assert states.shape[0] == 2
    assert torch.equal(states[0, :3], product[0, :3])
    assert torch.equal(states[1, :4], terminal[1, :4])
    assert (states[:, 0] == BOS_TOKEN).all()


def test_sample_intermediate_states_is_seed_reproducible():
    product, terminal = _states()
    torch.manual_seed(42)
    first = sample_intermediate_states(
        product, terminal, [0.5, 0.5], vocab_size=32,
        scheduler=LinearScheduler(),
    )
    torch.manual_seed(42)
    second = sample_intermediate_states(
        product, terminal, [0.5, 0.5], vocab_size=32,
        scheduler=LinearScheduler(),
    )
    assert torch.equal(first, second)


def test_guidance_records_round_trip_and_collate(tmp_path: Path):
    records = [
        make_guidance_record(
            product_tokens=[BOS_TOKEN, 4, 5],
            state_tokens=[BOS_TOKEN, 4],
            terminal_tokens=[BOS_TOKEN, 4, 6],
            time_step=0.5,
            reward=1.0,
            source_index=2,
            sample_index=0,
            time_index=50,
            sample_seed=11,
            coupling_seed=12,
        ),
        make_guidance_record(
            product_tokens=[BOS_TOKEN, 7],
            state_tokens=[BOS_TOKEN, 7, 8],
            terminal_tokens=[BOS_TOKEN, 7, 8, 9],
            time_step=0.25,
            reward=0.0,
            source_index=3,
            sample_index=1,
            time_index=25,
            sample_seed=13,
            coupling_seed=14,
        ),
    ]
    path = tmp_path / "guidance.pt"
    save_guidance_dataset(path, records, metadata={"split": "train"})
    loaded, metadata = load_guidance_dataset(path)
    assert loaded == records
    assert metadata == {"split": "train"}
    dataset = GuidanceDataset(path)
    assert len(dataset) == 2
    batch = collate_guidance_records([dataset[0], dataset[1]])
    assert batch["product_tokens"].shape == (2, 3)
    assert batch["state_tokens"].shape == (2, 3)
    assert batch["terminal_tokens"].shape == (2, 4)
    assert torch.equal(batch["reward"], torch.tensor([1.0, 0.0]))


def test_guidance_record_requires_bos_and_finite_reward():
    kwargs = dict(
        product_tokens=[BOS_TOKEN, 4],
        state_tokens=[BOS_TOKEN, 4],
        terminal_tokens=[BOS_TOKEN, 4],
        time_step=0.5,
        reward=1.0,
        source_index=0,
        sample_index=0,
        time_index=1,
        sample_seed=1,
        coupling_seed=2,
    )
    with pytest.raises(ValueError, match="product_tokens"):
        make_guidance_record(**{**kwargs, "product_tokens": [4]})
    with pytest.raises(ValueError, match="finite"):
        make_guidance_record(**{**kwargs, "reward": float("nan")})


def _group_records(group_count=3, group_size=4):
    return [
        {"source_index": source_index}
        for source_index in range(group_count)
        for _ in range(group_size)
    ]


def _batch_source_groups(batch, records):
    return {
        int(records[index]["source_index"])
        for index in batch
    }


def test_product_group_batch_sampler_keeps_complete_groups_and_tail():
    records = _group_records(group_count=3)
    sampler = ProductGroupBatchSampler(
        records, batch_size=8, group_size=4, shuffle=False,
    )
    batches = list(sampler)
    assert len(sampler) == 2
    assert [len(batch) for batch in batches] == [8, 4]
    assert all(len(_batch_source_groups(batch, records)) == 2 for batch in batches[:1])
    assert len(_batch_source_groups(batches[1], records)) == 1
    assert sorted(index for batch in batches for index in batch) == list(range(12))


def test_product_group_batch_sampler_shuffle_is_seed_and_epoch_reproducible():
    records = _group_records(group_count=8)
    first = ProductGroupBatchSampler(
        records, batch_size=8, group_size=4, seed=123,
    )
    second = ProductGroupBatchSampler(
        records, batch_size=8, group_size=4, seed=123,
    )
    assert list(first) == list(second)
    first.set_epoch(1)
    assert list(first) != list(second)
    second.set_epoch(1)
    assert list(first) == list(second)


def test_product_group_batch_sampler_drop_last_and_invalid_inputs():
    records = _group_records(group_count=3)
    sampler = ProductGroupBatchSampler(
        records, batch_size=8, group_size=4, drop_last=True,
    )
    assert len(sampler) == 1
    assert [len(batch) for batch in sampler] == [8]
    with pytest.raises(ValueError, match="divisible"):
        ProductGroupBatchSampler(records, batch_size=6, group_size=4)
    with pytest.raises(ValueError, match="exactly group_size"):
        ProductGroupBatchSampler(records[:-1], batch_size=8, group_size=4)
