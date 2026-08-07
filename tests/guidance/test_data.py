from pathlib import Path

import torch
import pytest

from edit_flows.core.scheduler import LinearScheduler
from edit_flows.guidance.data import (
    GuidanceDataset,
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
