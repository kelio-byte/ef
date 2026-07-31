import math

import pytest
import torch

from edit_flows.core.scheduler import LinearScheduler
from edit_flows.sampling.euler_beam import (
    _BranchState,
    _branch_sort_key,
    _step_log_p,
    sample_euler_beam,
)
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


def _scalar_triggered_only(actions, log_rates, log_ins, log_sub, adapt_h):
    rates = torch.exp(log_rates[0])
    result = 0.0
    eps = 1e-12
    for pos in actions["ins_mask"][0].nonzero(as_tuple=False).squeeze(-1).tolist():
        token = int(actions["ins_tokens"][0, pos])
        result += math.log(max(1.0 - math.exp(-adapt_h * rates[pos, 0].item()), eps))
        result += math.log(max(torch.exp(log_ins[0, pos, token]).item(), eps))
    for pos in actions["sub_mask"][0].nonzero(as_tuple=False).squeeze(-1).tolist():
        token = int(actions["sub_tokens"][0, pos])
        result += math.log(max(1.0 - math.exp(-adapt_h * rates[pos, 1].item()), eps))
        result += math.log(max(torch.exp(log_sub[0, pos, token]).item(), eps))
    for pos in actions["del_mask"][0].nonzero(as_tuple=False).squeeze(-1).tolist():
        result += math.log(max(1.0 - math.exp(-adapt_h * rates[pos, 2].item()), eps))
    return result


def _scalar_complete_step_log_p(actions, log_rates, log_ins, log_sub, adapt_h):
    rates = torch.exp(log_rates[0])
    result = 0.0
    eps = 1e-12
    for pos in range(rates.shape[0]):
        ins_rate, sub_rate, del_rate = rates[pos].tolist()
        if bool(actions["ins_mask"][0, pos]):
            token = int(actions["ins_tokens"][0, pos])
            result += math.log(max(1.0 - math.exp(-adapt_h * ins_rate), eps))
            result += math.log(max(torch.exp(log_ins[0, pos, token]).item(), eps))
        else:
            result -= adapt_h * ins_rate

        ds_rate = sub_rate + del_rate
        if bool(actions["sub_mask"][0, pos]):
            token = int(actions["sub_tokens"][0, pos])
            result += math.log(max(1.0 - math.exp(-adapt_h * ds_rate), eps))
            result += math.log(max(sub_rate / max(ds_rate, eps), eps))
            result += math.log(max(torch.exp(log_sub[0, pos, token]).item(), eps))
        elif bool(actions["del_mask"][0, pos]):
            result += math.log(max(1.0 - math.exp(-adapt_h * ds_rate), eps))
            result += math.log(max(del_rate / max(ds_rate, eps), eps))
        else:
            result -= adapt_h * ds_rate
    return result


def _random_actions(length, vocab_size):
    sub_mask = torch.rand(1, length) < 0.35
    del_mask = (torch.rand(1, length) < 0.25) & ~sub_mask
    return {
        "ins_mask": torch.rand(1, length) < 0.45,
        "sub_mask": sub_mask,
        "del_mask": del_mask,
        "ins_tokens": torch.randint(vocab_size, (1, length)),
        "sub_tokens": torch.randint(vocab_size, (1, length)),
    }


def test_step_log_p_matches_complete_scalar_implementation():
    torch.manual_seed(7)
    length, vocab_size = 13, 17
    log_rates = torch.randn(1, length, 3)
    log_ins = torch.log_softmax(torch.randn(1, length, vocab_size), dim=-1)
    log_sub = torch.log_softmax(torch.randn(1, length, vocab_size), dim=-1)
    actions = _random_actions(length, vocab_size)
    adapt_h = 0.037
    expected = _scalar_complete_step_log_p(
        actions, log_rates, log_ins, log_sub, adapt_h,
    )
    actual = _step_log_p(actions, log_rates, log_ins, log_sub, adapt_h)
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)


def test_step_log_p_no_events_includes_survival_probability():
    length, vocab_size = 5, 8
    actions = {
        "ins_mask": torch.zeros(1, length, dtype=torch.bool),
        "sub_mask": torch.zeros(1, length, dtype=torch.bool),
        "del_mask": torch.zeros(1, length, dtype=torch.bool),
        "ins_tokens": torch.zeros(1, length, dtype=torch.long),
        "sub_tokens": torch.zeros(1, length, dtype=torch.long),
    }
    adapt_h = 0.1
    result = _step_log_p(
        actions,
        torch.zeros(1, length, 3),
        torch.log_softmax(torch.zeros(1, length, vocab_size), dim=-1),
        torch.log_softmax(torch.zeros(1, length, vocab_size), dim=-1),
        adapt_h=adapt_h,
    )
    assert math.isclose(result, -length * 3 * adapt_h, abs_tol=1e-6)


def test_step_log_p_is_finite_for_extreme_rates():
    length, vocab_size = 4, 8
    for raw_log_rate in (-100.0, 20.0):
        log_rates = torch.full((1, length, 3), raw_log_rate)
        log_probs = torch.log_softmax(torch.randn(1, length, vocab_size), dim=-1)
        result = _step_log_p(
            _random_actions(length, vocab_size),
            log_rates, log_probs, log_probs, adapt_h=0.01,
        )
        assert math.isfinite(result)


def test_branch_sort_prefers_larger_log_probability_then_weight():
    high = _BranchState(torch.tensor([[BOS_TOKEN, 4]]), path_log_p=-2.0, weight=1.0)
    low = _BranchState(torch.tensor([[BOS_TOKEN, 5]]), path_log_p=-10.0, weight=100.0)
    assert max([low, high], key=_branch_sort_key) is high
    heavier = _BranchState(torch.tensor([[BOS_TOKEN, 6]]), path_log_p=-2.0, weight=2.0)
    assert max([high, heavier], key=_branch_sort_key) is heavier


class _StochasticModel(torch.nn.Module):
    def __init__(self, vocab_size=16):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, tokens, time_step, padding_mask, origin_mask=None):
        batch, length = tokens.shape
        log_rates = torch.full((batch, length, 3), -0.3, device=tokens.device)
        logits = torch.arange(
            self.vocab_size, dtype=torch.float, device=tokens.device,
        ).view(1, 1, -1).expand(batch, length, -1)
        log_probs = torch.log_softmax(logits / 4.0, dim=-1)
        mask = padding_mask.unsqueeze(-1)
        return (
            log_rates.masked_fill(mask, -1e9),
            log_probs.masked_fill(mask, -1e9),
            log_probs.masked_fill(mask, -1e9),
        )


def test_sample_euler_beam_same_seed_is_reproducible():
    model = _StochasticModel()
    x_0 = torch.tensor([
        [BOS_TOKEN, 4, 5, 6, PAD_TOKEN],
        [BOS_TOKEN, 7, 8, PAD_TOKEN, PAD_TOKEN],
    ])
    kwargs = dict(
        scheduler=LinearScheduler(), n_branches=3, n_steps=4,
        max_seq_len=32, base_seed=123,
    )
    assert torch.equal(
        sample_euler_beam(model, x_0, **kwargs),
        sample_euler_beam(model, x_0, **kwargs),
    )


def test_sample_euler_beam_different_seed_changes_output():
    model = _StochasticModel()
    x_0 = torch.tensor([[BOS_TOKEN, 4, 5, 6, 7, 8, PAD_TOKEN]])
    common = dict(
        scheduler=LinearScheduler(), n_branches=3, n_steps=6, max_seq_len=64,
    )
    first = sample_euler_beam(model, x_0, base_seed=100, **common)
    second = sample_euler_beam(model, x_0, base_seed=200, **common)
    assert not torch.equal(first, second)


def test_sample_euler_beam_one_branch_runs():
    model = _StochasticModel()
    x_0 = torch.tensor([
        [BOS_TOKEN, 4, 5, PAD_TOKEN],
        [BOS_TOKEN, 6, PAD_TOKEN, PAD_TOKEN],
    ])
    result = sample_euler_beam(
        model, x_0, LinearScheduler(), n_branches=1, n_steps=3,
        max_seq_len=32, base_seed=42,
    )
    assert result.shape[0] == 2


def test_sample_euler_beam_rejects_unsupported_origin_mask():
    model = _StochasticModel()
    x_0 = torch.tensor([[BOS_TOKEN, 4, PAD_TOKEN]])
    with pytest.raises(NotImplementedError, match="origin_mask"):
        sample_euler_beam(model, x_0, LinearScheduler(), use_origin_mask=True)


def test_sample_euler_beam_validates_sizes():
    model = _StochasticModel()
    x_0 = torch.tensor([[BOS_TOKEN, 4, PAD_TOKEN]])
    with pytest.raises(ValueError, match="n_branches"):
        sample_euler_beam(model, x_0, LinearScheduler(), n_branches=0)
    with pytest.raises(ValueError, match="n_steps"):
        sample_euler_beam(model, x_0, LinearScheduler(), n_steps=0)


def test_stateless_uniform_is_reproducible_distinct_and_order_independent():
    from edit_flows.sampling.euler_beam import _stateless_uniform

    seeds = torch.tensor([101, 202, 303], dtype=torch.int64)
    forward = _stateless_uniform(seeds, step=7, seq_len=64, stream=2, dtype=torch.float32)
    repeat = _stateless_uniform(seeds, step=7, seq_len=64, stream=2, dtype=torch.float32)
    reverse = _stateless_uniform(seeds.flip(0), step=7, seq_len=64, stream=2, dtype=torch.float32)
    other_stream = _stateless_uniform(seeds, step=7, seq_len=64, stream=3, dtype=torch.float32)

    assert torch.equal(forward, repeat)
    assert torch.equal(forward, reverse.flip(0))
    assert not torch.equal(forward[0], forward[1])
    assert not torch.equal(forward, other_stream)
    assert bool(((forward > 0) & (forward < 1)).all())


def test_stateless_uniform_has_reasonable_first_moments():
    from edit_flows.sampling.euler_beam import _stateless_uniform

    seeds = torch.arange(1, 20001, dtype=torch.int64)
    values = _stateless_uniform(
        seeds, step=11, seq_len=4, stream=0, dtype=torch.float64,
    )
    assert abs(values.mean().item() - 0.5) < 0.01
    assert abs(values.var().item() - (1.0 / 12.0)) < 0.005


def test_batch_step_log_p_matches_single_branch_wrapper():
    from edit_flows.sampling.euler_beam import _step_log_p, _step_log_p_batch

    torch.manual_seed(19)
    batch, length, vocab_size = 4, 9, 13
    log_rates = torch.randn(batch, length, 3)
    log_ins = torch.log_softmax(torch.randn(batch, length, vocab_size), dim=-1)
    log_sub = torch.log_softmax(torch.randn(batch, length, vocab_size), dim=-1)
    actions = _random_actions(length, vocab_size)
    actions = {key: value.expand(batch, -1).clone() for key, value in actions.items()}
    h = torch.tensor([[0.01], [0.03], [0.07], [0.11]])
    batch_result = _step_log_p_batch(actions, log_rates, log_ins, log_sub, h)
    scalar_result = torch.tensor([
        _step_log_p(
            {key: value[i:i + 1] for key, value in actions.items()},
            log_rates[i:i + 1], log_ins[i:i + 1], log_sub[i:i + 1],
            float(h[i, 0]),
        )
        for i in range(batch)
    ])
    assert torch.allclose(batch_result, scalar_result, rtol=1e-6, atol=1e-6)


def test_batch_apply_edits_matches_individual_application():
    from edit_flows.sampling.euler_beam import _apply_edits_batch

    x = torch.tensor([
        [1, 4, 5, 6, 0],
        [1, 7, 8, 0, 0],
        [1, 9, 10, 11, 12],
    ])
    actions = {
        "ins_mask": torch.tensor([
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, False, False],
        ]),
        "del_mask": torch.tensor([
            [False, False, True, False, False],
            [False, True, False, False, False],
            [False, False, False, True, False],
        ]),
        "sub_mask": torch.tensor([
            [False, False, False, True, False],
            [False, False, False, False, False],
            [False, True, False, False, False],
        ]),
        "ins_tokens": torch.tensor([
            [0, 13, 0, 0, 0],
            [0, 0, 14, 0, 0],
            [0, 0, 0, 0, 0],
        ]),
        "sub_tokens": torch.tensor([
            [0, 0, 0, 15, 0],
            [0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0],
        ]),
    }
    batch_result = _apply_edits_batch(x, actions, max_seq_len=32, pad_token=0)
    rows = []
    for i in range(x.shape[0]):
        row_actions = {key: value[i:i + 1] for key, value in actions.items()}
        rows.append(_apply_edits_batch(
            x[i:i + 1], row_actions, max_seq_len=32, pad_token=0,
        ))
    max_len = max(row.shape[1] for row in rows)
    expected = torch.zeros(len(rows), max_len, dtype=x.dtype)
    for i, row in enumerate(rows):
        expected[i, :row.shape[1]] = row
    assert torch.equal(batch_result, expected)


def test_sample_seeds_match_individual_runs_and_validate_length():
    model = _StochasticModel()
    x_single = torch.tensor([[BOS_TOKEN, 4, 5, 6, PAD_TOKEN]])
    x_batch = x_single.repeat(3, 1)
    seeds = [42, 1042, 2042]
    common = dict(
        scheduler=LinearScheduler(), n_branches=3, n_steps=4, max_seq_len=32,
    )
    batched = sample_euler_beam(
        model, x_batch, sample_seeds=seeds, **common,
    )
    individual_rows = [
        sample_euler_beam(model, x_single, base_seed=seed, **common)
        for seed in seeds
    ]
    max_len = max(row.shape[1] for row in individual_rows)
    individual = torch.full((len(seeds), max_len), PAD_TOKEN, dtype=torch.long)
    for i, row in enumerate(individual_rows):
        individual[i, :row.shape[1]] = row[0]
    assert torch.equal(batched, individual)

    with pytest.raises(ValueError, match="sample_seeds length"):
        sample_euler_beam(model, x_batch, sample_seeds=[42], **common)


def test_make_batch_is_product_major():
    from scripts.sample_retro import _make_batch

    batch = _make_batch([[4, 5], [7]], n_samples=3, pad_token=PAD_TOKEN)
    assert torch.equal(batch[0], batch[1])
    assert torch.equal(batch[1], batch[2])
    assert torch.equal(batch[3], batch[4])
    assert torch.equal(batch[4], batch[5])
    assert not torch.equal(batch[2], batch[3])


def test_legacy_score_mode_matches_triggered_only_reference():
    from edit_flows.sampling.euler_beam import (
        _legacy_branch_sort_key,
        _step_log_p_batch,
    )

    torch.manual_seed(29)
    batch, length, vocab_size = 3, 11, 15
    log_rates = torch.randn(batch, length, 3)
    log_ins = torch.log_softmax(torch.randn(batch, length, vocab_size), dim=-1)
    log_sub = torch.log_softmax(torch.randn(batch, length, vocab_size), dim=-1)
    actions = _random_actions(length, vocab_size)
    actions = {key: value.expand(batch, -1).clone() for key, value in actions.items()}
    h = torch.tensor([[0.01], [0.05], [0.1]])
    actual = _step_log_p_batch(
        actions, log_rates, log_ins, log_sub, h,
        score_mode="legacy_triggered_reverse",
    )
    expected = torch.tensor([
        _scalar_triggered_only(
            {key: value[i:i + 1] for key, value in actions.items()},
            log_rates[i:i + 1], log_ins[i:i + 1], log_sub[i:i + 1],
            float(h[i, 0]),
        )
        for i in range(batch)
    ])
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)

    a = _BranchState(torch.tensor([[BOS_TOKEN, 4]]), path_log_p=-2.0)
    b = _BranchState(torch.tensor([[BOS_TOKEN, 5]]), path_log_p=-10.0)
    assert max([a, b], key=_legacy_branch_sort_key) is b
