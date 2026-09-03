import torch

from edit_flows.guidance.sampling import apply_action_guidance


def _inputs():
    torch.manual_seed(4)
    log_rates = torch.randn(2, 3, 3)
    log_insert = torch.log_softmax(torch.randn(2, 3, 7), dim=-1)
    log_substitute = torch.log_softmax(torch.randn(2, 3, 7), dim=-1)
    h_insert = torch.rand(2, 3, 7) + 0.2
    h_substitute = torch.rand(2, 3, 7) + 0.2
    h_delete = torch.rand(2, 3, 1) + 0.2
    return (
        log_rates, log_insert, log_substitute,
        h_insert, h_substitute, h_delete,
    )


def test_beta_zero_is_exact_identity():
    values = _inputs()
    actual = apply_action_guidance(*values, beta=0.0)
    assert all(result is original for result, original in zip(actual, values[:3]))


def test_constant_guidance_is_exact_identity_after_rate_normalization():
    log_rates, log_insert, log_substitute, *_ = _inputs()
    h_insert = torch.full_like(log_insert, 7.0).exp()
    h_substitute = torch.full_like(log_substitute, 7.0).exp()
    h_delete = torch.full_like(log_rates[:, :, :1], 7.0).exp()
    actual = apply_action_guidance(
        log_rates, log_insert, log_substitute,
        h_insert, h_substitute, h_delete,
    )
    torch.testing.assert_close(actual[0], log_rates)
    torch.testing.assert_close(actual[1], log_insert)
    torch.testing.assert_close(actual[2], log_substitute)


def test_guidance_preserves_position_rate_and_normalizes_posteriors():
    values = _inputs()
    log_rates, log_insert, log_substitute = values[:3]
    actual = apply_action_guidance(*values, beta=1.0)
    guided_rates, guided_insert, guided_substitute = actual
    torch.testing.assert_close(
        guided_rates.exp().sum(dim=-1),
        log_rates.exp().sum(dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        guided_insert.exp().sum(dim=-1),
        torch.ones_like(log_insert[..., 0]),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        guided_substitute.exp().sum(dim=-1),
        torch.ones_like(log_substitute[..., 0]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_per_sample_guidance_preserves_editable_total_and_moves_rate():
    log_rates = torch.zeros(1, 3, 3)
    log_insert = torch.log_softmax(torch.zeros(1, 3, 2), dim=-1)
    log_substitute = torch.log_softmax(torch.zeros(1, 3, 2), dim=-1)
    h_insert = torch.ones(1, 3, 2)
    h_substitute = torch.ones(1, 3, 2)
    h_delete = torch.ones(1, 3, 1)
    h_insert[:, 2] = 9.0
    h_substitute[:, 2] = 9.0
    h_delete[:, 2] = 9.0
    editable = torch.tensor([[False, True, True]])

    guided_rates, _, _ = apply_action_guidance(
        log_rates,
        log_insert,
        log_substitute,
        h_insert,
        h_substitute,
        h_delete,
        rate_normalization="per_sample",
        position_mask=editable,
    )
    base = log_rates.exp().sum(dim=-1)
    guided = guided_rates.exp().sum(dim=-1)
    torch.testing.assert_close(guided[editable].sum(), base[editable].sum())
    assert guided[0, 2] > guided[0, 1]
    torch.testing.assert_close(guided[0, 0], base[0, 0])


def test_constant_guidance_is_identity_with_per_sample_normalization():
    values = _inputs()
    log_rates, log_insert, log_substitute = values[:3]
    constant = 4.0
    actual = apply_action_guidance(
        log_rates,
        log_insert,
        log_substitute,
        torch.full_like(log_insert, constant),
        torch.full_like(log_substitute, constant),
        torch.full_like(log_rates[:, :, :1], constant),
        rate_normalization="per_sample",
        position_mask=torch.tensor([[False, True, True], [False, True, False]]),
    )
    torch.testing.assert_close(actual[0], log_rates)
    torch.testing.assert_close(actual[1], log_insert)
    torch.testing.assert_close(actual[2], log_substitute)
