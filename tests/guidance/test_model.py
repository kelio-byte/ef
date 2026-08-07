import torch

from edit_flows.guidance.model import ProductConditionedGuidance


def test_product_conditioned_guidance_shapes_and_positive_outputs():
    model = ProductConditionedGuidance(
        vocab_size=23,
        hidden_dim=32,
        product_layers=1,
        state_layers=1,
        num_heads=4,
        dim_feedforward=64,
        max_seq_len=16,
    )
    product = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]], dtype=torch.long)
    state = torch.tensor([[2, 7, 8, 0, 0], [5, 9, 10, 11, 0]], dtype=torch.long)
    product_pad = product.eq(0)
    state_pad = state.eq(0)
    time = torch.tensor([[0.2], [0.8]])

    h_ins, h_sub, h_del = model(product, state, time, product_pad, state_pad)
    assert h_ins.shape == (2, 5, 23)
    assert h_sub.shape == (2, 5, 23)
    assert h_del.shape == (2, 5, 1)
    for output in (h_ins, h_sub, h_del):
        assert torch.isfinite(output).all()
        assert (output > 0).all()


def test_guidance_model_backward_and_parameter_scale():
    model = ProductConditionedGuidance(
        vocab_size=31,
        hidden_dim=64,
        product_layers=1,
        state_layers=2,
        num_heads=8,
        dim_feedforward=128,
        max_seq_len=12,
    )
    product = torch.randint(1, 31, (3, 5))
    state = torch.randint(1, 31, (3, 6))
    product_pad = torch.zeros_like(product, dtype=torch.bool)
    state_pad = torch.zeros_like(state, dtype=torch.bool)
    time = torch.rand(3, 1)
    outputs = model(product, state, time, product_pad, state_pad)
    loss = sum(item.mean() for item in outputs)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert sum(parameter.numel() for parameter in model.parameters()) < 10_000_000
