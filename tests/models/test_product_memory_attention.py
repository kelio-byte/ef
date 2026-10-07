import torch

from edit_flows.core.scheduler import CubicScheduler
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.sampling.euler import sample_euler
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


def _model() -> EditFlowsTransformer:
    return EditFlowsTransformer(
        vocab_size=24,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        dim_feedforward=32,
        max_seq_len=16,
        dropout=0.0,
        attention_dropout=0.0,
        use_product_memory=True,
        product_memory_encoder_layers=1,
        product_memory_fusion_after_layers=[1, 2],
    ).eval()


def test_product_memory_attention_capture_is_per_head_and_masks_padding():
    torch.manual_seed(37)
    model = _model()
    state = torch.tensor([
        [BOS_TOKEN, 4, 5, PAD_TOKEN],
        [BOS_TOKEN, 6, 7, 8],
    ])
    product = torch.tensor([
        [BOS_TOKEN, 4, 5, PAD_TOKEN],
        [BOS_TOKEN, 6, 7, 8],
    ])
    state_padding = state == PAD_TOKEN
    product_padding = product == PAD_TOKEN
    time = torch.tensor([[0.25], [0.75]])
    product_memory = model.encode_product(product, product_padding)

    baseline = model(
        state,
        time,
        state_padding,
        product_memory=product_memory,
        product_memory_padding_mask=product_padding,
    )
    captures = []

    def record(**payload):
        captures.append(payload)

    with model.capture_product_memory_attention(record):
        captured_output = model(
            state,
            time,
            state_padding,
            product_memory=product_memory,
            product_memory_padding_mask=product_padding,
        )

    for ordinary, captured in zip(baseline, captured_output):
        assert torch.allclose(ordinary, captured, atol=1e-6, rtol=1e-6)

    assert [item["layer_index"] for item in captures] == [1, 2]
    for capture in captures:
        weights = capture["attention_weights"]
        assert weights.shape == (2, 4, 4, 4)
        assert torch.equal(capture["state_tokens"], state)
        assert torch.equal(capture["state_padding_mask"], state_padding)
        assert torch.equal(
            capture["product_memory_padding_mask"], product_padding,
        )

        # Key padding must receive exactly zero probability.  Valid query rows
        # remain normalized independently for every attention head.
        assert torch.equal(weights[0, :, :, 3], torch.zeros_like(weights[0, :, :, 3]))
        valid_query_weights = weights[~state_padding].reshape(-1, weights.shape[-1])
        assert torch.allclose(
            valid_query_weights.sum(dim=-1),
            torch.ones(valid_query_weights.shape[0]),
            atol=1e-6,
            rtol=1e-6,
        )

    # The context manager must restore the ordinary forward path after use.
    assert model._product_memory_attention_callback is None


def test_product_memory_attention_capture_preserves_seeded_euler_sampling():
    model = _model()
    product = torch.tensor([
        [BOS_TOKEN, 4, 5, PAD_TOKEN],
        [BOS_TOKEN, 6, 7, PAD_TOKEN],
    ])

    torch.manual_seed(61)
    ordinary, _ = sample_euler(
        model,
        product,
        CubicScheduler(),
        n_steps=4,
        max_seq_len=12,
    )

    records = []
    torch.manual_seed(61)
    with model.capture_product_memory_attention(
        lambda **payload: records.append(payload),
    ):
        captured, _ = sample_euler(
            model,
            product,
            CubicScheduler(),
            n_steps=4,
            max_seq_len=12,
        )

    assert torch.equal(captured, ordinary)
    assert len(records) == 8  # two fusion layers for each of four model calls
