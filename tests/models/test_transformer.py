import torch
import torch.nn as nn

from edit_flows.models.transformer import PreNormEncoderLayer


class ZeroAttention(nn.Module):
    def forward(self, query, key, value, key_padding_mask=None):
        return torch.zeros_like(query), None


class TestPreNormEncoderLayer:
    def test_zero_blocks_preserve_residual_input(self):
        layer = PreNormEncoderLayer(
            d_model=4,
            nhead=2,
            dim_feedforward=8,
            dropout=0.0,
            attention_dropout=0.0,
        )
        layer.self_attn = ZeroAttention()
        layer.linear1.weight.data.zero_()
        layer.linear1.bias.data.zero_()
        layer.linear2.weight.data.zero_()
        layer.linear2.bias.data.zero_()

        src = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0]],
             [[5.0, 6.0, 7.0, 8.0]]]
        )
        out = layer(src)

        assert torch.allclose(out, src)
