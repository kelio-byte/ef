"""Product-conditioned guidance adapter for the first DGM experiments."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from edit_flows.guidance.dgm import positive_guidance
from edit_flows.models.transformer import (
    PreNormEncoderLayer,
    SinusoidalTimeEmbedding,
    sinusoidal_position_encoding,
)


class ProductConditionedGuidance(nn.Module):
    """A small action-level DGM guidance network.

    The base Edit Flows model remains frozen and this module has its own
    product encoder and current-state encoder.  It emits positive weights for
    insert, substitute and delete actions at every current-state position.
    This is intentionally an action-level approximation; it does not claim to
    be the exact fixed-coordinate Z-space construction for variable-length
    Edit Flows.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        hidden_dim: int = 256,
        product_layers: int = 2,
        state_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "relu",
        pos_encoding_scale: bool = True,
        pad_token: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if hidden_dim < 1 or num_heads < 1:
            raise ValueError("hidden_dim and num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if product_layers < 1 or state_layers < 1:
            raise ValueError("product_layers and state_layers must be positive")
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")

        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.pos_encoding_scale = bool(pos_encoding_scale)
        self.pad_token = int(pad_token)

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.product_layers = nn.ModuleList([
            PreNormEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation=activation,
            )
            for _ in range(product_layers)
        ])
        self.state_layers = nn.ModuleList([
            PreNormEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation=activation,
            )
            for _ in range(state_layers)
        ])
        self.product_norm = nn.LayerNorm(hidden_dim)
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.insert_head = self._make_head(hidden_dim, vocab_size)
        self.substitute_head = self._make_head(hidden_dim, vocab_size)
        self.delete_head = self._make_head(hidden_dim, 1)
        self._init_weights()

    @staticmethod
    def _make_head(hidden_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _embed_tokens(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError(
                f"tokens must have shape [batch, length], got {tuple(tokens.shape)}"
            )
        if tokens.shape[1] > self.max_seq_len:
            raise ValueError(
                f"sequence length {tokens.shape[1]} exceeds max_seq_len "
                f"{self.max_seq_len}"
            )
        if tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.vocab_size):
            raise ValueError("tokens contain an id outside the guidance vocabulary")
        x = self.token_embedding(tokens)
        if self.pos_encoding_scale:
            x = x * math.sqrt(self.hidden_dim)
        pos = sinusoidal_position_encoding(
            tokens.shape[1], self.hidden_dim, tokens.device,
        )
        return x + pos.unsqueeze(0)

    @staticmethod
    def _masked_mean(x: Tensor, padding_mask: Tensor) -> Tensor:
        valid = (~padding_mask).unsqueeze(-1).to(dtype=x.dtype)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (x * valid).sum(dim=1) / denom

    def forward(
        self,
        product_tokens: Tensor,
        state_tokens: Tensor,
        time_step: Tensor,
        product_padding_mask: Tensor,
        state_padding_mask: Tensor,
        *,
        return_raw: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return positive ``(H_ins, H_sub, H_del)`` tensors.

        Shapes are ``[B, L_state, V]``, ``[B, L_state, V]`` and
        ``[B, L_state, 1]``.  Padding positions are retained for shape
        stability and must be masked by the caller when constructing actions.
        """
        if product_tokens.ndim != 2 or state_tokens.ndim != 2:
            raise ValueError("product_tokens and state_tokens must be rank-2")
        if product_tokens.shape[0] != state_tokens.shape[0]:
            raise ValueError("product and state batches must have the same size")
        batch_size = product_tokens.shape[0]
        if time_step.ndim == 2 and time_step.shape[1] == 1:
            time_step = time_step[:, 0]
        if time_step.ndim != 1 or time_step.shape[0] != batch_size:
            raise ValueError("time_step must have shape [batch] or [batch, 1]")
        if product_padding_mask.shape != product_tokens.shape:
            raise ValueError("product_padding_mask must match product_tokens")
        if state_padding_mask.shape != state_tokens.shape:
            raise ValueError("state_padding_mask must match state_tokens")
        if product_padding_mask.dtype != torch.bool or state_padding_mask.dtype != torch.bool:
            raise TypeError("padding masks must be boolean tensors")

        product = self._embed_tokens(product_tokens).transpose(0, 1)
        for layer in self.product_layers:
            product = layer(product, src_key_padding_mask=product_padding_mask)
        product = self.product_norm(product).transpose(0, 1)
        product_context = self.context_projection(
            self._masked_mean(product, product_padding_mask),
        )

        state = self._embed_tokens(state_tokens)
        time = self.time_embedding(time_step).unsqueeze(1)
        state = state + time + product_context.unsqueeze(1)
        state = state.transpose(0, 1)
        for layer in self.state_layers:
            state = layer(state, src_key_padding_mask=state_padding_mask)
        state = self.state_norm(state).transpose(0, 1)

        raw_insert = self.insert_head(state)
        raw_substitute = self.substitute_head(state)
        raw_delete = self.delete_head(state)
        if return_raw:
            return raw_insert, raw_substitute, raw_delete
        return (
            positive_guidance(raw_insert),
            positive_guidance(raw_substitute),
            positive_guidance(raw_delete),
        )
