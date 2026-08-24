import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, t: Tensor) -> Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.hidden_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=t.device, dtype=t.dtype)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.hidden_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb


def sinusoidal_position_encoding(seq_len: int, hidden_dim: int, device: torch.device) -> Tensor:
    position = torch.arange(seq_len, device=device, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_dim, 2, device=device, dtype=torch.float)
        * (-math.log(10000.0) / hidden_dim)
    )
    pe = torch.zeros(seq_len, hidden_dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PreNormEncoderLayer(nn.Module):
    """Pre-Norm encoder layer with separate attention/FFN dropout and configurable activation."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=attention_dropout, batch_first=False,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, src: Tensor, src_key_padding_mask: Tensor = None) -> Tensor:
        x = src + self.dropout1(
            self.self_attn(
                self.norm1(src), self.norm1(src), self.norm1(src),
                key_padding_mask=src_key_padding_mask,
            )[0]
        )
        x = x + self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x)))))
        )
        return x


class PreNormCrossAttentionLayer(nn.Module):
    """Residual cross-attention from a dynamic edit state to static memory.

    The state remains the query sequence.  Product memory is only used as
    keys/values, so this layer never creates edit positions for the product
    sequence itself.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.state_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=attention_dropout,
            batch_first=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        state: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        normalized_memory = self.memory_norm(memory)
        attended = self.cross_attn(
            self.state_norm(state),
            normalized_memory,
            normalized_memory,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        return state + self.dropout(attended)


LOG_EPS = -1e9


def _log_softplus(x: Tensor, threshold: float = 20.0) -> Tensor:
    """log(F.softplus(x)) with truncation for numerical stability.

    For x <= -threshold:  softplus(x) ≈ exp(x), so log(softplus(x)) ≈ x.
    For x >  -threshold:  compute directly via log(F.softplus(x)).
    """
    result = torch.empty_like(x)
    safe = x > -threshold
    result[~safe] = x[~safe]
    sp = F.softplus(x[safe])
    result[safe] = torch.log(sp)
    return result


class EditFlowsTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "relu",
        pos_encoding_scale: bool = True,
        use_origin_mask: bool = False,
        use_product_memory: bool = False,
        product_memory_encoder_layers: int = 0,
        product_memory_fusion_after_layers: Sequence[int] | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.pos_encoding_scale = pos_encoding_scale
        self.use_product_memory = bool(use_product_memory)

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if use_origin_mask:
            self.origin_embedding = nn.Embedding(2, hidden_dim)
        else:
            self.origin_embedding = None

        self.layers = nn.ModuleList([
            PreNormEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(hidden_dim)

        if self.use_product_memory:
            if product_memory_encoder_layers <= 0:
                raise ValueError(
                    "use_product_memory=True requires "
                    "product_memory_encoder_layers >= 1"
                )
            if product_memory_fusion_after_layers is None:
                product_memory_fusion_after_layers = (num_layers,)
            fusion_after_layers = tuple(
                int(layer) for layer in product_memory_fusion_after_layers
            )
            if not fusion_after_layers:
                raise ValueError(
                    "use_product_memory=True requires at least one "
                    "product_memory_fusion_after_layers entry"
                )
            if (
                len(set(fusion_after_layers)) != len(fusion_after_layers)
                or any(layer < 1 or layer > num_layers for layer in fusion_after_layers)
            ):
                raise ValueError(
                    "product_memory_fusion_after_layers must contain unique "
                    f"1-based layer indices in [1, {num_layers}], got "
                    f"{fusion_after_layers}"
                )
            self.product_memory_encoder_layers = nn.ModuleList([
                PreNormEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation=activation,
                )
                for _ in range(product_memory_encoder_layers)
            ])
            self.product_memory_final_layer_norm = nn.LayerNorm(hidden_dim)
            self.product_memory_fusion_after_layers = fusion_after_layers
            self.product_memory_fusion_layers = nn.ModuleDict({
                str(layer): PreNormCrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for layer in fusion_after_layers
            })
        else:
            if product_memory_encoder_layers not in (0, None):
                raise ValueError(
                    "product_memory_encoder_layers requires "
                    "use_product_memory=True"
                )
            if product_memory_fusion_after_layers not in (None, (), []):
                raise ValueError(
                    "product_memory_fusion_after_layers requires "
                    "use_product_memory=True"
                )
            self.product_memory_encoder_layers = None
            self.product_memory_final_layer_norm = None
            self.product_memory_fusion_after_layers = ()
            self.product_memory_fusion_layers = None

        self.rates_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.ins_logits_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size),
        )
        self.sub_logits_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def encode_product(
        self,
        product_tokens: Tensor,
        product_padding_mask: Tensor,
    ) -> Tensor:
        """Encode the immutable initial product once for later reuse.

        This path deliberately omits the time embedding: it represents the
        original product ``x_0``, not a dynamic edit state.  Token embeddings
        are shared with the state encoder, while the contextual encoder is
        separate.
        """
        if not self.use_product_memory:
            raise RuntimeError("encode_product requires use_product_memory=True")
        if product_tokens.ndim != 2:
            raise ValueError(
                "product_tokens must have shape [batch, length], got "
                f"{tuple(product_tokens.shape)}"
            )
        if product_padding_mask.shape != product_tokens.shape:
            raise ValueError(
                "product_padding_mask must match product_tokens, got "
                f"{tuple(product_padding_mask.shape)} and "
                f"{tuple(product_tokens.shape)}"
            )
        batch_size, product_len = product_tokens.shape
        if product_len == 0:
            raise ValueError("product memory requires a non-empty x_0 sequence")

        token_emb = self.token_embedding(product_tokens)
        if self.pos_encoding_scale:
            token_emb = token_emb * math.sqrt(self.hidden_dim)
        pos_enc = sinusoidal_position_encoding(
            product_len, self.hidden_dim, product_tokens.device,
        )
        product = token_emb + pos_enc.unsqueeze(0).expand(batch_size, -1, -1)
        product = product.transpose(0, 1)

        for layer in self.product_memory_encoder_layers:
            product = layer(product, src_key_padding_mask=product_padding_mask)

        product = product.transpose(0, 1)
        return self.product_memory_final_layer_norm(product)

    def _resolve_product_memory(
        self,
        *,
        batch_size: int,
        product_tokens: Tensor | None,
        product_padding_mask: Tensor | None,
        product_memory: Tensor | None,
        product_memory_padding_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Validate or build static memory for a forward call."""
        supplied = (
            product_tokens,
            product_padding_mask,
            product_memory,
            product_memory_padding_mask,
        )
        if not self.use_product_memory:
            if any(value is not None for value in supplied):
                raise ValueError(
                    "product memory inputs were supplied but "
                    "use_product_memory=False"
                )
            return None, None

        if product_memory is None:
            if product_tokens is None or product_padding_mask is None:
                raise ValueError(
                    "use_product_memory=True requires either cached "
                    "product_memory/product_memory_padding_mask or "
                    "product_tokens/product_padding_mask"
                )
            product_memory = self.encode_product(
                product_tokens, product_padding_mask,
            )
            product_memory_padding_mask = product_padding_mask
        elif product_memory_padding_mask is None:
            raise ValueError(
                "cached product_memory requires product_memory_padding_mask"
            )

        if product_memory.ndim != 3:
            raise ValueError(
                "product_memory must have shape [batch, length, hidden], got "
                f"{tuple(product_memory.shape)}"
            )
        if product_memory.shape[0] != batch_size:
            raise ValueError(
                "product_memory batch size must match dynamic state, got "
                f"{product_memory.shape[0]} and {batch_size}"
            )
        if product_memory.shape[2] != self.hidden_dim:
            raise ValueError(
                "product_memory hidden dimension must match model hidden_dim, "
                f"got {product_memory.shape[2]} and {self.hidden_dim}"
            )
        if product_memory_padding_mask.shape != product_memory.shape[:2]:
            raise ValueError(
                "product_memory_padding_mask must match memory [batch, length]"
            )
        return product_memory, product_memory_padding_mask

    def forward(
        self,
        tokens: Tensor,
        time_step: Tensor,
        padding_mask: Tensor,
        origin_mask: Tensor = None,
        product_tokens: Tensor | None = None,
        product_padding_mask: Tensor | None = None,
        product_memory: Tensor | None = None,
        product_memory_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, seq_len = tokens.shape
        if seq_len == 0:
            rates = torch.empty(batch_size, 0, 3, device=tokens.device)
            ins = torch.empty(batch_size, 0, self.vocab_size, device=tokens.device)
            sub = torch.empty(batch_size, 0, self.vocab_size, device=tokens.device)
            return rates, ins, sub

        product_memory, product_memory_padding_mask = self._resolve_product_memory(
            batch_size=batch_size,
            product_tokens=product_tokens,
            product_padding_mask=product_padding_mask,
            product_memory=product_memory,
            product_memory_padding_mask=product_memory_padding_mask,
        )

        token_emb = self.token_embedding(tokens)
        if origin_mask is not None and self.origin_embedding is not None:
            token_emb = token_emb + self.origin_embedding(origin_mask.long())
        if self.pos_encoding_scale:
            token_emb = token_emb * math.sqrt(self.hidden_dim)

        time_emb = self.time_embedding(time_step)
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)

        pos_enc = sinusoidal_position_encoding(seq_len, self.hidden_dim, tokens.device)
        pos_emb = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)

        x = token_emb + time_emb + pos_emb
        x = x.transpose(0, 1)

        product_memory_t = (
            product_memory.transpose(0, 1)
            if product_memory is not None else None
        )
        for layer_index, layer in enumerate(self.layers, start=1):
            x = layer(x, src_key_padding_mask=padding_mask)
            if layer_index in self.product_memory_fusion_after_layers:
                x = self.product_memory_fusion_layers[str(layer_index)](
                    x,
                    product_memory_t,
                    memory_key_padding_mask=product_memory_padding_mask,
                )

        x = x.transpose(0, 1)
        x = self.final_layer_norm(x)

        ins_logits = self.ins_logits_out(x)
        sub_logits = self.sub_logits_out(x)

        log_rates = _log_softplus(self.rates_out(x))
        log_ins_probs = F.log_softmax(ins_logits, dim=-1)
        log_sub_probs = F.log_softmax(sub_logits, dim=-1)

        pad_mask_3d = padding_mask.unsqueeze(-1)
        log_rates = log_rates.masked_fill(pad_mask_3d, LOG_EPS)
        log_ins_probs = log_ins_probs.masked_fill(pad_mask_3d, LOG_EPS)
        log_sub_probs = log_sub_probs.masked_fill(pad_mask_3d, LOG_EPS)

        return log_rates, log_ins_probs, log_sub_probs
