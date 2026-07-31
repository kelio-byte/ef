import math

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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.pos_encoding_scale = pos_encoding_scale

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

    def forward(
        self,
        tokens: Tensor,
        time_step: Tensor,
        padding_mask: Tensor,
        origin_mask: Tensor = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, seq_len = tokens.shape
        if seq_len == 0:
            rates = torch.empty(batch_size, 0, 3, device=tokens.device)
            ins = torch.empty(batch_size, 0, self.vocab_size, device=tokens.device)
            sub = torch.empty(batch_size, 0, self.vocab_size, device=tokens.device)
            return rates, ins, sub

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

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)

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
