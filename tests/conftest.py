import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


V = 16
HIDDEN_DIM = 64
BATCH_SIZE = 4


@pytest.fixture
def vocab_size():
    return V + 3


@pytest.fixture
def batch_size():
    return BATCH_SIZE


@pytest.fixture
def device():
    return torch.device("cpu")


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4, batch_first=False,
            ),
            num_layers=2,
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.rates_head = nn.Linear(hidden_dim, 3)
        self.ins_head = nn.Linear(hidden_dim, vocab_size)
        self.sub_head = nn.Linear(hidden_dim, vocab_size)

    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.hidden_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=t.device, dtype=t.dtype)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.hidden_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, tokens, time_step, padding_mask, origin_mask=None):
        b, l = tokens.shape
        if l == 0:
            rates = torch.empty(b, 0, 3, device=tokens.device)
            ins_probs = torch.empty(b, 0, self.vocab_size, device=tokens.device)
            sub_probs = torch.empty(b, 0, self.vocab_size, device=tokens.device)
            return rates, ins_probs, sub_probs

        tok_emb = self.embed(tokens)
        t_emb = self._time_embed(time_step)
        t_emb_proj = self.time_mlp(t_emb).unsqueeze(1).expand(-1, l, -1)
        x = tok_emb + t_emb_proj
        x = x.transpose(0, 1)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = x.transpose(0, 1)
        x = self.ln(x)

        rates = F.softplus(self.rates_head(x))
        ins_probs = F.softmax(self.ins_head(x), dim=-1)
        sub_probs = F.softmax(self.sub_head(x), dim=-1)

        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        rates = rates * mask_expanded
        ins_probs = ins_probs * mask_expanded
        sub_probs = sub_probs * mask_expanded
        return rates, ins_probs, sub_probs


@pytest.fixture
def dummy_model(vocab_size):
    return DummyModel(vocab_size=vocab_size)


@pytest.fixture
def sample_sequences(vocab_size):
    torch.manual_seed(42)
    batch_size = 4
    lengths = [5, 8, 3, 6]
    max_len = max(lengths)
    x = torch.full((batch_size, max_len), 0, dtype=torch.long)
    for b, length in enumerate(lengths):
        x[b, :length] = torch.randint(3, vocab_size, (length,))
    return x
