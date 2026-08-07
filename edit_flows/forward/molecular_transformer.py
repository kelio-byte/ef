"""Modern-PyTorch compatibility loader for the legacy Molecular Transformer.

The checkpoint in ``new_checkpoints/`` was produced by OpenNMT-py 0.4.1 and
serializes a ``torchtext.vocab.Vocab`` object.  Installing that old dependency
into the Edit Flows environment would replace the working PyTorch stack, so
this module reconstructs only the small inference subset needed for a forward
reaction score.  The layer names and operations intentionally follow the
OpenNMT implementation closely enough for a strict state-dict load.

The model direction is reactants/reagents -> product.  ``score_batch`` accepts
ordinary (un-tokenized) SMILES; Edit Flows' ``#global#`` candidates can first
be converted with :func:`retro_global_to_smiles`.
"""

from __future__ import annotations

import contextlib
import importlib
import math
import re
import sys
import types
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# This is the tokenizer published by the Molecular Transformer repository.
_SMI_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|"
    r"\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)


def smi_tokenize(smiles: str) -> list[str]:
    """Tokenize one ordinary SMILES string using the official regex."""

    if not isinstance(smiles, str):
        raise TypeError(f"SMILES must be str, got {type(smiles)!r}")
    tokens = _SMI_TOKEN_PATTERN.findall(smiles)
    if "".join(tokens) != smiles:
        raise ValueError(f"SMILES contains unsupported characters: {smiles!r}")
    return tokens


def retro_global_to_smiles(value: str) -> str:
    """Convert a space-tokenized Edit Flows ``#global#`` candidate to SMILES."""

    compact = "".join(value.split())
    if not compact:
        return ""
    try:
        from scripts.preprocessing.global_align import inverse_global_align
    except ImportError:  # pragma: no cover - script invocation path
        from preprocessing.global_align import inverse_global_align
    return inverse_global_align(compact)


class _LegacyVocabStub:
    """Unpickling target for ``torchtext.vocab.Vocab`` without torchtext."""


@contextlib.contextmanager
def _legacy_torchtext_stub():
    """Temporarily provide the one legacy class referenced by the checkpoint."""

    names = ("torchtext", "torchtext.vocab")
    previous = {name: sys.modules.get(name) for name in names}
    try:
        torchtext_module = types.ModuleType("torchtext")
        vocab_module = types.ModuleType("torchtext.vocab")
        vocab_module.Vocab = _LegacyVocabStub
        torchtext_module.vocab = vocab_module
        sys.modules["torchtext"] = torchtext_module
        sys.modules["torchtext.vocab"] = vocab_module
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_legacy_payload(path: str | Path) -> dict[str, Any]:
    """Load an old OpenNMT payload without installing torchtext."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        if exc.name != "torchtext":
            raise
        with _legacy_torchtext_stub():
            return torch.load(path, map_location="cpu", weights_only=False)


class _LegacyLayerNorm(nn.Module):
    """OpenNMT 0.4.1 LayerNorm (including its unbiased std convention)."""

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class _LegacyMultiHeadedAttention(nn.Module):
    def __init__(self, head_count: int, model_dim: int, dropout: float):
        super().__init__()
        if model_dim % head_count != 0:
            raise ValueError("model_dim must be divisible by head_count")
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim, model_dim)
        self.linear_values = nn.Linear(model_dim, model_dim)
        self.linear_query = nn.Linear(model_dim, model_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(
        self,
        key: Tensor,
        value: Tensor,
        query: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = key.size(0)
        heads = self.head_count
        depth = self.dim_per_head

        def shape(x: Tensor) -> Tensor:
            return x.view(batch_size, -1, heads, depth).transpose(1, 2)

        def unshape(x: Tensor) -> Tensor:
            return x.transpose(1, 2).contiguous().view(batch_size, -1, heads * depth)

        key = shape(self.linear_keys(key))
        value = shape(self.linear_values(value))
        query = shape(self.linear_query(query))
        query = query / math.sqrt(depth)
        scores = torch.matmul(query, key.transpose(2, 3))
        if mask is not None:
            mask = mask.to(dtype=torch.bool).unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)
        attn = self.softmax(scores)
        context = unshape(torch.matmul(self.dropout(attn), value))
        output = self.final_linear(context)
        top_attn = attn[:, 0, :, :].contiguous()
        return output, top_attn


class _LegacyPositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = _LegacyLayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        return self.dropout_2(self.w_2(inter)) + x


class _LegacyPositionalEncoding(nn.Module):
    def __init__(self, dropout: float, dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float)
            * (-(math.log(10000.0) / dim))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))
        self.dropout = nn.Dropout(dropout)
        self.dim = dim

    def forward(self, emb: Tensor) -> Tensor:
        emb = emb * math.sqrt(self.dim)
        return self.dropout(emb + self.pe[: emb.size(0)])


class _LegacyEmbeddings(nn.Module):
    """The shared word lookup + sinusoidal encoding used by OpenNMT."""

    def __init__(self, vocab_size: int, dim: int, dropout: float, pad_id: int):
        super().__init__()
        # Keep these names identical to OpenNMT's state_dict paths.
        self.make_embedding = nn.Module()
        self.make_embedding.emb_luts = nn.ModuleList(
            [nn.Embedding(vocab_size, dim, padding_idx=pad_id)]
        )
        self.make_embedding.pe = _LegacyPositionalEncoding(dropout, dim)
        self.word_padding_idx = pad_id

    def forward(self, source: Tensor) -> Tensor:
        # source: [length, batch, 1]
        emb = self.make_embedding.emb_luts[0](source[:, :, 0])
        return self.make_embedding.pe(emb)


class _LegacyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = _LegacyMultiHeadedAttention(heads, d_model, dropout)
        self.feed_forward = _LegacyPositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = _LegacyLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        return self.feed_forward(self.dropout(context) + inputs)


class _LegacyTransformerEncoder(nn.Module):
    def __init__(
        self,
        layers: int,
        d_model: int,
        heads: int,
        d_ff: int,
        dropout: float,
        vocab_size: int,
        pad_id: int,
    ):
        super().__init__()
        self.embeddings = _LegacyEmbeddings(vocab_size, d_model, dropout, pad_id)
        self.transformer = nn.ModuleList(
            [_LegacyTransformerEncoderLayer(d_model, heads, d_ff, dropout) for _ in range(layers)]
        )
        self.layer_norm = _LegacyLayerNorm(d_model)

    def forward(self, src: Tensor) -> tuple[Tensor, Tensor]:
        emb = self.embeddings(src)
        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        batch, length = words.size()
        mask = words.eq(self.embeddings.word_padding_idx).unsqueeze(1).expand(batch, length, length)
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)
        return emb, out.transpose(0, 1).contiguous()


class _LegacyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = _LegacyMultiHeadedAttention(heads, d_model, dropout)
        self.context_attn = _LegacyMultiHeadedAttention(heads, d_model, dropout)
        self.feed_forward = _LegacyPositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = _LegacyLayerNorm(d_model)
        self.layer_norm_2 = _LegacyLayerNorm(d_model)
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(1, 5000, 5000, dtype=torch.uint8), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(
        self,
        inputs: Tensor,
        memory_bank: Tensor,
        src_pad_mask: Tensor,
        tgt_pad_mask: Tensor,
    ) -> Tensor:
        length = tgt_pad_mask.size(1)
        dec_mask = torch.gt(
            tgt_pad_mask.to(dtype=torch.int64) + self.mask[:, :length, :length].to(dtype=torch.int64),
            0,
        )
        input_norm = self.layer_norm_1(inputs)
        query, _ = self.self_attn(input_norm, input_norm, input_norm, mask=dec_mask)
        query = self.drop(query) + inputs
        query_norm = self.layer_norm_2(query)
        mid, _ = self.context_attn(memory_bank, memory_bank, query_norm, mask=src_pad_mask)
        return self.feed_forward(self.drop(mid) + query)


class _LegacyTransformerDecoder(nn.Module):
    def __init__(
        self,
        layers: int,
        d_model: int,
        heads: int,
        d_ff: int,
        dropout: float,
        vocab_size: int,
        pad_id: int,
    ):
        super().__init__()
        self.decoder_type = "transformer"
        self.num_layers = layers
        self.embeddings = _LegacyEmbeddings(vocab_size, d_model, dropout, pad_id)
        self.self_attn_type = "scaled-dot"
        self.transformer_layers = nn.ModuleList(
            [_LegacyTransformerDecoderLayer(d_model, heads, d_ff, dropout) for _ in range(layers)]
        )
        self.layer_norm = _LegacyLayerNorm(d_model)

    def forward(self, tgt: Tensor, memory_bank: Tensor, src: Tensor) -> Tensor:
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()
        src_pad_mask = src_words.eq(self.embeddings.word_padding_idx).unsqueeze(1).expand(
            src_batch, tgt_len, src_len
        )
        tgt_pad_mask = tgt_words.eq(self.embeddings.word_padding_idx).unsqueeze(1).expand(
            tgt_batch, tgt_len, tgt_len
        )
        output = self.embeddings(tgt).transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()
        for layer in self.transformer_layers:
            output = layer(output, src_memory_bank, src_pad_mask, tgt_pad_mask)
        return self.layer_norm(output).transpose(0, 1).contiguous()


class _LegacyNMT(nn.Module):
    def __init__(self, vocab_size: int, opt: Any, pad_id: int):
        super().__init__()
        d_model = int(getattr(opt, "rnn_size", getattr(opt, "word_vec_size", 256)))
        d_ff = int(getattr(opt, "transformer_ff", 2048))
        heads = int(getattr(opt, "heads", 8))
        dropout = float(getattr(opt, "dropout", 0.1))
        enc_layers = int(getattr(opt, "enc_layers", getattr(opt, "layers", 4)))
        dec_layers = int(getattr(opt, "dec_layers", getattr(opt, "layers", 4)))
        self.encoder = _LegacyTransformerEncoder(
            enc_layers, d_model, heads, d_ff, dropout, vocab_size, pad_id
        )
        self.decoder = _LegacyTransformerDecoder(
            dec_layers, d_model, heads, d_ff, dropout, vocab_size, pad_id
        )

    def forward(self, src_ids: Tensor, tgt_input_ids: Tensor) -> Tensor:
        src = src_ids.transpose(0, 1).unsqueeze(-1).contiguous()
        tgt = tgt_input_ids.transpose(0, 1).unsqueeze(-1).contiguous()
        _, memory = self.encoder(src)
        outputs = self.decoder(tgt, memory, src)
        return outputs


class MolecularTransformerScorer:
    """Teacher-forced forward likelihood scorer backed by the legacy checkpoint."""

    def __init__(self, model: _LegacyNMT, generator: nn.Module, vocab: Sequence[str], device: torch.device):
        self.model = model.to(device).eval()
        self.generator = generator.to(device).eval()
        self.vocab = list(vocab)
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.device = device
        self.pad_id = self.stoi["<blank>"]
        self.bos_id = self.stoi["<s>"]
        self.eos_id = self.stoi["</s>"]
        self.unk_id = self.stoi["<unk>"]

    @torch.inference_mode()
    def score_batch(
        self,
        source_smiles: Sequence[str],
        target_smiles: Sequence[str],
        *,
        batch_size: int = 32,
        reduction: str = "mean",
    ) -> Tensor:
        """Return per-example teacher-forced log-likelihoods.

        ``source_smiles`` are reactants/reagents and ``target_smiles`` are
        products.  ``reduction='mean'`` divides by the number of target tokens
        including EOS, making scores less length-biased; ``sum`` returns the
        sequence log-likelihood.  Unknown source tokens are mapped to `<unk>`.
        """

        if len(source_smiles) != len(target_smiles):
            raise ValueError("source_smiles and target_smiles must have equal length")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        all_scores: list[Tensor] = []
        for start in range(0, len(source_smiles), batch_size):
            src_chunk = source_smiles[start : start + batch_size]
            tgt_chunk = target_smiles[start : start + batch_size]
            src_ids = [self._encode(s, add_bos_eos=False) for s in src_chunk]
            tgt_ids = [self._encode(s, add_bos_eos=True) for s in tgt_chunk]
            src = self._pad(src_ids)
            tgt = self._pad(tgt_ids)
            # Decoder consumes BOS ... final pre-EOS token; targets are next tokens.
            decoder_input = tgt[:, :-1]
            expected = tgt[:, 1:]
            hidden = self.model(src, decoder_input)
            log_probs = F.log_softmax(self.generator(hidden), dim=-1).transpose(0, 1)
            token_logp = log_probs.gather(-1, expected.unsqueeze(-1)).squeeze(-1)
            valid = expected.ne(self.pad_id)
            summed = (token_logp * valid).sum(dim=1)
            if reduction == "mean":
                summed = summed / valid.sum(dim=1).clamp_min(1)
            all_scores.append(summed.cpu())
        return torch.cat(all_scores) if all_scores else torch.empty(0)

    def _encode(self, smiles: str, *, add_bos_eos: bool) -> list[int]:
        ids = [self.stoi.get(token, self.unk_id) for token in smi_tokenize(smiles)]
        if add_bos_eos:
            return [self.bos_id, *ids, self.eos_id]
        return ids or [self.unk_id]

    def _pad(self, sequences: Sequence[Sequence[int]]) -> Tensor:
        max_len = max(len(sequence) for sequence in sequences)
        result = torch.full((len(sequences), max_len), self.pad_id, dtype=torch.long)
        for row, sequence in enumerate(sequences):
            result[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        return result.to(self.device)


def load_molecular_transformer(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> MolecularTransformerScorer:
    """Load the legacy checkpoint with a strict state-dict compatibility check."""

    payload = _load_legacy_payload(checkpoint)
    required = {"vocab", "model", "opt", "generator"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint missing keys: {sorted(missing)}")
    vocab_entries = dict(payload["vocab"])
    if "src" not in vocab_entries or "tgt" not in vocab_entries:
        raise ValueError("checkpoint must contain src and tgt vocabularies")
    src_vocab = vocab_entries["src"]
    tgt_vocab = vocab_entries["tgt"]
    if list(src_vocab.itos) != list(tgt_vocab.itos):
        raise ValueError("checkpoint is not using a shared source/target vocabulary")
    vocab = list(src_vocab.itos)
    opt = payload["opt"]
    pad_id = vocab.index("<blank>")
    model = _LegacyNMT(len(vocab), opt, pad_id)
    model_state = payload["model"]
    generator_state = payload["generator"]
    model.load_state_dict(model_state, strict=True)
    # OpenNMT stores the projection inside a one-layer Sequential, hence the
    # ``0.weight``/``0.bias`` state-dict keys.
    generator = nn.Sequential(nn.Linear(model.encoder.layer_norm.a_2.numel(), len(vocab)))
    generator.load_state_dict(generator_state, strict=True)
    target_device = torch.device(device)
    return MolecularTransformerScorer(model, generator, vocab, target_device)


__all__ = [
    "MolecularTransformerScorer",
    "load_molecular_transformer",
    "retro_global_to_smiles",
    "smi_tokenize",
]
