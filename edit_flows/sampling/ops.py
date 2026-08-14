import torch
import torch.nn.functional as F
from torch import Tensor

from edit_flows.utils.tokens import BOS_TOKEN, GAP_TOKEN, PAD_TOKEN, UNK_TOKEN


# These IDs are model-internal structural symbols, not molecular output
# tokens.  Training targets never select them as INSERT/SUBSTITUTE outputs.
FORBIDDEN_OUTPUT_TOKEN_IDS = (PAD_TOKEN, BOS_TOKEN, GAP_TOKEN, UNK_TOKEN)
# Model padding paths use a large finite negative sentinel rather than -inf.
# Treat it as zero probability before conditional-Q renormalization, otherwise
# an all-masked row could be accidentally revived as a uniform distribution.
LOG_ZERO_CUTOFF = -1e8


def edit_position_masks(
    x_t: Tensor,
    *,
    pad_token: int = PAD_TOKEN,
) -> tuple[Tensor, Tensor]:
    """Return legal ``(insert, substitute/delete)`` position masks.

    Position 0 contains BOS.  It is immutable as a token, so substitution and
    deletion are forbidden there.  An insertion at position 0, however,
    means *insert immediately after BOS* in :func:`apply_ins_del_operations`.
    That is how a leading GAP in the aligned training target is represented,
    so it must remain a legal insertion anchor.
    """
    if x_t.ndim != 2:
        raise ValueError("x_t must have shape [batch, length]")
    non_pad = x_t != pad_token
    insert_positions = non_pad
    sub_del_positions = non_pad.clone()
    if sub_del_positions.shape[1] > 0:
        sub_del_positions[:, 0] = False
    return insert_positions, sub_del_positions


def legal_token_log_probs(
    log_probs: Tensor,
    *,
    current_tokens: Tensor | None = None,
    forbidden_token_ids: tuple[int, ...] = FORBIDDEN_OUTPUT_TOKEN_IDS,
) -> tuple[Tensor, Tensor]:
    """Restrict a token posterior to the training-supported action space.

    INSERT/SUBSTITUTE outputs may not be structural tokens.  For substitute,
    ``current_tokens`` additionally removes the identity/no-op token at each
    position.  The returned distribution is renormalized over the remaining
    legal tokens; ``log_normalizer`` is returned so callers that score sampled
    actions can use the same conditional probability.
    """
    if log_probs.ndim != 3:
        raise ValueError("log_probs must have shape [batch, length, vocab]")
    if current_tokens is not None and current_tokens.shape != log_probs.shape[:2]:
        raise ValueError(
            "current_tokens must have shape [batch, length] matching log_probs"
        )

    vocab_size = log_probs.shape[-1]
    allowed = torch.ones(vocab_size, dtype=torch.bool, device=log_probs.device)
    for token_id in forbidden_token_ids:
        if 0 <= int(token_id) < vocab_size:
            allowed[int(token_id)] = False

    masked = log_probs.masked_fill(log_probs <= LOG_ZERO_CUTOFF, float("-inf"))
    masked = masked.masked_fill(~allowed.view(1, 1, -1), float("-inf"))
    if current_tokens is not None:
        token_indices = current_tokens.clamp(0, vocab_size - 1).unsqueeze(-1)
        masked = masked.scatter(2, token_indices, float("-inf"))

    log_normalizer = torch.logsumexp(masked, dim=-1)
    has_legal_token = torch.isfinite(log_normalizer).unsqueeze(-1)
    normalized = masked - log_normalizer.unsqueeze(-1)
    normalized = torch.where(has_legal_token, normalized, masked)
    return normalized, log_normalizer


def apply_ins_del_operations(
    x_t: Tensor,
    ins_mask: Tensor,
    del_mask: Tensor,
    ins_tokens: Tensor,
    max_seq_len: int = 512,
    pad_token: int = PAD_TOKEN,
) -> Tensor:
    batch_size, seq_len = x_t.shape
    device = x_t.device

    replace_mask = ins_mask & del_mask
    x_t_modified = x_t.clone()
    x_t_modified[replace_mask] = ins_tokens[replace_mask]

    eff_ins_mask = ins_mask & ~replace_mask
    eff_del_mask = del_mask & ~replace_mask

    xt_pad_mask = x_t == pad_token
    xt_seq_lens = (~xt_pad_mask).sum(dim=1)
    new_lengths = xt_seq_lens + eff_ins_mask.sum(dim=1) - eff_del_mask.sum(dim=1)
    max_new_len = int(new_lengths.max().item())

    if max_new_len <= 0:
        return torch.full(
            (batch_size, 1), pad_token, dtype=x_t.dtype, device=device,
        )

    x_new = torch.full(
        (batch_size, max_new_len), pad_token, dtype=x_t.dtype, device=device,
    )

    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
    pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)

    cum_del = torch.cumsum(eff_del_mask.float(), dim=1)
    cum_ins = torch.cumsum(eff_ins_mask.float(), dim=1)
    cum_ins_before = F.pad(cum_ins[:, :-1], (1, 0), value=0)

    new_pos = pos_idx + cum_ins_before - cum_del
    keep_mask = ~eff_del_mask & (new_pos >= 0) & (new_pos < max_new_len)
    if keep_mask.any():
        x_new[
            batch_idx.expand(-1, seq_len)[keep_mask],
            new_pos[keep_mask].long(),
        ] = x_t_modified[keep_mask]

    if eff_ins_mask.any():
        ins_pos = new_pos + 1
        ins_valid = eff_ins_mask & (ins_pos >= 0) & (ins_pos < max_new_len)
        if ins_valid.any():
            x_new[
                batch_idx.expand(-1, seq_len)[ins_valid],
                ins_pos[ins_valid].long(),
            ] = ins_tokens[ins_valid]

    if max_new_len > max_seq_len:
        max_new_len = max_seq_len

    return x_new[:, :max_new_len]
