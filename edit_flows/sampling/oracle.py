"""Oracle rate computation for Edit Flows generation.

At each Euler step, given the current state x_t and the known target x_1,
computes the theoretically optimal edit rates by dynamically aligning x_t
with x_1 and aggregating Z-space edit demands to X-space rates.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from edit_flows.core.scheduler import KappaScheduler
from edit_flows.core.rate_scale import get_rate_scale
from edit_flows.core.z_space import rm_gap_tokens, make_ut_mask_from_z
from edit_flows.utils.tokens import PAD_TOKEN, GAP_TOKEN, BOS_TOKEN

LOG_EPS = -1e9
SMALL_RATE = 1e-9
LOG_SMALL_RATE = -20.72  # log(1e-9)


def _align_pair(seq_0: Tensor, seq_1: Tensor) -> tuple[list[int], list[int], int]:
    """Levenshtein DP alignment, returns (aligned_0, aligned_1, edit_distance)."""
    seq_0_np = seq_0.cpu().numpy()
    seq_1_np = seq_1.cpu().numpy()
    m, n = len(seq_0_np), len(seq_1_np)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_0_np[i - 1] == seq_1_np[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    aligned_0, aligned_1 = [], []
    i, j = m, n
    while i or j:
        if i and j and seq_0_np[i - 1] == seq_1_np[j - 1]:
            aligned_0.append(int(seq_0_np[i - 1]))
            aligned_1.append(int(seq_1_np[j - 1]))
            i, j = i - 1, j - 1
        elif i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
            aligned_0.append(int(seq_0_np[i - 1]))
            aligned_1.append(int(seq_1_np[j - 1]))
            i, j = i - 1, j - 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            aligned_0.append(int(seq_0_np[i - 1]))
            aligned_1.append(GAP_TOKEN)
            i -= 1
        else:
            aligned_0.append(GAP_TOKEN)
            aligned_1.append(int(seq_1_np[j - 1]))
            j -= 1

    return aligned_0[::-1], aligned_1[::-1], dp[m][n]


def compute_oracle_log_ux_cat(
    uz_mask: Tensor,
    z_gap_mask: Tensor,
    z_pad_mask: Tensor,
    t: Tensor,
    scheduler: KappaScheduler,
    model_vocab_size: int,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
) -> Tensor:
    """Construct optimal X-space log-rates given Z-space edit demands.

    Mirrors scripts/oracle_loss_profile.py:compute_oracle_log_ux_cat.
    """
    B, L_z, n_ops = uz_mask.shape
    device = uz_mask.device

    sched_coeff = get_rate_scale(t, scheduler, clamp_max=clamp_max, clamp_kappa=clamp_kappa)

    non_gap_mask = ~z_gap_mask
    indices = non_gap_mask.long().cumsum(dim=1) - 1

    valid = ~z_gap_mask & ~z_pad_mask
    x_lens = valid.sum(dim=1)
    L_x = int(x_lens.max().item())
    if L_x == 0:
        L_x = 1

    indices = indices.clamp(min=0, max=L_x - 1)

    weighted = uz_mask.float() * sched_coeff.unsqueeze(-1)

    ux_cat = torch.full(
        (B, L_x, n_ops), SMALL_RATE, dtype=torch.float, device=device,
    )

    for b in range(B):
        idx = indices[b]
        keep = ~z_pad_mask[b]
        if keep.any():
            ux_cat[b].index_add_(0, idx[keep], weighted[b, keep])

    x_pad_mask = torch.arange(L_x, device=device).unsqueeze(0) >= x_lens.unsqueeze(1)
    ux_cat[x_pad_mask] = SMALL_RATE
    ux_cat = ux_cat.clamp(min=SMALL_RATE)

    return torch.log(ux_cat)


def compute_oracle_model_output(
    x_t: Tensor,
    x_1: Tensor,
    t: Tensor,
    scheduler: KappaScheduler,
    vocab_size: int,
    pad_token: int = PAD_TOKEN,
    bos_token: int = BOS_TOKEN,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Oracle replacement for model.forward() during Euler sampling.

    Args:
        x_t: current state in X space (B, L_t), includes BOS and PAD
        x_1: target in X space (B, L_1), includes BOS and PAD
        t: current time (B, 1)
        scheduler: kappa scheduler
        vocab_size: full model vocabulary size (including special tokens)
        pad_token: PAD token id
        bos_token: BOS token id

    Returns:
        (log_rates, log_ins_probs, log_sub_probs, edit_dists) in model output format,
        where edit_dists is a list of ints (edit distance per sample).
    """
    B = x_t.shape[0]
    device = x_t.device

    # 1. Align each pair (strip PAD, DP align, re-pad to uniform Z length)
    z_t_list, z_1_list, edit_dists = [], [], []
    for b in range(B):
        xt_b = x_t[b][x_t[b] != pad_token]
        x1_b = x_1[b][x_1[b] != pad_token]
        zt, z1, dist = _align_pair(xt_b, x1_b)
        z_t_list.append(torch.tensor(zt, dtype=torch.long, device=device))
        z_1_list.append(torch.tensor(z1, dtype=torch.long, device=device))
        edit_dists.append(dist)

    max_z_len = max(len(z) for z in z_t_list)
    z_t_padded = torch.full((B, max_z_len), pad_token, dtype=torch.long, device=device)
    z_1_padded = torch.full((B, max_z_len), pad_token, dtype=torch.long, device=device)
    for b in range(B):
        Lz = len(z_t_list[b])
        z_t_padded[b, :Lz] = z_t_list[b]
        z_1_padded[b, :Lz] = z_1_list[b]

    # 2. Compute uz_mask: which Z positions need which edits
    uz_mask = make_ut_mask_from_z(z_t_padded, z_1_padded, vocab_size=vocab_size)

    # 3. rm_gap_tokens: extract X-space structure from z_t
    x_t_aligned, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z_t_padded)

    # 4. Compute oracle rates in X space
    log_ux_cat = compute_oracle_log_ux_cat(
        uz_mask, z_gap_mask, z_pad_mask, t, scheduler, vocab_size,
        clamp_kappa=clamp_kappa, clamp_max=clamp_max,
    )
    # log_ux_cat: (B, L_x, 2*V+1), channels: [V ins, V sub, 1 del]

    L_x = log_ux_cat.shape[1]
    log_ux_cat = log_ux_cat.to(device=device, dtype=torch.float)

    # 5. Split into model output format
    # log_ux_cat[:, :, :V] → insertion log-rates per token
    # log_ux_cat[:, :, V:2V] → substitution log-rates per token
    # log_ux_cat[:, :, 2V] → deletion log-rate
    log_ins_rates = log_ux_cat[:, :, :vocab_size]        # (B, L_x, V)
    log_sub_rates = log_ux_cat[:, :, vocab_size:2*vocab_size]  # (B, L_x, V)
    log_del_rate = log_ux_cat[:, :, 2*vocab_size]         # (B, L_x)

    # Aggregate total rates per edit type (logsumexp for numerical stability)
    log_lambda_ins = torch.logsumexp(log_ins_rates, dim=-1)   # (B, L_x)
    log_lambda_sub = torch.logsumexp(log_sub_rates, dim=-1)   # (B, L_x)
    log_lambda_del = log_del_rate                              # (B, L_x)

    # Token probabilities: log_softmax of per-token rates
    log_ins_probs = F.log_softmax(log_ins_rates, dim=-1)      # (B, L_x, V)
    log_sub_probs = F.log_softmax(log_sub_rates, dim=-1)      # (B, L_x, V)

    # Stack rates into (B, L_x, 3)
    log_rates = torch.stack([log_lambda_ins, log_lambda_sub, log_lambda_del], dim=-1)

    # 6. Pad/crop to match x_t's sequence length for Euler sampler compatibility
    L_t = x_t.shape[1]
    if L_x < L_t:
        pad_len = L_t - L_x
        log_rates = F.pad(log_rates, (0, 0, 0, pad_len), value=LOG_EPS)
        log_ins_probs = F.pad(log_ins_probs, (0, 0, 0, pad_len), value=LOG_EPS)
        log_sub_probs = F.pad(log_sub_probs, (0, 0, 0, pad_len), value=LOG_EPS)
    elif L_x > L_t:
        log_rates = log_rates[:, :L_t, :]
        log_ins_probs = log_ins_probs[:, :L_t, :]
        log_sub_probs = log_sub_probs[:, :L_t, :]

    # 7. Mask PAD positions (use x_t's pad mask for consistency)
    xt_pad_mask = x_t == pad_token
    log_rates[xt_pad_mask] = LOG_EPS
    log_ins_probs[xt_pad_mask] = LOG_EPS
    log_sub_probs[xt_pad_mask] = LOG_EPS

    return log_rates, log_ins_probs, log_sub_probs, edit_dists
