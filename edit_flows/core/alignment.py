from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from edit_flows.utils.tokens import PAD_TOKEN, GAP_TOKEN


def identity_align_xs_to_zs(
    x_0: Tensor, x_1: Tensor
) -> Tuple[Tensor, Tensor]:
    return x_0, x_1


def _align_pair(
    seq_0: Tensor, seq_1: Tensor
) -> Tuple[List[int], List[int]]:
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

    return aligned_0[::-1], aligned_1[::-1]


def opt_align_xs_to_zs(
    x_0: Tensor, x_1: Tensor
) -> Tuple[Tensor, Tensor]:
    # ``collate_fn`` pads each side of a batch independently.  Strip those
    # PAD tokens before DP; otherwise padding is treated as a real symbol and
    # batches with different sequence lengths can produce invalid alignments.
    aligned_pairs = []
    for b in range(x_0.shape[0]):
        len_0 = int((x_0[b] != PAD_TOKEN).sum().item())
        len_1 = int((x_1[b] != PAD_TOKEN).sum().item())
        aligned_pairs.append(_align_pair(x_0[b, :len_0], x_1[b, :len_1]))

    max_z_len = max((len(pair[0]) for pair in aligned_pairs), default=0)
    z_0 = torch.full(
        (x_0.shape[0], max_z_len), PAD_TOKEN,
        dtype=x_0.dtype, device=x_0.device,
    )
    z_1 = torch.full(
        (x_1.shape[0], max_z_len), PAD_TOKEN,
        dtype=x_1.dtype, device=x_1.device,
    )
    for b, (aligned_0, aligned_1) in enumerate(aligned_pairs):
        z_0[b, :len(aligned_0)] = torch.tensor(
            aligned_0, dtype=x_0.dtype, device=x_0.device,
        )
        z_1[b, :len(aligned_1)] = torch.tensor(
            aligned_1, dtype=x_1.dtype, device=x_1.device,
        )
    return z_0, z_1


def naive_align_xs_to_zs(
    x_0: Tensor, x_1: Tensor
) -> Tuple[Tensor, Tensor]:
    max_len = max(x_0.shape[1], x_1.shape[1])
    z_0 = F.pad(x_0, (0, max_len - x_0.shape[1]), value=GAP_TOKEN)
    z_1 = F.pad(x_1, (0, max_len - x_1.shape[1]), value=GAP_TOKEN)
    return z_0, z_1


def shifted_align_xs_to_zs(
    x_0: Tensor, x_1: Tensor
) -> Tuple[Tensor, Tensor]:
    batch_size = x_0.shape[0]
    x0_lens = (x_0 != PAD_TOKEN).sum(dim=1)
    x1_lens = (x_1 != PAD_TOKEN).sum(dim=1)
    z_lens = x0_lens + x1_lens
    max_z_len = int(z_lens.max().item())

    z_0 = torch.full(
        (batch_size, max_z_len), GAP_TOKEN,
        dtype=x_0.dtype, device=x_0.device,
    )
    z_1 = torch.full(
        (batch_size, max_z_len), GAP_TOKEN,
        dtype=x_1.dtype, device=x_1.device,
    )

    batch_indices = torch.arange(batch_size, device=x_0.device).unsqueeze(1)

    for b in range(batch_size):
        l0 = int(x0_lens[b].item())
        l1 = int(x1_lens[b].item())
        z_0[b, :l0] = x_0[b, :l0]
        z_1[b, l0:l0 + l1] = x_1[b, :l1]
        z_0[b, l0 + l1:] = PAD_TOKEN
        z_1[b, l0 + l1:] = PAD_TOKEN

    return z_0, z_1
