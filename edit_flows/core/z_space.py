"""用途：处理含 GAP 的对齐状态及其编辑流掩码。
输入：对齐后的 token 状态和调度时间。
输出：去 GAP 序列、掩码或按条件采样的中间状态。
"""

import torch
from torch import Tensor

from edit_flows.utils.tokens import PAD_TOKEN, GAP_TOKEN


def rm_gap_tokens(
    z: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """作用：从对齐状态中移除 GAP。输入：含 GAP、PAD 的状态批次。输出：去 GAP 序列及三种掩码。"""
    B, L = z.shape
    device = z.device

    valid = (z != PAD_TOKEN) & (z != GAP_TOKEN)
    counts = valid.sum(dim=1)
    max_len = int(counts.max().item()) if B > 0 else 0

    if max_len == 0:
        x = z.new_zeros((B, 0)).long()
        x_pad_mask = torch.empty((B, 0), dtype=torch.bool, device=device)
        z_gap_mask = z == GAP_TOKEN
        z_pad_mask = z == PAD_TOKEN
        return x, x_pad_mask, z_gap_mask, z_pad_mask

    x = torch.full((B, max_len), PAD_TOKEN, dtype=z.dtype, device=device)

    dest_col = valid.long().cumsum(dim=1) - 1

    row_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, L)
    x[row_idx[valid], dest_col[valid]] = z[valid]

    x_pad_mask = x == PAD_TOKEN
    z_gap_mask = z == GAP_TOKEN
    z_pad_mask = z == PAD_TOKEN
    return x, x_pad_mask, z_gap_mask, z_pad_mask


def fill_gap_tokens_with_repeats_log(
    log_x_ut: Tensor,
    z_gap_mask: Tensor,
    z_pad_mask: Tensor,
    log_eps: float = -1e9,
) -> Tensor:
    """作用：将非 GAP 位置的对数分布扩展回对齐长度。输入：分布与 GAP/PAD 掩码。输出：扩展后的对数分布。"""
    batch_size = z_gap_mask.shape[0]
    x_seq_len = log_x_ut.shape[1]

    non_gap_mask = ~z_gap_mask
    indices = non_gap_mask.cumsum(dim=1) - 1
    indices = indices.clamp(min=0, max=x_seq_len - 1)

    batch_indices = torch.arange(batch_size, device=log_x_ut.device).unsqueeze(1)
    result = log_x_ut[batch_indices, indices]
    result[z_pad_mask] = log_eps
    return result


def make_ut_mask_from_z(
    z_t: Tensor,
    z_1: Tensor,
    vocab_size: int,
) -> Tensor:
    """作用：标记目标相对当前状态所需的编辑操作。输入：中间/目标状态和词表大小。输出：插入、替换、删除操作掩码。"""
    batch_size, z_seq_len = z_t.shape
    n_ops = 2 * vocab_size + 1

    z_neq = (z_t != z_1) & (z_t != PAD_TOKEN) & (z_1 != PAD_TOKEN)
    z_ins = (z_t == GAP_TOKEN) & (z_1 != GAP_TOKEN) & z_neq
    z_del = (z_t != GAP_TOKEN) & (z_1 == GAP_TOKEN) & z_neq
    z_sub = z_neq & ~z_ins & ~z_del

    u_mask = torch.zeros(
        (batch_size, z_seq_len, n_ops),
        dtype=torch.bool,
        device=z_t.device,
    )
    u_mask[z_ins, z_1[z_ins]] = True
    u_mask[z_sub, z_1[z_sub] + vocab_size] = True
    u_mask[:, :, -1][z_del] = True

    return u_mask


def sample_cond_zt(
    z_0: Tensor,
    z_1: Tensor,
    t: Tensor,
    vocab_size: int,
    kappa_fn,
    return_pick: bool = False,
) -> Tensor:
    """作用：按条件桥采样中间对齐状态。输入：源、目标、时间和 κ 调度器。输出：采样状态，可选返回目标选择掩码。"""
    kappa_t = kappa_fn(t)
    rand = torch.rand_like(z_0, dtype=torch.float)
    pick_z1 = rand < kappa_t
    z_t = torch.where(pick_z1, z_1, z_0)
    if return_pick:
        return z_t, pick_z1
    return z_t
