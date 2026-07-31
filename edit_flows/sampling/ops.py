import torch
import torch.nn.functional as F
from torch import Tensor

from edit_flows.utils.tokens import PAD_TOKEN


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
