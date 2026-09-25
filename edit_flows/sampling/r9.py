"""用途：执行正式 R9K1M2 采样中的单步 K1M2 分支转移。
输入：模型、当前状态、产品记忆、时间步及随机种子。
输出：下一步候选状态及其采样信息。
"""

from dataclasses import dataclass
import math
import torch
from torch import Tensor
from .common import get_adaptive_h
from .r9_helpers import (
    _mix_child_seed,
    _token_keys_batch,
    _sample_actions_per_branch,
    _set_second_child_noop,
    _step_log_p_batch,
    _apply_edits_batch,
    _select_k1_m2_children,
)
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN


@dataclass
class Branch:
    x_t: Tensor
    t: float
    seed: int
    state_key: tuple
    log_mass: float = 0.0
    path_log_p: float = 0.0
    weight: float = 1.0


@torch.inference_mode()
def sample_r9(
    model,
    x_0,
    scheduler,
    sample_seeds,
    product_memory,
    product_memory_padding_mask,
    n_steps=100,
    max_seq_len=96,
):
    """Retain the original tensor batching, seed mixing and K1M2 selection."""
    device = x_0.device
    batch_size = x_0.shape[0]
    if len(sample_seeds) != batch_size or n_steps < 1:
        raise ValueError("Invalid seed count or step count")
    origin_keys = _token_keys_batch(x_0, PAD_TOKEN, BOS_TOKEN)
    branches = [
        Branch(x_0[b : b + 1], 0.0, sample_seeds[b], origin_keys[b])
        for b in range(batch_size)
    ]
    noop_step = min(n_steps - 1, int(0.9 * n_steps))
    for step in range(n_steps):
        flat = [(b, s) for b, s in enumerate(branches) if s.t < 1.0]
        if not flat:
            break
        branch_tensors = [s.x_t for _, s in flat]
        widths = [x.shape[1] for x in branch_tensors]
        max_l = max(widths)
        if all(width == max_l for width in widths):
            x_batch = torch.cat(branch_tensors, dim=0)
        else:
            x_batch = torch.full(
                (len(flat), max_l), PAD_TOKEN, dtype=torch.long, device=device
            )
            for i, tensor in enumerate(branch_tensors):
                x_batch[i, : tensor.shape[1]] = tensor
        t_vals = torch.tensor(
            [s.t for _, s in flat], dtype=torch.float, device=device
        ).unsqueeze(-1)
        parent_sample_indices = torch.tensor(
            [b for b, _ in flat], dtype=torch.long, device=device
        )
        memory = product_memory.index_select(0, parent_sample_indices)
        memory_mask = product_memory_padding_mask.index_select(0, parent_sample_indices)
        rates, ins, sub = model(
            x_batch,
            t_vals,
            x_batch == PAD_TOKEN,
            product_memory=memory,
            product_memory_padding_mask=memory_mask,
        )
        h = get_adaptive_h(1.0 / n_steps, t_vals, scheduler)
        parent_values = [i for i in range(len(flat)) for _ in range(2)]
        parent_indices = torch.tensor(parent_values, dtype=torch.long, device=device)
        x_children = x_batch.index_select(0, parent_indices)
        lr = rates.index_select(0, parent_indices)
        li = ins.index_select(0, parent_indices)
        ls = sub.index_select(0, parent_indices)
        hc = h.index_select(0, parent_indices)
        seed_values = [
            _mix_child_seed(s.seed, step, c) for _, s in flat for c in range(2)
        ]
        seeds = torch.tensor(seed_values, dtype=torch.int64, device=device)
        actions = _sample_actions_per_branch(
            seeds,
            x_children,
            lr,
            li,
            ls,
            hc,
            pad_token=PAD_TOKEN,
            event_prob_mode="poisson",
            step=step,
        )
        if step == noop_step:
            _set_second_child_noop(actions, 2)
        step_log_ps = (
            _step_log_p_batch(
                actions, lr, li, ls, hc, state_tokens=x_children, pad_token=PAD_TOKEN
            )
            .cpu()
            .tolist()
        )
        h_values = hc.squeeze(-1).cpu().tolist()
        x_next = _apply_edits_batch(x_children, actions, max_seq_len, PAD_TOKEN)
        keys = _token_keys_batch(x_next, PAD_TOKEN, BOS_TOKEN)
        candidates = {b: [] for b, _ in flat}
        child_keys = {b: [] for b, _ in flat}
        for i, parent_i in enumerate(parent_values):
            b, s = flat[parent_i]
            candidates[b].append(
                Branch(
                    x_next[i : i + 1],
                    s.t + h_values[i],
                    seed_values[i],
                    keys[i],
                    s.log_mass - math.log(2),
                    s.path_log_p + step_log_ps[i],
                    1.0,
                )
            )
            child_keys[b].append(keys[i])
        for b, _ in flat:
            branches[b] = _select_k1_m2_children(
                candidates[b], child_keys[b], origin_keys[b], 0.5
            )
    out_len = max(s.x_t.shape[1] for s in branches)
    out = torch.full((batch_size, out_len), PAD_TOKEN, dtype=torch.long, device=device)
    for row, state in enumerate(branches):
        out[row, : state.x_t.shape[1]] = state.x_t
    return out
