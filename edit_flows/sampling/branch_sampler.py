"""用途：执行每条独立采样运行中的 K=1、M 可配置分支转移。
输入：模型、当前状态、产品记忆、时间步及随机种子。
输出：每条运行最终保留的候选状态；n_runs 次运行由调用方展开。
"""

from dataclasses import dataclass
import torch
from torch import Tensor
from .common import get_adaptive_h
from .branch_sampler_helpers import (
    _mix_child_seed,
    _token_keys_batch,
    _sample_actions_per_branch,
    _apply_edits_batch,
    _select_k1m_child,
)
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN


def _adaptive_h_to_list(adapt_h):
    """Convert per-branch step sizes to host values for Python branch state."""
    return adapt_h.squeeze(-1).cpu().tolist()


@dataclass
class Branch:
    """作用：保存一条采样分支的状态。输入：状态、时间和随机种子。输出：可逐步更新的分支记录。"""

    x_t: Tensor
    t: float
    seed: int


@torch.inference_mode()
def sample_branches(
    model,
    x_0,
    scheduler,
    sample_seeds,
    product_memory,
    product_memory_padding_mask,
    n_steps=100,
    max_seq_len=96,
    n_children=2,
    changed_state_bonus=0.5,
    statistics=None,
    diagnostics=None,
):
    """作用：批量推进每条输入的 K=1、M 子候选分支。输入：模型、初始状态、调度器、种子、产品记忆及 M。输出：每条输入的最终 token 状态。

    独立运行数由 inference.predict 构造；本函数每步采样 M 个候选，按状态频数和变化奖励保留一个。
    """
    device = x_0.device
    batch_size = x_0.shape[0]
    if len(sample_seeds) != batch_size or n_steps < 1:
        raise ValueError("Invalid seed count or step count")
    if (
        not isinstance(n_children, int)
        or isinstance(n_children, bool)
        or n_children < 1
    ):
        raise ValueError("n_children must be a positive integer")
    origin_keys = _token_keys_batch(x_0, PAD_TOKEN, BOS_TOKEN)
    branch_keys = list(origin_keys)
    if diagnostics is not None:
        diagnostics.begin_batch()
    branches = [
        Branch(x_0[b : b + 1], 0.0, sample_seeds[b])
        for b in range(batch_size)
    ]
    nfe_counts = [0] * batch_size if statistics is not None else None
    model_calls = 0
    for step in range(n_steps):
        flat = [(b, s) for b, s in enumerate(branches) if s.t < 1.0]
        if not flat:
            break
        if nfe_counts is not None:
            model_calls += 1
            for b, _ in flat:
                nfe_counts[b] += 1
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
        parent_values = [i for i in range(len(flat)) for _ in range(n_children)]
        parent_indices = torch.tensor(parent_values, dtype=torch.long, device=device)
        x_children = x_batch.index_select(0, parent_indices)
        lr = rates.index_select(0, parent_indices)
        li = ins.index_select(0, parent_indices)
        ls = sub.index_select(0, parent_indices)
        hc = h.index_select(0, parent_indices)
        seed_values = [
            _mix_child_seed(s.seed, step, c)
            for _, s in flat
            for c in range(n_children)
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
        h_values = _adaptive_h_to_list(hc)
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
                )
            )
            child_keys[b].append(keys[i])
        parent_ids = [b for b, _ in flat]
        parent_keys = [branch_keys[b] for b in parent_ids]
        selected_keys = []
        for b, _ in flat:
            selected = _select_k1m_child(
                candidates[b], child_keys[b], origin_keys[b], changed_state_bonus
            )
            branches[b] = selected
            selected_key = next(
                key
                for candidate, key in zip(candidates[b], child_keys[b])
                if candidate is selected
            )
            branch_keys[b] = selected_key
            selected_keys.append(selected_key)
        if diagnostics is not None:
            diagnostics.record_step(
                step=step,
                parent_ids=parent_ids,
                parent_keys=parent_keys,
                selected_keys=selected_keys,
                x_batch=x_batch,
                log_rates=rates,
                log_ins_probs=ins,
                log_sub_probs=sub,
                actions=actions,
                n_children=n_children,
            )
    out_len = max(s.x_t.shape[1] for s in branches)
    out = torch.full((batch_size, out_len), PAD_TOKEN, dtype=torch.long, device=device)
    for row, state in enumerate(branches):
        out[row, : state.x_t.shape[1]] = state.x_t
    if nfe_counts is not None:
        statistics["model_calls"] = statistics.get("model_calls", 0) + model_calls
        statistics["nfe_total"] = statistics.get("nfe_total", 0) + sum(nfe_counts)
        histogram = statistics.setdefault("nfe_histogram", {})
        for count in nfe_counts:
            histogram[str(count)] = histogram.get(str(count), 0) + 1
    return out
