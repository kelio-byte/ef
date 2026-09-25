"""用途：提供编辑采样共用的事件概率与步长计算。
输入：模型速率、调度器和当前采样时间。
输出：事件概率或自适应采样步长。
"""

from __future__ import annotations
import torch
from torch import Tensor
from edit_flows.core.scheduler import KappaScheduler


def _event_probability(mu: Tensor, mode: str) -> Tensor:
    """作用：将事件速率换算为事件概率。输入：累计速率和模式。输出：同形状概率张量。"""
    if mode == "poisson":
        return 1 - torch.exp(-mu)
    if mode == "linear":
        return torch.clamp(mu, min=0.0, max=1.0)
    raise ValueError(f"Unsupported event_prob_mode: {mode}")


def get_adaptive_h(h: float, t: Tensor, scheduler: KappaScheduler) -> Tensor:
    """作用：按调度曲线限制采样步长。输入：基础步长、当前时间和调度器。输出：逐样本步长张量。"""
    coeff = (1 - scheduler(t)) / scheduler.derivative(t)
    h_adapt = torch.minimum(h * torch.ones_like(t), coeff)
    return h_adapt
