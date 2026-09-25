"""用途：提供编辑采样共用的事件概率与步长计算。
输入：模型速率、调度器和当前采样时间。
输出：事件概率或自适应采样步长。
"""

from __future__ import annotations
import torch
from torch import Tensor
from edit_flows.core.scheduler import KappaScheduler


def _event_probability(mu: Tensor, mode: str) -> Tensor:
    if mode == "poisson":
        return 1 - torch.exp(-mu)
    if mode == "linear":
        return torch.clamp(mu, min=0.0, max=1.0)
    raise ValueError(f"Unsupported event_prob_mode: {mode}")


def get_adaptive_h(h: float, t: Tensor, scheduler: KappaScheduler) -> Tensor:
    coeff = (1 - scheduler(t)) / scheduler.derivative(t)
    h_adapt = torch.minimum(h * torch.ones_like(t), coeff)
    return h_adapt
