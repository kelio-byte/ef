import math

import torch
from torch import Tensor

from edit_flows.core.scheduler import KappaScheduler

LOG_EPS = -1e9


def get_rate_scale(
    t: Tensor,
    scheduler: KappaScheduler,
    clamp_max: float = 50.0,
    clamp_kappa: bool = False,
    eps: float = 1e-8,
) -> Tensor:
    kappa_t = scheduler(t)
    deriv_t = scheduler.derivative(t)
    if clamp_kappa:
        scale = deriv_t * torch.clamp(1.0 / (1.0 - kappa_t + eps), max=clamp_max)
    else:
        scale = torch.clamp(deriv_t / (1.0 - kappa_t + eps), max=clamp_max)
    return scale


def apply_rate_parameterization(
    log_base_rates: Tensor,
    t: Tensor,
    scheduler: KappaScheduler,
    use_rate_reparam: bool = False,
    clamp_max: float = 50.0,
    clamp_kappa: bool = False,
    log_eps: float = LOG_EPS,
) -> Tensor:
    if not use_rate_reparam:
        return log_base_rates

    scale = get_rate_scale(t, scheduler, clamp_max=clamp_max, clamp_kappa=clamp_kappa)
    log_scale = torch.log(scale.clamp_min(1e-12)).unsqueeze(1)
    return log_base_rates + log_scale
