import torch
from torch import Tensor

from edit_flows.core.rate_scale import get_rate_scale


def bregman_loss(
    log_ux_cat: Tensor,
    z_gap_mask: Tensor,
    z_pad_mask: Tensor,
    uz_mask: Tensor,
    t: Tensor,
    scheduler,
    use_rate_reparam: bool = False,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
) -> Tensor:
    from edit_flows.core.z_space import fill_gap_tokens_with_repeats_log

    ux_cat = torch.exp(log_ux_cat)
    sched_coeff = get_rate_scale(t, scheduler, clamp_max=clamp_max, clamp_kappa=clamp_kappa)
    log_uz_cat = fill_gap_tokens_with_repeats_log(log_ux_cat, z_gap_mask, z_pad_mask)
    if use_rate_reparam:
        u_tot = ux_cat.sum(dim=(1, 2))
        ce_term = (log_uz_cat * uz_mask).sum(dim=(1, 2))
        per_sample_loss = sched_coeff.squeeze(-1) * (u_tot - ce_term)
    else:
        u_tot = ux_cat.sum(dim=(1, 2))
        ce_term = (log_uz_cat * uz_mask * sched_coeff.unsqueeze(-1)).sum(dim=(1, 2))
        per_sample_loss = u_tot - ce_term

    loss = per_sample_loss.mean()
    return loss
