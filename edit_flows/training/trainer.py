from typing import Callable, Optional

import torch
from torch import Tensor

from edit_flows.core.coupling import Coupling
from edit_flows.core.rate_scale import apply_rate_parameterization, get_rate_scale
from edit_flows.core.scheduler import KappaScheduler
from edit_flows.core.z_space import rm_gap_tokens, make_ut_mask_from_z, sample_cond_zt, project_mask_z_to_x
from edit_flows.training.loss import bregman_loss
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN, GAP_TOKEN


def prepare_batch(
    x_0: Tensor,
    x_1: Tensor,
    scheduler: KappaScheduler,
    align_fn: Callable,
    model_vocab_size: int,
    bos_token: int = BOS_TOKEN,
    pad_token: int = PAD_TOKEN,
    use_origin_mask: bool = False,
) -> dict:
    B = x_0.shape[0]
    device = x_0.device

    z_0, z_1 = align_fn(x_0, x_1)

    B_z, L_z = z_0.shape
    seq_lens_0 = (z_0 != pad_token).sum(dim=1)
    seq_lens_1 = (z_1 != pad_token).sum(dim=1)
    max_sl = int(torch.maximum(seq_lens_0, seq_lens_1).max().item())
    new_L = max_sl + 1

    z_0_padded = torch.full((B_z, new_L), pad_token, dtype=z_0.dtype, device=device)
    z_1_padded = torch.full((B_z, new_L), pad_token, dtype=z_1.dtype, device=device)

    z_0_padded[:, 0] = bos_token
    z_1_padded[:, 0] = bos_token

    cols = torch.arange(L_z, device=device).unsqueeze(0).expand(B_z, -1)
    copy_0 = (cols < seq_lens_0.unsqueeze(1))[:, :max_sl]
    copy_1 = (cols < seq_lens_1.unsqueeze(1))[:, :max_sl]
    z_0_padded[:, 1:][copy_0] = z_0[:, :max_sl][copy_0]
    z_1_padded[:, 1:][copy_1] = z_1[:, :max_sl][copy_1]

    t = torch.rand(B, 1, device=device)

    z_t, pick_z1 = sample_cond_zt(z_0_padded, z_1_padded, t, model_vocab_size, scheduler, return_pick=True)
    x_t, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z_t)

    uz_mask = make_ut_mask_from_z(z_t, z_1_padded, vocab_size=model_vocab_size)

    batch = {
        "x_t": x_t,
        "x_pad_mask": x_pad_mask,
        "z_gap_mask": z_gap_mask,
        "z_pad_mask": z_pad_mask,
        "uz_mask": uz_mask,
        "t": t,
    }

    if use_origin_mask:
        # A token is "original" iff the current Z-state still carries the
        # source token at that position.  This stays true for unchanged
        # positions where z_0 == z_1, unlike using ~pick_z1 directly.
        origin_mask_z = (z_t == z_0_padded) & (z_0_padded != GAP_TOKEN)
        origin_mask = project_mask_z_to_x(
            origin_mask_z, z_gap_mask, z_pad_mask, x_t.shape,
        )
        batch["origin_mask"] = origin_mask

    return batch


def _forward_loss_and_metrics(
    model,
    batch_data: dict,
    scheduler: KappaScheduler,
    use_rate_reparam: bool = False,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
    time_input: str = "t",
) -> tuple[Tensor, dict]:
    """Run one forward pass and compute loss/rate diagnostics.

    Keeping this path shared by training and validation prevents TensorBoard
    validation curves from silently using a different objective than training.
    """
    if time_input not in {"t", "kappa"}:
        raise ValueError(f"Unsupported time_input: {time_input}")

    device = next(model.parameters()).device
    x_t = batch_data["x_t"].to(device)
    x_pad_mask = batch_data["x_pad_mask"].to(device)
    z_gap_mask = batch_data["z_gap_mask"].to(device)
    z_pad_mask = batch_data["z_pad_mask"].to(device)
    uz_mask = batch_data["uz_mask"].to(device)
    t = batch_data["t"].to(device)

    t_model = scheduler(t) if time_input == "kappa" else t

    origin_mask = batch_data.get("origin_mask")
    if origin_mask is not None:
        origin_mask = origin_mask.to(device)

    log_rates, log_ins_probs, log_sub_probs = model(
        x_t, t_model, x_pad_mask, origin_mask=origin_mask,
    )

    log_lambda_ins = log_rates[:, :, 0]
    log_lambda_sub = log_rates[:, :, 1]
    log_lambda_del = log_rates[:, :, 2]

    log_u_tia_ins = log_lambda_ins.unsqueeze(-1) + log_ins_probs
    log_u_tia_sub = log_lambda_sub.unsqueeze(-1) + log_sub_probs
    log_u_tia_del = log_lambda_del.unsqueeze(-1)
    log_ux_cat = torch.cat([log_u_tia_ins, log_u_tia_sub, log_u_tia_del], dim=-1)

    loss = bregman_loss(
        log_ux_cat=log_ux_cat,
        z_gap_mask=z_gap_mask,
        z_pad_mask=z_pad_mask,
        uz_mask=uz_mask,
        t=t,
        scheduler=scheduler,
        use_rate_reparam=use_rate_reparam,
        clamp_kappa=clamp_kappa,
        clamp_max=clamp_max,
    )

    with torch.no_grad():
        log_rates_eff = apply_rate_parameterization(
            log_rates, t, scheduler, use_rate_reparam=use_rate_reparam,
            clamp_kappa=clamp_kappa, clamp_max=clamp_max,
        )
        rates = torch.exp(log_rates_eff)
        u_ins = rates[:, :, 0].sum(dim=1).mean()
        u_del = rates[:, :, 2].sum(dim=1).mean()
        u_sub = rates[:, :, 1].sum(dim=1).mean()
        log_u_tia_ins_eff = log_rates_eff[:, :, 0].unsqueeze(-1) + log_ins_probs
        log_u_tia_sub_eff = log_rates_eff[:, :, 1].unsqueeze(-1) + log_sub_probs
        log_u_tia_del_eff = log_rates_eff[:, :, 2].unsqueeze(-1)
        log_ux_cat_eff = torch.cat(
            [log_u_tia_ins_eff, log_u_tia_sub_eff, log_u_tia_del_eff], dim=-1,
        )
        u_tot = torch.exp(log_ux_cat_eff).sum(dim=(1, 2)).mean()

    return loss, {
        "u_tot": u_tot.item(),
        "u_ins": u_ins.item(),
        "u_del": u_del.item(),
        "u_sub": u_sub.item(),
        # These aliases make the TensorBoard names explicit: each lambda is
        # the batch-mean sum of the corresponding per-position edit rate.
        "lambda_total": u_tot.item(),
        "lambda_ins": u_ins.item(),
        "lambda_del": u_del.item(),
        "lambda_sub": u_sub.item(),
        "t_mean": t.mean().item(),
        "kappa_mean": scheduler(t).mean().item(),
        "rate_scale_mean": get_rate_scale(
            t, scheduler, clamp_max=clamp_max, clamp_kappa=clamp_kappa,
        ).mean().item(),
        "rate_scale_max": get_rate_scale(
            t, scheduler, clamp_max=clamp_max, clamp_kappa=clamp_kappa,
        ).max().item(),
    }


def train_step(
    model,
    batch_data: dict,
    scheduler: KappaScheduler,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
    use_rate_reparam: bool = False,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
    time_input: str = "t",
) -> dict:
    """Run one optimizer update and return scalar diagnostics."""
    loss, metrics = _forward_loss_and_metrics(
        model, batch_data, scheduler,
        use_rate_reparam=use_rate_reparam,
        clamp_kappa=clamp_kappa,
        clamp_max=clamp_max,
        time_input=time_input,
    )
    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    return {"loss": loss.item(), **metrics}


@torch.no_grad()
def evaluate_step(
    model,
    batch_data: dict,
    scheduler: KappaScheduler,
    use_rate_reparam: bool = False,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
    time_input: str = "t",
) -> dict:
    """Compute training-objective metrics without changing model parameters."""
    loss, metrics = _forward_loss_and_metrics(
        model, batch_data, scheduler,
        use_rate_reparam=use_rate_reparam,
        clamp_kappa=clamp_kappa,
        clamp_max=clamp_max,
        time_input=time_input,
    )
    return {"loss": loss.item(), **metrics}
