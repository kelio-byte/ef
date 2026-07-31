#!/usr/bin/env python
"""Oracle loss distribution analysis.

Computes the theoretically optimal model output (rates that minimize the
Bregman divergence given the data) and evaluates the resulting loss
distribution.  This separates inherent method variance (from t-sampling,
z_t-sampling, and alignment structure) from model-specific issues.
"""

import argparse
import os
import sys

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from edit_flows.data.dataset import (
    PreAlignedDataset, RetroDataset, load_vocab, collate_fn,
)
from edit_flows.core.scheduler import CubicScheduler
from edit_flows.core.alignment import (
    opt_align_xs_to_zs, identity_align_xs_to_zs,
)
from edit_flows.core.z_space import fill_gap_tokens_with_repeats
from edit_flows.training.trainer import prepare_batch
from edit_flows.training.loss import bregman_loss
from edit_flows.utils.tokens import PAD_TOKEN


LOG_EPS = -1e9
SMALL_RATE = 1e-9  # near-zero rate for unused edit channels
LOG_SMALL_RATE = -20.72  # log(1e-9)


def compute_oracle_log_ux_cat(
    uz_mask: Tensor,
    z_gap_mask: Tensor,
    z_pad_mask: Tensor,
    t: Tensor,
    scheduler: CubicScheduler,
    model_vocab_size: int,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
) -> Tensor:
    """Construct the optimal X-space log-rates that minimize the Bregman loss.

    For each X position and edit channel c, optimal u = K_c * sched_coeff,
    where K_c is the number of Z positions mapping to this X position that
    need edit channel c.  Unused channels → tiny rate.

    The Z→X mapping mirrors fill_gap_tokens_with_repeats: each Z position
    inherits the rate vector of the nearest preceding non-GAP position.
    """
    B, L_z, n_ops = uz_mask.shape  # n_ops = 2*V + 1
    device = uz_mask.device

    # Scheduler coefficient: κ̇ / (1-κ), clamped as in training.
    from edit_flows.core.rate_scale import get_rate_scale
    sched_coeff = get_rate_scale(t, scheduler, clamp_max=clamp_max, clamp_kappa=clamp_kappa)

    # --- Z → X mapping (mirrors fill_gap_tokens_with_repeats) ---
    # non_gap_mask includes PAD positions (PAD != GAP), matching the
    # original implementation.  PAD positions get clamped + zeroed.
    non_gap_mask = ~z_gap_mask  # (B, L_z)
    indices = non_gap_mask.long().cumsum(dim=1) - 1  # (B, L_z)

    # Correct L_x: number of real X tokens (non-GAP, non-PAD), same as
    # rm_gap_tokens uses.  Take max across batch for padding.
    valid = ~z_gap_mask & ~z_pad_mask  # (B, L_z)
    x_lens = valid.sum(dim=1)  # (B,)
    L_x = int(x_lens.max().item())
    if L_x == 0:
        L_x = 1  # guard against degenerate empty sequences

    indices = indices.clamp(min=0, max=L_x - 1)  # (B, L_z)

    # --- Aggregate Z demands → X ---
    # weighted[b, i, c] = sched_coeff[b] if Z position i needs edit c
    weighted = uz_mask.float() * sched_coeff.unsqueeze(-1)  # (B, L_z, n_ops)

    ux_cat = torch.full(
        (B, L_x, n_ops), SMALL_RATE, dtype=torch.float, device=device,
    )

    # Scatter-add: accumulate weighted demands onto X positions
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, L_z)
    # Use index_add for correct gradient-free accumulation
    for b in range(B):
        idx = indices[b]  # (L_z,)
        # Only aggregate from non-PAD Z positions
        keep = ~z_pad_mask[b]
        if keep.any():
            ux_cat[b].index_add_(0, idx[keep], weighted[b, keep])

    # X positions that received no demand stay at SMALL_RATE;
    # pad-only positions are zeroed below.
    x_pad_mask = torch.arange(L_x, device=device).unsqueeze(0) >= x_lens.unsqueeze(1)
    ux_cat[x_pad_mask] = SMALL_RATE

    ux_cat = ux_cat.clamp(min=SMALL_RATE)
    return torch.log(ux_cat)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze oracle loss distribution")
    parser.add_argument("--config", type=str, default="configs/retro.yaml")
    parser.add_argument("--num_batches", type=int, default=200,
                        help="Number of batches to sample")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    cfg = config["retro"]
    device = torch.device(args.device)

    data_dir = cfg["data_dir"]
    vocab_path = os.path.join(data_dir, cfg["vocab_file"])
    token2id, model_vocab = load_vocab(vocab_path)

    # Use pre-aligned data if available
    train_aligned_src = os.path.join(data_dir, "train", "train_aligned_src.txt")
    train_aligned_tgt = os.path.join(data_dir, "train", "train_aligned_tgt.txt")
    if os.path.exists(train_aligned_src) and os.path.exists(train_aligned_tgt):
        dataset = PreAlignedDataset(train_aligned_src, train_aligned_tgt, token2id)
        align_fn = identity_align_xs_to_zs
        print(f"Using pre-aligned data: {len(dataset):,} pairs")
    else:
        dataset = RetroDataset(
            src_path=os.path.join(data_dir, "train", "src-train.txt"),
            tgt_path=os.path.join(data_dir, "train", "tgt-train.txt"),
            token2id=token2id,
        )
        align_fn = opt_align_xs_to_zs
        print(f"Using on-the-fly DP alignment: {len(dataset):,} pairs")

    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )

    scheduler = CubicScheduler()

    losses = []
    u_tots = []
    ce_terms = []
    sched_coeffs = []
    n_edits_list = []

    print(f"Sampling {args.num_batches} batches (batch_size={cfg['batch_size']})...")
    clamp_kappa = cfg.get("clamp_kappa", False)
    clamp_max = cfg.get("clamp_max", 50.0)
    for batch_idx, (x_0, x_1) in enumerate(loader):
        if batch_idx >= args.num_batches:
            break

        batch = prepare_batch(
            x_0, x_1, scheduler, align_fn,
            model_vocab_size=model_vocab,
        )

        t = batch["t"]
        uz_mask = batch["uz_mask"]
        z_gap_mask = batch["z_gap_mask"]
        z_pad_mask = batch["z_pad_mask"]

        # Compute optimal X-space log-rates
        log_ux_cat = compute_oracle_log_ux_cat(
            uz_mask, z_gap_mask, z_pad_mask, t, scheduler, model_vocab,
            clamp_kappa=clamp_kappa, clamp_max=clamp_max,
        )

        # Compute Bregman loss with oracle rates (per-sample, before mean)
        with torch.no_grad():
            ux_cat = torch.exp(log_ux_cat)
            u_tot = ux_cat.sum(dim=(1, 2))

            from edit_flows.core.z_space import fill_gap_tokens_with_repeats_log
            log_uz_cat = fill_gap_tokens_with_repeats_log(
                log_ux_cat, z_gap_mask, z_pad_mask,
            )
            from edit_flows.core.rate_scale import get_rate_scale as _get_rate_scale
            sc = _get_rate_scale(t, scheduler, clamp_max=clamp_max, clamp_kappa=clamp_kappa)
            ce = (log_uz_cat * uz_mask.float() * sc.unsqueeze(-1)).sum(dim=(1, 2))
            n_edits = uz_mask.float().sum(dim=(1, 2))
            per_sample_loss = u_tot - ce

        losses.extend(per_sample_loss.tolist())
        u_tots.extend(u_tot.tolist())
        ce_terms.extend(ce.tolist())
        sched_coeffs.extend(sc.squeeze(-1).tolist())
        n_edits_list.extend(n_edits.tolist())

        if (batch_idx + 1) % 50 == 0:
            print(f"  batch {batch_idx + 1}/{args.num_batches}")

    # --- Statistics ---
    losses_t = torch.tensor(losses)
    u_tots_t = torch.tensor(u_tots)
    ce_terms_t = torch.tensor(ce_terms)
    sc_t = torch.tensor(sched_coeffs)
    edits_t = torch.tensor(n_edits_list)

    print("\n" + "=" * 60)
    print("ORACLE LOSS DISTRIBUTION")
    print("=" * 60)
    print(f"  Samples:        {len(losses):,}")
    print(f"  Loss mean:      {losses_t.mean():.4f}")
    print(f"  Loss std:       {losses_t.std():.4f}")
    print(f"  Loss min:       {losses_t.min():.4f}")
    print(f"  Loss max:       {losses_t.max():.4f}")
    print(f"  Loss median:    {losses_t.median():.4f}")
    neg_frac = (losses_t < 0).float().mean().item()
    print(f"  % negative:     {neg_frac * 100:.1f}%")
    print(f"  u_tot mean:     {u_tots_t.mean():.4f}")
    print(f"  u_tot std:      {u_tots_t.std():.4f}")
    print(f"  CE term mean:   {ce_terms_t.mean():.4f}")
    print(f"  CE term std:    {ce_terms_t.std():.4f}")
    print(f"  sched_coeff mean: {sc_t.mean():.4f}")
    print(f"  sched_coeff std:  {sc_t.std():.4f}")
    print(f"  #edits mean:    {edits_t.mean():.2f}")
    print(f"  #edits std:     {edits_t.std():.2f}")

    # Loss breakdown by sched_coeff bucket
    print("\n--- Loss by sched_coeff range ---")
    buckets = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 50)]
    for lo, hi in buckets:
        mask = (sc_t >= lo) & (sc_t < hi)
        if mask.sum() == 0:
            continue
        sub = losses_t[mask]
        print(f"  sc∈[{lo:2d},{hi:2d}): n={mask.sum():5d}  "
              f"mean={sub.mean():8.4f}  std={sub.std():8.4f}  "
              f"min={sub.min():8.4f}  max={sub.max():8.4f}  "
              f"neg={((sub < 0).float().mean() * 100):5.1f}%")

    # Percentile distribution
    print("\n--- Loss percentiles ---")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = torch.quantile(losses_t, p / 100.0)
        print(f"  P{p:2d}: {val:10.4f}")

    # --- Decompose variance sources ---
    print("\n--- Variance decomposition ---")
    sc_t_cpu = sc_t
    edits_t_cpu = edits_t
    # Bucket by sched_coeff and compute per-bucket stats
    sc_buckets = [(0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 50)]
    for lo, hi in sc_buckets:
        mask = (sc_t_cpu >= lo) & (sc_t_cpu < hi)
        if mask.sum() < 10:
            continue
        sub = losses_t[mask]
        print(f"  sc∈[{lo:4.1f},{hi:4.1f}): n={mask.sum():5d}  "
              f"loss_mean={sub.mean():8.2f}  loss_std={sub.std():8.2f}  "
              f"u_tot_mean={u_tots_t[mask].mean():6.2f}  "
              f"edits_mean={edits_t_cpu[mask].mean():5.1f}")

    # Pearson correlation between key variables
    print("\n--- Correlations ---")
    stacked = torch.stack([losses_t, u_tots_t, ce_terms_t, sc_t_cpu, edits_t_cpu], dim=1)
    corr = torch.corrcoef(stacked.T)
    labels = ["loss", "u_tot", "ce_term", "sched_coeff", "n_edits"]
    print(f"  {'':>12s} " + " ".join(f"{l:>10s}" for l in labels))
    for i, li in enumerate(labels):
        row = " ".join(f"{corr[i,j]:10.4f}" for j in range(len(labels)))
        print(f"  {li:>12s} {row}")

    print("\n--- Oracle vs Training gap ---")
    train_loss_typical = 11.0  # approximate from training log
    gap = train_loss_typical - losses_t.mean().item()
    print(f"  Training loss (typical):  {train_loss_typical:.2f}")
    print(f"  Oracle loss (mean):       {losses_t.mean():.2f}")
    print(f"  Gap:                      {gap:.2f}")
    print(f"  Gap / oracle_std:         {gap / losses_t.std():.2f}")
    print(f"  (gap < 1 std → model is within natural noise envelope)")
    print(f"  Interpretation: model is {'close to' if abs(gap) < losses_t.std() else 'far from'} oracle regime")


if __name__ == "__main__":
    main()
