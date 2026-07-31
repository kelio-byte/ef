#!/usr/bin/env python
"""Analyze oracle trajectory: how does edit distance d(x_t, x_1) evolve over time?

Compares empirical completion rates against theoretical kappa(t), and contrasts
different schedulers (Cubic, Linear) and datasets (Standard, #global#).

Usage:
  PYTHONPATH=. python scripts/oracle_trajectory_analysis.py \
      --traj_cubic_std train_subsets/eval/oracle_standard_cubic/trajectory.pt \
      --traj_linear_std train_subsets/eval/oracle_standard_linear/trajectory.pt \
      --traj_cubic_global train_subsets/eval/oracle_global_cubic/trajectory.pt \
      --traj_linear_global train_subsets/eval/oracle_global_linear/trajectory.pt
"""

import argparse
import os
import sys
from collections import defaultdict

import torch
import numpy as np


def load_trajectory(path: str) -> list[dict]:
    """Load trajectory data in per-batch format.

    Returns list of per-batch dicts: [{"ts": [...], "dists": [...]}, ...].
    Also handles legacy format: {"ts": [...], "dists": [...]}.
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "trajectories" in data:
        return data["trajectories"]
    # Legacy format: wrap in a single-batch list
    return [{"ts": data["ts"], "dists": data["dists"]}]


def _extract_sample_trajectories(
    batches: list[dict],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract per-sample (t, d) trajectories from per-batch data.

    Returns list of (t_arr, d_arr) pairs, one per sample.
    """
    samples = []
    for batch in batches:
        ts_list = batch["ts"]     # list of (B,) tensors
        dists_list = batch["dists"]  # list of (B,) tensors
        B = ts_list[0].shape[0]
        n_steps = len(ts_list)
        # Stack into (B, n_steps)
        t_arr = torch.stack(ts_list, dim=1).numpy()    # (B, n_steps)
        d_arr = torch.stack(dists_list, dim=1).numpy()  # (B, n_steps)
        for i in range(B):
            samples.append((t_arr[i], d_arr[i]))
    return samples


def interpolate_onto_grid(
    batches: list[dict], n_grid: int = 100,
) -> np.ndarray:
    """Interpolate per-sample completion rates onto a common t grid.

    Returns: (N, n_grid) array of completion rates.
    """
    samples = _extract_sample_trajectories(batches)

    # Completion rate: c(t) = 1 - d(t)/d(0)
    t_grid = np.linspace(0, 1, n_grid)
    result = np.full((len(samples), n_grid), np.nan)

    for i, (t_i, d_i) in enumerate(samples):
        d0 = d_i[0]
        if d0 == 0:
            continue  # already matched, no edits needed
        c_i = 1.0 - d_i / d0

        # Remove duplicate t values and sort
        _, unique_idx = np.unique(t_i, return_index=True)
        t_i = t_i[np.sort(unique_idx)]
        c_i = c_i[np.sort(unique_idx)]

        if len(t_i) >= 2:
            result[i] = np.interp(t_grid, t_i, c_i)

    return result


def compute_stats_at_grid(c_grid: np.ndarray) -> dict:
    """Compute mean, std, percentiles at each t grid point."""
    n_grid = c_grid.shape[1]
    return {
        "mean": np.nanmean(c_grid, axis=0),
        "std": np.nanstd(c_grid, axis=0),
        "p10": np.nanpercentile(c_grid, 10, axis=0),
        "p25": np.nanpercentile(c_grid, 25, axis=0),
        "p50": np.nanpercentile(c_grid, 50, axis=0),
        "p75": np.nanpercentile(c_grid, 75, axis=0),
        "p90": np.nanpercentile(c_grid, 90, axis=0),
        "n_valid": np.sum(~np.isnan(c_grid), axis=0),
    }


def compute_final_stats(batches: list[dict]) -> dict:
    """Compute statistics of the final state (t≈1)."""
    samples = _extract_sample_trajectories(batches)

    d_init = np.array([s[1][0] for s in samples])
    d_final = np.array([s[1][-1] for s in samples])

    valid = d_init > 0
    d_final = d_final[valid]
    d_init = d_init[valid]

    # Categorize by initial edit distance
    bins = [(1, 5), (6, 15), (16, 100)]
    strat_stats = {}
    for lo, hi in bins:
        mask = (d_init >= lo) & (d_init <= hi)
        if mask.sum() == 0:
            continue
        d_f = d_final[mask]
        strat_stats[f"L∈[{lo},{hi}]"] = {
            "n": int(mask.sum()),
            "mean_init": float(d_init[mask].mean()),
            "mean_final": float(d_f.mean()),
            "frac_zero": float((d_f == 0).mean()),
            "frac_one": float((d_f == 1).mean()),
            "hist": np.bincount(d_f.astype(int), minlength=11).tolist()[:11],
        }

    return {
        "n_total": int(len(d_init)),
        "mean_init": float(d_init.mean()),
        "mean_final": float(d_final.mean()),
        "frac_zero": float((d_final == 0).mean()),
        "frac_one": float((d_final == 1).mean()),
        "frac_le_one": float((d_final <= 1).mean()),
        "stratified": strat_stats,
    }


def theoretical_kappa(t: np.ndarray, name: str) -> np.ndarray:
    if name == "cubic":
        return t ** 3
    elif name == "linear":
        return t
    return t


def print_report(stats: dict, label: str, theo_name: str):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    fs = stats["final"]
    print(f"\nSamples: {fs['n_total']}")
    print(f"Mean initial edit distance: {fs['mean_init']:.1f}")
    print(f"Mean final edit distance:   {fs['mean_final']:.2f}")
    print(f"Fraction d=0 (perfect):     {fs['frac_zero']*100:.1f}%")
    print(f"Fraction d=1 (one missed):  {fs['frac_one']*100:.1f}%")
    print(f"Fraction d≤1:               {fs['frac_le_one']*100:.1f}%")

    print(f"\n--- Final Edit Distance by L (initial edits) ---")
    for name, s in fs["stratified"].items():
        print(f"  {name} (n={s['n']}):")
        print(f"    Mean init: {s['mean_init']:.1f}, Mean final: {s['mean_final']:.2f}")
        print(f"    d=0: {s['frac_zero']*100:.1f}%, d=1: {s['frac_one']*100:.1f}%")
        hist = s["hist"]
        if any(hist):
            bar = "  ".join(f"d={i}:{c}" for i, c in enumerate(hist) if c > 0)
            print(f"    Hist: {bar}")

    # Completion rate at key time points
    print(f"\n--- Completion Rate vs κ(t) at Key Time Points ---")
    cr = stats["completion"]
    t = np.linspace(0, 1, len(cr["mean"]))
    kappa = theoretical_kappa(t, theo_name)
    key_ts = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
    print(f"  {'t':>6s}  {'κ(t)':>6s}  {'c_mean':>7s}  {'c_std':>6s}  {'c_p50':>7s}  {'Δ(c-κ)':>7s}")
    for kt in key_ts:
        idx = np.argmin(np.abs(t - kt))
        kappa_v = kappa[idx]
        delta = cr["mean"][idx] - kappa_v
        print(f"  {kt:6.3f}  {kappa_v:6.3f}  {cr['mean'][idx]:7.3f}  "
              f"{cr['std'][idx]:6.3f}  {cr['p50'][idx]:7.3f}  {delta:+7.3f}")

    # Final gap decomposition
    print(f"\n--- Final Gap ---")
    final_idx = -1
    gap = 1.0 - cr["mean"][final_idx]
    print(f"  κ(t≈1): {kappa[final_idx]:.4f}")
    print(f"  c_mean(t≈1): {cr['mean'][final_idx]:.4f}")
    print(f"  Gap (1 - c_mean): {gap:.4f}")
    print(f"  Gap std: {cr['std'][final_idx]:.4f}")


def export_comparison_data(
    all_stats: dict,
    output_dir: str,
):
    """Export CSV data for plotting: completion rate curves for all configs."""
    t = np.linspace(0, 1, 100)

    rows = ["t,kappa_cubic,kappa_linear," + ",".join(
        f"{key}_mean,{key}_std,{key}_p50,{key}_nvalid"
        for key in all_stats)]
    for i in range(100):
        parts = [
            f"{t[i]:.4f}",
            f"{theoretical_kappa(t, 'cubic')[i]:.4f}",
            f"{theoretical_kappa(t, 'linear')[i]:.4f}",
        ]
        for key, stats in all_stats.items():
            cr = stats["completion"]
            parts.append(f"{cr['mean'][i]:.6f}")
            parts.append(f"{cr['std'][i]:.6f}")
            parts.append(f"{cr['p50'][i]:.6f}")
            parts.append(f"{int(cr['n_valid'][i])}")
        rows.append(",".join(parts))

    csv_path = os.path.join(output_dir, "trajectory_comparison.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(rows))
    print(f"\nCSV exported to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze oracle edit distance trajectories")
    parser.add_argument("--traj_cubic_std", type=str, default="")
    parser.add_argument("--traj_linear_std", type=str, default="")
    parser.add_argument("--traj_cubic_global", type=str, default="")
    parser.add_argument("--traj_linear_global", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="train_subsets/eval")
    parser.add_argument("--n_grid", type=int, default=100)
    args = parser.parse_args()

    configs = {}
    if args.traj_cubic_std:
        configs["cubic_std"] = (args.traj_cubic_std, "cubic", "Standard (Cubic)")
    if args.traj_linear_std:
        configs["linear_std"] = (args.traj_linear_std, "linear", "Standard (Linear)")
    if args.traj_cubic_global:
        configs["cubic_global"] = (args.traj_cubic_global, "cubic", "#global# (Cubic)")
    if args.traj_linear_global:
        configs["linear_global"] = (args.traj_linear_global, "linear", "#global# (Linear)")

    if not configs:
        print("ERROR: No trajectory files provided.")
        sys.exit(1)

    all_stats = {}

    for key, (path, theo_name, label) in configs.items():
        print(f"\nLoading {label} from {path}...")
        batches = load_trajectory(path)
        n_samples = sum(b["ts"][0].shape[0] for b in batches)
        n_steps = sum(len(b["ts"]) for b in batches)
        print(f"  {len(batches)} batches, {n_samples} samples, {n_steps} total steps")

        c_grid = interpolate_onto_grid(batches, args.n_grid)
        comp_stats = compute_stats_at_grid(c_grid)
        final_stats = compute_final_stats(batches)

        stats = {"completion": comp_stats, "final": final_stats}
        all_stats[key] = stats
        print_report(stats, label, theo_name)

    # Export comparison CSV if multiple configs
    if len(configs) > 1:
        export_comparison_data(all_stats, args.output_dir)

    # Cross-scheduler comparison
    print(f"\n{'=' * 60}")
    print(f"  CROSS-SCHEDULER COMPARISON")
    print(f"{'=' * 60}")

    for dataset in ["std", "global"]:
        k_cub = f"cubic_{dataset}"
        k_lin = f"linear_{dataset}"
        if k_cub not in all_stats or k_lin not in all_stats:
            continue

        s_cub = all_stats[k_cub]
        s_lin = all_stats[k_lin]

        print(f"\n--- {dataset.upper()} Dataset: Cubic vs Linear ---")
        print(f"  {'Metric':<30s}  {'Cubic':>8s}  {'Linear':>8s}")
        print(f"  {'-'*48}")
        print(f"  {'Final mean edit distance':<30s}  "
              f"{s_cub['final']['mean_final']:8.3f}  {s_lin['final']['mean_final']:8.3f}")
        print(f"  {'Final d=0 (perfect)':<30s}  "
              f"{s_cub['final']['frac_zero']*100:7.1f}%  {s_lin['final']['frac_zero']*100:7.1f}%")
        print(f"  {'Final d=1 (one missed)':<30s}  "
              f"{s_cub['final']['frac_one']*100:7.1f}%  {s_lin['final']['frac_one']*100:7.1f}%")
        print(f"  {'Final d≤1':<30s}  "
              f"{s_cub['final']['frac_le_one']*100:7.1f}%  {s_lin['final']['frac_le_one']*100:7.1f}%")

        # Gap improvement
        gap_cub = 1.0 - s_cub["completion"]["mean"][-1]
        gap_lin = 1.0 - s_lin["completion"]["mean"][-1]
        print(f"  {'Final gap (1-c_mean)':<30s}  "
              f"{gap_cub:8.4f}  {gap_lin:8.4f}")
        if gap_cub > 0:
            print(f"  {'Gap reduction':<30s}  "
                  f"{(1 - gap_lin/gap_cub)*100:7.1f}%")

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
