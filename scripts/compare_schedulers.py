#!/usr/bin/env python
"""Compare sched_coeff = κ̇/(1-κ) distributions for different κ_t schedules."""

import torch
import math


def analyze_scheduler(name, kappa_fn, deriv_fn, clamp=50.0, n_samples=100000):
    t = torch.rand(n_samples)
    kappa = kappa_fn(t)
    deriv = deriv_fn(t)
    sc = deriv / (1 - kappa + 1e-8)
    sc_clamped = torch.clamp(sc, max=clamp)

    # Stats
    print(f"\n{'='*50}")
    print(f"  {name}: κ_t = {name.split('(')[0]}")
    print(f"{'='*50}")
    print(f"  Mean κ_t:           {kappa.mean():.4f}")
    print(f"  Mean κ̇_t:           {deriv.mean():.4f}")
    print(f"  Mean sched_coeff:   {sc_clamped.mean():.4f}")
    print(f"  Std sched_coeff:    {sc_clamped.std():.4f}")
    print(f"  Min sched_coeff:    {sc_clamped.min():.4f}")
    print(f"  Median sched_coeff: {sc_clamped.median():.4f}")

    # Key thresholds
    frac_lt_1 = (sc_clamped < 1.0).float().mean()
    frac_gt_e = (sc_clamped > math.e).float().mean()
    frac_clamped = (sc > clamp).float().mean()
    print(f"  P(sc < 1):          {frac_lt_1:.3f}  (positive loss regime)")
    print(f"  P(sc > e≈2.718):    {frac_gt_e:.3f}  (negative loss regime)")
    print(f"  P(clamped at {clamp}):   {frac_clamped:.3f}")

    # Percentiles
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = torch.quantile(sc_clamped, p / 100)
        print(f"  P{p:2d}: {val:8.4f}")

    # sc at which loss contribution turns negative
    # loss per edit = sc * (1 - log(sc)), negative when sc > e
    print(f"  Efficiency: {frac_lt_1:.1%} samples in 'clean' regime (sc<1)")
    print(f"              {frac_gt_e:.1%} samples in 'negative loss' regime (sc>e)")

    return sc_clamped


# Cubic:  κ_t = t³,    κ̇_t = 3t²
# Quad:   κ_t = t²,    κ̇_t = 2t
# Linear: κ_t = t,     κ̇_t = 1

sc_cubic = analyze_scheduler("Cubic  (t³)", lambda t: t**3, lambda t: 3*t**2)
sc_quad  = analyze_scheduler("Quad   (t²)", lambda t: t**2, lambda t: 2*t)
sc_linear = analyze_scheduler("Linear (t¹)", lambda t: t, lambda t: torch.ones_like(t))

# Summary comparison
print(f"\n{'='*50}")
print(f"  SUMMARY")
print(f"{'='*50}")
print(f"  {'':20s} {'Cubic(t³)':>12s} {'Quad(t²)':>12s} {'Linear(t¹)':>12s}")
print(f"  {'Mean sc':20s} {sc_cubic.mean():12.4f} {sc_quad.mean():12.4f} {sc_linear.mean():12.4f}")
print(f"  {'Std sc':20s} {sc_cubic.std():12.4f} {sc_quad.std():12.4f} {sc_linear.std():12.4f}")
print(f"  {'Median sc':20s} {sc_cubic.median():12.4f} {sc_quad.median():12.4f} {sc_linear.median():12.4f}")
print(f"  {'P(sc<1)':20s} {(sc_cubic<1).float().mean():11.3f} {(sc_quad<1).float().mean():11.3f} {(sc_linear<1).float().mean():11.3f}")
print(f"  {'P(sc>e)':20s} {(sc_cubic>math.e).float().mean():11.3f} {(sc_quad>math.e).float().mean():11.3f} {(sc_linear>math.e).float().mean():11.3f}")
print(f"  {'P(sc>10)':20s} {(sc_cubic>10).float().mean():11.3f} {(sc_quad>10).float().mean():11.3f} {(sc_linear>10).float().mean():11.3f}")
print(f"  {'P(sc>50 clamped)':20s} {(sc_cubic>50).float().mean():11.3f} {(sc_quad>50).float().mean():11.3f} {(sc_linear>50).float().mean():11.3f}")

# "Warm-up" analysis: what t gives sc < 1?
print(f"\n  --- Warm-up duration (t where sc < 1) ---")
for name, fn in [("Cubic", lambda t: 3*t**2/(1-t**3)),
                  ("Quad", lambda t: 2*t/(1-t**2)),
                  ("Linear", lambda t: 1/(1-t))]:
    # Find t where sc = 1
    if name == "Linear":
        t_crit = 0.0  # sc=1 at t=0
    elif name == "Quad":
        # 2t/(1-t²) = 1 → t² + 2t - 1 = 0 → t = √2-1
        t_crit = math.sqrt(2) - 1
    else:  # Cubic
        # 3t²/(1-t³) = 1 → t³ + 3t² - 1 = 0, solve numerically
        # t ≈ 0.532
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if fn(mid) < 1:
                lo = mid
            else:
                hi = mid
        t_crit = lo
    print(f"  {name}: sc < 1 for t < {t_crit:.4f}  (≈{t_crit*100:.0f}% of t-range)")
