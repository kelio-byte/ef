"""Time policies for single-edit beam/greedy sampling.

A TimePolicy schedules the kappa value passed (indirectly, via *t*) to the
model at each edit step.  All policies speak kappa internally; the caller is
responsible for converting kappa to *t* before the model forward::

    kappa = policy.get_kappa(step)       # (B, 1)  — before model forward
    t = scheduler.inverse(kappa)         # (B, 1)  — κ → t
    t_model = _compute_model_time(t, ...)
    ...
    u_tot = compute_base_rate_total      # after model forward
    stop = policy.update(kappa.squeeze(-1), u_tot)  # feed back for next step
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from edit_flows.core.scheduler import KappaScheduler


class TimePolicy(ABC):
    """Abstract time policy for single-edit sampling.

    The policy operates in κ-space.  The caller converts κ to *t* for the
    model and passes κ back to ``update()`` so state-aware policies can track
    progress without a round-trip through the scheduler.
    """

    @abstractmethod
    def reset(self, batch_size: int, device: torch.device, max_edits: int) -> None:
        """(Re-)initialise per-sample state before the first edit step."""
        ...

    @abstractmethod
    def get_kappa(self, step: int) -> Tensor:
        """Return ``(B, 1)`` kappa tensor for the current step.

        Called **before** the model forward.  For state-aware policies this
        uses internal state accumulated from the preceding ``update()``
        call(s).  At step 0 an initial / default kappa is returned.
        """
        ...

    @abstractmethod
    def update(self, kappa: Tensor, u_tot_base: Tensor) -> Tensor:
        """Ingest model output and return a per-sample stop signal.

        Called **after** the model forward.

        Parameters
        ----------
        kappa:
            ``(B,)`` — κ used for the current step (same value returned by
            the preceding ``get_kappa`` call).
        u_tot_base:
            ``(B,)`` — base-rate total from the model output (unscaled by
            :math:`k(t)`, i.e. the raw model output when ``use_rate_reparam``
            is on).

        Returns
        -------
        stop:
            ``(B,)`` bool tensor.  Samples marked ``True`` will be excluded
            from further editing by the caller (in addition to any external
            ``stop_u_tot_base`` threshold).
        """
        ...

    def clone(self) -> "TimePolicy":
        """Return a deep copy of the policy and its internal state."""
        return copy.deepcopy(self)

    @abstractmethod
    def state_key(self) -> tuple:
        """Return a hashable snapshot of internal state for beam dedup."""
        ...


# ---------------------------------------------------------------------------
# State-agnostic policies
# ---------------------------------------------------------------------------


class DepthTimePolicy(TimePolicy):
    """Map discrete edit depth to kappa via the flow scheduler.

    ``t = (step + 1) / (max_edits + 1)``, then ``κ = scheduler(t)``.
    Identical for every sample.
    """

    def __init__(self, scheduler: KappaScheduler) -> None:
        self._scheduler = scheduler

    def reset(self, batch_size: int, device: torch.device, max_edits: int) -> None:
        self._batch_size = batch_size
        self._device = device
        self._max_edits = max_edits

    def get_kappa(self, step: int) -> Tensor:
        if self._max_edits <= 0:
            raise ValueError("max_edits must be positive")
        t_val = (step + 1) / (self._max_edits + 1)
        t = torch.full((self._batch_size, 1), t_val, device=self._device)
        return self._scheduler(t)

    def update(self, kappa: Tensor, u_tot_base: Tensor) -> Tensor:
        return torch.zeros(self._batch_size, dtype=torch.bool, device=self._device)

    def state_key(self) -> tuple:
        return ("depth", self._max_edits)


class FixedTimePolicy(TimePolicy):
    """Return a constant kappa for every sample and every step."""

    def __init__(self, scheduler: KappaScheduler, time_const: float = 0.5) -> None:
        self._scheduler = scheduler
        self._time_const = time_const

    def reset(self, batch_size: int, device: torch.device, max_edits: int) -> None:
        self._batch_size = batch_size
        self._device = device

    def get_kappa(self, step: int) -> Tensor:
        t = torch.full((self._batch_size, 1), self._time_const, device=self._device)
        return self._scheduler(t)

    def update(self, kappa: Tensor, u_tot_base: Tensor) -> Tensor:
        return torch.zeros(self._batch_size, dtype=torch.bool, device=self._device)

    def state_key(self) -> tuple:
        return ("fixed", self._time_const)


# ---------------------------------------------------------------------------
# State-aware policies
# ---------------------------------------------------------------------------


class RatioTimePolicy(TimePolicy):
    """Kappa driven by the ratio of remaining edit mass.

    At steps 0–1 a default kappa (depth-based) is used because the ratio
    ``u_prev / u_init`` is still 1.0 until the first post-edit model forward.
    For subsequent steps::

        kappa = clamp(1 - u_prev / u_init, ε, 1)

    The intuition: when 80 % of the initial edit mass has been consumed the
    sample is ~80 % done, so kappa should reflect a late-training state.
    """

    def __init__(self, scheduler: KappaScheduler) -> None:
        self._scheduler = scheduler

    def reset(self, batch_size: int, device: torch.device, max_edits: int) -> None:
        self._batch_size = batch_size
        self._device = device
        self._max_edits = max_edits
        self._u_init = torch.zeros(batch_size, device=device)
        self._u_prev = torch.zeros(batch_size, device=device)
        self._initialized = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def get_kappa(self, step: int) -> Tensor:
        # The first two steps use depth-based κ because at step 1 the ratio
        # u_prev/u_init is still 1.0 (u_prev came from step 0's model output,
        # before the first edit was applied).  By step 2 the model has seen the
        # post-edit state and u_prev reflects actual progress.
        if step <= 1:
            t_val = (step + 1.0) / (self._max_edits + 1.0)
            t = torch.full((self._batch_size, 1), t_val, device=self._device)
            return self._scheduler(t)

        kappa = torch.zeros(self._batch_size, device=self._device)
        mask = self._initialized & (self._u_init > 1e-8)
        if mask.any():
            kappa[mask] = 1.0 - self._u_prev[mask] / self._u_init[mask].clamp_min(1e-8)
        kappa = kappa.clamp(1e-8, 1.0)
        return kappa.unsqueeze(-1)

    def update(self, kappa: Tensor, u_tot_base: Tensor) -> Tensor:
        self._u_init = torch.where(
            ~self._initialized,
            u_tot_base,
            self._u_init,
        )
        self._u_prev = u_tot_base
        self._initialized[:] = True
        return torch.zeros(self._batch_size, dtype=torch.bool, device=self._device)

    def state_key(self) -> tuple:
        return (
            "ratio",
            tuple(bool(v) for v in self._initialized.detach().cpu().tolist()),
            tuple(float(v) for v in self._u_init.detach().cpu().tolist()),
            tuple(float(v) for v in self._u_prev.detach().cpu().tolist()),
        )


class KappaTimePolicy(TimePolicy):
    r"""Iterative kappa-based time from the conservation-like recurrence.

    At step 0 an initial kappa is set from the default (depth) *t*.  After
    each model forward the policy advances kappa according to::

        kappa_next = 1 - (1 - kappa_cur) * (u_tot - 1) / u_tot

    The intuition is that :math:`(1-\kappa)` represents remaining probability
    mass and ``u_tot`` represents remaining edit demand.  When they shrink at
    the same rate the time stays aligned with the model's internal expectation.

    Samples where ``u_tot < 1`` are flagged as stopped — the model estimates
    less than one edit remaining.
    """

    def __init__(self, scheduler: KappaScheduler) -> None:
        self._scheduler = scheduler

    def reset(self, batch_size: int, device: torch.device, max_edits: int) -> None:
        self._batch_size = batch_size
        self._device = device
        self._max_edits = max_edits
        self._kappa_cur = torch.zeros(batch_size, device=device)
        self._step0_done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def get_kappa(self, step: int) -> Tensor:
        if step == 0:
            t_val = 1.0 / (self._max_edits + 1.0)
            t = torch.full((self._batch_size, 1), t_val, device=self._device)
            self._kappa_cur = self._scheduler(t).squeeze(-1)
            self._step0_done[:] = True
            return self._scheduler(t)

        return self._kappa_cur.unsqueeze(-1)

    def update(self, kappa: Tensor, u_tot_base: Tensor) -> Tensor:
        # Advance kappa for the next step: kappa' = 1 - (1-kappa) * (u-1)/u
        safe_u = u_tot_base.clamp_min(1e-8)
        ratio = (safe_u - 1.0) / safe_u
        kappa_next = 1.0 - (1.0 - kappa) * ratio
        self._kappa_cur = kappa_next.clamp(0.0, 1.0)
        return u_tot_base < 1.0

    def state_key(self) -> tuple:
        return (
            "kappa",
            tuple(float(v) for v in self._kappa_cur.detach().cpu().tolist()),
            tuple(bool(v) for v in self._step0_done.detach().cpu().tolist()),
        )
