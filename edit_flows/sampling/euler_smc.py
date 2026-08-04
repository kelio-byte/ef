"""Independent Sequential Monte Carlo mechanics for discrete samplers.

This module deliberately does not call the Edit Flows model or change the
Euler-Beam sampler.  It provides the particle/weight/resampling primitives
needed to validate an Euler-SMC implementation on synthetic transition
systems before introducing a chemistry reward or a new checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import Tensor


def _validate_log_weights(log_weights: Tensor) -> None:
    if log_weights.ndim != 1:
        raise ValueError(
            f"log_weights must be 1-D, got shape {tuple(log_weights.shape)}"
        )
    if log_weights.numel() < 1:
        raise ValueError("log_weights must contain at least one particle")
    if not torch.isfinite(log_weights).all():
        raise ValueError("log_weights must contain only finite values")


def normalize_log_weights(log_weights: Tensor) -> tuple[Tensor, Tensor]:
    """Return normalized log weights and their log normalizer."""
    _validate_log_weights(log_weights)
    log_normalizer = torch.logsumexp(log_weights, dim=0)
    return log_weights - log_normalizer, log_normalizer


def effective_sample_size(log_weights: Tensor) -> Tensor:
    """Compute ESS from unnormalized log weights without probability underflow."""
    normalized_log_weights, _ = normalize_log_weights(log_weights)
    return torch.exp(-torch.logsumexp(2.0 * normalized_log_weights, dim=0))


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _mix_seed(base_seed: int, product_index: int, step: int) -> int:
    """Stable per-product/per-step seed independent of batch layout."""
    if base_seed < 0 or product_index < 0 or step < 0:
        raise ValueError("base_seed, product_index and step must be >= 0")
    mask = (1 << 64) - 1
    value = (
        (int(base_seed) & mask)
        ^ (((int(product_index) + 1) * 0x9E3779B97F4A7C15) & mask)
        ^ (((int(step) + 1) * 0xD1B54A32D192ED03) & mask)
    )
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & mask
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & mask
    value ^= value >> 31
    return value & ((1 << 63) - 1)


def systematic_resample(
    log_weights: Tensor,
    *,
    seed: int,
) -> Tensor:
    """Return systematic-resampling indices for one particle population.

    The random offset is generated from a local generator, so the result does
    not depend on unrelated global RNG consumption or on the order of other
    product populations.
    """
    normalized_log_weights, _ = normalize_log_weights(log_weights)
    weights = torch.exp(normalized_log_weights)
    n_particles = weights.numel()
    generator = _make_generator(weights.device, seed)
    offset = torch.rand(
        (), dtype=weights.dtype, device=weights.device, generator=generator,
    )
    positions = (
        offset + torch.arange(
            n_particles, dtype=weights.dtype, device=weights.device,
        )
    ) / n_particles
    cdf = torch.cumsum(weights, dim=0).clamp_max(1.0)
    return torch.searchsorted(cdf, positions, right=False).clamp_max(
        n_particles - 1,
    )


def systematic_resample_batch(
    log_weights: Tensor,
    *,
    base_seed: int,
    step: int,
    product_indices: Optional[Tensor] = None,
) -> Tensor:
    """Resample each product row with a layout-independent derived seed."""
    if log_weights.ndim != 2:
        raise ValueError(
            "log_weights must be 2-D (products, particles), got shape "
            f"{tuple(log_weights.shape)}"
        )
    if log_weights.shape[0] < 1 or log_weights.shape[1] < 1:
        raise ValueError("batch must contain at least one product and particle")
    if product_indices is None:
        product_indices = torch.arange(
            log_weights.shape[0], dtype=torch.long,
        )
    if product_indices.ndim != 1 or product_indices.numel() != log_weights.shape[0]:
        raise ValueError(
            "product_indices must be 1-D with one entry per product row"
        )
    indices = [
        systematic_resample(
            log_weights[row],
            seed=_mix_seed(base_seed, int(product_indices[row]), step),
        )
        for row in range(log_weights.shape[0])
    ]
    return torch.stack(indices, dim=0)


@dataclass
class SMCParticleSet:
    """Particle states and genealogy for one product population."""

    states: Tensor
    log_weights: Tensor
    ancestor_ids: Tensor

    def __post_init__(self) -> None:
        if self.states.ndim < 1 or self.states.shape[0] < 1:
            raise ValueError("states must have a non-empty particle dimension")
        n_particles = self.states.shape[0]
        if self.log_weights.shape != (n_particles,):
            raise ValueError("log_weights must have one value per particle")
        if self.ancestor_ids.shape != (n_particles,):
            raise ValueError("ancestor_ids must have one value per particle")
        _validate_log_weights(self.log_weights)
        if self.ancestor_ids.dtype != torch.long:
            raise ValueError("ancestor_ids must use torch.long")

    @classmethod
    def initial(cls, states: Tensor) -> "SMCParticleSet":
        """Create an equally weighted population with self ancestors."""
        if states.ndim < 1 or states.shape[0] < 1:
            raise ValueError("states must have a non-empty particle dimension")
        n_particles = states.shape[0]
        return cls(
            states=states,
            log_weights=torch.full(
                (n_particles,),
                -math.log(n_particles),
                dtype=torch.float32,
                device=states.device,
            ),
            ancestor_ids=torch.arange(
                n_particles, dtype=torch.long, device=states.device,
            ),
        )

    @property
    def n_particles(self) -> int:
        return self.states.shape[0]


@dataclass
class SMCStepResult:
    """Result and diagnostics from one weighted-transition/resampling step."""

    particles: SMCParticleSet
    ess_before_resampling: float
    resampled: bool
    resample_indices: Optional[Tensor]
    log_evidence_increment: float


def advance_particles(
    particles: SMCParticleSet,
    next_states: Tensor,
    parent_indices: Tensor,
    log_target_increment: Tensor,
    log_proposal_increment: Tensor,
    *,
    ess_threshold: Optional[float] = None,
    resample_seed: Optional[int] = None,
) -> SMCStepResult:
    """Advance particles with an importance ratio and optional resampling.

    ``next_states[i]`` is proposed from ``particles[parent_indices[i]]``.
    The target/proposal log-ratio is added to that parent's log weight.  When
    ESS is below ``ess_threshold``, systematic resampling resets weights to
    equal mass while preserving each selected particle's ancestor id.
    """
    if next_states.ndim < 1 or next_states.shape[0] < 1:
        raise ValueError("next_states must have a non-empty particle dimension")
    n_next = next_states.shape[0]
    for name, values in (
        ("parent_indices", parent_indices),
        ("log_target_increment", log_target_increment),
        ("log_proposal_increment", log_proposal_increment),
    ):
        if values.shape != (n_next,):
            raise ValueError(
                f"{name} must have shape ({n_next},), got {tuple(values.shape)}"
            )
    if parent_indices.dtype != torch.long:
        raise ValueError("parent_indices must use torch.long")
    if parent_indices.numel() and (
        int(parent_indices.min()) < 0
        or int(parent_indices.max()) >= particles.n_particles
    ):
        raise ValueError("parent_indices contains an invalid particle index")
    if not torch.isfinite(log_target_increment).all() or \
       not torch.isfinite(log_proposal_increment).all():
        raise ValueError("importance increments must contain finite values")
    if ess_threshold is not None and ess_threshold <= 0:
        raise ValueError("ess_threshold must be > 0 when provided")
    if ess_threshold is not None and resample_seed is None:
        raise ValueError("resample_seed is required when ESS resampling is enabled")

    normalized_parent_log_weights, _ = normalize_log_weights(
        particles.log_weights,
    )
    parent_log_weights = normalized_parent_log_weights.index_select(
        0, parent_indices,
    )
    log_ratio = log_target_increment - log_proposal_increment
    proposed_log_weights = parent_log_weights + log_ratio
    _, log_normalizer = normalize_log_weights(proposed_log_weights)
    ess = float(effective_sample_size(proposed_log_weights).item())

    should_resample = (
        ess_threshold is not None and ess < float(ess_threshold)
    )
    if not should_resample:
        next_particles = SMCParticleSet(
            states=next_states,
            log_weights=proposed_log_weights,
            ancestor_ids=particles.ancestor_ids.index_select(
                0, parent_indices,
            ),
        )
        return SMCStepResult(
            particles=next_particles,
            ess_before_resampling=ess,
            resampled=False,
            resample_indices=None,
            log_evidence_increment=float(log_normalizer.item()),
        )

    resample_indices = systematic_resample(
        proposed_log_weights, seed=int(resample_seed),
    )
    n_particles = resample_indices.numel()
    next_particles = SMCParticleSet(
        states=next_states.index_select(0, resample_indices),
        log_weights=torch.full(
            (n_particles,),
            -math.log(n_particles),
            dtype=proposed_log_weights.dtype,
            device=proposed_log_weights.device,
        ),
        ancestor_ids=particles.ancestor_ids.index_select(
            0, parent_indices.index_select(0, resample_indices),
        ),
    )
    return SMCStepResult(
        particles=next_particles,
        ess_before_resampling=ess,
        resampled=True,
        resample_indices=resample_indices,
        log_evidence_increment=float(log_normalizer.item()),
    )
