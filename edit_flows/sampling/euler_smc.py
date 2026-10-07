"""Independent Sequential Monte Carlo mechanics for discrete samplers.

This module deliberately does not call the Edit Flows model or change the
Euler-Beam sampler.  It provides the particle/weight/resampling primitives
needed to validate an Euler-SMC implementation on synthetic transition
systems before introducing a chemistry reward or a new checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, List, Optional, Sequence

import torch
from torch import Tensor

from edit_flows.core.rate_scale import apply_rate_parameterization, get_rate_scale
from edit_flows.core.scheduler import KappaScheduler
from edit_flows.sampling.euler import _compute_model_time, get_adaptive_h
from edit_flows.sampling.euler_beam import (
    _apply_edits_batch,
    _apply_q_temperature,
    _sample_actions_per_branch,
    _step_log_p_batch,
)
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


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


@dataclass
class EulerTransitionResult:
    """One batched Euler proposal and its exact Poisson log probability."""

    next_states: Tensor
    log_proposal_increment: Tensor
    step_size: Tensor
    actions: dict


@dataclass
class EulerSMCBootstrapResult:
    """Diagnostics from an isolated target=proposal Euler-SMC rollout."""

    particles: SMCParticleSet
    ess_history: List[float]
    resampling_steps: List[int]
    log_evidence: float


@dataclass
class EulerSMCTerminalTwistResult:
    """Result from a bootstrap proposal followed by one terminal twist."""

    particles: SMCParticleSet
    ess_history: List[float]
    resampling_steps: List[int]
    log_evidence: float
    terminal_ess_before_resampling: float
    terminal_resampled: bool
    terminal_log_evidence_increment: float


def terminal_twist_target_increment(
    log_proposal_increment: Tensor,
    terminal_reward: Tensor,
    *,
    beta: float = 1.0,
) -> Tensor:
    """Return a terminal-twisted target increment.

    The first isolated reward adapter uses the exponential tilt

    ``q(y) ∝ p_proposal(y) * exp(beta * R(y))``.

    This helper only forms the log target increment; it does not sample, alter
    the proposal, or inspect targets.  Keeping the arithmetic separate makes
    the identity limit (``beta=0``) and synthetic importance-ratio checks
    explicit before any chemistry sampler is exposed.
    """
    if log_proposal_increment.ndim != 1:
        raise ValueError(
            "log_proposal_increment must be 1-D, got shape "
            f"{tuple(log_proposal_increment.shape)}"
        )
    if terminal_reward.shape != log_proposal_increment.shape:
        raise ValueError(
            "terminal_reward must match log_proposal_increment shape, got "
            f"{tuple(terminal_reward.shape)} vs "
            f"{tuple(log_proposal_increment.shape)}"
        )
    if not math.isfinite(beta):
        raise ValueError(f"beta must be finite, got {beta}")
    if not torch.isfinite(log_proposal_increment).all():
        raise ValueError("log_proposal_increment must contain finite values")
    if not torch.isfinite(terminal_reward).all():
        raise ValueError("terminal_reward must contain finite values")
    log_increment = log_proposal_increment + float(beta) * terminal_reward
    if not torch.isfinite(log_increment).all():
        raise ValueError("terminal twist produced non-finite increments")
    return log_increment


def apply_terminal_twist(
    particles: SMCParticleSet,
    terminal_reward: Tensor,
    *,
    beta: float = 1.0,
    ess_threshold: Optional[float] = None,
    resample_seed: Optional[int] = None,
) -> SMCStepResult:
    """Apply one terminal reward factor to an existing particle population.

    The particles are already at their terminal states.  Therefore the
    proposal increment is zero and the target/proposal ratio is exactly
    ``exp(beta * terminal_reward)``.  This is intentionally a separate
    operation from :func:`run_euler_smc_bootstrap`: no intermediate twisting,
    learned proposal, or default sampler behavior is changed.
    """
    if terminal_reward.shape != (particles.n_particles,):
        raise ValueError(
            "terminal_reward must have one value per particle, got shape "
            f"{tuple(terminal_reward.shape)} for {particles.n_particles}"
        )
    parent_indices = torch.arange(
        particles.n_particles,
        dtype=torch.long,
        device=particles.states.device,
    )
    zero_increment = torch.zeros(
        particles.n_particles,
        dtype=particles.log_weights.dtype,
        device=particles.states.device,
    )
    log_target_increment = terminal_twist_target_increment(
        zero_increment,
        terminal_reward.to(
            device=particles.states.device,
            dtype=particles.log_weights.dtype,
        ),
        beta=beta,
    )
    return advance_particles(
        particles,
        particles.states,
        parent_indices,
        log_target_increment=log_target_increment,
        log_proposal_increment=zero_increment,
        ess_threshold=ess_threshold,
        resample_seed=resample_seed,
    )


@torch.inference_mode()
def euler_transition_step(
    model,
    states: Tensor,
    scheduler: KappaScheduler,
    *,
    step: int,
    n_steps: int,
    seeds: Sequence[int] | Tensor,
    t: float | Tensor = 0.0,
    max_seq_len: int = 512,
    pad_token: int = PAD_TOKEN,
    bos_token: int = BOS_TOKEN,
    use_rate_reparam: bool = False,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
    time_input: str = "t",
    train_scheduler: Optional[KappaScheduler] = None,
    event_prob_mode: str = "poisson",
    q_temperature: float = 1.0,
) -> EulerTransitionResult:
    """Propose one Euler step for a particle population.

    This is deliberately an adapter, not a sampler: it performs one model
    forward, one stateless Euler action draw per particle, and returns the
    resulting states together with the proposal log probability.  The
    proposal uses the same Poisson event semantics and vectorized action
    helpers as Euler-Beam, so a first SMC smoke test can set target=proposal
    without changing the existing sampler.

    ``states`` must be a padded ``(N, L)`` tensor and ``seeds`` must contain
    one stable seed per row.  All rows share the same Euler step index but may
    carry different times, which is useful when testing batch/layout invariance.
    ``event_prob_mode='linear'`` is intentionally rejected until its exact
    scorer is implemented; using the current Poisson log-probability for a
    linear proposal would silently invalidate the importance ratio.
    """
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError(
            "states must be a non-empty 2-D padded tensor, got "
            f"shape {tuple(states.shape)}"
        )
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if max_seq_len < 1:
        raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len}")
    if event_prob_mode != "poisson":
        raise ValueError(
            "Euler-SMC currently supports only event_prob_mode='poisson'; "
            "a linear proposal needs a matching exact log-probability"
        )
    if not math.isfinite(q_temperature) or q_temperature <= 0:
        raise ValueError(
            f"q_temperature must be finite and > 0, got {q_temperature}"
        )

    device = next(model.parameters()).device
    states = states.to(device=device, dtype=torch.long)
    n_particles = states.shape[0]
    seeds = torch.as_tensor(seeds, dtype=torch.long, device=device)
    if seeds.ndim != 1 or seeds.numel() != n_particles:
        raise ValueError(
            "seeds must be 1-D with one value per state row, got "
            f"shape {tuple(seeds.shape)} for {n_particles} rows"
        )
    if (seeds < 0).any():
        raise ValueError("seeds must be non-negative")

    if isinstance(t, Tensor):
        times = t.to(device=device, dtype=torch.float32)
        if times.ndim == 0:
            times = times.expand(n_particles)
        elif times.ndim == 2 and times.shape == (n_particles, 1):
            times = times[:, 0]
        elif times.ndim != 1 or times.shape[0] != n_particles:
            raise ValueError(
                "t must be scalar, (N,), or (N, 1), got "
                f"shape {tuple(times.shape)}"
            )
    else:
        times = torch.full(
            (n_particles,), float(t), dtype=torch.float32, device=device,
        )
    if not torch.isfinite(times).all() or (times < 0).any():
        raise ValueError("t must contain finite non-negative values")
    t_column = times.unsqueeze(-1)

    x_pad_mask = states == pad_token
    t_model = _compute_model_time(
        t_column, scheduler, time_input, train_scheduler,
    )
    log_rates, log_ins_probs, log_sub_probs = model(
        states, t_model, x_pad_mask,
    )

    if not use_rate_reparam and train_scheduler is not None and \
       scheduler.name != train_scheduler.name:
        k_sample = get_rate_scale(
            t_column, scheduler,
            clamp_kappa=clamp_kappa, clamp_max=clamp_max,
        )
        k_train = get_rate_scale(
            t_model, train_scheduler,
            clamp_kappa=clamp_kappa, clamp_max=clamp_max,
        )
        log_rates = log_rates + torch.log(
            k_sample / k_train.clamp_min(1e-2)
        ).unsqueeze(1)

    log_rates = apply_rate_parameterization(
        log_rates, t_column, scheduler,
        use_rate_reparam=use_rate_reparam,
        clamp_kappa=clamp_kappa, clamp_max=clamp_max,
    )
    log_ins_probs = _apply_q_temperature(log_ins_probs, q_temperature)
    log_sub_probs = _apply_q_temperature(log_sub_probs, q_temperature)

    default_h = torch.full_like(t_column, 1.0 / n_steps)
    step_size = get_adaptive_h(default_h, t_column, scheduler)
    actions = _sample_actions_per_branch(
        seeds,
        states,
        log_rates,
        log_ins_probs,
        log_sub_probs,
        step_size,
        pad_token=pad_token,
        event_prob_mode=event_prob_mode,
        step=step,
    )
    done = times >= 1.0
    if done.any():
        actions["ins_mask"][done] = False
        actions["del_mask"][done] = False
        actions["sub_mask"][done] = False

    log_proposal = _step_log_p_batch(
        actions,
        log_rates,
        log_ins_probs,
        log_sub_probs,
        step_size,
        score_mode="full_probability",
        state_tokens=states,
        pad_token=pad_token,
    )
    next_states = _apply_edits_batch(
        states, actions, max_seq_len=max_seq_len, pad_token=pad_token,
    )
    return EulerTransitionResult(
        next_states=next_states,
        log_proposal_increment=log_proposal,
        step_size=step_size[:, 0],
        actions=actions,
    )


@torch.inference_mode()
def run_euler_smc_bootstrap(
    model,
    initial_states: Tensor,
    scheduler: KappaScheduler,
    *,
    n_steps: int,
    n_particles: Optional[int] = None,
    base_seed: int = 0,
    product_index: int = 0,
    max_seq_len: int = 512,
    pad_token: int = PAD_TOKEN,
    use_rate_reparam: bool = False,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
    time_input: str = "t",
    train_scheduler: Optional[KappaScheduler] = None,
    ess_threshold: Optional[float] = None,
    q_temperature: float = 1.0,
) -> EulerSMCBootstrapResult:
    """Run an isolated multi-step bootstrap SMC population.

    ``target=proposal`` is intentional: this function validates time/seed/
    state plumbing and ESS diagnostics, but cannot improve model accuracy.  A
    future production sampler must replace the equal target increment with an
    independently justified reward or twisted target before being exposed in
    ``sample_retro.py``.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if base_seed < 0 or product_index < 0:
        raise ValueError("base_seed and product_index must be >= 0")
    if n_particles is not None and n_particles < 1:
        raise ValueError("n_particles must be >= 1 when provided")
    if initial_states.ndim != 2 or initial_states.shape[0] < 1:
        raise ValueError("initial_states must be a non-empty 2-D tensor")
    if n_particles is not None and initial_states.shape[0] != n_particles:
        raise ValueError(
            "initial_states row count must equal n_particles when provided"
        )

    particles = SMCParticleSet.initial(initial_states)
    n_particles = particles.n_particles
    current_seeds = torch.tensor(
        [
            _mix_seed(base_seed, product_index, particle_index)
            for particle_index in range(n_particles)
        ],
        dtype=torch.long,
        device=particles.states.device,
    )
    current_t = 0.0
    ess_history: List[float] = []
    resampling_steps: List[int] = []
    log_evidence = 0.0

    for step in range(n_steps):
        draw_seeds = torch.tensor(
            [
                _mix_seed(int(seed), product_index, step)
                for seed in current_seeds.detach().cpu().tolist()
            ],
            dtype=torch.long,
            device=particles.states.device,
        )
        transition = euler_transition_step(
            model,
            particles.states,
            scheduler,
            step=step,
            n_steps=n_steps,
            seeds=draw_seeds,
            t=current_t,
            max_seq_len=max_seq_len,
            pad_token=pad_token,
            use_rate_reparam=use_rate_reparam,
            clamp_kappa=clamp_kappa,
            clamp_max=clamp_max,
            time_input=time_input,
            train_scheduler=train_scheduler,
            q_temperature=q_temperature,
        )
        parent_indices = torch.arange(
            n_particles, dtype=torch.long, device=particles.states.device,
        )
        resample_seed = _mix_seed(base_seed, product_index, step)
        step_result = advance_particles(
            particles,
            transition.next_states,
            parent_indices,
            log_target_increment=transition.log_proposal_increment,
            log_proposal_increment=transition.log_proposal_increment,
            ess_threshold=ess_threshold,
            resample_seed=resample_seed if ess_threshold is not None else None,
        )
        particles = step_result.particles
        ess_history.append(step_result.ess_before_resampling)
        if step_result.resampled:
            resampling_steps.append(step)
            current_seeds = current_seeds.index_select(
                0, step_result.resample_indices,
            )
        log_evidence += step_result.log_evidence_increment

        step_size = float(transition.step_size[0].item())
        if not torch.allclose(
            transition.step_size,
            transition.step_size[0].expand_as(transition.step_size),
            atol=1e-6,
        ):
            raise RuntimeError(
                "bootstrap rollout requires one shared time step per product"
            )
        current_t = min(1.0, current_t + step_size)

    return EulerSMCBootstrapResult(
        particles=particles,
        ess_history=ess_history,
        resampling_steps=resampling_steps,
        log_evidence=log_evidence,
    )


@torch.inference_mode()
def run_euler_smc_terminal_twist(
    model,
    initial_states: Tensor,
    scheduler: KappaScheduler,
    *,
    terminal_reward_fn: Callable[[Tensor], Tensor],
    n_steps: int,
    n_particles: Optional[int] = None,
    base_seed: int = 0,
    product_index: int = 0,
    max_seq_len: int = 512,
    pad_token: int = PAD_TOKEN,
    use_rate_reparam: bool = False,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
    time_input: str = "t",
    train_scheduler: Optional[KappaScheduler] = None,
    beta: float = 1.0,
    ess_threshold: Optional[float] = None,
    q_temperature: float = 1.0,
) -> EulerSMCTerminalTwistResult:
    """Run Euler proposal dynamics and twist only the terminal population.

    This isolated entry point intentionally keeps all intermediate transitions
    equal to the existing bootstrap proposal.  ``terminal_reward_fn`` is
    called exactly once on the final particle states and must return one finite
    scalar per particle.  The optional ESS threshold applies only to this final
    reward reweighting; intermediate bootstrap weights remain uniform because
    target=proposal there.
    """
    bootstrap = run_euler_smc_bootstrap(
        model,
        initial_states,
        scheduler,
        n_steps=n_steps,
        n_particles=n_particles,
        base_seed=base_seed,
        product_index=product_index,
        max_seq_len=max_seq_len,
        pad_token=pad_token,
        use_rate_reparam=use_rate_reparam,
        clamp_kappa=clamp_kappa,
        clamp_max=clamp_max,
        time_input=time_input,
        train_scheduler=train_scheduler,
        ess_threshold=None,
        q_temperature=q_temperature,
    )
    terminal_reward = terminal_reward_fn(bootstrap.particles.states)
    if not isinstance(terminal_reward, Tensor):
        raise TypeError("terminal_reward_fn must return a torch.Tensor")
    terminal_step = apply_terminal_twist(
        bootstrap.particles,
        terminal_reward,
        beta=beta,
        ess_threshold=ess_threshold,
        resample_seed=(
            _mix_seed(base_seed, product_index, n_steps)
            if ess_threshold is not None else None
        ),
    )
    return EulerSMCTerminalTwistResult(
        particles=terminal_step.particles,
        ess_history=bootstrap.ess_history,
        resampling_steps=bootstrap.resampling_steps,
        log_evidence=(
            bootstrap.log_evidence + terminal_step.log_evidence_increment
        ),
        terminal_ess_before_resampling=terminal_step.ess_before_resampling,
        terminal_resampled=terminal_step.resampled,
        terminal_log_evidence_increment=terminal_step.log_evidence_increment,
    )


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
