import math

import pytest
import torch

from edit_flows.sampling.euler_smc import (
    SMCParticleSet,
    advance_particles,
    euler_transition_step,
    effective_sample_size,
    normalize_log_weights,
    systematic_resample,
    systematic_resample_batch,
)
from edit_flows.core.scheduler import LinearScheduler
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


def test_normalize_log_weights_and_ess_are_stable():
    log_weights = torch.tensor([1000.0, 999.0, 998.0])
    normalized, log_normalizer = normalize_log_weights(log_weights)

    assert torch.isfinite(normalized).all()
    assert torch.isfinite(log_normalizer)
    assert torch.allclose(
        torch.exp(normalized).sum(), torch.tensor(1.0), atol=2e-5,
    )
    assert effective_sample_size(log_weights).item() == pytest.approx(
        1.0 / sum(
            float(prob) ** 2
            for prob in torch.softmax(log_weights, dim=0)
        ),
        rel=1e-4,
    )


def test_systematic_resampling_is_seeded_and_in_range():
    log_weights = torch.log(torch.tensor([0.1, 0.2, 0.7]))
    first = systematic_resample(log_weights, seed=17)
    second = systematic_resample(log_weights, seed=17)

    assert torch.equal(first, second)
    assert first.shape == (3,)
    assert int(first.min()) >= 0
    assert int(first.max()) < 3


def test_batch_resampling_is_independent_of_unrelated_rows():
    log_weights = torch.log(torch.tensor([
        [0.1, 0.2, 0.7],
        [0.6, 0.3, 0.1],
    ]))
    product_indices = torch.tensor([10, 20], dtype=torch.long)
    batched = systematic_resample_batch(
        log_weights, base_seed=42, step=3,
        product_indices=product_indices,
    )
    # Compare against the same helper with the exact per-product seed by
    # checking each row through a one-row batch; unrelated rows cannot affect it.
    for row, product_index in enumerate(product_indices.tolist()):
        single = systematic_resample_batch(
            log_weights[row:row + 1],
            base_seed=42,
            step=3,
            product_indices=torch.tensor([product_index]),
        )[0]
        assert torch.equal(batched[row], single)
def test_bootstrap_target_equals_proposal_keeps_uniform_weights():
    states = torch.arange(4).unsqueeze(-1)
    particles = SMCParticleSet.initial(states)
    result = advance_particles(
        particles,
        next_states=states + 10,
        parent_indices=torch.arange(4, dtype=torch.long),
        log_target_increment=torch.zeros(4),
        log_proposal_increment=torch.zeros(4),
        ess_threshold=2.0,
        resample_seed=7,
    )

    assert not result.resampled
    assert result.ess_before_resampling == pytest.approx(4.0)
    assert result.log_evidence_increment == pytest.approx(0.0)
    assert torch.allclose(
        result.particles.log_weights,
        torch.full((4,), -math.log(4)),
    )
    assert torch.equal(result.particles.ancestor_ids, torch.arange(4))


def test_importance_ratio_and_genealogy_survive_resampling():
    states = torch.arange(3).unsqueeze(-1)
    particles = SMCParticleSet.initial(states)
    target = torch.tensor([0.6, 0.3, 0.1])
    # Uniform proposal: target/proposal = 3 * target.
    log_ratio = torch.log(3.0 * target)
    weighted = advance_particles(
        particles,
        next_states=states + 1,
        parent_indices=torch.arange(3, dtype=torch.long),
        log_target_increment=log_ratio,
        log_proposal_increment=torch.zeros(3),
    )
    assert not weighted.resampled
    assert torch.allclose(
        torch.softmax(weighted.particles.log_weights, dim=0), target,
        atol=1e-6,
    )

    resampled = advance_particles(
        particles,
        next_states=states + 1,
        parent_indices=torch.arange(3, dtype=torch.long),
        log_target_increment=log_ratio,
        log_proposal_increment=torch.zeros(3),
        ess_threshold=2.5,
        resample_seed=11,
    )
    assert resampled.resampled
    assert resampled.resample_indices is not None
    assert torch.allclose(
        resampled.particles.log_weights,
        torch.full((3,), -math.log(3)),
    )
    assert set(resampled.particles.ancestor_ids.tolist()).issubset({0, 1, 2})


def test_particle_validation_rejects_missing_resampling_seed():
    particles = SMCParticleSet.initial(torch.arange(3).unsqueeze(-1))
    with pytest.raises(ValueError, match="resample_seed"):
        advance_particles(
            particles,
            next_states=particles.states,
            parent_indices=torch.arange(3, dtype=torch.long),
            log_target_increment=torch.tensor([2.0, 0.0, -2.0]),
            log_proposal_increment=torch.zeros(3),
            ess_threshold=2.9,
        )


class _TransitionModel(torch.nn.Module):
    def __init__(self, vocab_size=16):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, tokens, time_step, padding_mask, origin_mask=None):
        batch, length = tokens.shape
        log_rates = torch.full(
            (batch, length, 3), -0.3, device=tokens.device,
        )
        logits = torch.arange(
            self.vocab_size, dtype=torch.float, device=tokens.device,
        ).view(1, 1, -1).expand(batch, length, -1)
        log_probs = torch.log_softmax(logits / 4.0, dim=-1)
        pad_mask = padding_mask.unsqueeze(-1)
        return (
            log_rates.masked_fill(pad_mask, -1e9),
            log_probs.masked_fill(pad_mask, -1e9),
            log_probs.masked_fill(pad_mask, -1e9),
        )


def test_euler_transition_is_seeded_and_closes_bootstrap_smc():
    model = _TransitionModel()
    states = torch.tensor([
        [BOS_TOKEN, 4, 5, PAD_TOKEN],
        [BOS_TOKEN, 7, 8, PAD_TOKEN],
    ])
    kwargs = dict(
        scheduler=LinearScheduler(),
        step=0,
        n_steps=4,
        seeds=torch.tensor([11, 29]),
        max_seq_len=16,
    )
    first = euler_transition_step(model, states, **kwargs)
    second = euler_transition_step(model, states, **kwargs)

    assert torch.equal(first.next_states, second.next_states)
    assert torch.equal(
        first.log_proposal_increment, second.log_proposal_increment,
    )
    assert torch.isfinite(first.log_proposal_increment).all()
    assert (first.next_states != PAD_TOKEN).any(dim=1).all()

    particles = SMCParticleSet.initial(states)
    identity = torch.arange(states.shape[0], dtype=torch.long)
    bootstrap = advance_particles(
        particles,
        next_states=first.next_states,
        parent_indices=identity,
        log_target_increment=first.log_proposal_increment,
        log_proposal_increment=first.log_proposal_increment,
        ess_threshold=1.5,
        resample_seed=123,
    )
    assert not bootstrap.resampled
    assert bootstrap.ess_before_resampling == pytest.approx(2.0)
    assert bootstrap.log_evidence_increment == pytest.approx(0.0)


def test_euler_transition_at_terminal_time_is_identity():
    model = _TransitionModel()
    states = torch.tensor([[BOS_TOKEN, 4, 5, PAD_TOKEN]])
    result = euler_transition_step(
        model,
        states,
        LinearScheduler(),
        step=3,
        n_steps=4,
        seeds=[19],
        t=1.0,
        max_seq_len=16,
    )

    # apply_ins_del_operations canonicalizes trailing PAD columns, so compare
    # the logical token sequence rather than the storage width.
    assert torch.equal(result.next_states, states[:, :3])
    assert torch.allclose(result.step_size, torch.zeros(1))
    assert torch.allclose(
        result.log_proposal_increment, torch.zeros(1), atol=1e-6,
    )


def test_euler_transition_rejects_unmatched_linear_probability():
    with pytest.raises(ValueError, match="event_prob_mode='poisson'"):
        euler_transition_step(
            _TransitionModel(),
            torch.tensor([[BOS_TOKEN, 4, PAD_TOKEN]]),
            LinearScheduler(),
            step=0,
            n_steps=4,
            seeds=[1],
            event_prob_mode="linear",
        )
