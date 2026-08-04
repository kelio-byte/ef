import math

import pytest
import torch

from edit_flows.sampling.euler_smc import (
    SMCParticleSet,
    advance_particles,
    effective_sample_size,
    normalize_log_weights,
    systematic_resample,
    systematic_resample_batch,
)


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
