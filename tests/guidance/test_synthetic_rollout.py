"""A small two-step categorical DGM rollout with a known target distribution."""

import torch

from edit_flows.guidance.dgm import guided_log_probs
from edit_flows.sampling.euler_smc import effective_sample_size


def test_two_step_guidance_recovers_known_terminal_distribution():
    # Base chain: start -> {A, B} -> {y0, y1, y2}.
    p_intermediate = torch.tensor([0.7, 0.3])
    p_terminal_given_intermediate = torch.tensor([
        [0.8, 0.2, 0.0],
        [0.1, 0.2, 0.7],
    ])
    reward = torch.tensor([1.0, 4.0, 2.0])

    base_terminal = p_intermediate @ p_terminal_given_intermediate
    target_terminal = base_terminal * reward
    target_terminal = target_terminal / target_terminal.sum()

    # Exact conditional guidance is E[r(Y) | intermediate] at the first step
    # and r(y) at the terminal step.
    h_intermediate = p_terminal_given_intermediate @ reward
    guided_intermediate = guided_log_probs(
        p_intermediate.log(), h_intermediate,
    ).exp()
    guided_terminal_given_intermediate = guided_log_probs(
        p_terminal_given_intermediate.clamp_min(1e-12).log(),
        reward.expand(2, -1),
    ).exp()

    generator = torch.Generator().manual_seed(20260807)
    n_samples = 200_000
    intermediate = torch.multinomial(
        guided_intermediate,
        num_samples=n_samples,
        replacement=True,
        generator=generator,
    )
    terminal = torch.empty(n_samples, dtype=torch.long)
    for group in range(2):
        selected = intermediate == group
        count = int(selected.sum().item())
        terminal[selected] = torch.multinomial(
            guided_terminal_given_intermediate[group],
            num_samples=count,
            replacement=True,
            generator=generator,
        )

    empirical = torch.bincount(terminal, minlength=3).float() / n_samples
    torch.testing.assert_close(empirical, target_terminal, atol=0.01, rtol=0.0)

    # The source proposal's terminal importance estimate should recover the
    # same normalizer in expectation.  This is a mechanics check, not a claim
    # that a chemistry reward has been learned.
    proposal_terminal = torch.multinomial(
        base_terminal,
        num_samples=n_samples,
        replacement=True,
        generator=generator,
    )
    proposal_rewards = reward[proposal_terminal]
    normalizer_estimate = proposal_rewards.mean()
    torch.testing.assert_close(
        normalizer_estimate,
        (base_terminal * reward).sum(),
        atol=0.01,
        rtol=0.0,
    )

    log_weights = proposal_rewards.log()
    ess = effective_sample_size(log_weights)
    assert 1.0 <= float(ess) <= n_samples
