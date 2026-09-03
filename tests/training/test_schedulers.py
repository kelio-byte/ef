import math

import torch

from edit_flows.training.schedulers import NoamScheduler


def test_noam_step_zero_is_safe_and_step_sets_update_lr():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=0.0)
    scheduler = NoamScheduler(
        optimizer, d_model=256, warmup_steps=8000, factor=1.0,
    )

    assert scheduler.get_lr() == 0.0
    expected = scheduler.get_lr(step=1)
    actual = scheduler.step()

    assert scheduler.state_dict() == {"_step": 1}
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(
        optimizer.param_groups[0]["lr"], expected, rel_tol=0.0, abs_tol=1e-15,
    )


def test_noam_state_round_trip_preserves_next_learning_rate():
    parameter_a = torch.nn.Parameter(torch.tensor(1.0))
    optimizer_a = torch.optim.Adam([parameter_a], lr=0.0)
    scheduler_a = NoamScheduler(optimizer_a, d_model=64, warmup_steps=4)
    for _ in range(3):
        scheduler_a.step()

    parameter_b = torch.nn.Parameter(torch.tensor(1.0))
    optimizer_b = torch.optim.Adam([parameter_b], lr=0.0)
    scheduler_b = NoamScheduler(optimizer_b, d_model=64, warmup_steps=4)
    scheduler_b.load_state_dict(scheduler_a.state_dict())

    assert scheduler_b.step() == scheduler_a.get_lr(step=4)
