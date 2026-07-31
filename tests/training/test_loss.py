import torch
from edit_flows.training.loss import bregman_loss
from edit_flows.core.scheduler import CubicScheduler


class TestBregmanLoss:
    def test_loss_positive(self):
        scheduler = CubicScheduler()
        log_ux_cat = torch.log(torch.clamp(torch.rand(2, 5, 33), min=1e-6))
        z_gap_mask = torch.zeros(2, 7, dtype=torch.bool)
        z_pad_mask = torch.zeros(2, 7, dtype=torch.bool)
        z_pad_mask[:, -2:] = True
        z_gap_mask[:, 2] = True
        z_gap_mask[:, 4] = True
        uz_mask = torch.zeros(2, 7, 33, dtype=torch.bool)
        uz_mask[:, 2, 5] = True
        uz_mask[:, 4, -1] = True
        t = torch.rand(2, 1) * 0.8 + 0.1

        loss = bregman_loss(log_ux_cat, z_gap_mask, z_pad_mask, uz_mask, t, scheduler)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_no_edits_low_loss(self):
        scheduler = CubicScheduler()
        log_ux_cat = torch.full((1, 3, 33), -18.42068)  # log(1e-8)
        z_gap_mask = torch.zeros(1, 3, dtype=torch.bool)
        z_pad_mask = torch.zeros(1, 3, dtype=torch.bool)
        uz_mask = torch.zeros(1, 3, 33, dtype=torch.bool)
        t = torch.tensor([[0.5]])

        loss = bregman_loss(log_ux_cat, z_gap_mask, z_pad_mask, uz_mask, t, scheduler)
        assert loss.item() < 1.0

    def test_loss_shape_scalar(self):
        scheduler = CubicScheduler()
        log_ux_cat = torch.log(torch.clamp(torch.rand(4, 5, 33), min=1e-6))
        z_gap_mask = torch.zeros(4, 7, dtype=torch.bool)
        z_pad_mask = torch.zeros(4, 7, dtype=torch.bool)
        uz_mask = torch.zeros(4, 7, 33, dtype=torch.bool)
        t = torch.rand(4, 1)

        loss = bregman_loss(log_ux_cat, z_gap_mask, z_pad_mask, uz_mask, t, scheduler)
        assert loss.dim() == 0

    def test_gradient_flow(self):
        scheduler = CubicScheduler()
        log_ux_cat = torch.rand(2, 3, 33, requires_grad=True)
        z_gap_mask = torch.zeros(2, 5, dtype=torch.bool)
        z_pad_mask = torch.zeros(2, 5, dtype=torch.bool)
        uz_mask = torch.zeros(2, 5, 33, dtype=torch.bool)
        uz_mask[:, 1, 3] = True
        t = torch.tensor([[0.3], [0.7]])

        loss = bregman_loss(log_ux_cat, z_gap_mask, z_pad_mask, uz_mask, t, scheduler)
        loss.backward()
        assert log_ux_cat.grad is not None
        assert (log_ux_cat.grad != 0).any()

    def test_rate_reparam_matches_equivalent_scaled_loss(self):
        scheduler = CubicScheduler()
        log_base = torch.log(torch.full((1, 2, 5), 0.7))
        z_gap_mask = torch.zeros(1, 2, dtype=torch.bool)
        z_pad_mask = torch.zeros(1, 2, dtype=torch.bool)
        uz_mask = torch.zeros(1, 2, 5, dtype=torch.bool)
        uz_mask[0, 0, 1] = True
        t = torch.tensor([[0.5]])

        scale = torch.clamp(scheduler.derivative(t) / (1 - scheduler(t) + 1e-8), max=50.0)
        loss_reparam = bregman_loss(
            log_base, z_gap_mask, z_pad_mask, uz_mask, t, scheduler,
            use_rate_reparam=True,
        )
        loss_direct = bregman_loss(
            log_base + torch.log(scale).unsqueeze(1), z_gap_mask, z_pad_mask, uz_mask, t, scheduler,
            use_rate_reparam=False,
        )
        expected_offset = -(scale * torch.log(scale)).sum()
        assert torch.allclose(loss_direct - loss_reparam, expected_offset, atol=1e-6)
