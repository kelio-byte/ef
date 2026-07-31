import torch

from edit_flows.core.rate_scale import apply_rate_parameterization, get_rate_scale
from edit_flows.core.scheduler import CubicScheduler


class TestRateScale:
    def test_get_rate_scale_matches_scheduler_formula(self):
        scheduler = CubicScheduler()
        t = torch.tensor([[0.5]])
        scale = get_rate_scale(t, scheduler)
        expected = torch.tensor([[3 * 0.5**2 / (1 - 0.5**3)]])
        assert torch.allclose(scale, expected)

    def test_apply_rate_parameterization_disabled_is_identity(self):
        scheduler = CubicScheduler()
        log_rates = torch.tensor([[[0.1, 0.2, 0.3]]])
        t = torch.tensor([[0.5]])
        out = apply_rate_parameterization(log_rates, t, scheduler, use_rate_reparam=False)
        assert torch.allclose(out, log_rates)

    def test_apply_rate_parameterization_enabled_adds_log_scale(self):
        scheduler = CubicScheduler()
        log_rates = torch.zeros(1, 2, 3)
        t = torch.tensor([[0.5]])
        out = apply_rate_parameterization(log_rates, t, scheduler, use_rate_reparam=True)
        expected = torch.log(get_rate_scale(t, scheduler)).unsqueeze(1).expand_as(out)
        assert torch.allclose(out, expected)
