import torch
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler


class TestCubicScheduler:
    def test_endpoints(self):
        sched = CubicScheduler()
        t0 = torch.tensor([[0.0]])
        t1 = torch.tensor([[1.0]])
        assert torch.allclose(sched(t0), torch.tensor([[0.0]]), atol=1e-6)
        assert torch.allclose(sched(t1), torch.tensor([[1.0]]), atol=1e-6)

    def test_monotonic(self):
        sched = CubicScheduler()
        ts = torch.linspace(0, 1, 100).reshape(-1, 1)
        vals = sched(ts)
        assert (vals[1:] >= vals[:-1]).all()

    def test_derivative_positive(self):
        sched = CubicScheduler()
        ts = torch.linspace(0.01, 0.99, 100).reshape(-1, 1)
        derivs = sched.derivative(ts)
        assert (derivs > 0).all()

    def test_derivative_consistency(self):
        sched = CubicScheduler()
        t = torch.tensor([[0.5]])
        eps = 1e-4
        numerical = (sched(t + eps) - sched(t - eps)) / (2 * eps)
        analytical = sched.derivative(t)
        assert torch.allclose(numerical, analytical, atol=1e-3)

    def test_cubic_a1_b1(self):
        sched = CubicScheduler(a=1.0, b=1.0)
        t = torch.tensor([[0.5]])
        assert torch.allclose(sched(t), torch.tensor([[0.125]]), atol=1e-6)
        assert torch.allclose(sched.derivative(t), torch.tensor([[0.75]]), atol=1e-6)

    def test_batched_input(self):
        sched = CubicScheduler()
        t = torch.rand(4, 1)
        out = sched(t)
        deriv = sched.derivative(t)
        assert out.shape == (4, 1)
        assert deriv.shape == (4, 1)

    def test_inverse_roundtrip(self):
        sched = CubicScheduler()
        t = torch.linspace(0, 1, 20).reshape(-1, 1)
        kappa = sched(t)
        recovered = sched.inverse(kappa)
        assert torch.allclose(recovered, t, atol=1e-6)

    def test_name(self):
        assert CubicScheduler().name == "cubic"


class TestLinearScheduler:
    def test_endpoints(self):
        sched = LinearScheduler()
        assert torch.allclose(sched(torch.tensor([[0.0]])), torch.tensor([[0.0]]))
        assert torch.allclose(sched(torch.tensor([[1.0]])), torch.tensor([[1.0]]))

    def test_derivative(self):
        sched = LinearScheduler()
        ts = torch.rand(10, 1)
        assert torch.allclose(sched.derivative(ts), torch.ones_like(ts))

    def test_inverse_roundtrip(self):
        sched = LinearScheduler()
        t = torch.linspace(0, 1, 20).reshape(-1, 1)
        kappa = sched(t)
        recovered = sched.inverse(kappa)
        assert torch.allclose(recovered, t, atol=1e-6)

    def test_name(self):
        assert LinearScheduler().name == "linear"
