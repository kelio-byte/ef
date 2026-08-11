import pytest
import torch
import torch.nn as nn
from edit_flows.sampling.euler import (
    _compute_model_time,
    _event_probability,
    get_euler_step_times,
    _sample_edit_actions,
    get_adaptive_h,
    sample_euler,
)
from edit_flows.core.rate_scale import get_rate_scale
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN


class TestGetAdaptiveH:
    def test_basic(self):
        scheduler = CubicScheduler()
        t = torch.tensor([[0.5]])
        h = get_adaptive_h(0.1, t, scheduler)
        assert h.shape == (1, 1)
        assert 0 < h.item() <= 0.1 + 1e-6

    def test_decreases_near_one(self):
        scheduler = CubicScheduler()
        t_mid = torch.tensor([[0.5]])
        t_end = torch.tensor([[0.999]])
        h_mid = get_adaptive_h(0.1, t_mid, scheduler)
        h_end = get_adaptive_h(0.1, t_end, scheduler)
        assert h_end.item() < h_mid.item()

    def test_step_times_use_actual_adaptive_endpoint(self):
        times = get_euler_step_times(100, CubicScheduler())
        assert abs(times[50] - 0.5) < 1e-6
        assert times[-1] >= 1.0


class TestEventProbability:
    def test_poisson_mode(self):
        mu = torch.tensor([0.0, 0.5, 2.0])
        out = _event_probability(mu, "poisson")
        expected = 1 - torch.exp(-mu)
        assert torch.allclose(out, expected)

    def test_linear_mode(self):
        mu = torch.tensor([-1.0, 0.5, 2.0])
        out = _event_probability(mu, "linear")
        expected = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(out, expected)


class TestSampleEuler:
    def test_explicit_zero_start_time_matches_baseline(self, dummy_model):
        x_0 = torch.tensor([[BOS_TOKEN, 3, 4, PAD_TOKEN]])
        torch.manual_seed(1234)
        baseline, _ = sample_euler(
            dummy_model, x_0, CubicScheduler(), n_steps=4, max_seq_len=32,
        )
        torch.manual_seed(1234)
        explicit, _ = sample_euler(
            dummy_model, x_0, CubicScheduler(), n_steps=4, max_seq_len=32,
            start_time=0.0,
        )
        assert torch.equal(baseline, explicit)

    def test_start_time_one_is_a_noop(self, dummy_model):
        x_0 = torch.tensor([[BOS_TOKEN, 3, 4, PAD_TOKEN]])
        result, trajectory = sample_euler(
            dummy_model, x_0, CubicScheduler(), n_steps=4, max_seq_len=32,
            start_time=1.0, record_trajectory=True,
        )
        assert torch.equal(result, x_0)
        assert len(trajectory) == 1
        assert torch.equal(trajectory[0], x_0)

    def test_guidance_beta_zero_matches_baseline(self, dummy_model):
        class ConstantGuidance(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))

            def forward(
                self, product, state, time, product_padding, state_padding,
            ):
                b, l = state.shape
                h = torch.ones(
                    b, l, 19, device=state.device,
                ) + self.anchor * 0
                return h, h.clone(), h[:, :, :1].clone()

        x_0 = torch.tensor([[BOS_TOKEN, 3, 4, PAD_TOKEN]])
        guidance = ConstantGuidance()
        torch.manual_seed(99)
        baseline, _ = sample_euler(
            dummy_model, x_0, CubicScheduler(), n_steps=4, max_seq_len=32,
        )
        torch.manual_seed(99)
        guided, _ = sample_euler(
            dummy_model, x_0, CubicScheduler(), n_steps=4, max_seq_len=32,
            guidance_model=guidance, guidance_product=x_0, guidance_beta=0.0,
            guidance_rate_normalization="per_sample",
        )
        assert torch.equal(baseline, guided)

    def test_bos_is_never_sampled_as_an_edit_position(self):
        x_t = torch.tensor([[BOS_TOKEN, 7, PAD_TOKEN]])
        log_rates = torch.full((1, 3, 3), 20.0)
        log_probs = torch.log_softmax(torch.zeros(1, 3, 16), dim=-1)
        actions = _sample_edit_actions(
            x_t,
            log_rates,
            log_probs,
            log_probs,
            torch.tensor([[0.1]]),
            pad_token=PAD_TOKEN,
        )
        assert not bool(actions["ins_mask"][0, 0])
        assert not bool(actions["sub_mask"][0, 0])
        assert not bool(actions["del_mask"][0, 0])

    def test_empty_prior_generates(self, dummy_model):
        dummy_model.eval()
        scheduler = CubicScheduler()
        x_0 = torch.empty((2, 0), dtype=torch.long)
        result, _ = sample_euler(
            dummy_model, x_0, scheduler,
            n_steps=10, max_seq_len=64,
        )
        assert result.shape[0] == 2

    def test_nonempty_prior(self, dummy_model):
        dummy_model.eval()
        scheduler = CubicScheduler()
        x_0 = torch.tensor([
            [BOS_TOKEN, 3, 4, PAD_TOKEN, PAD_TOKEN],
            [BOS_TOKEN, 5, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN],
        ])
        result, _ = sample_euler(
            dummy_model, x_0, scheduler,
            n_steps=10, max_seq_len=64,
        )
        assert result.shape[0] == 2
        assert not torch.isnan(result).any()

    def test_trajectory_recording(self, dummy_model):
        dummy_model.eval()
        scheduler = CubicScheduler()
        x_0 = torch.tensor([
            [BOS_TOKEN, 3, PAD_TOKEN, PAD_TOKEN],
        ])
        result, trajectory = sample_euler(
            dummy_model, x_0, scheduler,
            n_steps=5, max_seq_len=32,
            record_trajectory=True,
        )
        assert len(trajectory) > 0
        assert trajectory[0].shape[0] == 1

    def test_capped_trajectory_keeps_first_post_step_without_changing_terminal(
        self, dummy_model,
    ):
        """Diagnostic state storage must not alter the sampled trajectory."""
        dummy_model.eval()
        scheduler = CubicScheduler()
        x_0 = torch.tensor([[BOS_TOKEN, 3, 4, PAD_TOKEN]])
        torch.manual_seed(2468)
        baseline, _ = sample_euler(
            dummy_model, x_0, scheduler, n_steps=5, max_seq_len=32,
        )
        torch.manual_seed(2468)
        capped, trajectory = sample_euler(
            dummy_model, x_0, scheduler, n_steps=5, max_seq_len=32,
            record_trajectory=True,
            max_recorded_trajectory_steps=1,
        )
        assert torch.equal(baseline, capped)
        assert len(trajectory) == 2
        assert torch.equal(trajectory[0], x_0)

    def test_capped_trajectory_requires_trajectory_recording(self, dummy_model):
        with pytest.raises(ValueError, match="requires record_trajectory"):
            sample_euler(
                dummy_model,
                torch.tensor([[BOS_TOKEN, 3, PAD_TOKEN]]),
                CubicScheduler(),
                n_steps=5,
                max_recorded_trajectory_steps=1,
            )

    def test_first_event_recording(self, dummy_model):
        dummy_model.eval()
        scheduler = CubicScheduler()
        x_0 = torch.tensor([
            [BOS_TOKEN, 3, 4, PAD_TOKEN],
        ])
        x_1 = torch.tensor([
            [BOS_TOKEN, 3, 5, PAD_TOKEN],
        ])
        result, _, first_events = sample_euler(
            dummy_model, x_0, scheduler,
            n_steps=5, max_seq_len=32,
            record_first_events=True,
            x_1=x_1,
            vocab_size=16,
        )
        assert result.shape[0] == 1
        assert len(first_events) == 1

    def test_all_event_recording_does_not_change_sampling(self, dummy_model):
        dummy_model.eval()
        scheduler = CubicScheduler()
        x_0 = torch.tensor([[BOS_TOKEN, 3, 4, PAD_TOKEN]])

        torch.manual_seed(123)
        result_plain, _ = sample_euler(
            dummy_model, x_0, scheduler,
            n_steps=5, max_seq_len=32,
        )
        torch.manual_seed(123)
        result_recorded, _, _ = sample_euler(
            dummy_model, x_0, scheduler,
            n_steps=5, max_seq_len=32, record_all_events=True,
        )

        assert torch.equal(result_plain, result_recorded)


class TestModelTimeMapping:
    def test_compute_model_time_same_scheduler_returns_t(self):
        t = torch.tensor([[0.5]])
        scheduler = CubicScheduler()
        out = _compute_model_time(t, scheduler, "t", scheduler)
        assert torch.allclose(out, t)

    def test_compute_model_time_kappa_mode_returns_sample_kappa(self):
        t = torch.tensor([[0.5]])
        sample_scheduler = CubicScheduler()
        out = _compute_model_time(t, sample_scheduler, "kappa", LinearScheduler())
        assert torch.allclose(out, sample_scheduler(t))

    def test_compute_model_time_cross_scheduler_maps_via_kappa(self):
        t = torch.tensor([[0.5]])
        sample_scheduler = LinearScheduler()
        train_scheduler = CubicScheduler()
        out = _compute_model_time(t, sample_scheduler, "t", train_scheduler)
        expected = train_scheduler.inverse(sample_scheduler(t))
        assert torch.allclose(out, expected)

    def test_compute_model_time_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported time_input"):
            _compute_model_time(torch.tensor([[0.5]]), CubicScheduler(), "bad")


class TestCrossSchedulerCorrection:
    def test_raw_rate_correction_matches_scale_ratio(self):
        t = torch.tensor([[0.5]])
        sample_scheduler = LinearScheduler()
        train_scheduler = CubicScheduler()
        t_model = _compute_model_time(t, sample_scheduler, "t", train_scheduler)
        k_sample = get_rate_scale(t, sample_scheduler)
        k_train = get_rate_scale(t_model, train_scheduler)
        correction = torch.log(k_sample / k_train.clamp_min(1e-12))
        expected = torch.log(torch.tensor([[2.0 / (3 * (0.5 ** (2/3)) / (1 - 0.5))]]))
        assert torch.allclose(correction, expected, atol=1e-6)


class OriginMaskProbeModel(nn.Module):
    def __init__(self, op: str, target_token: int = 9):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.op = op
        self.target_token = target_token
        self.observed_masks = []

    def forward(self, tokens, time_step, padding_mask, origin_mask=None):
        if origin_mask is None:
            self.observed_masks.append(None)
        else:
            self.observed_masks.append(origin_mask.detach().clone())

        b, l = tokens.shape
        log_rates = torch.full((b, l, 3), -30.0, device=tokens.device)
        log_ins_probs = torch.full((b, l, 16), -1e9, device=tokens.device)
        log_sub_probs = torch.full((b, l, 16), -1e9, device=tokens.device)
        log_ins_probs[:, :, self.target_token] = 0.0
        log_sub_probs[:, :, self.target_token] = 0.0

        if len(self.observed_masks) == 1 and l > 1:
            if self.op == "sub":
                log_rates[:, 1, 1] = 20.0
            elif self.op == "ins":
                log_rates[:, 1, 0] = 20.0
            elif self.op == "del":
                log_rates[:, 1, 2] = 20.0
            elif self.op == "replace":
                log_rates[:, 1, 0] = 20.0
                log_rates[:, 1, 2] = 20.0
            else:
                raise ValueError(f"Unsupported op: {self.op}")

        log_rates = log_rates.masked_fill(padding_mask.unsqueeze(-1), -30.0)
        return log_rates, log_ins_probs, log_sub_probs


class TestOriginMaskSampling:
    @pytest.mark.parametrize(
        ("op", "expected_tokens", "expected_mask"),
        [
            ("sub", torch.tensor([[BOS_TOKEN, 9, 4]]), torch.tensor([[True, False, True]])),
            ("ins", torch.tensor([[BOS_TOKEN, 3, 9, 4]]), torch.tensor([[True, True, False, True]])),
            ("del", torch.tensor([[BOS_TOKEN, 4]]), torch.tensor([[True, True]])),
            ("replace", torch.tensor([[BOS_TOKEN, 9, 4]]), torch.tensor([[True, False, True]])),
        ],
    )
    def test_origin_mask_tracks_edit_history(
        self,
        monkeypatch: pytest.MonkeyPatch,
        op: str,
        expected_tokens: torch.Tensor,
        expected_mask: torch.Tensor,
    ):
        monkeypatch.setattr(
            torch, "rand_like",
            lambda x, **kwargs: torch.zeros_like(x, dtype=kwargs.get("dtype", x.dtype)),
        )

        model = OriginMaskProbeModel(op=op)
        x_0 = torch.tensor([[BOS_TOKEN, 3, 4, PAD_TOKEN]])

        result, _ = sample_euler(
            model, x_0, LinearScheduler(),
            n_steps=2, max_seq_len=16, use_origin_mask=True,
        )

        result = result[:, :expected_tokens.shape[1]]
        assert torch.equal(result, expected_tokens)
        assert torch.equal(model.observed_masks[0], torch.tensor([[True, True, True, True]]))
        assert torch.equal(model.observed_masks[1], expected_mask)

    def test_all_event_recording_contains_exact_post_edit_state(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            torch, "rand_like",
            lambda x, **kwargs: torch.zeros_like(
                x, dtype=kwargs.get("dtype", x.dtype),
            ),
        )
        model = OriginMaskProbeModel(op="ins", target_token=9)
        x_0 = torch.tensor([[BOS_TOKEN, 3, 4, PAD_TOKEN]])

        result, _, all_events = sample_euler(
            model, x_0, LinearScheduler(),
            n_steps=2, max_seq_len=16, record_all_events=True,
        )

        assert len(all_events[0]) == 1
        assert torch.equal(all_events[0][0]["x_t"], x_0[0])
        assert torch.equal(all_events[0][0]["x_next"], result[0])
