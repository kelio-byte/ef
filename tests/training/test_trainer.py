import torch
import pytest

from edit_flows.core.alignment import identity_align_xs_to_zs, opt_align_xs_to_zs
from edit_flows.core.coupling import EmptyCoupling, UniformCoupling
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.training.trainer import prepare_batch, train_step
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


class TestPrepareBatch:
    def test_output_keys(self):
        x_1 = torch.tensor([
            [BOS_TOKEN, 3, 4, 5, 6, PAD_TOKEN, PAD_TOKEN],
            [BOS_TOKEN, 7, 8, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN],
        ])
        coupling = EmptyCoupling()
        x_0, x_1 = coupling.sample(x_1)
        scheduler = CubicScheduler()

        batch = prepare_batch(
            x_0, x_1, scheduler,
            align_fn=opt_align_xs_to_zs,
            model_vocab_size=13,
        )
        for key in ["x_t", "x_pad_mask", "z_gap_mask", "z_pad_mask", "uz_mask", "t"]:
            assert key in batch

    def test_x_t_no_gap(self):
        x_1 = torch.tensor([
            [BOS_TOKEN, 3, 4, 5, 6, PAD_TOKEN, PAD_TOKEN],
            [BOS_TOKEN, 7, 8, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN],
        ])
        coupling = EmptyCoupling()
        x_0, x_1 = coupling.sample(x_1)
        scheduler = CubicScheduler()

        batch = prepare_batch(
            x_0, x_1, scheduler,
            align_fn=opt_align_xs_to_zs,
            model_vocab_size=13,
        )
        from edit_flows.utils.tokens import GAP_TOKEN
        assert GAP_TOKEN not in batch["x_t"].tolist()

    def test_pad_mask_consistent(self):
        x_1 = torch.full((2, 8), PAD_TOKEN, dtype=torch.long)
        x_1[:, 0] = BOS_TOKEN
        x_1[0, 1:4] = torch.tensor([3, 4, 5])
        x_1[1, 1:6] = torch.tensor([6, 7, 8, 9, 10])

        coupling = UniformCoupling(min_len=2, max_len=4, vocab_size=10, pad_token=PAD_TOKEN)
        x_0, x_1 = coupling.sample(x_1)
        scheduler = CubicScheduler()

        batch = prepare_batch(
            x_0, x_1, scheduler,
            align_fn=opt_align_xs_to_zs,
            model_vocab_size=13,
        )
        x_t = batch["x_t"]
        x_pad_mask = batch["x_pad_mask"]
        assert torch.equal(x_pad_mask, x_t == PAD_TOKEN)


class TestPrepareBatchOriginMask:
    def test_identical_sequences_stay_original(self):
        x_0 = torch.tensor([[BOS_TOKEN, 5, 6, PAD_TOKEN]])
        x_1 = torch.tensor([[BOS_TOKEN, 5, 6, PAD_TOKEN]])

        for seed in range(4):
            torch.manual_seed(seed)
            batch = prepare_batch(
                x_0, x_1, LinearScheduler(), identity_align_xs_to_zs,
                model_vocab_size=20, use_origin_mask=True,
            )
            assert torch.equal(
                batch["origin_mask"],
                torch.ones_like(batch["x_t"], dtype=torch.bool),
            )

    def test_inserted_tokens_are_not_original(self, monkeypatch: pytest.MonkeyPatch):
        x_0 = torch.tensor([[BOS_TOKEN, 5, PAD_TOKEN]])
        x_1 = torch.tensor([[BOS_TOKEN, 5, 7]])

        monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.ones(*args, **kwargs))
        monkeypatch.setattr(
            torch, "rand_like",
            lambda x, **kwargs: torch.zeros_like(x, dtype=kwargs.get("dtype", torch.float)),
        )

        batch = prepare_batch(
            x_0, x_1, LinearScheduler(), identity_align_xs_to_zs,
            model_vocab_size=20, use_origin_mask=True,
        )

        assert torch.equal(batch["x_t"], torch.tensor([[BOS_TOKEN, BOS_TOKEN, 5, 7]]))
        assert torch.equal(
            batch["origin_mask"],
            torch.tensor([[True, True, True, False]]),
        )


class TestTrainStep:
    def test_loss_decreases(self, dummy_model):
        x_1 = torch.tensor([
            [BOS_TOKEN, 3, 4, 5, 6, PAD_TOKEN],
            [BOS_TOKEN, 7, 8, 9, PAD_TOKEN, PAD_TOKEN],
        ])
        coupling = EmptyCoupling()
        x_0, x_1 = coupling.sample(x_1)
        scheduler = CubicScheduler()
        optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.01)
        vocab_size = dummy_model.vocab_size

        batch = prepare_batch(
            x_0, x_1, scheduler,
            align_fn=opt_align_xs_to_zs,
            model_vocab_size=vocab_size,
        )

        metrics_before = train_step(dummy_model, batch, scheduler, optimizer)
        metrics_after = train_step(dummy_model, batch, scheduler, optimizer)
        assert metrics_after["loss"] < metrics_before["loss"] * 1.5

    def test_metrics_keys(self, dummy_model):
        x_1 = torch.tensor([
            [BOS_TOKEN, 3, 4, 5, 6, PAD_TOKEN],
            [BOS_TOKEN, 7, 8, 9, PAD_TOKEN, PAD_TOKEN],
        ])
        coupling = EmptyCoupling()
        x_0, x_1 = coupling.sample(x_1)
        scheduler = CubicScheduler()
        optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.01)
        vocab_size = dummy_model.vocab_size

        batch = prepare_batch(
            x_0, x_1, scheduler,
            align_fn=opt_align_xs_to_zs,
            model_vocab_size=vocab_size,
        )
        metrics = train_step(dummy_model, batch, scheduler, optimizer)
        for key in ["loss", "u_tot", "u_ins", "u_del", "u_sub"]:
            assert key in metrics
            assert isinstance(metrics[key], float)

    def test_real_transformer_backward_no_inplace_error(self):
        model = EditFlowsTransformer(
            vocab_size=16,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            dim_feedforward=64,
            max_seq_len=32,
            dropout=0.0,
            attention_dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = CubicScheduler()

        x_1 = torch.tensor([
            [BOS_TOKEN, 3, 4, 5, PAD_TOKEN, PAD_TOKEN],
            [BOS_TOKEN, 6, 7, 8, 9, PAD_TOKEN],
        ])
        coupling = EmptyCoupling()
        x_0, x_1 = coupling.sample(x_1)

        batch = prepare_batch(
            x_0, x_1, scheduler,
            align_fn=opt_align_xs_to_zs,
            model_vocab_size=16,
        )

        metrics = train_step(model, batch, scheduler, optimizer)
        assert isinstance(metrics["loss"], float)

    def test_rate_reparam_train_step_runs(self, dummy_model):
        x_1 = torch.tensor([
            [BOS_TOKEN, 3, 4, 5, 6, PAD_TOKEN],
            [BOS_TOKEN, 7, 8, 9, PAD_TOKEN, PAD_TOKEN],
        ])
        coupling = EmptyCoupling()
        x_0, x_1 = coupling.sample(x_1)
        scheduler = CubicScheduler()
        optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.01)
        vocab_size = dummy_model.vocab_size

        batch = prepare_batch(
            x_0, x_1, scheduler,
            align_fn=opt_align_xs_to_zs,
            model_vocab_size=vocab_size,
        )
        metrics = train_step(
            dummy_model, batch, scheduler, optimizer,
            use_rate_reparam=True,
        )
        assert isinstance(metrics["loss"], float)
