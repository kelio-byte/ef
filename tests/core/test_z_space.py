import torch
from edit_flows.core.z_space import (
    rm_gap_tokens, rv_gap_tokens, fill_gap_tokens_with_repeats,
    make_ut_mask_from_z, sample_cond_zt, project_mask_z_to_x,
)
from edit_flows.core.scheduler import CubicScheduler
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN, GAP_TOKEN


class TestRmGapTokens:
    def test_basic_removal(self):
        z = torch.tensor([
            [BOS_TOKEN, 3, GAP_TOKEN, 4, PAD_TOKEN],
            [BOS_TOKEN, GAP_TOKEN, 5, PAD_TOKEN, PAD_TOKEN],
        ])
        x, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z)
        assert x.shape[0] == 2
        assert GAP_TOKEN not in x[0].tolist()
        assert GAP_TOKEN not in x[1].tolist()

    def test_gap_mask_correct(self):
        z = torch.tensor([
            [BOS_TOKEN, 3, GAP_TOKEN, 4, PAD_TOKEN],
        ])
        _, _, z_gap_mask, _ = rm_gap_tokens(z)
        assert z_gap_mask[0, 2].item()
        assert not z_gap_mask[0, 0].item()
        assert not z_gap_mask[0, 1].item()

    def test_roundtrip(self):
        z = torch.tensor([
            [BOS_TOKEN, 3, GAP_TOKEN, 4, PAD_TOKEN],
            [BOS_TOKEN, GAP_TOKEN, 5, PAD_TOKEN, PAD_TOKEN],
        ])
        x, _, z_gap_mask, z_pad_mask = rm_gap_tokens(z)
        zr = rv_gap_tokens(x, z_gap_mask, z_pad_mask)
        assert torch.equal(z, zr)


class TestFillGapTokensWithRepeats:
    def test_repeat_fill(self):
        ux = torch.tensor([
            [[0.1, 0.2, 0.3, 0.4],
             [0.5, 0.6, 0.7, 0.8],
             [0.9, 1.0, 1.1, 1.2]],
        ])
        z_gap_mask = torch.tensor([
            [False, True, False, True, False],
        ])
        z_pad_mask = torch.tensor([
            [False, False, False, False, True],
        ])
        result = fill_gap_tokens_with_repeats(ux, z_gap_mask, z_pad_mask)
        assert result.shape == (1, 5, 4)
        assert torch.allclose(result[0, 0], ux[0, 0])
        assert torch.allclose(result[0, 1], ux[0, 0])
        assert torch.allclose(result[0, 2], ux[0, 1])
        assert torch.allclose(result[0, 3], ux[0, 1])
        assert result[0, 4].sum() == 0


class TestMakeUtMaskFromZ:
    def test_insert_mask(self):
        z_t = torch.tensor([[BOS_TOKEN, GAP_TOKEN, PAD_TOKEN]])
        z_1 = torch.tensor([[BOS_TOKEN, 6, PAD_TOKEN]])
        vocab_size = 10
        mask = make_ut_mask_from_z(z_t, z_1, vocab_size=vocab_size)
        assert mask.shape == (1, 3, 2 * vocab_size + 1)
        assert mask[0, 1, 6].item()

    def test_delete_mask(self):
        z_t = torch.tensor([[BOS_TOKEN, 6, PAD_TOKEN]])
        z_1 = torch.tensor([[BOS_TOKEN, GAP_TOKEN, PAD_TOKEN]])
        vocab_size = 10
        mask = make_ut_mask_from_z(z_t, z_1, vocab_size=vocab_size)
        assert mask[0, 1, -1].item()

    def test_substitute_mask(self):
        z_t = torch.tensor([[BOS_TOKEN, 6, PAD_TOKEN]])
        z_1 = torch.tensor([[BOS_TOKEN, 8, PAD_TOKEN]])
        vocab_size = 10
        mask = make_ut_mask_from_z(z_t, z_1, vocab_size=vocab_size)
        assert mask[0, 1, 8 + vocab_size].item()

    def test_no_edit_positions_zero(self):
        z_t = torch.tensor([[BOS_TOKEN, 5, 6, PAD_TOKEN]])
        z_1 = torch.tensor([[BOS_TOKEN, 5, 6, PAD_TOKEN]])
        vocab_size = 10
        mask = make_ut_mask_from_z(z_t, z_1, vocab_size=vocab_size)
        assert mask.sum() == 0

    def test_pad_positions_ignored(self):
        z_t = torch.tensor([[BOS_TOKEN, 5, PAD_TOKEN, PAD_TOKEN]])
        z_1 = torch.tensor([[BOS_TOKEN, 6, PAD_TOKEN, PAD_TOKEN]])
        vocab_size = 10
        mask = make_ut_mask_from_z(z_t, z_1, vocab_size=vocab_size)
        assert mask[:, 2:, :].sum() == 0


class TestSampleCondZt:
    def test_output_shape(self):
        z_0 = torch.tensor([[BOS_TOKEN, 3, GAP_TOKEN, PAD_TOKEN]])
        z_1 = torch.tensor([[BOS_TOKEN, GAP_TOKEN, 4, PAD_TOKEN]])
        t = torch.tensor([[0.3]])
        scheduler = CubicScheduler()
        z_t = sample_cond_zt(z_0, z_1, t, vocab_size=20, kappa_fn=scheduler)
        assert z_t.shape == z_0.shape

    def test_at_t0(self):
        z_0 = torch.tensor([[BOS_TOKEN, 3, GAP_TOKEN, PAD_TOKEN]])
        z_1 = torch.tensor([[BOS_TOKEN, GAP_TOKEN, 4, PAD_TOKEN]])
        t = torch.tensor([[0.0]])
        scheduler = CubicScheduler()
        torch.manual_seed(42)
        z_t = sample_cond_zt(z_0, z_1, t, vocab_size=20, kappa_fn=scheduler)
        assert torch.equal(z_t, z_0)

    def test_at_t1(self):
        z_0 = torch.tensor([[BOS_TOKEN, 3, GAP_TOKEN, PAD_TOKEN]])
        z_1 = torch.tensor([[BOS_TOKEN, GAP_TOKEN, 4, PAD_TOKEN]])
        t = torch.tensor([[1.0]])
        scheduler = CubicScheduler()
        torch.manual_seed(42)
        z_t = sample_cond_zt(z_0, z_1, t, vocab_size=20, kappa_fn=scheduler)
        assert torch.equal(z_t, z_1)

    def test_return_pick(self):
        z_0 = torch.tensor([[BOS_TOKEN, 3, GAP_TOKEN, PAD_TOKEN]])
        z_1 = torch.tensor([[BOS_TOKEN, GAP_TOKEN, 4, PAD_TOKEN]])
        t = torch.tensor([[1.0]])
        scheduler = CubicScheduler()
        z_t, pick_z1 = sample_cond_zt(
            z_0, z_1, t, vocab_size=20, kappa_fn=scheduler, return_pick=True,
        )
        assert z_t.shape == z_0.shape
        assert pick_z1.shape == z_0.shape
        assert pick_z1.dtype == torch.bool
        # at t=1.0, all positions should pick from z_1 (non-PAD)
        assert pick_z1[0, :3].all()


class TestProjectMaskZtoX:
    def test_basic_projection(self):
        mask_z = torch.tensor([
            [True, False, True, False, False],
            [True, True, False, False, False],
        ])
        z_gap_mask = torch.tensor([
            [False, False, True, False, True],
            [False, True, False, False, True],
        ])
        z_pad_mask = torch.tensor([
            [False, False, False, False, True],
            [False, False, False, False, True],
        ])
        x_shape = (2, 4)
        mask_x = project_mask_z_to_x(mask_z, z_gap_mask, z_pad_mask, x_shape)
        # Row 0: valid positions are 0(T), 1(F), 3(F) → [T, F, F, PAD]
        assert mask_x[0, 0].item() is True
        assert mask_x[0, 1].item() is False
        assert mask_x[0, 2].item() is False
        assert mask_x[0, 3].item() is False  # PAD default
        # Row 1: valid positions are 0(T), 2(F) → [T, F, PAD, PAD]
        assert mask_x[1, 0].item() is True
        assert mask_x[1, 1].item() is False
        assert mask_x[1, 2].item() is False  # PAD default

    def test_all_gap(self):
        mask_z = torch.tensor([[True, True, True]])
        z_gap_mask = torch.tensor([[False, True, False]])
        z_pad_mask = torch.tensor([[False, False, True]])
        x_shape = (1, 2)
        mask_x = project_mask_z_to_x(mask_z, z_gap_mask, z_pad_mask, x_shape)
        # Valid: 0(T), 2(T but PAD) → only position 0 maps
        assert mask_x[0, 0].item() is True
        assert mask_x[0, 1].item() is False  # PAD default
