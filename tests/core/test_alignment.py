import torch
from edit_flows.core.alignment import (
    opt_align_xs_to_zs, naive_align_xs_to_zs, shifted_align_xs_to_zs,
)
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN, GAP_TOKEN


class TestOptAlign:
    def test_equal_length(self):
        x_0 = torch.tensor([[BOS_TOKEN, 3, 4, 5, 6]])
        x_1 = torch.tensor([[BOS_TOKEN, 3, 4, 5, 6]])
        z_0, z_1 = opt_align_xs_to_zs(x_0, x_1)
        assert z_0.shape == z_1.shape
        assert torch.equal(z_0, z_1)

    def test_insertion(self):
        x_0 = torch.tensor([[BOS_TOKEN, 3, 5]])
        x_1 = torch.tensor([[BOS_TOKEN, 3, 4, 5]])
        z_0, z_1 = opt_align_xs_to_zs(x_0, x_1)
        assert z_0.shape[1] == 4
        assert GAP_TOKEN in z_0[0].tolist()
        assert 4 in z_1[0].tolist()

    def test_deletion(self):
        x_0 = torch.tensor([[BOS_TOKEN, 3, 4, 5]])
        x_1 = torch.tensor([[BOS_TOKEN, 3, 5]])
        z_0, z_1 = opt_align_xs_to_zs(x_0, x_1)
        assert z_0.shape[1] == 4
        assert GAP_TOKEN in z_1[0].tolist()

    def test_substitution(self):
        x_0 = torch.tensor([[BOS_TOKEN, 3, 4, 5]])
        x_1 = torch.tensor([[BOS_TOKEN, 3, 6, 5]])
        z_0, z_1 = opt_align_xs_to_zs(x_0, x_1)
        assert z_0.shape[1] == 4
        assert 4 in z_0[0].tolist()
        assert 6 in z_1[0].tolist()

    def test_edit_distance_minimal(self):
        x_0 = torch.tensor([[3, 4, 5]])
        x_1 = torch.tensor([[3, 6, 5]])
        z_0, z_1 = opt_align_xs_to_zs(x_0, x_1)
        n_edits = (z_0 != z_1).sum().item()
        assert n_edits == 1

    def test_batch_padding_is_not_aligned_as_a_real_token(self):
        x_0 = torch.tensor([
            [BOS_TOKEN, 3, 4, PAD_TOKEN],
            [BOS_TOKEN, 5, PAD_TOKEN, PAD_TOKEN],
        ])
        x_1 = torch.tensor([
            [BOS_TOKEN, 3, 4, 6],
            [BOS_TOKEN, 5, 7, PAD_TOKEN],
        ])

        z_0, z_1 = opt_align_xs_to_zs(x_0, x_1)

        assert z_0.shape == z_1.shape
        assert torch.equal(
            (z_0[0] != PAD_TOKEN),
            torch.tensor([True, True, True, True]),
        )
        assert torch.equal(
            (z_0[1] != PAD_TOKEN),
            torch.tensor([True, True, True, False]),
        )
        assert PAD_TOKEN not in z_0[0].tolist()
        assert PAD_TOKEN not in z_1[0].tolist()


class TestNaiveAlign:
    def test_equal_length_output(self):
        x_0 = torch.tensor([[3, 4, 5, PAD_TOKEN, PAD_TOKEN]])
        x_1 = torch.tensor([[3, 4, 5, 6, 7]])
        z_0, z_1 = naive_align_xs_to_zs(x_0, x_1)
        assert z_0.shape[1] == z_1.shape[1]


class TestShiftedAlign:
    def test_correct_structure(self):
        x_0 = torch.tensor([[3, 4, PAD_TOKEN]])
        x_1 = torch.tensor([[5, 6, 7]])
        z_0, z_1 = shifted_align_xs_to_zs(x_0, x_1)
        z_0_non_pad = z_0[0][z_0[0] != PAD_TOKEN]
        z_1_non_pad = z_1[0][z_1[0] != PAD_TOKEN]
        assert 3 in z_0_non_pad.tolist()
        assert 4 in z_0_non_pad.tolist()
        assert 5 in z_1_non_pad.tolist()
        assert 6 in z_1_non_pad.tolist()
        assert 7 in z_1_non_pad.tolist()
