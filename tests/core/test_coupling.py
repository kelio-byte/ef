import torch
from edit_flows.core.coupling import EmptyCoupling, UniformCoupling, ExtendedCoupling
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN


class TestEmptyCoupling:
    def test_empty_prior(self):
        x1 = torch.randint(3, 13, (4, 8))
        coupling = EmptyCoupling()
        x0, x1_out = coupling.sample(x1)
        assert x0.shape == (4, 0)
        assert torch.equal(x1_out, x1)

    def test_different_lengths(self):
        x1 = torch.full((2, 10), 3, dtype=torch.long)
        x1[0, 7:] = PAD_TOKEN
        coupling = EmptyCoupling()
        x0, _ = coupling.sample(x1)
        assert x0.shape[0] == 2
        assert x0.shape[1] == 0


class TestUniformCoupling:
    def test_shape(self):
        x1 = torch.randint(3, 13, (4, 8))
        coupling = UniformCoupling(min_len=3, max_len=6, vocab_size=13, pad_token=PAD_TOKEN)
        x0, _ = coupling.sample(x1)
        assert x0.shape[0] == 4
        assert 3 <= x0.shape[1] <= 6

    def test_pad_positions(self):
        coupling = UniformCoupling(min_len=3, max_len=6, vocab_size=10, pad_token=PAD_TOKEN)
        x1 = torch.randint(3, 13, (4, 8))
        x0, _ = coupling.sample(x1)
        for b in range(4):
            assert 3 <= x0.shape[1] <= 6

    def test_mirror_len(self):
        coupl = UniformCoupling(
            min_len=2, max_len=10, vocab_size=7, mirror_len=True, pad_token=PAD_TOKEN,
        )
        x1 = torch.full((4, 10), PAD_TOKEN, dtype=torch.long)
        x1[:, 0] = BOS_TOKEN
        x1[0, 1:6] = torch.tensor([3, 4, 5, 6, 7])
        x1[1, 1:4] = torch.tensor([3, 4, 5])
        x1[2, 1:9] = torch.tensor([3, 4, 5, 6, 7, 8, 9, 3])
        x1[3, 1:3] = torch.tensor([3, 4])
        x0, _ = coupl.sample(x1)
        assert x0.shape == x1.shape


class TestExtendedCoupling:
    def test_shape(self):
        x1 = torch.randint(3, 13, (4, 12))
        n_insert = 5
        coupling = ExtendedCoupling(n_insert=n_insert, vocab_size=13, pad_token=PAD_TOKEN)
        x0, _ = coupling.sample(x1)
        assert x0.shape[1] == x1.shape[1] + n_insert

    def test_preserves_original(self):
        x1 = torch.randint(3, 13, (2, 12))
        coupling = ExtendedCoupling(n_insert=3, vocab_size=13, pad_token=PAD_TOKEN)
        x0, _ = coupling.sample(x1)
        for b in range(2):
            x0_no_pad = x0[b][x0[b] != PAD_TOKEN]
            x1_no_pad = x1[b][x1[b] != PAD_TOKEN]
            assert all(t in x0_no_pad.tolist() for t in x1_no_pad.tolist())
