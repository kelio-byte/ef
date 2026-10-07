import torch
from edit_flows.sampling.ops import (
    apply_ins_del_operations,
    legal_token_log_probs,
)
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


class TestApplyInsDelOperations:
    def test_bos_anchor_insertion_preserves_sentinel(self):
        x_t = torch.tensor([[BOS_TOKEN, 4, PAD_TOKEN]])
        ins_mask = torch.tensor([[True, False, False]])
        del_mask = torch.tensor([[False, False, False]])
        ins_tokens = torch.tensor([[9, PAD_TOKEN, PAD_TOKEN]])

        result = apply_ins_del_operations(
            x_t, ins_mask, del_mask, ins_tokens,
            max_seq_len=10, pad_token=PAD_TOKEN,
        )

        assert torch.equal(
            result[0][result[0] != PAD_TOKEN],
            torch.tensor([BOS_TOKEN, 9, 4]),
        )

    def test_pure_insertion(self):
        x_t = torch.tensor([[1, 2, 3, PAD_TOKEN, PAD_TOKEN]])
        ins_mask = torch.tensor([[False, True, False, False, False]])
        del_mask = torch.tensor([[False, False, False, False, False]])
        ins_tokens = torch.tensor([[PAD_TOKEN, 9, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]])

        result = apply_ins_del_operations(
            x_t, ins_mask, del_mask, ins_tokens,
            max_seq_len=10, pad_token=PAD_TOKEN,
        )
        result_no_pad = result[0][result[0] != PAD_TOKEN]
        expected = torch.tensor([1, 2, 9, 3])
        assert torch.equal(result_no_pad, expected)

    def test_pure_deletion(self):
        x_t = torch.tensor([[1, 2, 3, 4, PAD_TOKEN]])
        ins_mask = torch.tensor([[False, False, False, False, False]])
        del_mask = torch.tensor([[False, True, False, False, False]])
        ins_tokens = torch.full((1, 5), PAD_TOKEN, dtype=torch.long)

        result = apply_ins_del_operations(
            x_t, ins_mask, del_mask, ins_tokens,
            max_seq_len=10, pad_token=PAD_TOKEN,
        )
        result_no_pad = result[0][result[0] != PAD_TOKEN]
        expected = torch.tensor([1, 3, 4])
        assert torch.equal(result_no_pad, expected)

    def test_simultaneous_ins_del_becomes_sub(self):
        x_t = torch.tensor([[1, 2, 3, PAD_TOKEN]])
        ins_mask = torch.tensor([[False, True, False, False]])
        del_mask = torch.tensor([[False, True, False, False]])
        ins_tokens = torch.tensor([[PAD_TOKEN, 9, PAD_TOKEN, PAD_TOKEN]])

        result = apply_ins_del_operations(
            x_t, ins_mask, del_mask, ins_tokens,
            max_seq_len=10, pad_token=PAD_TOKEN,
        )
        result_no_pad = result[0][result[0] != PAD_TOKEN]
        expected = torch.tensor([1, 9, 3])
        assert torch.equal(result_no_pad, expected)

    def test_mixed_operations(self):
        x_t = torch.tensor([[1, 2, 3, 4, 5, PAD_TOKEN, PAD_TOKEN]])
        ins_mask = torch.tensor([[False, True, False, False, False, False, False]])
        del_mask = torch.tensor([[False, False, False, True, False, False, False]])
        ins_tokens = torch.tensor([[PAD_TOKEN, 9, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]])

        result = apply_ins_del_operations(
            x_t, ins_mask, del_mask, ins_tokens,
            max_seq_len=10, pad_token=PAD_TOKEN,
        )
        result_no_pad = result[0][result[0] != PAD_TOKEN]
        expected = torch.tensor([1, 2, 9, 3, 5])
        assert torch.equal(result_no_pad, expected)

    def test_all_deleted(self):
        x_t = torch.tensor([[1, PAD_TOKEN, PAD_TOKEN]])
        ins_mask = torch.tensor([[False, False, False]])
        del_mask = torch.tensor([[True, False, False]])
        ins_tokens = torch.full((1, 3), PAD_TOKEN, dtype=torch.long)

        result = apply_ins_del_operations(
            x_t, ins_mask, del_mask, ins_tokens,
            max_seq_len=10, pad_token=PAD_TOKEN,
        )
        assert result.shape[1] == 1
        assert result[0, 0] == PAD_TOKEN

    def test_multi_insertion_ordering(self):
        x_t = torch.tensor([[1, 4, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]])
        ins_mask = torch.tensor([[True, True, False, False, False]])
        del_mask = torch.tensor([[False, False, False, False, False]])
        ins_tokens = torch.tensor([[2, 3, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]])

        result = apply_ins_del_operations(
            x_t, ins_mask, del_mask, ins_tokens,
            max_seq_len=10, pad_token=PAD_TOKEN,
        )
        result_no_pad = result[0][result[0] != PAD_TOKEN]
        expected = torch.tensor([1, 2, 4, 3])
        assert torch.equal(result_no_pad, expected)

    def test_batched(self):
        x_t = torch.tensor([
            [1, 2, 3, PAD_TOKEN],
            [4, 5, 6, PAD_TOKEN],
        ])
        ins_mask = torch.tensor([
            [False, True, False, False],
            [True, False, False, False],
        ])
        del_mask = torch.tensor([
            [False, False, False, False],
            [False, False, True, False],
        ])
        ins_tokens = torch.tensor([
            [PAD_TOKEN, 9, PAD_TOKEN, PAD_TOKEN],
            [8, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN],
        ])

        result = apply_ins_del_operations(
            x_t, ins_mask, del_mask, ins_tokens,
            max_seq_len=10, pad_token=PAD_TOKEN,
        )
        r0 = result[0][result[0] != PAD_TOKEN]
        r1 = result[1][result[1] != PAD_TOKEN]
        assert torch.equal(r0, torch.tensor([1, 2, 9, 3]))
        assert torch.equal(r1, torch.tensor([4, 8, 5]))


def test_legal_token_log_probs_does_not_revive_masked_sentinel_mass():
    log_probs = torch.full((1, 1, 8), -1e9)
    log_probs[0, 0, PAD_TOKEN] = 0.0

    normalized, normalizer = legal_token_log_probs(log_probs)

    assert not torch.isfinite(normalizer[0, 0])
    assert not torch.isfinite(normalized[0, 0]).any()
