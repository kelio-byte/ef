from edit_flows.utils.tokens import (
    BOS_TOKEN,
    GAP_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    bos_token_id,
    gap_token_id,
    model_vocab_size,
    pad_token_id,
    z_vocab_size,
)


class TestTokenHelpers:
    def test_special_token_ids_match_constants(self):
        assert pad_token_id(123) == PAD_TOKEN
        assert bos_token_id(123) == BOS_TOKEN
        assert gap_token_id(123) == GAP_TOKEN
        assert UNK_TOKEN == 3

    def test_vocab_size_helpers_include_all_special_tokens(self):
        assert model_vocab_size(68) == 72
        assert z_vocab_size(68) == 72
