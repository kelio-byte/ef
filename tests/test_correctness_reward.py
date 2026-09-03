from scripts.train_correctness_reward import (
    INVALID_SCORE,
    _auc_from_pairs,
    _feature_vector,
    _rank_metrics,
)


def test_correctness_features_are_label_free_and_deterministic():
    first = _feature_vector(
        [1, 2, 3, 0], [1, 4, 0],
        product_canonical="CCO", candidate_canonical="CCN",
        raw_forward_reward=0.5, vocab_size=8,
    )
    second = _feature_vector(
        [1, 2, 3, 0], [1, 4, 0],
        product_canonical="CCO", candidate_canonical="CCN",
        raw_forward_reward=0.5, vocab_size=8,
    )
    assert first == second
    assert len(first) == 9 + 3 * 8


def test_auc_gives_half_credit_to_ties():
    auc, wins, ties, pairs = _auc_from_pairs(
        [0.5, 0.5, 0.1], [True, False, False],
    )
    assert auc == 0.75
    assert (wins, ties, pairs) == (1, 1, 2)


def test_endpoint_rerank_excludes_invalid_candidates():
    examples = [
        {"product_index": 0, "record_index": 0, "label": False, "valid": True},
        {"product_index": 0, "record_index": 1, "label": True, "valid": True},
        {"product_index": 0, "record_index": 2, "label": False, "valid": False},
    ]
    result = _rank_metrics(examples, [0.1, 0.9, INVALID_SCORE], order="correctness_reward")
    assert result["top_k"]["1"]["percent"] == 100.0
    assert result["oracle_percent"] == 100.0
    assert result["invalid_candidate_percent"] == 100.0 / 3.0

