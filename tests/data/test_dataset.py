import pytest

from edit_flows.data.dataset import PreAlignedDataset, RetroDataset, load_vocab


TOKEN2ID = {"<UNK>": 3, "a": 4, "b": 5}


@pytest.mark.parametrize("text", ["C\nC\n", "<PAD>\nC\n", "C\n\n"])
def test_vocab_rejects_ambiguous_ids(tmp_path, text):
    path = tmp_path / "vocab.txt"
    path.write_text(text)
    with pytest.raises(ValueError, match="vocabulary"):
        load_vocab(path)


def test_vocab_preserves_order_and_ignores_counts(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_text("O 9\nC 100\n")
    tokens, size = load_vocab(path)
    assert tokens["O"] == 4 and tokens["C"] == 5
    assert tokens["<PAD>"] == 0
    assert sorted(tokens.values()) == list(range(size))


def _write_pair(tmp_path, left_name, right_name, left, right):
    left_path = tmp_path / left_name
    right_path = tmp_path / right_name
    left_path.write_text(left)
    right_path.write_text(right)
    return str(left_path), str(right_path)


def test_raw_dataset_rejects_line_count_mismatch(tmp_path):
    src, tgt = _write_pair(
        tmp_path, "src.txt", "tgt.txt", "a\nb\n", "a\n",
    )

    with pytest.raises(ValueError, match="line-count mismatch"):
        RetroDataset(src, tgt, TOKEN2ID)


def test_pre_aligned_dataset_rejects_length_mismatch(tmp_path):
    src, tgt = _write_pair(
        tmp_path, "z0.txt", "z1.txt", "a b\n", "a\n",
    )

    with pytest.raises(ValueError, match="aligned pair length mismatch"):
        PreAlignedDataset(src, tgt, TOKEN2ID)
