import pytest

from edit_flows.data.dataset import PreAlignedDataset, RetroDataset


TOKEN2ID = {"<UNK>": 3, "a": 4, "b": 5}


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
