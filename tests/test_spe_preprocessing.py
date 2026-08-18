from pathlib import Path

import pytest

from scripts.preprocessing.preprocess_spe import (
    build_vocab,
    preprocess,
    restore_smiles,
    tokenize_smiles,
)
from scripts.preprocessing.spe_stats import audit_dataset


def test_restore_smiles_removes_only_display_whitespace():
    assert restore_smiles(" C ( N ) C [C@@H] ") == "C(N)C[C@@H]"


def test_preprocess_is_lossless_and_builds_train_only_vocab(tmp_path):
    source = tmp_path / "source"
    codes = Path("scripts/preprocessing/SPE_ChEMBL.txt")
    for split in ("train", "val", "test"):
        split_dir = source / split
        split_dir.mkdir(parents=True)
        (split_dir / f"src-{split}.txt").write_text("C C O\nC ( N ) C\n")
        (split_dir / f"tgt-{split}.txt").write_text("C O\nC N C\n")
    output = tmp_path / "source_SPE"
    metadata = preprocess(
        source,
        output,
        codes,
        merges=1,
        max_lines=1,
        cache_reset_interval=1,
    )
    assert metadata["splits"]["train"]["pair_count"] == 1
    assert metadata["merges"] == 1
    assert (output / "train/src-train.txt").read_text().splitlines() == ["C C O"]
    assert (output / "example.vocab.src").is_file()
    assert "<GAP>" not in (output / "train/src-train.txt").read_text()


def test_preprocess_rejects_invalid_merge_count(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="merges"):
        preprocess(
            source,
            tmp_path / "output",
            Path("scripts/preprocessing/SPE_ChEMBL.txt"),
            merges=-2,
        )


def test_preprocess_rejects_writing_over_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="must differ"):
        preprocess(source, source, Path("scripts/preprocessing/SPE_ChEMBL.txt"))


def test_spe_stats_checks_alignment_and_reports_oov(tmp_path):
    dataset = tmp_path / "dataset"
    original = tmp_path / "original"
    (dataset / "train").mkdir(parents=True)
    for split in ("train", "val", "test"):
        split_dir = dataset / split
        split_dir.mkdir(exist_ok=True)
        original_split_dir = original / split
        original_split_dir.mkdir(parents=True)
        (split_dir / f"src-{split}.txt").write_text("C O\n")
        (split_dir / f"tgt-{split}.txt").write_text("C N\n")
        (original_split_dir / f"src-{split}.txt").write_text("C O\n")
        (original_split_dir / f"tgt-{split}.txt").write_text("C N\n")
        (split_dir / f"{split}_aligned_src.txt").write_text("C O\n")
        (split_dir / f"{split}_aligned_tgt.txt").write_text("C N\n")
    (dataset / "example.vocab.src").write_text("C\t3\nO\t1\n")
    result = audit_dataset(
        dataset, original_dir=original, check_round_trip=True,
    )
    assert result["vocab"]["token_count"] == 2
    assert result["splits"]["val"]["alignment"]["operations"]["SUB"]["count"] == 1
    assert result["splits"]["val"]["oov"]["tgt"]["oov_token_count"] == 1
    assert result["splits"]["val"]["integrity"]["aligned_projection_match"]
