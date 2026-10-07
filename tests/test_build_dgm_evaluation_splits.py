import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_dgm_evaluation_splits.py"
SPEC = importlib.util.spec_from_file_location("build_dgm_evaluation_splits", SCRIPT_PATH)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _write_augmented_pair(tmp_path: Path, reactions: int, augmentation: int) -> tuple[Path, Path]:
    src = tmp_path / "src.txt"
    tgt = tmp_path / "tgt.txt"
    source_rows = []
    target_rows = []
    for reaction_index in range(reactions):
        for augmentation_index in range(augmentation):
            source_rows.append(
                f"P {reaction_index} AUG {augmentation_index} "
                + "X " * (reaction_index % 4)
            )
            target_rows.append(
                f"R {reaction_index} "
                + (". R2 " if reaction_index % 3 else "")
                + f"AUG {augmentation_index}"
            )
    src.write_text("\n".join(source_rows) + "\n")
    tgt.write_text("\n".join(target_rows) + "\n")
    return src, tgt


def test_builds_nonoverlapping_complete_augmentation_blocks(tmp_path):
    src, tgt = _write_augmented_pair(tmp_path, reactions=12, augmentation=2)
    output_root = tmp_path / "evaluation_v2"

    manifest = builder.build_evaluation_splits(
        src_path=src,
        tgt_path=tgt,
        output_root=output_root,
        augmentation=2,
        excluded_ranges=[range(0, 2)],
        split_sizes={"dev": 3, "confirm": 3, "final": 3},
        seed=7,
    )

    seen_indices = set()
    for name, expected_count in (("dev", 3), ("confirm", 3), ("final", 3)):
        info = manifest["splits"][name]
        assert info["original_reaction_count"] == expected_count
        assert info["augmented_input_line_count"] == expected_count * 2
        indices = set(info["original_reaction_indices"])
        assert not (seen_indices & indices)
        seen_indices.update(indices)

        src_rows = (output_root / name / "src.txt").read_text().splitlines()
        tgt_rows = (output_root / name / "tgt.txt").read_text().splitlines()
        assert len(src_rows) == len(tgt_rows) == expected_count * 2
        for row_index in range(0, len(src_rows), 2):
            assert src_rows[row_index].split()[1] == src_rows[row_index + 1].split()[1]
            assert tgt_rows[row_index].split()[1] == tgt_rows[row_index + 1].split()[1]

    assert seen_indices.isdisjoint({0, 1})
    assert manifest["reserve"]["original_reaction_count"] == 1
    stored_manifest = json.loads((output_root / "manifest.json").read_text())
    assert stored_manifest["selection_seed"] == 7


def test_rejects_partial_augmentation_blocks(tmp_path):
    src = tmp_path / "src.txt"
    tgt = tmp_path / "tgt.txt"
    src.write_text("a\nb\nc\n")
    tgt.write_text("x\ny\nz\n")

    with pytest.raises(ValueError, match="complete augmentation blocks"):
        builder.load_reaction_groups(
            src, tgt, augmentation=2, excluded_ranges=[],
        )


def test_refuses_to_overwrite_an_existing_output_root(tmp_path):
    src, tgt = _write_augmented_pair(tmp_path, reactions=4, augmentation=2)
    output_root = tmp_path / "already_exists"
    output_root.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.build_evaluation_splits(
            src_path=src,
            tgt_path=tgt,
            output_root=output_root,
            augmentation=2,
            excluded_ranges=[],
            split_sizes={"dev": 1, "confirm": 1, "final": 1},
            seed=1,
        )
