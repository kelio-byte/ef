#!/usr/bin/env python3
"""Project a reaction-level split onto another tokenized representation.

The source and target files in this project are arranged as complete
augmentation blocks.  A split manifest stores the original reaction indices;
this utility selects those same blocks from another tokenizer's files and,
optionally, verifies the un-tokenized strings against a reference split.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence


def _read_rows(path: Path) -> list[str]:
    with path.open() as handle:
        return [line.rstrip("\n") for line in handle]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unspace(value: str) -> str:
    return "".join(value.split())


def _write_rows(path: Path, rows: Sequence[str]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(f"{row}\n")


def project_split(
    *,
    source_src: Path,
    source_tgt: Path,
    reference_src: Path | None,
    reference_tgt: Path | None,
    manifest_path: Path,
    split_name: str,
    output_dir: Path,
    augmentation: int,
) -> dict:
    if augmentation <= 0:
        raise ValueError("augmentation must be positive")
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {output_dir}; refusing to overwrite"
        )

    manifest = json.loads(manifest_path.read_text())
    split = manifest["splits"][split_name]
    indices = [int(index) for index in split["original_reaction_indices"]]
    src_rows = _read_rows(source_src)
    tgt_rows = _read_rows(source_tgt)
    if len(src_rows) != len(tgt_rows):
        raise ValueError(
            f"source/target row count mismatch: {len(src_rows)} != {len(tgt_rows)}"
        )
    if len(src_rows) % augmentation:
        raise ValueError(
            f"source rows {len(src_rows)} are not divisible by augmentation "
            f"{augmentation}"
        )
    total_reactions = len(src_rows) // augmentation
    if any(index < 0 or index >= total_reactions for index in indices):
        raise ValueError("split manifest contains an out-of-range reaction index")

    selected_src: list[str] = []
    selected_tgt: list[str] = []
    for index in indices:
        begin = index * augmentation
        end = begin + augmentation
        selected_src.extend(src_rows[begin:end])
        selected_tgt.extend(tgt_rows[begin:end])

    if reference_src is not None or reference_tgt is not None:
        if reference_src is None or reference_tgt is None:
            raise ValueError("reference_src and reference_tgt must be supplied together")
        reference_src_rows = _read_rows(reference_src)
        reference_tgt_rows = _read_rows(reference_tgt)
        if len(reference_src_rows) != len(selected_src):
            raise ValueError(
                "reference source has a different number of selected rows: "
                f"{len(reference_src_rows)} != {len(selected_src)}"
            )
        if len(reference_tgt_rows) != len(selected_tgt):
            raise ValueError(
                "reference target has a different number of selected rows: "
                f"{len(reference_tgt_rows)} != {len(selected_tgt)}"
            )
        for row_number, (actual, reference) in enumerate(
            zip(selected_src, reference_src_rows), start=1
        ):
            if _unspace(actual) != _unspace(reference):
                raise ValueError(
                    f"source representation mismatch at selected row {row_number}"
                )
        for row_number, (actual, reference) in enumerate(
            zip(selected_tgt, reference_tgt_rows), start=1
        ):
            if _unspace(actual) != _unspace(reference):
                raise ValueError(
                    f"target representation mismatch at selected row {row_number}"
                )

    output_dir.mkdir(parents=True)
    output_src = output_dir / "src.txt"
    output_tgt = output_dir / "tgt.txt"
    _write_rows(output_src, selected_src)
    _write_rows(output_tgt, selected_tgt)

    output_manifest = {
        "schema_version": 1,
        "purpose": "Matched tokenized projection of a reaction-level evaluation split",
        "split": split_name,
        "augmentation": augmentation,
        "original_reaction_indices": indices,
        "original_reaction_count": len(indices),
        "augmented_input_line_count": len(selected_src),
        "selection_manifest": str(manifest_path.resolve()),
        "source": {
            "src_path": str(source_src.resolve()),
            "tgt_path": str(source_tgt.resolve()),
            "src_sha256": _sha256(source_src),
            "tgt_sha256": _sha256(source_tgt),
        },
        "output": {
            "src_path": str(output_src.resolve()),
            "tgt_path": str(output_tgt.resolve()),
            "src_sha256": _sha256(output_src),
            "tgt_sha256": _sha256(output_tgt),
        },
        "reference_check": (
            {
                "src_path": str(reference_src.resolve()),
                "tgt_path": str(reference_tgt.resolve()),
                "unspaced_strings_match": True,
            }
            if reference_src is not None and reference_tgt is not None
            else None
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n"
    )
    return output_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project one reaction-level split onto another tokenization."
    )
    parser.add_argument("--source_src", required=True, type=Path)
    parser.add_argument("--source_tgt", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--reference_src", type=Path)
    parser.add_argument("--reference_tgt", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = project_split(
        source_src=args.source_src,
        source_tgt=args.source_tgt,
        reference_src=args.reference_src,
        reference_tgt=args.reference_tgt,
        manifest_path=args.manifest,
        split_name=args.split,
        output_dir=args.output_dir,
        augmentation=args.augmentation,
    )
    print(json.dumps({
        "split": result["split"],
        "original_reaction_count": result["original_reaction_count"],
        "augmented_input_line_count": result["augmented_input_line_count"],
        "output_dir": str(args.output_dir),
        "reference_check": result["reference_check"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
