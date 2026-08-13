#!/usr/bin/env python
"""Build a SPE-tokenized shadow copy of the USPTO retro dataset.

The input files are already atom-tokenized by the historical pipeline.  This
script first removes those display separators, reconstructs each SMILES, and
then applies the pre-trained SmilesPE tokenizer.  It never reads an aligned
file and refuses to write into the source dataset.

The output contains only the six unaligned split files and a training-only
``example.vocab.src``.  Levenshtein alignments are intentionally produced by
the existing ``scripts/precompute_alignments.py`` command after this script.

Examples
--------
Sanity check the first 50 pairs of each split into a temporary directory::

    PYTHONPATH=. python scripts/preprocessing/preprocess_spe.py \
        --max-lines 50 --output-dir /tmp/edit_flows_spe_sanity

Build the full shadow dataset::

    PYTHONPATH=. python scripts/preprocessing/preprocess_spe.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence


DEFAULT_SOURCE_DIR = Path("datasets/USPTO_50K_PtoR_aug20_#global#")
DEFAULT_OUTPUT_DIR = Path(
    "datasets/USPTO_50K_PtoR_aug20_#global#_SPE"
)
DEFAULT_CODES_PATH = Path("scripts/preprocessing/SPE_ChEMBL.txt")
SPLITS = ("train", "val", "test")


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_smiles(tokenized_line: str) -> str:
    """Undo the historical whitespace-only token display format."""
    return "".join(tokenized_line.strip().split())


def _paired_lines(
    src_path: Path,
    tgt_path: Path,
    *,
    max_lines: int | None,
) -> Iterator[tuple[int, str, str]]:
    """Yield paired raw lines and fail on a length mismatch."""
    with src_path.open() as src_handle, tgt_path.open() as tgt_handle:
        line_no = 0
        while True:
            src_line = src_handle.readline()
            tgt_line = tgt_handle.readline()
            if not src_line and not tgt_line:
                break
            line_no += 1
            if not src_line or not tgt_line:
                raise ValueError(
                    "source/target line-count mismatch at line "
                    f"{line_no}: {src_path} vs {tgt_path}"
                )
            yield line_no, src_line, tgt_line
            if max_lines is not None and line_no >= max_lines:
                break


def _load_tokenizer(codes_path: Path):
    try:
        from SmilesPE.tokenizer import SPE_Tokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "SmilesPE is required; install it with "
            "python -m pip install 'SmilesPE==0.0.3'"
        ) from exc
    codes = codes_path.open()
    try:
        tokenizer = SPE_Tokenizer(codes)
    finally:
        codes.close()
    return tokenizer


def tokenize_smiles(tokenizer, smiles: str) -> list[str]:
    """Tokenize one complete SMILES with deterministic standard SPE."""
    tokens = tokenizer.tokenize(smiles, dropout=0).split()
    restored = "".join(tokens)
    if restored != smiles:
        raise ValueError(
            "SPE tokenization is not lossless: "
            f"original={smiles!r}, restored={restored!r}"
        )
    return tokens


def _tokenize_split(
    source_dir: Path,
    output_dir: Path,
    split: str,
    tokenizer,
    *,
    max_lines: int | None,
    cache_reset_interval: int,
) -> dict:
    split_source_dir = source_dir / split
    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)
    src_path = split_source_dir / f"src-{split}.txt"
    tgt_path = split_source_dir / f"tgt-{split}.txt"
    out_src_path = split_output_dir / f"src-{split}.txt"
    out_tgt_path = split_output_dir / f"tgt-{split}.txt"
    for path in (src_path, tgt_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    pair_count = 0
    src_token_count = 0
    tgt_token_count = 0
    src_max_tokens = 0
    tgt_max_tokens = 0
    with (
        out_src_path.open("w") as out_src,
        out_tgt_path.open("w") as out_tgt,
    ):
        for line_no, src_line, tgt_line in _paired_lines(
            src_path, tgt_path, max_lines=max_lines,
        ):
            src_smiles = restore_smiles(src_line)
            tgt_smiles = restore_smiles(tgt_line)
            if not src_smiles or not tgt_smiles:
                raise ValueError(
                    f"empty reconstructed SMILES at {split}:{line_no}"
                )
            src_tokens = tokenize_smiles(tokenizer, src_smiles)
            tgt_tokens = tokenize_smiles(tokenizer, tgt_smiles)
            if "<GAP>" in src_tokens or "<GAP>" in tgt_tokens:
                raise ValueError(
                    f"<GAP> appeared in unaligned SPE tokens at "
                    f"{split}:{line_no}"
                )
            out_src.write(" ".join(src_tokens) + "\n")
            out_tgt.write(" ".join(tgt_tokens) + "\n")
            pair_count += 1
            src_token_count += len(src_tokens)
            tgt_token_count += len(tgt_tokens)
            src_max_tokens = max(src_max_tokens, len(src_tokens))
            tgt_max_tokens = max(tgt_max_tokens, len(tgt_tokens))
            if (
                cache_reset_interval > 0
                and pair_count % cache_reset_interval == 0
            ):
                # SmilesPE caches every unique complete SMILES.  Bounding the
                # cache keeps the full 2M-line run memory-stable without
                # changing deterministic tokenization.
                tokenizer.cache.clear()

    return {
        "pair_count": pair_count,
        "src_token_count": src_token_count,
        "tgt_token_count": tgt_token_count,
        "src_mean_tokens": src_token_count / pair_count if pair_count else 0.0,
        "tgt_mean_tokens": tgt_token_count / pair_count if pair_count else 0.0,
        "src_max_tokens": src_max_tokens,
        "tgt_max_tokens": tgt_max_tokens,
        "source_sha256": {
            "src": _sha256(src_path),
            "tgt": _sha256(tgt_path),
        },
        "output_sha256": {
            "src": _sha256(out_src_path),
            "tgt": _sha256(out_tgt_path),
        },
    }


def build_vocab(output_dir: Path, *, train_split: str = "train") -> dict:
    """Build the same frequency-sorted, train-union vocab as the project."""
    counter: Counter[str] = Counter()
    for side in ("src", "tgt"):
        path = output_dir / train_split / f"{side}-{train_split}.txt"
        with path.open() as handle:
            for line in handle:
                counter.update(line.split())
    vocab_path = output_dir / "example.vocab.src"
    with vocab_path.open("w") as handle:
        for token, count in counter.most_common():
            handle.write(f"{token}\t{count}\n")
    return {
        "path": str(vocab_path),
        "token_count": len(counter),
        "model_vocab_size": len(counter) + 4,
        "sha256": _sha256(vocab_path),
        "total_training_token_count": sum(counter.values()),
    }


def _validate_paths(source_dir: Path, output_dir: Path, codes_path: Path) -> None:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("output_dir must differ from the source #global# directory")
    if not codes_path.is_file():
        raise FileNotFoundError(codes_path)
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)


def preprocess(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    codes_path: Path = DEFAULT_CODES_PATH,
    *,
    splits: Sequence[str] = SPLITS,
    max_lines: int | None = None,
    cache_reset_interval: int = 50_000,
) -> dict:
    """Create the SPE unaligned files and training-only vocabulary."""
    _validate_paths(source_dir, output_dir, codes_path)
    if max_lines is not None and max_lines < 1:
        raise ValueError("max_lines must be positive when provided")
    if cache_reset_interval < 0:
        raise ValueError("cache_reset_interval must be non-negative")
    unknown_splits = sorted(set(splits) - set(SPLITS))
    if unknown_splits:
        raise ValueError(f"unknown split(s): {unknown_splits}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = _load_tokenizer(codes_path)
    split_stats = {}
    for split in splits:
        split_stats[split] = _tokenize_split(
            source_dir,
            output_dir,
            split,
            tokenizer,
            max_lines=max_lines,
            cache_reset_interval=cache_reset_interval,
        )
    vocab_stats = None
    if "train" in splits:
        vocab_stats = build_vocab(output_dir)

    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "codes_path": str(codes_path.resolve()),
        "codes_sha256": _sha256(codes_path),
        "smilespe_version": "0.0.3",
        "dropout": 0,
        "max_lines": max_lines,
        "cache_reset_interval": cache_reset_interval,
        "source_kind": "unaligned src/tgt only; old aligned files ignored",
        "splits": split_stats,
        "vocab": vocab_stats,
    }
    metadata_path = output_dir / "spe_preprocessing_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--codes", type=Path, default=DEFAULT_CODES_PATH)
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=list(SPLITS),
    )
    parser.add_argument(
        "--max-lines", type=int, default=None,
        help="Process only the first N paired lines per selected split",
    )
    parser.add_argument(
        "--cache-reset-interval", type=int, default=50_000,
        help="Clear SmilesPE's complete-SMILES cache periodically; 0 disables",
    )
    args = parser.parse_args(argv)
    metadata = preprocess(
        args.source_dir,
        args.output_dir,
        args.codes,
        splits=args.splits,
        max_lines=args.max_lines,
        cache_reset_interval=args.cache_reset_interval,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
