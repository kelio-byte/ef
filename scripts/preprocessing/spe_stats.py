#!/usr/bin/env python
"""Audit and compare the original and SPE retro-tokenized datasets.

The edit distance and INS/DEL/SUB counts are read from the aligned files
produced by ``scripts/precompute_alignments.py``.  This script does not
reimplement alignment.  It also checks the SPE round-trip property, paired
line counts, aligned lengths, and the expected placement of ``<GAP>``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence


SPLITS = ("train", "val", "test")
GAP = "<GAP>"
SPECIAL_TOKEN_COUNT = 4


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _summary(values: list[int], *, threshold: int | None = None) -> dict:
    result = {
        "count": len(values),
        "mean": sum(values) / len(values) if values else 0.0,
        "median": _percentile(values, 0.5),
        "p90": _percentile(values, 0.9),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values) if values else 0,
    }
    if threshold is not None:
        result["over_threshold_count"] = sum(value > threshold for value in values)
        result["over_threshold_rate"] = (
            result["over_threshold_count"] / len(values) if values else 0.0
        )
    return result


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_vocab(path: Path) -> set[str]:
    tokens = set()
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"invalid vocab line {path}:{line_no}")
            tokens.add(fields[0])
    return tokens


def _iter_paired(
    first: Path,
    second: Path,
) -> Iterable[tuple[int, str, str]]:
    with first.open() as first_handle, second.open() as second_handle:
        line_no = 0
        while True:
            first_line = first_handle.readline()
            second_line = second_handle.readline()
            if not first_line and not second_line:
                return
            line_no += 1
            if not first_line or not second_line:
                raise ValueError(
                    f"line-count mismatch at line {line_no}: {first} vs {second}"
                )
            yield line_no, first_line.rstrip("\n"), second_line.rstrip("\n")


def _oov_summary(
    token_lines: Iterable[list[str]],
    vocab: set[str],
) -> dict:
    token_count = 0
    oov_count = 0
    line_count = 0
    oov_line_count = 0
    for tokens in token_lines:
        line_count += 1
        line_oov = sum(token not in vocab for token in tokens)
        token_count += len(tokens)
        oov_count += line_oov
        oov_line_count += bool(line_oov)
    return {
        "token_count": token_count,
        "oov_token_count": oov_count,
        "oov_token_rate": oov_count / token_count if token_count else 0.0,
        "line_count": line_count,
        "oov_line_count": oov_line_count,
        "oov_line_rate": oov_line_count / line_count if line_count else 0.0,
    }


def _operation_stats(
    aligned_pairs: Iterable[tuple[list[str], list[str]]],
    *,
    raw_pairs: Iterable[tuple[int, str, str]] | None = None,
) -> dict:
    counts = Counter()
    distances = []
    aligned_lengths = []
    edit_densities = []
    gap_count = 0
    pair_count = 0
    projection_mismatch_count = 0
    for src_tokens, tgt_tokens in aligned_pairs:
        if len(src_tokens) != len(tgt_tokens):
            raise ValueError("aligned source/target token lengths differ")
        if raw_pairs is not None:
            try:
                raw_line_no, raw_src_line, raw_tgt_line = next(raw_pairs)
            except StopIteration as exc:
                raise ValueError(
                    "aligned files contain more rows than raw src/tgt files"
                ) from exc
            raw_src_tokens = raw_src_line.split()
            raw_tgt_tokens = raw_tgt_line.split()
            if (
                [token for token in src_tokens if token != GAP] != raw_src_tokens
                or [token for token in tgt_tokens if token != GAP] != raw_tgt_tokens
            ):
                projection_mismatch_count += 1
        distance = 0
        for src_token, tgt_token in zip(src_tokens, tgt_tokens):
            if src_token == GAP and tgt_token == GAP:
                raise ValueError("alignment contains a GAP/GAP column")
            if src_token == GAP:
                counts["INS"] += 1
                gap_count += 1
                distance += 1
            elif tgt_token == GAP:
                counts["DEL"] += 1
                gap_count += 1
                distance += 1
            elif src_token != tgt_token:
                counts["SUB"] += 1
                distance += 1
        distances.append(distance)
        aligned_lengths.append(len(src_tokens))
        edit_densities.append(distance / len(src_tokens) if src_tokens else 0.0)
        pair_count += 1
    total_edits = sum(counts.values())
    total_aligned_tokens = sum(aligned_lengths)
    counts_with_rates = {
        name: {"count": int(counts[name]), "rate": counts[name] / total_edits if total_edits else 0.0}
        for name in ("INS", "DEL", "SUB")
    }
    return {
        "pair_count": pair_count,
        "edit_distance": _summary(distances),
        "edit_density": _summary(edit_densities),
        "aligned_length": _summary(aligned_lengths),
        "operations": counts_with_rates,
        "total_edit_operations": total_edits,
        "total_aligned_token_count": total_aligned_tokens,
        "keep_token_count": total_aligned_tokens - total_edits,
        "keep_rate": (
            (total_aligned_tokens - total_edits) / total_aligned_tokens
            if total_aligned_tokens else 0.0
        ),
        "gap_token_count": gap_count,
        "projection_mismatch_count": projection_mismatch_count,
    }


def _audit_split(
    dataset_dir: Path,
    split: str,
    vocab: set[str],
    *,
    max_seq_len: int,
    original_dir: Path | None,
    check_round_trip: bool,
) -> dict:
    split_dir = dataset_dir / split
    raw_src = split_dir / f"src-{split}.txt"
    raw_tgt = split_dir / f"tgt-{split}.txt"
    aligned_src = split_dir / f"{split}_aligned_src.txt"
    aligned_tgt = split_dir / f"{split}_aligned_tgt.txt"
    original_src = (
        original_dir / split / f"src-{split}.txt"
        if original_dir is not None else None
    )
    original_tgt = (
        original_dir / split / f"tgt-{split}.txt"
        if original_dir is not None else None
    )
    required = (raw_src, raw_tgt, aligned_src, aligned_tgt)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))

    src_lengths = []
    tgt_lengths = []
    src_token_count = 0
    tgt_token_count = 0
    src_oov_count = 0
    tgt_oov_count = 0
    src_oov_line_count = 0
    tgt_oov_line_count = 0
    combined_oov_count = 0
    combined_oov_line_count = 0
    round_trip_failures = 0
    unaligned_gap_count = 0
    raw_pairs = _iter_paired(raw_src, raw_tgt)
    if check_round_trip and (original_src is None or original_tgt is None):
        raise ValueError(
            "check_round_trip requires original_dir so SPE tokens can be "
            "compared with the source unaligned SMILES"
        )
    original_pairs = (
        _iter_paired(original_src, original_tgt)
        if check_round_trip and original_src is not None and original_tgt is not None
        else None
    )
    for line_no, src_line, tgt_line in raw_pairs:
        src_tokens = src_line.split()
        tgt_tokens = tgt_line.split()
        src_lengths.append(len(src_tokens))
        tgt_lengths.append(len(tgt_tokens))
        src_line_oov = sum(token not in vocab for token in src_tokens)
        tgt_line_oov = sum(token not in vocab for token in tgt_tokens)
        src_token_count += len(src_tokens)
        tgt_token_count += len(tgt_tokens)
        src_oov_count += src_line_oov
        tgt_oov_count += tgt_line_oov
        src_oov_line_count += bool(src_line_oov)
        tgt_oov_line_count += bool(tgt_line_oov)
        combined_oov_count += src_line_oov + tgt_line_oov
        combined_oov_line_count += bool(src_line_oov or tgt_line_oov)
        unaligned_gap_count += src_tokens.count(GAP) + tgt_tokens.count(GAP)
        if check_round_trip:
            assert original_pairs is not None
            original_line_no, original_src_line, original_tgt_line = next(
                original_pairs
            )
            if original_line_no != line_no:
                raise ValueError(
                    f"original/SPE line mismatch at {split}:{line_no}"
                )
            if "".join(src_tokens) != "".join(original_src_line.split()):
                round_trip_failures += 1
            if "".join(tgt_tokens) != "".join(original_tgt_line.split()):
                round_trip_failures += 1
    if check_round_trip:
        assert original_pairs is not None
        try:
            extra_original_line = next(original_pairs)
        except StopIteration:
            extra_original_line = None
        if extra_original_line is not None:
            raise ValueError(
                f"original dataset has extra lines after SPE split {split}: "
                f"line {extra_original_line[0]}"
            )

    aligned_pairs = (
        (src_line.split(), tgt_line.split())
        for _, src_line, tgt_line in _iter_paired(aligned_src, aligned_tgt)
    )
    alignment_stats = _operation_stats(
        aligned_pairs,
        raw_pairs=_iter_paired(raw_src, raw_tgt),
    )
    raw_pair_count = len(src_lengths)
    if alignment_stats["pair_count"] != raw_pair_count:
        raise ValueError(
            f"raw/aligned pair count mismatch for {split}: "
            f"{raw_pair_count} != {alignment_stats['pair_count']}"
        )
    if alignment_stats["projection_mismatch_count"]:
        raise ValueError(
            f"aligned tokens do not project back to raw tokens in "
            f"{dataset_dir}/{split}: "
            f"{alignment_stats['projection_mismatch_count']} rows"
        )
    if round_trip_failures:
        raise ValueError(f"SPE round-trip failures in {dataset_dir}/{split}")
    if unaligned_gap_count:
        raise ValueError(f"unaligned data contains <GAP> in {dataset_dir}/{split}")

    def _count_summary(
        token_count: int,
        oov_count: int,
        line_count: int,
        oov_line_count: int,
    ) -> dict:
        return {
            "token_count": token_count,
            "oov_token_count": oov_count,
            "oov_token_rate": oov_count / token_count if token_count else 0.0,
            "line_count": line_count,
            "oov_line_count": oov_line_count,
            "oov_line_rate": oov_line_count / line_count if line_count else 0.0,
        }

    src_oov = _count_summary(
        src_token_count, src_oov_count, raw_pair_count, src_oov_line_count,
    )
    tgt_oov = _count_summary(
        tgt_token_count, tgt_oov_count, raw_pair_count, tgt_oov_line_count,
    )
    combined_oov = _count_summary(
        src_token_count + tgt_token_count,
        combined_oov_count,
        raw_pair_count,
        combined_oov_line_count,
    )
    return {
        "pair_count": raw_pair_count,
        "src_length": _summary(src_lengths, threshold=max_seq_len),
        "tgt_length": _summary(tgt_lengths, threshold=max_seq_len),
        "alignment": alignment_stats,
        "oov": {"src": src_oov, "tgt": tgt_oov, "combined": combined_oov},
        "integrity": {
            "source_target_line_count_match": True,
            "raw_aligned_line_count_match": True,
            "aligned_token_length_match": True,
            "aligned_projection_match": True,
            "spe_round_trip_failure_count": round_trip_failures,
            "unaligned_gap_token_count": unaligned_gap_count,
        },
    }


def audit_dataset(
    dataset_dir: Path,
    *,
    max_seq_len: int = 256,
    original_dir: Path | None = None,
    check_round_trip: bool = False,
) -> dict:
    """Return split-level and aggregate tokenizer/alignment statistics."""
    vocab_path = dataset_dir / "example.vocab.src"
    vocab = _load_vocab(vocab_path)
    splits = {
        split: _audit_split(
            dataset_dir,
            split,
            vocab,
            max_seq_len=max_seq_len,
            original_dir=original_dir,
            check_round_trip=check_round_trip,
        )
        for split in SPLITS
    }
    total_pairs = sum(item["pair_count"] for item in splits.values())
    total_edits = sum(
        item["alignment"]["total_edit_operations"] for item in splits.values()
    )
    total_operations = Counter()
    for item in splits.values():
        for name, operation in item["alignment"]["operations"].items():
            total_operations[name] += operation["count"]
    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "vocab": {
            "path": str(vocab_path.resolve()),
            "token_count": len(vocab),
            "model_vocab_size": len(vocab) + SPECIAL_TOKEN_COUNT,
            "sha256": _sha256(vocab_path),
        },
        "max_seq_len": max_seq_len,
        "splits": splits,
        "aggregate": {
            "pair_count": total_pairs,
            "edit_distance_total": total_edits,
            "operations": {
                name: {
                    "count": int(total_operations[name]),
                    "rate": total_operations[name] / total_edits if total_edits else 0.0,
                }
                for name in ("INS", "DEL", "SUB")
            },
        },
    }


def compare_datasets(
    baseline_dir: Path,
    spe_dir: Path,
    *,
    max_seq_len: int = 256,
    check_round_trip: bool = True,
) -> dict:
    baseline = audit_dataset(
        baseline_dir,
        max_seq_len=max_seq_len,
        check_round_trip=False,
    )
    spe = audit_dataset(
        spe_dir,
        max_seq_len=max_seq_len,
        original_dir=baseline_dir,
        check_round_trip=check_round_trip,
    )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline,
        "spe": spe,
        "comparison": {
            "vocab_token_count_delta": spe["vocab"]["token_count"] - baseline["vocab"]["token_count"],
            "model_vocab_size_delta": spe["vocab"]["model_vocab_size"] - baseline["vocab"]["model_vocab_size"],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir", type=Path,
        default=Path("datasets/USPTO_50K_PtoR_aug20_#global#"),
    )
    parser.add_argument(
        "--spe-dir", type=Path,
        default=Path("datasets/USPTO_50K_PtoR_aug20_#global#_SPE"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument(
        "--skip-round-trip",
        action="store_true",
        help="Skip the extra SPE unaligned round-trip audit",
    )
    args = parser.parse_args(argv)
    result = compare_datasets(
        args.baseline_dir,
        args.spe_dir,
        max_seq_len=args.max_seq_len,
        check_round_trip=not args.skip_round_trip,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
