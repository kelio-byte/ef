#!/usr/bin/env python
"""Build reproducible reaction-level evaluation splits for DGM experiments.

The augmented USPTO files contain consecutive blocks of equivalent SMILES
representations for one original reaction.  This utility treats the complete
block as indivisible: augmentation rows are never counted as independent
reactions and are never split across development/confirmation/final sets.

The output is intentionally self-contained.  Each split has matching ``src``
and ``tgt`` files, while ``manifest.json`` records source hashes, original
reaction indices, split sizes, and lightweight difficulty strata.  The subset
files can therefore be passed directly to ``scripts/eval.py`` with
``--start_product 0``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import median
from typing import Iterable, Sequence


DEFAULT_EXCLUDED_RANGES = ("0:600",)
DEFAULT_SPLIT_SIZES = {
    "dev_unique1000_aug20": 1000,
    "confirm_unique1000_aug20": 1000,
    "final_unique2000_aug20": 2000,
}


@dataclass(frozen=True)
class ReactionGroup:
    """One indivisible original reaction and its augmentation rows."""

    original_index: int
    src_rows: tuple[str, ...]
    tgt_rows: tuple[str, ...]
    product_token_count: int
    reactant_component_count: int
    stratum: tuple[int, str]


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_excluded_ranges(values: Iterable[str]) -> list[range]:
    """Parse half-open reaction-index ranges written as ``START:STOP``."""
    ranges: list[range] = []
    for value in values:
        try:
            start_text, stop_text = value.split(":", 1)
            start, stop = int(start_text), int(stop_text)
        except ValueError as exc:
            raise ValueError(
                f"invalid excluded range {value!r}; use START:STOP"
            ) from exc
        if start < 0 or stop <= start:
            raise ValueError(
                f"invalid excluded range {value!r}; require 0 <= START < STOP"
            )
        ranges.append(range(start, stop))
    return ranges


def _read_rows(path: Path) -> list[str]:
    with path.open() as handle:
        return [line.rstrip("\n") for line in handle]


def _quartile_cutpoints(values: Sequence[int]) -> tuple[int, int, int]:
    if not values:
        raise ValueError("cannot compute strata for an empty candidate pool")
    ordered = sorted(values)
    return tuple(
        ordered[(len(ordered) * quantile - 1) // 4]
        for quantile in (1, 2, 3)
    )


def _length_bin(value: int, cutpoints: tuple[int, int, int]) -> int:
    for index, cutoff in enumerate(cutpoints):
        if value <= cutoff:
            return index
    return len(cutpoints)


def _component_bucket(component_count: int) -> str:
    if component_count <= 1:
        return "1"
    if component_count == 2:
        return "2"
    return "3+"


def _group_feature(rows: Sequence[str], *, target: bool) -> int:
    values: list[int] = []
    for row in rows:
        tokens = row.split()
        values.append(tokens.count(".") + 1 if target else len(tokens))
    return int(round(median(values)))


def load_reaction_groups(
    src_path: Path,
    tgt_path: Path,
    *,
    augmentation: int,
    excluded_ranges: Sequence[range],
) -> tuple[list[ReactionGroup], dict[str, int | list[int]]]:
    """Load complete augmentation blocks and derive reproducible strata."""
    if augmentation <= 0:
        raise ValueError("augmentation must be positive")
    src_rows = _read_rows(src_path)
    tgt_rows = _read_rows(tgt_path)
    if len(src_rows) != len(tgt_rows):
        raise ValueError(
            "source and target line counts differ: "
            f"{len(src_rows)} != {len(tgt_rows)}"
        )
    if len(src_rows) % augmentation:
        raise ValueError(
            "source/target rows do not contain complete augmentation blocks: "
            f"{len(src_rows)} % {augmentation} != 0"
        )

    total_reactions = len(src_rows) // augmentation
    excluded = {
        index
        for index in range(total_reactions)
        if any(index in item for item in excluded_ranges)
    }
    candidate_rows: list[tuple[int, tuple[str, ...], tuple[str, ...], int, int]] = []
    for original_index in range(total_reactions):
        if original_index in excluded:
            continue
        begin = original_index * augmentation
        end = begin + augmentation
        src_block = tuple(src_rows[begin:end])
        tgt_block = tuple(tgt_rows[begin:end])
        candidate_rows.append((
            original_index,
            src_block,
            tgt_block,
            _group_feature(src_block, target=False),
            _group_feature(tgt_block, target=True),
        ))

    cutpoints = _quartile_cutpoints([item[3] for item in candidate_rows])
    groups = [
        ReactionGroup(
            original_index=original_index,
            src_rows=src_block,
            tgt_rows=tgt_block,
            product_token_count=product_tokens,
            reactant_component_count=reactant_components,
            stratum=(
                _length_bin(product_tokens, cutpoints),
                _component_bucket(reactant_components),
            ),
        )
        for (
            original_index,
            src_block,
            tgt_block,
            product_tokens,
            reactant_components,
        ) in candidate_rows
    ]
    return groups, {
        "total_source_rows": len(src_rows),
        "total_original_reactions": total_reactions,
        "excluded_original_reactions": len(excluded),
        "candidate_original_reactions": len(groups),
        "product_token_quartile_cutpoints": list(cutpoints),
    }


def _quota_by_stratum(
    groups_by_stratum: dict[tuple[int, str], list[ReactionGroup]],
    count: int,
) -> dict[tuple[int, str], int]:
    available = sum(len(groups) for groups in groups_by_stratum.values())
    if count > available:
        raise ValueError(
            f"requested {count} reactions but only {available} remain"
        )
    if count == 0:
        return {stratum: 0 for stratum in groups_by_stratum}

    exact = {
        stratum: len(groups) * count / available
        for stratum, groups in groups_by_stratum.items()
    }
    quotas = {stratum: int(value) for stratum, value in exact.items()}
    remaining = count - sum(quotas.values())
    # Largest-remainder allocation preserves the overall stratum mix while
    # deterministic tie-breaking keeps the manifest reproducible.
    for stratum in sorted(
        groups_by_stratum,
        key=lambda key: (-(exact[key] - quotas[key]), key),
    ):
        if remaining == 0:
            break
        if quotas[stratum] < len(groups_by_stratum[stratum]):
            quotas[stratum] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("could not allocate the requested stratified quota")
    return quotas


def split_groups(
    groups: Sequence[ReactionGroup],
    *,
    split_sizes: dict[str, int],
    seed: int,
) -> tuple[dict[str, list[ReactionGroup]], list[ReactionGroup]]:
    """Create non-overlapping stratified splits, retaining any reserve pool."""
    if any(size <= 0 for size in split_sizes.values()):
        raise ValueError("every split size must be positive")
    if sum(split_sizes.values()) > len(groups):
        raise ValueError(
            f"requested {sum(split_sizes.values())} reactions but only "
            f"{len(groups)} candidates are available"
        )

    rng = random.Random(seed)
    remaining_by_stratum: dict[tuple[int, str], list[ReactionGroup]] = defaultdict(list)
    for group in groups:
        remaining_by_stratum[group.stratum].append(group)
    for group_list in remaining_by_stratum.values():
        rng.shuffle(group_list)

    selected: dict[str, list[ReactionGroup]] = {}
    for split_name, requested in split_sizes.items():
        quotas = _quota_by_stratum(remaining_by_stratum, requested)
        current: list[ReactionGroup] = []
        for stratum in sorted(remaining_by_stratum):
            quota = quotas[stratum]
            current.extend(remaining_by_stratum[stratum][:quota])
            del remaining_by_stratum[stratum][:quota]
        rng.shuffle(current)
        if len(current) != requested:
            raise RuntimeError(
                f"split {split_name} has {len(current)}, expected {requested}"
            )
        selected[split_name] = current

    reserve = [
        group
        for stratum in sorted(remaining_by_stratum)
        for group in remaining_by_stratum[stratum]
    ]
    rng.shuffle(reserve)
    return selected, reserve


def _split_summary(groups: Sequence[ReactionGroup], augmentation: int) -> dict:
    strata = Counter(
        f"product_length_quartile_{length_bin + 1}|reactant_components_{components}"
        for length_bin, components in (group.stratum for group in groups)
    )
    component_counts = Counter(
        _component_bucket(group.reactant_component_count) for group in groups
    )
    return {
        "original_reaction_count": len(groups),
        "augmented_input_line_count": len(groups) * augmentation,
        "original_reaction_indices": [group.original_index for group in groups],
        "product_token_count": {
            "min": min(group.product_token_count for group in groups),
            "median": int(round(median(group.product_token_count for group in groups))),
            "max": max(group.product_token_count for group in groups),
        },
        "reactant_component_bucket_counts": dict(sorted(component_counts.items())),
        "joint_stratum_counts": dict(sorted(strata.items())),
    }


def _write_split(
    output_dir: Path,
    groups: Sequence[ReactionGroup],
    augmentation: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    src_path = output_dir / "src.txt"
    tgt_path = output_dir / "tgt.txt"
    with src_path.open("w") as src_handle, tgt_path.open("w") as tgt_handle:
        for group in groups:
            if len(group.src_rows) != augmentation or len(group.tgt_rows) != augmentation:
                raise RuntimeError("encountered a partial augmentation block")
            for source, target in zip(group.src_rows, group.tgt_rows):
                src_handle.write(f"{source}\n")
                tgt_handle.write(f"{target}\n")
    summary = _split_summary(groups, augmentation)
    summary.update({
        "src_file": str(src_path),
        "tgt_file": str(tgt_path),
        "src_sha256": sha256_file(src_path),
        "tgt_sha256": sha256_file(tgt_path),
    })
    return summary


def build_evaluation_splits(
    *,
    src_path: Path,
    tgt_path: Path,
    output_root: Path,
    augmentation: int,
    excluded_ranges: Sequence[range],
    split_sizes: dict[str, int],
    seed: int,
) -> dict:
    """Build the three reaction-level files and return their manifest."""
    if output_root.exists():
        raise FileExistsError(
            f"output root already exists: {output_root}; refusing to overwrite"
        )
    groups, source_summary = load_reaction_groups(
        src_path, tgt_path,
        augmentation=augmentation,
        excluded_ranges=excluded_ranges,
    )
    selected, reserve = split_groups(
        groups,
        split_sizes=split_sizes,
        seed=seed,
    )

    output_root.mkdir(parents=True, exist_ok=False)
    split_summaries = {
        name: _write_split(output_root / name, split_groups_, augmentation)
        for name, split_groups_ in selected.items()
    }
    all_selected_indices = [
        group.original_index
        for split_groups_ in selected.values()
        for group in split_groups_
    ]
    if len(set(all_selected_indices)) != len(all_selected_indices):
        raise RuntimeError("constructed splits overlap in original reaction index")

    manifest = {
        "schema_version": 1,
        "purpose": (
            "DGM method-level evaluation. One original reaction, including all "
            "of its augmentation rows, is one indivisible statistical unit."
        ),
        "augmentation": augmentation,
        "selection_seed": seed,
        "excluded_original_reaction_ranges": [
            [item.start, item.stop] for item in excluded_ranges
        ],
        "source": {
            "src_path": str(src_path.resolve()),
            "tgt_path": str(tgt_path.resolve()),
            "src_sha256": sha256_file(src_path),
            "tgt_sha256": sha256_file(tgt_path),
            **source_summary,
        },
        "split_sizes_requested": split_sizes,
        "splits": split_summaries,
        "reserve": _split_summary(reserve, augmentation),
    }
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create non-overlapping, reaction-level DGM evaluation splits "
            "from an augmented source/target pair."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--tgt", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument(
        "--exclude_range",
        action="append",
        default=None,
        metavar="START:STOP",
        help=(
            "Half-open original-reaction index range to quarantine because it "
            "was previously used. May be supplied more than once. Defaults to "
            "the historical range 0:600."
        ),
    )
    parser.add_argument("--dev_size", type=int, default=1000)
    parser.add_argument("--confirm_size", type=int, default=1000)
    parser.add_argument("--final_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    excluded_ranges = parse_excluded_ranges(
        DEFAULT_EXCLUDED_RANGES if args.exclude_range is None else args.exclude_range
    )
    split_sizes = {
        "dev_unique1000_aug20": args.dev_size,
        "confirm_unique1000_aug20": args.confirm_size,
        "final_unique2000_aug20": args.final_size,
    }
    manifest = build_evaluation_splits(
        src_path=args.src,
        tgt_path=args.tgt,
        output_root=args.output_root,
        augmentation=args.augmentation,
        excluded_ranges=excluded_ranges,
        split_sizes=split_sizes,
        seed=args.seed,
    )
    print(json.dumps({
        "output_root": str(args.output_root),
        "split_counts": {
            name: info["original_reaction_count"]
            for name, info in manifest["splits"].items()
        },
        "reserve_original_reaction_count": manifest["reserve"]["original_reaction_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
