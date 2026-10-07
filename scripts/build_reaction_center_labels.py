#!/usr/bin/env python3
"""Build reaction-center labels and raw-to-global crosswalks.

Large per-reaction JSONL files are written under ``results/`` (gitignored).
Compact reports and hashes are written under ``after_spe/results/stage1``.
No model, checkpoint, target prediction, or GPU is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable

from edit_flows.chem.reaction_center import (
    canonical_reaction_key,
    canonical_reaction_key_achiral,
    extract_reaction_center,
    reaction_key_sha256,
)
from scripts.preprocessing.global_align import inverse_global_align


REACTION_COLUMN = "reactants>reagents>production"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_worker(item: tuple[int, str, str, str]) -> dict[str, Any]:
    raw_index, reaction_id, reaction_class, reaction_smiles = item
    return extract_reaction_center(
        reaction_smiles,
        reaction_id=reaction_id,
        reaction_class=reaction_class,
        raw_index=raw_index,
    )


def _processed_worker(item: tuple[int, str, str]) -> dict[str, Any]:
    block_index, source, target = item
    try:
        product = "".join(source.split())
        global_reactants = "".join(target.split())
        reactants = inverse_global_align(global_reactants)
        key = canonical_reaction_key(product, reactants)
        achiral_key = canonical_reaction_key_achiral(product, reactants)
        return {
            "status": "ok",
            "block_index": block_index,
            "reaction_key": key,
            "reaction_key_sha256": reaction_key_sha256(key),
            "achiral_reaction_key": achiral_key,
            "achiral_reaction_key_sha256": reaction_key_sha256(achiral_key),
        }
    except Exception as exc:
        return {
            "status": "parse_error",
            "block_index": block_index,
            "error": str(exc),
        }


def _ordered_map(function, items: list, workers: int) -> list:
    if workers == 1:
        return [function(item) for item in items]
    with Pool(processes=workers) as pool:
        return list(pool.imap(function, items, chunksize=128))


def _load_raw(path: Path) -> list[tuple[int, str, str, str]]:
    records = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or REACTION_COLUMN not in reader.fieldnames:
            raise ValueError(f"missing {REACTION_COLUMN!r} column in {path}")
        for raw_index, row in enumerate(reader):
            records.append(
                (
                    raw_index,
                    row.get("id", str(raw_index)),
                    row.get("class", "UNK"),
                    row[REACTION_COLUMN],
                )
            )
    return records


def _load_processed_blocks(
    source_path: Path,
    target_path: Path,
    *,
    augmentation: int,
) -> tuple[list[tuple[int, str, str]], int]:
    blocks: list[tuple[int, str, str]] = []
    row_count = 0
    with source_path.open() as source_handle, target_path.open() as target_handle:
        while True:
            source_rows = [source_handle.readline() for _ in range(augmentation)]
            target_rows = [target_handle.readline() for _ in range(augmentation)]
            if not any(source_rows) and not any(target_rows):
                break
            if not all(source_rows) or not all(target_rows):
                raise ValueError(
                    "processed source/target files do not contain complete "
                    f"augmentation={augmentation} blocks"
                )
            block_index = len(blocks)
            blocks.append((block_index, source_rows[0], target_rows[0]))
            row_count += augmentation
        if source_handle.readline() or target_handle.readline():
            raise ValueError("processed source/target line-count mismatch")
    return blocks, row_count


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _crosswalk(
    raw_labels: list[dict[str, Any]],
    processed_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    duplicate_keys: list[dict[str, Any]] = []
    unmatched_raw = [item for item in raw_labels if item["status"] == "ok"]
    unmatched_processed = [
        item for item in processed_records if item["status"] == "ok"
    ]
    method_counts: Counter[str] = Counter()

    def pair_by_key(key_field: str, method: str) -> None:
        nonlocal unmatched_raw, unmatched_processed
        raw_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        processed_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in unmatched_raw:
            raw_by_key[record[key_field]].append(record)
        for record in unmatched_processed:
            processed_by_key[record[key_field]].append(record)
        used_raw: set[int] = set()
        used_processed: set[int] = set()
        for key in sorted(set(raw_by_key) & set(processed_by_key)):
            raw_group = sorted(
                raw_by_key[key], key=lambda item: item["raw_index"]
            )
            processed_group = sorted(
                processed_by_key[key], key=lambda item: item["block_index"]
            )
            if len(raw_group) > 1 or len(processed_group) > 1:
                duplicate_keys.append(
                    {
                        "match_method": method,
                        "reaction_key_sha256": reaction_key_sha256(key),
                        "raw_count": len(raw_group),
                        "processed_count": len(processed_group),
                        "raw_indices": [item["raw_index"] for item in raw_group],
                        "processed_block_indices": [
                            item["block_index"] for item in processed_group
                        ],
                    }
                )
            paired_count = min(len(raw_group), len(processed_group))
            for occurrence in range(paired_count):
                raw_record = raw_group[occurrence]
                processed_record = processed_group[occurrence]
                ambiguous = len(raw_group) > 1 or len(processed_group) > 1
                matches.append(
                    {
                        "processed_block_index": processed_record["block_index"],
                        "raw_index": raw_record["raw_index"],
                        "reaction_id": raw_record["reaction_id"],
                        "reaction_key_sha256": raw_record["reaction_key_sha256"],
                        "match_method": method,
                        "key_occurrence": occurrence,
                        "key_is_ambiguous": ambiguous,
                    }
                )
                method_counts[method] += 1
                used_raw.add(raw_record["raw_index"])
                used_processed.add(processed_record["block_index"])
        unmatched_raw = [
            item for item in unmatched_raw if item["raw_index"] not in used_raw
        ]
        unmatched_processed = [
            item
            for item in unmatched_processed
            if item["block_index"] not in used_processed
        ]

    pair_by_key("reaction_key", "exact_isomeric")
    # The historical global-alignment pipeline flips the written @/@@ form in
    # about 1% of reaction pairs.  Match only the still-unmatched records after
    # removing stereo, retain the mapped raw label, and mark this fallback so
    # stereochemical analyses can exclude it.
    pair_by_key("achiral_reaction_key", "achiral_fallback")

    raw_only = unmatched_raw
    processed_only = unmatched_processed
    matches.sort(key=lambda item: item["processed_block_index"])
    summary = {
        "matched_count": len(matches),
        "unique_unambiguous_match_count": sum(
            not item["key_is_ambiguous"] for item in matches
        ),
        "ambiguous_occurrence_match_count": sum(
            item["key_is_ambiguous"] for item in matches
        ),
        "duplicate_key_count": len(duplicate_keys),
        "match_method_counts": dict(sorted(method_counts.items())),
        "raw_only_count": len(raw_only),
        "processed_only_count": len(processed_only),
        "raw_only": [
            {
                "raw_index": item["raw_index"],
                "reaction_id": item["reaction_id"],
                "reaction_key_sha256": item["reaction_key_sha256"],
                "status": item["status"],
            }
            for item in raw_only[:100]
        ],
        "processed_only": [
            {
                "processed_block_index": item["block_index"],
                "reaction_key_sha256": item["reaction_key_sha256"],
                "status": item["status"],
            }
            for item in processed_only[:100]
        ],
        "duplicate_keys": duplicate_keys[:100],
    }
    return matches, summary


def _label_summary(labels: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(record["status"] for record in labels)
    valid = [record for record in labels if record["status"] == "ok"]
    component_counts = Counter(record["center_component_count"] for record in valid)
    center_type_counts = Counter()
    atom_field_counts = Counter()
    bond_kind_counts = Counter()
    for record in valid:
        bond_kind_counts.update(event["kind"] for event in record["changed_bonds"])
        for event in record["atom_changes"]:
            atom_field_counts.update(event["changed_fields"])
        for component in record["center_components"]:
            center_type_counts.update(component["center_types"])
    return {
        "status_counts": dict(sorted(status_counts.items())),
        "valid_reaction_count": len(valid),
        "no_center_count": sum(not record["has_center"] for record in valid),
        "multi_center_count": sum(
            record["center_component_count"] > 1 for record in valid
        ),
        "over_three_center_count": sum(
            record["center_component_count"] > 3 for record in valid
        ),
        "reagent_field_nonempty_count": sum(
            record["reagent_field_nonempty"] for record in valid
        ),
        "zero_map_product_reaction_count": sum(
            record["zero_map_product_atom_count"] > 0 for record in valid
        ),
        "zero_map_reactant_reaction_count": sum(
            record["zero_map_reactant_atom_count"] > 0 for record in valid
        ),
        "component_count_histogram": {
            str(key): value for key, value in sorted(component_counts.items())
        },
        "center_type_counts": dict(sorted(center_type_counts.items())),
        "bond_kind_counts": dict(sorted(bond_kind_counts.items())),
        "atom_changed_field_counts": dict(sorted(atom_field_counts.items())),
    }


def process_split(
    *,
    split: str,
    raw_csv: Path,
    processed_dir: Path,
    cache_dir: Path,
    augmentation: int,
    workers: int,
) -> dict[str, Any]:
    raw_items = _load_raw(raw_csv)
    raw_labels = _ordered_map(_raw_worker, raw_items, workers)
    split_dir = processed_dir / split
    source_path = split_dir / f"src-{split}.txt"
    target_path = split_dir / f"tgt-{split}.txt"
    blocks, processed_row_count = _load_processed_blocks(
        source_path, target_path, augmentation=augmentation
    )
    processed_records = _ordered_map(_processed_worker, blocks, workers)
    matches, crosswalk_summary = _crosswalk(raw_labels, processed_records)

    labels_path = cache_dir / f"reaction_centers_{split}.jsonl"
    processed_path = cache_dir / f"processed_keys_{split}.jsonl"
    crosswalk_path = cache_dir / f"raw_to_processed_{split}.jsonl"
    _write_jsonl(labels_path, raw_labels)
    _write_jsonl(processed_path, processed_records)
    _write_jsonl(crosswalk_path, matches)
    return {
        "raw_csv": str(raw_csv),
        "raw_sha256": _sha256(raw_csv),
        "raw_reaction_count": len(raw_labels),
        "processed_source": str(source_path),
        "processed_target": str(target_path),
        "processed_source_sha256": _sha256(source_path),
        "processed_target_sha256": _sha256(target_path),
        "processed_row_count": processed_row_count,
        "processed_reaction_block_count": len(blocks),
        "processed_status_counts": dict(
            sorted(Counter(item["status"] for item in processed_records).items())
        ),
        "labels": _label_summary(raw_labels),
        "crosswalk": crosswalk_summary,
        "cache": {
            "labels_path": str(labels_path),
            "labels_sha256": _sha256(labels_path),
            "processed_keys_path": str(processed_path),
            "processed_keys_sha256": _sha256(processed_path),
            "crosswalk_path": str(crosswalk_path),
            "crosswalk_sha256": _sha256(crosswalk_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw_dir", type=Path, default=Path("datasets/USPTO_50K")
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("datasets/USPTO_50K_PtoR_aug20_#global#"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("results/after_spe_stage1/cache"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("after_spe/results/stage1/s1_crosswalk_report.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.augmentation <= 0:
        raise ValueError("augmentation must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    unknown = sorted(set(args.splits) - {"train", "val", "test"})
    if unknown:
        raise ValueError(f"unknown splits: {unknown}")

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Reaction-center labels and raw-to-global R-SMILES crosswalk",
        "augmentation": args.augmentation,
        "workers": args.workers,
        "rdkit_aromatic_normalization": (
            "Sanitized RDKit aromaticity; aromatic bonds use one normalized "
            "AROMATIC signature before graph comparison."
        ),
        "splits": {},
    }
    for split in args.splits:
        raw_csv = args.raw_dir / f"raw_{split}.csv"
        report["splits"][split] = process_split(
            split=split,
            raw_csv=raw_csv,
            processed_dir=args.processed_dir,
            cache_dir=args.cache_dir,
            augmentation=args.augmentation,
            workers=args.workers,
        )
        split_report = report["splits"][split]
        print(
            f"{split}: raw={split_report['raw_reaction_count']} "
            f"processed={split_report['processed_reaction_block_count']} "
            f"matched={split_report['crosswalk']['matched_count']} "
            f"raw_only={split_report['crosswalk']['raw_only_count']} "
            f"processed_only={split_report['crosswalk']['processed_only_count']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
