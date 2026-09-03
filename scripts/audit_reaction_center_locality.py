#!/usr/bin/env python3
"""Audit graph-center projection and edit locality in global SPE-M500 data."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from multiprocessing import Pool
from pathlib import Path
import random
from statistics import mean
from typing import Any, Iterable

from rdkit import Chem

from edit_flows.chem.reaction_center import split_reaction_smiles
from edit_flows.chem.spe_provenance import (
    graph_distances,
    insertion_anchor_atoms,
    load_spe_codes,
    map_raw_product_atoms,
    minimum_distance,
    project_syntax_tokens,
    tokenize_with_provenance,
)


GAP = "<GAP>"
_WORKER_CODES = None
_WORKER_MAX_ISOMORPHISMS = 1024


def _init_worker(codes_path: str, merges: int, max_isomorphisms: int) -> None:
    global _WORKER_CODES, _WORKER_MAX_ISOMORPHISMS
    _WORKER_CODES = load_spe_codes(Path(codes_path), merges=merges)
    _WORKER_MAX_ISOMORPHISMS = max_isomorphisms


def _extract_edits(
    aligned_source: list[str], aligned_target: list[str], source_length: int
) -> list[dict[str, Any]]:
    if len(aligned_source) != len(aligned_target):
        raise ValueError("aligned source/target lengths differ")
    edits: list[dict[str, Any]] = []
    source_index = 0
    column = 0
    while column < len(aligned_source):
        source_token = aligned_source[column]
        target_token = aligned_target[column]
        if source_token == GAP:
            if target_token == GAP:
                raise ValueError("alignment contains GAP/GAP")
            begin = column
            inserted_tokens = []
            while column < len(aligned_source) and aligned_source[column] == GAP:
                if aligned_target[column] == GAP:
                    raise ValueError("alignment contains GAP/GAP")
                inserted_tokens.append(aligned_target[column])
                column += 1
            edits.append(
                {
                    "kind": "insertion_run",
                    "anchor": source_index,
                    "aligned_begin": begin,
                    "run_length": len(inserted_tokens),
                }
            )
            continue
        if target_token == GAP:
            edits.append({"kind": "existing_token", "mode": "DEL", "index": source_index})
        elif source_token != target_token:
            edits.append({"kind": "existing_token", "mode": "SUB", "index": source_index})
        source_index += 1
        column += 1
    if source_index != source_length:
        raise ValueError(
            f"aligned source projects to {source_index} tokens, expected {source_length}"
        )
    return edits


def _audit_view(
    *,
    raw_product: str,
    center_components: list[dict[str, Any]],
    global_source: str,
    m500_source: str,
    aligned_source: str,
    aligned_target: str,
) -> dict[str, Any]:
    if _WORKER_CODES is None:
        raise RuntimeError("worker SPE codes were not initialized")
    product_smiles = "".join(global_source.split())
    expected_tokens = m500_source.split()
    provenance_tokens = tokenize_with_provenance(product_smiles, _WORKER_CODES)
    actual_tokens = [token.surface for token in provenance_tokens]
    tokenization_exact = actual_tokens == expected_tokens
    if not tokenization_exact:
        raise ValueError("provenance SPE replay differs from saved M500 source")

    mapping = map_raw_product_atoms(
        raw_product,
        product_smiles,
        max_isomorphisms=_WORKER_MAX_ISOMORPHISMS,
    )
    projected_components: list[set[int]] = []
    for component in center_components:
        projected = set()
        for atom_map in component["atom_maps"]:
            if atom_map not in mapping.atom_map_to_processed_indices:
                raise ValueError(f"center atom map {atom_map} is absent from product mapping")
            projected.update(mapping.atom_map_to_processed_indices[atom_map])
        if projected:
            projected_components.append(projected)
    center_indices = set().union(*projected_components) if projected_components else set()
    molecule = Chem.MolFromSmiles(product_smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse global product")
    distances = graph_distances(molecule, center_indices)
    projected_token_atoms = project_syntax_tokens(provenance_tokens)
    token_distances = [
        minimum_distance(atom_indices, distances)
        for atom_indices in projected_token_atoms
    ]

    collisions = 0
    for token in provenance_tokens:
        touched_components = sum(
            bool(token.atom_indices.intersection(component))
            for component in projected_components
        )
        collisions += touched_components > 1

    aligned_source_tokens = aligned_source.split()
    aligned_target_tokens = aligned_target.split()
    if [token for token in aligned_source_tokens if token != GAP] != expected_tokens:
        raise ValueError("aligned source does not project to saved M500 source")
    edits = _extract_edits(
        aligned_source_tokens, aligned_target_tokens, len(expected_tokens)
    )
    edit_counts = Counter()
    covered = {str(radius): Counter() for radius in (0, 1, 2)}
    mode_counts = Counter()
    for edit in edits:
        kind = edit["kind"]
        edit_counts[kind] += 1
        if kind == "insertion_run":
            atoms = insertion_anchor_atoms(provenance_tokens, edit["anchor"])
        else:
            atoms = projected_token_atoms[edit["index"]]
            mode_counts[edit["mode"]] += 1
        distance = minimum_distance(atoms, distances)
        for radius in (0, 1, 2):
            if distance is not None and distance <= radius:
                covered[str(radius)][kind] += 1

    return {
        "tokenization_exact": tokenization_exact,
        "token_count": len(provenance_tokens),
        "direct_atom_token_count": sum(bool(token.atom_indices) for token in provenance_tokens),
        "syntax_only_token_count": sum(not token.atom_indices for token in provenance_tokens),
        "projected_token_count": sum(bool(value) for value in projected_token_atoms),
        "center_token_count": {
            str(radius): sum(
                distance is not None and distance <= radius
                for distance in token_distances
            )
            for radius in (0, 1, 2)
        },
        "component_collision_token_count": collisions,
        "isomorphism_count": mapping.isomorphism_count,
        "isomorphism_limit_reached": mapping.isomorphism_limit_reached,
        "mapping_used_chirality": mapping.used_chirality,
        "edit_counts": dict(edit_counts),
        "edit_mode_counts": dict(mode_counts),
        "covered_edit_counts": {
            radius: dict(counts) for radius, counts in covered.items()
        },
    }


def _audit_reaction(task: dict[str, Any]) -> dict[str, Any]:
    try:
        views = [
            _audit_view(
                raw_product=task["raw_product"],
                center_components=task["center_components"],
                **view,
            )
            for view in task["views"]
        ]
        return {
            "status": "ok",
            "processed_block_index": task["processed_block_index"],
            "raw_index": task["raw_index"],
            "reaction_id": task["reaction_id"],
            "match_method": task["match_method"],
            "key_is_ambiguous": task["key_is_ambiguous"],
            "center_component_count": len(task["center_components"]),
            "views": views,
        }
    except Exception as exc:
        return {
            "status": "error",
            "processed_block_index": task["processed_block_index"],
            "raw_index": task["raw_index"],
            "reaction_id": task["reaction_id"],
            "error": str(exc),
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_products(path: Path) -> dict[int, str]:
    products = {}
    with path.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            _, _, product = split_reaction_smiles(
                row["reactants>reagents>production"]
            )
            products[index] = product
    return products


def _selected_indices(
    available: list[int], sample_reactions: int | None, seed: int
) -> list[int]:
    if sample_reactions is None or sample_reactions >= len(available):
        return sorted(available)
    if sample_reactions <= 0:
        raise ValueError("sample_reactions must be positive")
    return sorted(random.Random(seed).sample(available, sample_reactions))


def _load_views(
    *,
    split: str,
    global_dir: Path,
    m500_dir: Path,
    selected: set[int],
    augmentation: int,
) -> dict[int, list[dict[str, str]]]:
    paths = {
        "global_source": global_dir / split / f"src-{split}.txt",
        "m500_source": m500_dir / split / f"src-{split}.txt",
        "aligned_source": m500_dir / split / f"{split}_aligned_src.txt",
        "aligned_target": m500_dir / split / f"{split}_aligned_tgt.txt",
    }
    handles = {name: path.open() for name, path in paths.items()}
    views: dict[int, list[dict[str, str]]] = {index: [] for index in selected}
    row_count = 0
    try:
        while True:
            rows = {name: handle.readline() for name, handle in handles.items()}
            if not any(rows.values()):
                break
            if not all(rows.values()):
                raise ValueError("S2 input files have different line counts")
            block_index = row_count // augmentation
            if block_index in selected:
                views[block_index].append(
                    {name: value.rstrip("\n") for name, value in rows.items()}
                )
            row_count += 1
    finally:
        for handle in handles.values():
            handle.close()
    if row_count % augmentation:
        raise ValueError("S2 input rows do not form complete augmentation blocks")
    incomplete = {
        index: len(block_views)
        for index, block_views in views.items()
        if len(block_views) != augmentation
    }
    if incomplete:
        raise ValueError(f"selected blocks have incomplete views: {incomplete}")
    return views


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summarize(results: list[dict[str, Any]], augmentation: int) -> dict[str, Any]:
    status_counts = Counter(result["status"] for result in results)
    valid = [result for result in results if result["status"] == "ok"]
    all_views = [view for result in valid for view in result["views"]]
    total_tokens = sum(view["token_count"] for view in all_views)
    summary: dict[str, Any] = {
        "reaction_status_counts": dict(sorted(status_counts.items())),
        "reaction_count": len(results),
        "valid_reaction_count": len(valid),
        "crosswalk_match_method_counts": dict(
            sorted(Counter(result["match_method"] for result in valid).items())
        ),
        "ambiguous_crosswalk_reaction_count": sum(
            result["key_is_ambiguous"] for result in valid
        ),
        "center_component_count_histogram": {
            str(key): value
            for key, value in sorted(
                Counter(
                    result["center_component_count"] for result in valid
                ).items()
            )
        },
        "view_count": len(all_views),
        "expected_views_per_reaction": augmentation,
        "tokenization_exact_count": sum(
            view["tokenization_exact"] for view in all_views
        ),
        "tokenization_exact_rate": _ratio(
            sum(view["tokenization_exact"] for view in all_views), len(all_views)
        ),
        "mapping_chirality_fallback_view_count": sum(
            not view["mapping_used_chirality"] for view in all_views
        ),
        "mapping_isomorphism_limit_view_count": sum(
            view["isomorphism_limit_reached"] for view in all_views
        ),
        "mapping_isomorphism_limit_reaction_count": sum(
            any(view["isomorphism_limit_reached"] for view in result["views"])
            for result in valid
        ),
        "mapping_multi_isomorphism_view_count": sum(
            view["isomorphism_count"] > 1 for view in all_views
        ),
        "max_isomorphism_count": max(
            (view["isomorphism_count"] for view in all_views), default=0
        ),
        "token_projection": {
            "total_token_count": total_tokens,
            "direct_atom_token_rate": _ratio(
                sum(view["direct_atom_token_count"] for view in all_views),
                total_tokens,
            ),
            "syntax_only_token_rate": _ratio(
                sum(view["syntax_only_token_count"] for view in all_views),
                total_tokens,
            ),
            "locatable_after_syntax_projection_rate": _ratio(
                sum(view["projected_token_count"] for view in all_views),
                total_tokens,
            ),
            "component_collision_token_rate": _ratio(
                sum(
                    view["component_collision_token_count"] for view in all_views
                ),
                total_tokens,
            ),
        },
        "center_token_sparsity": {},
        "edit_locality": {},
    }
    for radius in (0, 1, 2):
        key = str(radius)
        counts = [view["center_token_count"][key] for view in all_views]
        summary["center_token_sparsity"][key] = {
            "micro_token_fraction": _ratio(sum(counts), total_tokens),
            "macro_view_fraction": mean(
                count / view["token_count"]
                for count, view in zip(counts, all_views)
                if view["token_count"]
            ) if all_views else None,
        }
        for kind in ("insertion_run", "existing_token"):
            total = sum(view["edit_counts"].get(kind, 0) for view in all_views)
            covered = sum(
                view["covered_edit_counts"][key].get(kind, 0)
                for view in all_views
            )
            reaction_recalls = []
            any_complete = 0
            all_complete = 0
            reaction_with_events = 0
            for result in valid:
                reaction_total = sum(
                    view["edit_counts"].get(kind, 0) for view in result["views"]
                )
                if not reaction_total:
                    continue
                reaction_with_events += 1
                reaction_covered = sum(
                    view["covered_edit_counts"][key].get(kind, 0)
                    for view in result["views"]
                )
                reaction_recalls.append(reaction_covered / reaction_total)
                eligible_views = [
                    view for view in result["views"]
                    if view["edit_counts"].get(kind, 0) > 0
                ]
                complete = [
                    view["covered_edit_counts"][key].get(kind, 0)
                    == view["edit_counts"].get(kind, 0)
                    for view in eligible_views
                ]
                any_complete += any(complete)
                all_complete += all(complete)
            summary["edit_locality"].setdefault(key, {})[kind] = {
                "event_count": total,
                "covered_event_count": covered,
                "micro_recall": _ratio(covered, total),
                "macro_reaction_recall": (
                    mean(reaction_recalls) if reaction_recalls else None
                ),
                "reaction_with_event_count": reaction_with_events,
                "any_augmentation_complete_recall_rate": _ratio(
                    any_complete, reaction_with_events
                ),
                "all_augmentations_complete_recall_rate": _ratio(
                    all_complete, reaction_with_events
                ),
            }
    summary["edit_mode_counts"] = dict(
        sum((Counter(view["edit_mode_counts"]) for view in all_views), Counter())
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument(
        "--raw_csv", type=Path, default=Path("datasets/USPTO_50K/raw_train.csv")
    )
    parser.add_argument(
        "--global_dir",
        type=Path,
        default=Path("datasets/USPTO_50K_PtoR_aug20_#global#"),
    )
    parser.add_argument(
        "--m500_dir",
        type=Path,
        default=Path("datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("results/after_spe_stage1/cache/reaction_centers_train.jsonl"),
    )
    parser.add_argument(
        "--crosswalk",
        type=Path,
        default=Path("results/after_spe_stage1/cache/raw_to_processed_train.jsonl"),
    )
    parser.add_argument(
        "--codes", type=Path, default=Path("scripts/preprocessing/SPE_ChEMBL.txt")
    )
    parser.add_argument("--merges", type=int, default=500)
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--sample_reactions", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_isomorphisms", type=int, default=1024)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("after_spe/results/stage1/rc0_locality.json"),
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("after_spe/results/stage1/s2_mapping_examples.jsonl"),
    )
    parser.add_argument(
        "--details",
        type=Path,
        default=Path("results/after_spe_stage1/cache/rc0_reaction_details.jsonl"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    labels = {
        record["raw_index"]: record for record in _read_jsonl(args.labels)
    }
    crosswalk = _read_jsonl(args.crosswalk)
    crosswalk_by_block = {
        record["processed_block_index"]: record for record in crosswalk
    }
    selected = _selected_indices(
        list(crosswalk_by_block), args.sample_reactions, args.seed
    )
    selected_set = set(selected)
    views = _load_views(
        split=args.split,
        global_dir=args.global_dir,
        m500_dir=args.m500_dir,
        selected=selected_set,
        augmentation=args.augmentation,
    )
    raw_products = _raw_products(args.raw_csv)
    tasks = []
    for block_index in selected:
        match = crosswalk_by_block[block_index]
        label = labels[match["raw_index"]]
        tasks.append(
            {
                "processed_block_index": block_index,
                "raw_index": match["raw_index"],
                "reaction_id": match["reaction_id"],
                "match_method": match["match_method"],
                "key_is_ambiguous": match["key_is_ambiguous"],
                "raw_product": raw_products[match["raw_index"]],
                "center_components": label["center_components"],
                "views": views[block_index],
            }
        )

    if args.workers == 1:
        _init_worker(str(args.codes), args.merges, args.max_isomorphisms)
        results = [_audit_reaction(task) for task in tasks]
    else:
        with Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(str(args.codes), args.merges, args.max_isomorphisms),
        ) as pool:
            results = list(pool.imap(_audit_reaction, tasks, chunksize=4))

    exact_unambiguous = [
        result
        for result in results
        if result["status"] == "ok"
        and result["match_method"] == "exact_isomeric"
        and not result["key_is_ambiguous"]
    ]
    args.details.parent.mkdir(parents=True, exist_ok=True)
    with args.details.open("w") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "selection": {
            "available_reaction_count": len(crosswalk),
            "selected_reaction_count": len(selected),
            "sample_reactions": args.sample_reactions,
            "seed": args.seed,
            "selected_block_indices": selected,
        },
        "protocol": {
            "augmentation": args.augmentation,
            "merges": args.merges,
            "workers": args.workers,
            "max_isomorphisms": args.max_isomorphisms,
            "syntax_projection": "nearest left/right atom union",
            "insertion_unit": "maximal aligned GAP-to-token run",
        },
        "summary": _summarize(results, args.augmentation),
        "exact_unambiguous_sensitivity": _summarize(
            exact_unambiguous, args.augmentation
        ),
        "errors": [result for result in results if result["status"] != "ok"][:100],
        "details_cache": {
            "path": str(args.details),
            "sha256": _sha256(args.details),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.examples.parent.mkdir(parents=True, exist_ok=True)
    with args.examples.open("w") as handle:
        for result in results[:20]:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"Wrote {args.output} and {args.examples}")
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
