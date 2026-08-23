#!/usr/bin/env python3
"""Build oracle and same-product pseudo-center score sidecars for RC1."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from multiprocessing import Pool
from pathlib import Path
import random
from typing import Any

from rdkit import Chem

from edit_flows.chem.reaction_center import split_reaction_smiles
from edit_flows.chem.spe_provenance import (
    component_mode_position_scores,
    load_spe_codes,
    map_raw_product_atoms,
    tokenize_with_provenance,
)


_WORKER_CODES = None
_WORKER_MAX_ISOMORPHISMS = 1024


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def _raw_rows(path: Path) -> dict[int, dict[str, str]]:
    result = {}
    with path.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            _, _, product = split_reaction_smiles(
                row["reactants>reagents>production"]
            )
            result[index] = {
                "reaction_id": row.get("id", str(index)),
                "product": product,
            }
    return result


def _product_map_graph(
    product_smiles: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    molecule = Chem.MolFromSmiles(product_smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse raw mapped product")
    atom_maps = sorted(
        atom.GetAtomMapNum()
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum() > 0
    )
    bonds = []
    for bond in molecule.GetBonds():
        first = bond.GetBeginAtom().GetAtomMapNum()
        second = bond.GetEndAtom().GetAtomMapNum()
        if first > 0 and second > 0:
            bonds.append(tuple(sorted((first, second))))
    return atom_maps, sorted(bonds)


def _pseudo_components(
    label: dict[str, Any], raw_product: str, *, seed: int
) -> list[dict[str, Any]]:
    atom_maps, bonds = _product_map_graph(raw_product)
    true_center = set(label["center_atom_maps"])
    radius_1 = set().union(
        *(
            set(component["radius_1_atom_maps"])
            for component in label["center_components"]
        )
    )
    radius_2 = set().union(
        *(
            set(component["radius_2_atom_maps"])
            for component in label["center_components"]
        )
    )
    rng = random.Random(seed)
    used: set[int] = set()
    result = []
    for component in label["center_components"]:
        wants_bond = bool(component["has_product_bond_change"])
        selected = None
        relaxation = None
        for relaxation_name, excluded in (
            ("outside_true_radius_2", radius_2 | used),
            ("outside_true_radius_1", radius_1 | used),
            ("outside_true_center", true_center | used),
            ("avoid_previous_only", used),
            ("unavoidable_true_center", set()),
        ):
            if wants_bond:
                candidates = [
                    list(bond)
                    for bond in bonds
                    if not set(bond).intersection(excluded)
                ]
            else:
                candidates = [
                    [atom_map]
                    for atom_map in atom_maps
                    if atom_map not in excluded
                ]
            if candidates:
                candidates.sort()
                selected = candidates[rng.randrange(len(candidates))]
                relaxation = relaxation_name
                break
        if selected is None:
            raise ValueError("product has no mapped atom for pseudo center")
        used.update(selected)
        result.append(
            {
                "component_id": component["component_id"],
                "atom_maps": selected,
                "matched_kind": "bond" if wants_bond else "atom",
                "relaxation": relaxation,
            }
        )
    return result


def _init_worker(codes_path: str, merges: int, max_isomorphisms: int) -> None:
    global _WORKER_CODES, _WORKER_MAX_ISOMORPHISMS
    _WORKER_CODES = load_spe_codes(Path(codes_path), merges=merges)
    _WORKER_MAX_ISOMORPHISMS = max_isomorphisms


def _score_components(
    *,
    raw_product: str,
    product_smiles: str,
    token_surfaces: list[str],
    components: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if _WORKER_CODES is None:
        raise RuntimeError("worker tokenizer is not initialized")
    provenance = tokenize_with_provenance(product_smiles, _WORKER_CODES)
    if [token.surface for token in provenance] != token_surfaces:
        raise ValueError("SPE replay differs from center-sidecar input")
    mapping = map_raw_product_atoms(
        raw_product,
        product_smiles,
        max_isomorphisms=_WORKER_MAX_ISOMORPHISMS,
    )
    scored = []
    for component in components:
        processed_indices = set()
        for atom_map in component["atom_maps"]:
            processed_indices.update(
                mapping.atom_map_to_processed_indices[atom_map]
            )
        scored.append(
            {
                **component,
                "processed_atom_indices": sorted(processed_indices),
                "position_scores": component_mode_position_scores(
                    product_smiles, provenance, processed_indices
                ),
            }
        )
    mapping_info = {
        "isomorphism_count": mapping.isomorphism_count,
        "isomorphism_limit_reached": mapping.isomorphism_limit_reached,
        "used_chirality": mapping.used_chirality,
    }
    return scored, mapping_info


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    try:
        product_smiles = "".join(task["global_source"].split())
        token_surfaces = task["m500_source"].split()
        if "".join(token_surfaces) != product_smiles:
            raise ValueError("global and M500 product strings differ")
        oracle_components = [
            {
                "component_id": component["component_id"],
                "atom_maps": component["atom_maps"],
                "center_types": component["center_types"],
                "has_product_bond_change": component[
                    "has_product_bond_change"
                ],
            }
            for component in task["center_components"]
        ]
        oracle, mapping = _score_components(
            raw_product=task["raw_product"],
            product_smiles=product_smiles,
            token_surfaces=token_surfaces,
            components=oracle_components,
        )
        pseudo, pseudo_mapping = _score_components(
            raw_product=task["raw_product"],
            product_smiles=product_smiles,
            token_surfaces=token_surfaces,
            components=task["pseudo_components"],
        )
        return {
            "status": "ok",
            "input_row_index": task["input_row_index"],
            "reaction_position": task["reaction_position"],
            "augmentation_index": task["augmentation_index"],
            "processed_block_index": task["processed_block_index"],
            "raw_index": task["raw_index"],
            "reaction_id": task["reaction_id"],
            "token_count": len(token_surfaces),
            "crosswalk_match_method": task["crosswalk_match_method"],
            "crosswalk_ambiguous": task["crosswalk_ambiguous"],
            "oracle_components": oracle,
            "pseudo_components": pseudo,
            "mapping": mapping,
            "pseudo_mapping": pseudo_mapping,
        }
    except Exception as exc:
        return {
            "status": "error",
            "input_row_index": task["input_row_index"],
            "reaction_position": task["reaction_position"],
            "augmentation_index": task["augmentation_index"],
            "error": str(exc),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/manifest.json"
        ),
    )
    parser.add_argument("--evaluation_split", default="dev_unique1000_aug20")
    parser.add_argument(
        "--global_products",
        type=Path,
        default=Path(
            "datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/"
            "dev_unique1000_aug20/src.txt"
        ),
    )
    parser.add_argument(
        "--m500_products",
        type=Path,
        default=Path(
            "datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/evaluation_v2/"
            "dev_unique1000_aug20/src.txt"
        ),
    )
    parser.add_argument(
        "--raw_csv",
        type=Path,
        default=Path("datasets/USPTO_50K/raw_val.csv"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(
            "results/after_spe_stage1/cache/reaction_centers_val.jsonl"
        ),
    )
    parser.add_argument(
        "--crosswalk",
        type=Path,
        default=Path(
            "results/after_spe_stage1/cache/raw_to_processed_val.jsonl"
        ),
    )
    parser.add_argument(
        "--codes",
        type=Path,
        default=Path("scripts/preprocessing/SPE_ChEMBL.txt"),
    )
    parser.add_argument("--merges", type=int, default=500)
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_isomorphisms", type=int, default=1024)
    parser.add_argument("--max_reactions", type=int)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "results/after_spe_stage1/center_sidecars/dev_unique1000_aug20"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = json.loads(args.manifest.read_text())
    reaction_indices = [
        int(value)
        for value in manifest["splits"][args.evaluation_split][
            "original_reaction_indices"
        ]
    ]
    if args.max_reactions is not None:
        if args.max_reactions <= 0:
            raise ValueError("max_reactions must be positive")
        reaction_indices = reaction_indices[: args.max_reactions]
    expected_rows = len(reaction_indices) * args.augmentation
    global_rows = args.global_products.read_text().splitlines()[:expected_rows]
    m500_rows = args.m500_products.read_text().splitlines()[:expected_rows]
    if len(global_rows) != expected_rows or len(m500_rows) != expected_rows:
        raise ValueError("evaluation product files have too few rows")

    labels = {
        record["raw_index"]: record for record in _read_jsonl(args.labels)
    }
    crosswalk = {
        record["processed_block_index"]: record
        for record in _read_jsonl(args.crosswalk)
    }
    raw = _raw_rows(args.raw_csv)
    tasks = []
    pseudo_relaxations = {}
    for reaction_position, block_index in enumerate(reaction_indices):
        match = crosswalk[block_index]
        label = labels[match["raw_index"]]
        raw_row = raw[match["raw_index"]]
        pseudo = _pseudo_components(
            label,
            raw_row["product"],
            seed=args.seed + 1000003 * match["raw_index"],
        )
        pseudo_relaxations[str(reaction_position)] = [
            component["relaxation"] for component in pseudo
        ]
        for augmentation_index in range(args.augmentation):
            input_row_index = (
                reaction_position * args.augmentation + augmentation_index
            )
            tasks.append(
                {
                    "input_row_index": input_row_index,
                    "reaction_position": reaction_position,
                    "augmentation_index": augmentation_index,
                    "processed_block_index": block_index,
                    "raw_index": match["raw_index"],
                    "reaction_id": raw_row["reaction_id"],
                    "raw_product": raw_row["product"],
                    "center_components": label["center_components"],
                    "pseudo_components": pseudo,
                    "global_source": global_rows[input_row_index],
                    "m500_source": m500_rows[input_row_index],
                    "crosswalk_match_method": match["match_method"],
                    "crosswalk_ambiguous": match["key_is_ambiguous"],
                }
            )

    if args.workers == 1:
        _init_worker(str(args.codes), args.merges, args.max_isomorphisms)
        records = [_worker(task) for task in tasks]
    else:
        with Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(str(args.codes), args.merges, args.max_isomorphisms),
        ) as pool:
            records = list(pool.imap(_worker, tasks, chunksize=32))
    errors = [record for record in records if record["status"] != "ok"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "scores.jsonl"
    with scores_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "RC1 oracle true-center and same-product pseudo-center scores"
        ),
        "deployable": False,
        "evaluation_split": args.evaluation_split,
        "reaction_count": len(reaction_indices),
        "augmentation": args.augmentation,
        "input_row_count": expected_rows,
        "seed": args.seed,
        "merges": args.merges,
        "component_assignment": (
            "trajectory_index modulo up-to-three components"
        ),
        "score_definition": {
            "distance_0": 1.0,
            "distance_1": 0.5,
            "distance_ge_2": 0.0,
        },
        "files": {
            "manifest": {
                "path": str(args.manifest),
                "sha256": _sha256(args.manifest),
            },
            "global_products": {
                "path": str(args.global_products),
                "sha256": _sha256(args.global_products),
            },
            "m500_products": {
                "path": str(args.m500_products),
                "sha256": _sha256(args.m500_products),
            },
            "labels": {
                "path": str(args.labels),
                "sha256": _sha256(args.labels),
            },
            "crosswalk": {
                "path": str(args.crosswalk),
                "sha256": _sha256(args.crosswalk),
            },
            "scores": {
                "path": str(scores_path),
                "sha256": _sha256(scores_path),
            },
        },
        "status_counts": {
            status: sum(record["status"] == status for record in records)
            for status in sorted({record["status"] for record in records})
        },
        "pseudo_relaxations": pseudo_relaxations,
        "errors": errors[:100],
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Built {len(records)} rows for {len(reaction_indices)} reactions; "
        f"errors={len(errors)}; output={args.output_dir}"
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
