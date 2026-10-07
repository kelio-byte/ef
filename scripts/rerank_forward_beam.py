#!/usr/bin/env python3
"""Rerank existing retrosynthesis candidates by forward beam reconstruction."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from rdkit import Chem, RDLogger
import torch

from edit_flows.forward import (
    forward_beam_reconstruction_rank,
    load_molecular_transformer,
)
from edit_flows.forward.molecular_transformer import retro_global_to_smiles


RDLogger.DisableLog("rdApp.*")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_global(value: str) -> str:
    try:
        molecule = Chem.MolFromSmiles(retro_global_to_smiles(value))
    except (TypeError, ValueError):
        return ""
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def _pairwise_auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    positive = scores[labels]
    negative = scores[~labels]
    if not positive.numel() or not negative.numel():
        return None
    wins = 0
    ties = 0
    for chunk in positive.split(256):
        difference = chunk[:, None] - negative[None, :]
        wins += int((difference > 0).sum())
        ties += int((difference == 0).sum())
    return (wins + 0.5 * ties) / (positive.numel() * negative.numel())


def run(args: argparse.Namespace) -> dict:
    prediction_path = Path(args.predictions)
    metadata_path = prediction_path.parent / "sampling_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"sampling metadata is required beside predictions: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text())
    beam_size = int(metadata["output_beam_size"])
    product_count = int(metadata["product_count"])
    prediction_lines = prediction_path.read_text().splitlines()
    if len(prediction_lines) != product_count * beam_size:
        raise ValueError("prediction count does not match sampling metadata")
    if metadata.get("output_sha256") != _sha256(prediction_path):
        raise ValueError("prediction SHA does not match sampling metadata")
    input_metadata = metadata.get("input", {})
    selection_start = int(input_metadata.get("selection_start_product", 0))
    selection_end = selection_start + product_count

    products = Path(args.products_file).read_text().splitlines()
    targets = Path(args.targets_file).read_text().splitlines()
    if selection_end > len(products) or selection_end > len(targets):
        raise ValueError("selected prediction interval exceeds product/target files")
    products = products[selection_start:selection_end]
    targets = targets[selection_start:selection_end]
    canonical_targets = [_canonical_global(value) for value in targets]
    repeated_products = [
        products[index // beam_size] for index in range(len(prediction_lines))
    ]

    scorer = load_molecular_transformer(args.checkpoint, device=args.device)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cache: dict[str, list[str]] = {}
    generation_stats: dict[str, int] = {}
    started = time.perf_counter()
    ranks = forward_beam_reconstruction_rank(
        scorer,
        prediction_lines,
        repeated_products,
        beam_size=args.forward_beam_size,
        max_length=args.max_length,
        min_length=args.min_length,
        batch_size=args.batch_size,
        forbid_unk=args.forbid_unk,
        canonicalize_source=args.canonicalize_source,
        cache=cache,
        stats=generation_stats,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    rewards = torch.where(
        ranks > 0,
        ranks.float().reciprocal(),
        torch.zeros_like(ranks, dtype=torch.float32),
    )

    labels = torch.zeros(len(prediction_lines), dtype=torch.bool)
    for index, prediction in enumerate(prediction_lines):
        canonical_prediction = _canonical_global(prediction)
        labels[index] = bool(
            canonical_prediction
            and canonical_prediction == canonical_targets[index // beam_size]
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reranked = []
    for start in range(0, len(prediction_lines), beam_size):
        order = sorted(
            range(beam_size),
            key=lambda offset: (-float(rewards[start + offset]), offset),
        )
        reranked.extend(prediction_lines[start + offset] for offset in order)
    output_predictions = output_dir / "predictions.txt"
    output_predictions.write_text("\n".join(reranked) + "\n")

    positive = labels
    negative = ~labels
    rank_counts = {
        str(rank): int((ranks == rank).sum())
        for rank in range(args.forward_beam_size + 1)
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "forward_beam_reconstruction_rerank",
        "source_predictions": str(prediction_path.resolve()),
        "source_predictions_sha256": metadata["output_sha256"],
        "forward_checkpoint": str(Path(args.checkpoint).resolve()),
        "candidate_count": len(prediction_lines),
        "candidate_beam_size": beam_size,
        "forward_beam_size": args.forward_beam_size,
        "max_length": args.max_length,
        "min_length": args.min_length,
        "batch_size": args.batch_size,
        "canonicalize_source": args.canonicalize_source,
        "unique_generated_sources": len(cache),
        "generation_input_stats": generation_stats,
        "reconstruction_rank_counts": rank_counts,
        "reconstruction_hit_rate": float((ranks > 0).float().mean()),
        "correct_candidate_count": int(positive.sum()),
        "incorrect_candidate_count": int(negative.sum()),
        "correct_reconstruction_hit_rate": (
            float((ranks[positive] > 0).float().mean()) if positive.any() else None
        ),
        "incorrect_reconstruction_hit_rate": (
            float((ranks[negative] > 0).float().mean()) if negative.any() else None
        ),
        "pairwise_correctness_auc": _pairwise_auc(rewards, labels),
        "generation_seconds": elapsed,
        "candidates_per_second": len(prediction_lines) / elapsed,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda" else 0
        ),
        "reranked_predictions": str(output_predictions.resolve()),
    }
    report_path = output_dir / "forward_beam_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    mixing_metadata = {
        "schema_version": 1,
        "created_at_utc": report["created_at_utc"],
        "method": report["method"],
        "augmentation": metadata.get("augmentation"),
        "product_count": product_count,
        "output_beam_size": beam_size,
        "output_line_count": len(reranked),
        "output_sha256": _sha256(output_predictions),
        "input": input_metadata,
        "source_sampling_metadata": str(metadata_path.resolve()),
        "forward_beam_report": str(report_path.resolve()),
    }
    (output_dir / "mixing_metadata.json").write_text(
        json.dumps(mixing_metadata, indent=2, sort_keys=True)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--products_file", required=True)
    parser.add_argument("--targets_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--forward_beam_size", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--forbid_unk", action="store_true")
    parser.add_argument(
        "--canonicalize_source",
        action="store_true",
        help=(
            "Canonicalize candidate reactants before forward generation; this "
            "makes the reward representation-invariant and improves cache reuse."
        ),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
