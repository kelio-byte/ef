#!/usr/bin/env python3
"""Attach Molecular Transformer forward rewards to guidance records.

This script never reads reaction targets.  It only consumes the product and
terminal proposal already stored in a guidance-data file, so it is safe for
train/validation reward construction.  The original reward is preserved as
``validity_reward``.  Teacher-forced likelihood and forward-beam product
reconstruction are separate, auditable reward modes.  ``append_likelihood``
adds a raw likelihood field without replacing an existing reward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

import torch
from rdkit import RDLogger

from edit_flows.data.dataset import load_vocab
from edit_flows.forward import (
    forward_beam_reconstruction_rank,
    forward_log_likelihood_reward,
    load_molecular_transformer,
    positive_forward_reward,
)


RDLogger.DisableLog("rdApp.*")


def _ids_to_global(ids: list[int], id2token: dict[int, str]) -> str:
    """Decode Edit Flows model ids to a space-tokenized global string."""

    return " ".join(
        id2token[int(token)]
        for token in ids
        if int(token) not in (0, 1)  # PAD/BOS are serialization markers.
    )


def _group_reward_statistics(
    records: Sequence[dict], reward: torch.Tensor,
) -> dict:
    """Summarize whether same-product samples provide contrasting targets."""

    groups: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(int(record["source_index"]), []).append(index)
    ranges = []
    unique_terminals = []
    for indices in groups.values():
        values = reward[indices]
        ranges.append(float(values.max() - values.min()))
        unique_terminals.append(len({
            tuple(records[index]["terminal_tokens"]) for index in indices
        }))
    group_sizes = [len(indices) for indices in groups.values()]
    variable_count = sum(value > 1e-12 for value in ranges)
    return {
        "product_group_count": len(groups),
        "group_size_min": min(group_sizes) if group_sizes else 0,
        "group_size_max": max(group_sizes) if group_sizes else 0,
        "variable_reward_group_count": variable_count,
        "variable_reward_group_fraction": (
            variable_count / len(groups) if groups else None
        ),
        "mean_reward_range": (
            sum(ranges) / len(ranges) if ranges else None
        ),
        "mean_unique_terminals": (
            sum(unique_terminals) / len(unique_terminals)
            if unique_terminals else None
        ),
        "multiple_terminal_group_fraction": (
            sum(value > 1 for value in unique_terminals) / len(unique_terminals)
            if unique_terminals else None
        ),
    }


def attach_forward_rewards(
    records: Sequence[dict],
    id2token: dict[int, str],
    scorer,
    *,
    reward_mode: str,
    batch_size: int,
    reward_temperature: float = 1.0,
    forward_beam_size: int = 5,
    max_length: int = 200,
    min_length: int = 1,
    forbid_unk: bool = False,
    canonicalize_source: bool = False,
) -> tuple[list[dict], dict, dict]:
    """Attach one auditable forward reward to serialized guidance records."""

    reactants = [_ids_to_global(r["terminal_tokens"], id2token) for r in records]
    products = [_ids_to_global(r["product_tokens"], id2token) for r in records]
    started = time.perf_counter()
    if reward_mode == "likelihood":
        cache: dict[tuple[str, str], float] = {}
        raw = forward_log_likelihood_reward(
            scorer,
            reactants,
            products,
            batch_size=batch_size,
            cache=cache,
        )
        reward = positive_forward_reward(raw, temperature=reward_temperature)
        output_records = []
        for record, raw_value, reward_value in zip(
            records, raw.tolist(), reward.tolist(),
        ):
            updated = dict(record)
            updated["validity_reward"] = float(record.get("reward", 0.0))
            updated["forward_log_likelihood"] = float(raw_value)
            updated["reward"] = float(reward_value)
            output_records.append(updated)
        metadata = {
            "reward": "forward_log_likelihood_exp",
            "reward_temperature": reward_temperature,
            "forward_reward_unique_pairs": len(cache),
            "forward_reward_cache_hit_count": len(records) - len(cache),
            "forward_reward_invalid_score": -20.0,
        }
        report = {
            "raw_log_likelihood": {
                "min": float(raw.min()) if raw.numel() else None,
                "mean": float(raw.mean()) if raw.numel() else None,
                "max": float(raw.max()) if raw.numel() else None,
            },
            "reward": {
                "min": float(reward.min()) if reward.numel() else None,
                "mean": float(reward.mean()) if reward.numel() else None,
                "max": float(reward.max()) if reward.numel() else None,
            },
        }
    elif reward_mode == "beam_reconstruction":
        beam_cache: dict[str, Sequence[str]] = {}
        generation_stats: dict[str, int] = {}
        ranks = forward_beam_reconstruction_rank(
            scorer,
            reactants,
            products,
            beam_size=forward_beam_size,
            max_length=max_length,
            min_length=min_length,
            batch_size=batch_size,
            forbid_unk=forbid_unk,
            canonicalize_source=canonicalize_source,
            cache=beam_cache,
            stats=generation_stats,
        )
        reward = torch.where(
            ranks > 0,
            ranks.float().reciprocal(),
            torch.zeros_like(ranks, dtype=torch.float32),
        )
        output_records = []
        for record, rank, reward_value in zip(
            records, ranks.tolist(), reward.tolist(),
        ):
            updated = dict(record)
            updated["validity_reward"] = float(record.get("reward", 0.0))
            updated["forward_beam_rank"] = int(rank)
            updated["reward"] = float(reward_value)
            output_records.append(updated)
        rank_counts = {
            str(rank): int((ranks == rank).sum())
            for rank in range(forward_beam_size + 1)
        }
        metadata = {
            "reward": "forward_beam_reconstruction_reciprocal_rank",
            "forward_beam_size": forward_beam_size,
            "forward_max_length": max_length,
            "forward_min_length": min_length,
            "forward_forbid_unk": forbid_unk,
            "forward_canonicalize_source": canonicalize_source,
            "forward_reward_unique_sources": len(beam_cache),
            "forward_reward_generation_stats": generation_stats,
        }
        report = {
            "reconstruction_rank_counts": rank_counts,
            "reconstruction_hit_rate": (
                float((ranks > 0).float().mean()) if ranks.numel() else None
            ),
            "reward": {
                "min": float(reward.min()) if reward.numel() else None,
                "mean": float(reward.mean()) if reward.numel() else None,
                "max": float(reward.max()) if reward.numel() else None,
            },
        }
    elif reward_mode == "append_likelihood":
        # Keep the existing reward untouched.  This enables a controlled
        # downstream calibration experiment without accidentally training on
        # a different reward field.
        cache: dict[tuple[str, str], float] = {}
        raw = forward_log_likelihood_reward(
            scorer,
            reactants,
            products,
            batch_size=batch_size,
            cache=cache,
        )
        output_records = []
        for record, raw_value in zip(records, raw.tolist()):
            updated = dict(record)
            updated["forward_log_likelihood"] = float(raw_value)
            output_records.append(updated)
        # ``_group_reward_statistics`` remains useful provenance in every
        # mode; for this non-destructive mode it summarizes the original one.
        reward = torch.tensor(
            [float(record.get("reward", 0.0)) for record in records],
            dtype=torch.float32,
        )
        metadata = {
            "forward_log_likelihood_attached": True,
            "forward_log_likelihood_unique_pairs": len(cache),
            "forward_log_likelihood_cache_hit_count": len(records) - len(cache),
            "forward_log_likelihood_invalid_score": -20.0,
        }
        report = {
            "forward_log_likelihood": {
                "min": float(raw.min()) if raw.numel() else None,
                "mean": float(raw.mean()) if raw.numel() else None,
                "max": float(raw.max()) if raw.numel() else None,
            },
        }
    else:
        raise ValueError(f"unsupported reward_mode: {reward_mode}")
    reward_wall_seconds = time.perf_counter() - started
    metadata["forward_reward_wall_seconds"] = reward_wall_seconds
    report.update({
        "reward_mode": reward_mode,
        "reward_wall_seconds": reward_wall_seconds,
        "product_group_statistics": _group_reward_statistics(records, reward),
    })
    return output_records, metadata, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--output_data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab_file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--reward_mode",
        choices=("likelihood", "beam_reconstruction", "append_likelihood"),
        default="likelihood",
    )
    parser.add_argument("--reward_temperature", type=float, default=1.0)
    parser.add_argument("--forward_beam_size", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--forbid_unk", action="store_true")
    parser.add_argument("--canonicalize_source", action="store_true")
    parser.add_argument("--max_records", type=int, default=-1)
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.max_records == 0 or args.max_records < -1:
        raise ValueError("max_records must be -1 or positive")

    payload = torch.load(args.input_data, map_location="cpu", weights_only=False)
    records = list(payload["records"])
    if args.max_records > 0:
        records = records[: args.max_records]
    token2id, _ = load_vocab(args.vocab_file)
    id2token = {value: key for key, value in token2id.items()}

    scorer = load_molecular_transformer(args.checkpoint, device=args.device)
    output_records, reward_metadata, report = attach_forward_rewards(
        records,
        id2token,
        scorer,
        reward_mode=args.reward_mode,
        batch_size=args.batch_size,
        reward_temperature=args.reward_temperature,
        forward_beam_size=args.forward_beam_size,
        max_length=args.max_length,
        min_length=args.min_length,
        forbid_unk=args.forbid_unk,
        canonicalize_source=args.canonicalize_source,
    )
    metadata = dict(payload.get("metadata", {}))
    metadata.update({
        "record_count": len(output_records),
        "reward_checkpoint": str(Path(args.checkpoint).resolve()),
        "reward_vocab_file": str(Path(args.vocab_file).resolve()),
    })
    metadata.update(reward_metadata)
    output = {
        "schema_version": payload.get("schema_version", 1),
        "records": output_records,
        "metadata": metadata,
    }
    output_path = Path(args.output_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    report.update({
        "input_records": len(records),
        "output": str(output_path.resolve()),
        "reward_checkpoint": str(Path(args.checkpoint).resolve()),
    })
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
