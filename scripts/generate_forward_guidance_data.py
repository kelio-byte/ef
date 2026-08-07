#!/usr/bin/env python3
"""Attach Molecular Transformer forward rewards to guidance records.

This script never reads reaction targets.  It only consumes the product and
terminal proposal already stored in a guidance-data file, so it is safe for
train/validation reward construction.  The original reward is preserved as
``validity_reward`` and the new raw/positive forward values are recorded for
auditability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from edit_flows.data.dataset import load_vocab
from edit_flows.forward import (
    forward_log_likelihood_reward,
    load_molecular_transformer,
    positive_forward_reward,
)


def _ids_to_global(ids: list[int], id2token: dict[int, str]) -> str:
    """Decode Edit Flows model ids to a space-tokenized global string."""

    return " ".join(
        id2token[int(token)]
        for token in ids
        if int(token) not in (0, 1)  # PAD/BOS are serialization markers.
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--output_data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab_file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--reward_temperature", type=float, default=1.0)
    parser.add_argument("--max_records", type=int, default=-1)
    args = parser.parse_args()
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
    reactants = [_ids_to_global(r["terminal_tokens"], id2token) for r in records]
    products = [_ids_to_global(r["product_tokens"], id2token) for r in records]

    scorer = load_molecular_transformer(args.checkpoint, device=args.device)
    cache: dict[tuple[str, str], float] = {}
    raw = forward_log_likelihood_reward(
        scorer,
        reactants,
        products,
        batch_size=args.batch_size,
        cache=cache,
    )
    positive = positive_forward_reward(raw, temperature=args.reward_temperature)
    output_records = []
    for record, raw_value, positive_value in zip(records, raw.tolist(), positive.tolist()):
        updated = dict(record)
        updated["validity_reward"] = float(record.get("reward", 0.0))
        updated["forward_log_likelihood"] = float(raw_value)
        updated["reward"] = float(positive_value)
        output_records.append(updated)
    metadata = dict(payload.get("metadata", {}))
    metadata.update({
        "record_count": len(output_records),
        "reward": "forward_log_likelihood_exp",
        "reward_temperature": args.reward_temperature,
        "reward_checkpoint": str(Path(args.checkpoint).resolve()),
        "reward_vocab_file": str(Path(args.vocab_file).resolve()),
        "forward_reward_unique_pairs": len(cache),
        "forward_reward_cache_hit_count": len(output_records) - len(cache),
        "forward_reward_invalid_score": -20.0,
    })
    output = {
        "schema_version": payload.get("schema_version", 1),
        "records": output_records,
        "metadata": metadata,
    }
    output_path = Path(args.output_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    report = {
        "input_records": len(records),
        "output": str(output_path.resolve()),
        "unique_pairs": len(cache),
        "cache_hits": len(output_records) - len(cache),
        "raw_log_likelihood": {
            "min": float(raw.min()),
            "mean": float(raw.mean()),
            "max": float(raw.max()),
        },
        "positive_reward": {
            "min": float(positive.min()),
            "mean": float(positive.mean()),
            "max": float(positive.max()),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
