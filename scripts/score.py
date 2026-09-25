"""Frozen global-SMILES ranking: 20 augmentations x 9 candidates per reaction."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from edit_flows.scoring import canonicalize_smiles_clear_map, compute_rank
from edit_flows.inference import sha256


def score(predictions, targets, workers=8):
    pred_path = Path(predictions)
    prediction_lines = pred_path.read_text().splitlines()
    target_lines = Path(targets).read_text().splitlines()
    if not prediction_lines or len(prediction_lines) % 180:
        raise ValueError("Predictions must have 180 lines per complete reaction")
    count = len(prediction_lines) // 180
    if len(target_lines) % 20 or len(target_lines) < count * 20:
        raise ValueError("Targets must cover complete 20-augmentation blocks")
    meta_path = pred_path.with_name("sampling_metadata.json")
    if not meta_path.exists():
        raise FileNotFoundError("Sampling metadata required")
    meta = json.loads(meta_path.read_text())
    if (
        meta["n_products"] * 9 != len(prediction_lines)
        or meta["outputs_per_product"] != 9
    ):
        raise ValueError("Prediction layout disagrees with metadata")
    if meta["predictions_sha256"] != sha256(pred_path):
        raise ValueError("Predictions changed since sampling")
    strings = ["".join(x.strip().split(" ")) for x in prediction_lines]
    target_strings = [
        "".join(target_lines[i].strip().split(" ")) for i in range(0, count * 20, 20)
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        canonical = list(pool.map(canonicalize_smiles_clear_map, strings, chunksize=64))
        truth = list(
            pool.map(canonicalize_smiles_clear_map, target_strings, chunksize=64)
        )
    if any(not t[0] for t in truth):
        raise ValueError("An evaluation target cannot be canonicalized")
    hits = [0] * 10
    oracle_hits = invalid1 = 0
    ranks = []
    for i in range(count):
        block = canonical[i * 180 : (i + 1) * 180]
        views = [block[a * 9 : (a + 1) * 9] for a in range(20)]
        rank, invalid = compute_rank(views, beam_size=9)
        ranked = sorted(rank, key=rank.get, reverse=True)[:10]
        found = next((j + 1 for j, c in enumerate(ranked) if c[0] == truth[i][0]), None)
        ranks.append(found)
        if found is not None:
            for k in range(found - 1, 10):
                hits[k] += 1
        oracle_hits += int(any(c[0] == truth[i][0] for c in block))
        invalid1 += invalid[0]
    result = {
        "reaction_count": count,
        "top_k_accuracy_percent": {
            str(k + 1): 100 * v / count for k, v in enumerate(hits)
        },
        "top_k_hits": hits,
        "oracle_any_percent": 100 * oracle_hits / count,
        "invalid_at_1_percent": 100 * invalid1 / (count * 20),
        "target_ranks": ranks,
        "aggregation_mode": "legacy_best_rank",
        "score_alpha": 1.0,
        "augmentation": 20,
        "beam_size": 9,
        "predictions_sha256": sha256(pred_path),
        "targets_sha256": sha256(targets),
    }
    output = pred_path.with_name("metrics.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k != "target_ranks"}, indent=2)
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    score(args.predictions, args.targets, args.workers)
