"""用途：按正式规则聚合候选并计算逆合成 Top-k 指标。
输入：候选预测文件、目标文件及对应采样元数据。
输出：各反应排名和汇总评测指标 JSON。
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from edit_flows.scoring import canonicalize_smiles_clear_map, compute_rank
from edit_flows.inference import sha256


def score(predictions, targets, workers=8):
    """作用：对预测文件评分并保存汇总结果。输入：预测文件、目标文件和进程数。输出：指标字典，并写入 metrics.json。"""
    pred_path = Path(predictions)
    prediction_lines = pred_path.read_text().splitlines()
    target_lines = Path(targets).read_text().splitlines()
    meta_path = pred_path.with_name("sampling_metadata.json")
    if not meta_path.exists():
        raise FileNotFoundError("Sampling metadata required")
    meta = json.loads(meta_path.read_text())
    n_runs = meta.get("n_runs")
    n_children = meta.get("n_children")
    augmentation = meta.get("augmentation", 20)
    if not isinstance(n_runs, int) or isinstance(n_runs, bool) or n_runs < 1:
        raise ValueError("Sampling metadata must contain a positive integer n_runs")
    if n_children is not None and (
        not isinstance(n_children, int)
        or isinstance(n_children, bool)
        or n_children < 1
    ):
        raise ValueError("Sampling metadata n_children must be a positive integer")
    if (
        not isinstance(augmentation, int)
        or isinstance(augmentation, bool)
        or augmentation < 1
    ):
        raise ValueError("Sampling metadata must contain a positive augmentation")
    lines_per_reaction = augmentation * n_runs
    if not prediction_lines or len(prediction_lines) % lines_per_reaction:
        raise ValueError(
            f"Predictions must have {lines_per_reaction} lines per complete reaction"
        )
    count = len(prediction_lines) // lines_per_reaction
    if len(target_lines) % augmentation or len(target_lines) < count * augmentation:
        raise ValueError(
            f"Targets must cover complete {augmentation}-augmentation blocks"
        )
    n_products = meta.get("n_products")
    if (
        not isinstance(n_products, int)
        or isinstance(n_products, bool)
        or n_products < 1
        or n_products * n_runs != len(prediction_lines)
        or meta.get("outputs_per_product") != n_runs
        or n_products % augmentation != 0
    ):
        raise ValueError("Prediction layout disagrees with metadata")
    if meta["predictions_sha256"] != sha256(pred_path):
        raise ValueError("Predictions changed since sampling")
    strings = ["".join(x.strip().split(" ")) for x in prediction_lines]
    target_strings = [
        "".join(target_lines[i].strip().split(" "))
        for i in range(0, count * augmentation, augmentation)
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
        block = canonical[
            i * lines_per_reaction : (i + 1) * lines_per_reaction
        ]
        views = [
            block[a * n_runs : (a + 1) * n_runs] for a in range(augmentation)
        ]
        rank, invalid = compute_rank(views, beam_size=n_runs)
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
        "invalid_at_1_percent": 100 * invalid1 / (count * augmentation),
        "target_ranks": ranks,
        "aggregation_mode": "legacy_best_rank",
        "score_alpha": 1.0,
        "augmentation": augmentation,
        "n_runs": n_runs,
        "n_children": n_children,
        "beam_size": n_runs,
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
