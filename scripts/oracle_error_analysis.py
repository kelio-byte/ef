#!/usr/bin/env python
"""Analyze oracle generation failures: why does the oracle miss some edits?

For each product where the top-1 oracle prediction doesn't match the target,
categorize the error (invalid SMILES vs valid-but-wrong) and analyze the
remaining edit distance, missed edit types, K-values, and positions.

Usage:
  PYTHONPATH=. python scripts/oracle_error_analysis.py \
      --predictions train_subsets/eval/oracle_standard_linear/predictions.txt \
      --targets train_subsets/eval/oracle_standard_linear/targets_subset.txt \
      --products train_subsets/USPTO_50K_PtoR_aug20/test/src-test.txt \
      --vocab_file /data6/duanbh/desktop/retrosynthesis/dataset/USPTO_50K_PtoR_aug20/example.vocab.src \
      --deduplicate 20 --n_samples 10
"""

import argparse
import os
import sys
from collections import Counter

import torch
from torch import Tensor
from rdkit import Chem

from edit_flows.data.dataset import load_vocab
from edit_flows.sampling.oracle import _align_pair
from edit_flows.core.z_space import make_ut_mask_from_z, rm_gap_tokens
from edit_flows.utils.tokens import PAD_TOKEN, GAP_TOKEN, BOS_TOKEN


def tokenize(smiles: str, token2id: dict) -> Tensor:
    tokens = smiles.strip().split()
    unk = token2id.get("<unk>", token2id.get("__unk__", 0))
    ids = [token2id.get(t, unk) for t in tokens]
    return torch.tensor(ids, dtype=torch.long)


def detokenize(ids: Tensor, id2token: dict) -> str:
    return " ".join(
        id2token[int(t)] for t in ids
        if int(t) not in (PAD_TOKEN, BOS_TOKEN, GAP_TOKEN)
    )


def smiles_valid(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def compute_edit_breakdown(
    pred_ids: Tensor,
    tgt_ids: Tensor,
) -> tuple[int, int, int, int]:
    """Return (total_edits, n_ins, n_del, n_sub) from pred→target alignment."""
    aligned_p, aligned_t, _ = _align_pair(pred_ids, tgt_ids)
    n_ins, n_del, n_sub = 0, 0, 0
    for a, b in zip(aligned_p, aligned_t):
        if a == b:
            continue
        if a == GAP_TOKEN:
            n_ins += 1
        elif b == GAP_TOKEN:
            n_del += 1
        else:
            n_sub += 1
    return n_ins + n_del + n_sub, n_ins, n_del, n_sub


def compute_missed_k_values(
    pred_ids: Tensor,
    tgt_ids: Tensor,
    vocab_size: int,
) -> dict:
    """Align pred→target, compute K-values for each missed edit position.

    Returns {k_value: count} mapping and per-edit details.
    """
    aligned_p, aligned_t, _ = _align_pair(pred_ids, tgt_ids)
    L_z = len(aligned_p)

    z_pred = torch.tensor(aligned_p, dtype=torch.long).unsqueeze(0)
    z_tgt = torch.tensor(aligned_t, dtype=torch.long).unsqueeze(0)

    uz_mask = make_ut_mask_from_z(z_pred, z_tgt, vocab_size=vocab_size)
    # uz_mask: (1, L_z, 2*V+1)

    # rm_gap_tokens to get X-space structure
    x_aligned, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z_pred)

    if x_aligned.shape[1] == 0:
        return {"k_values": [], "details": []}

    L_x = x_aligned.shape[1]
    n_ops = uz_mask.shape[2]

    # Map Z→X: non-GAP positions index into X
    non_gap = ~z_gap_mask  # (1, L_z)
    indices = non_gap.long().cumsum(dim=1) - 1
    indices = indices.clamp(min=0, max=L_x - 1)

    # Sum K per (X_pos, edit_channel)
    k_per_pos = {}
    z_idx = indices[0]  # (L_z,)
    uz = uz_mask[0]     # (L_z, n_ops)
    for j in range(L_z):
        if z_pad_mask[0, j]:
            continue
        xp = int(z_idx[j].item())
        for c in range(n_ops):
            if uz[j, c]:
                key = (xp, c)
                k_per_pos[key] = k_per_pos.get(key, 0) + 1

    k_values = list(k_per_pos.values())
    details = [
        {"x_pos": xp, "channel": c, "k": k}
        for (xp, c), k in k_per_pos.items()
    ]

    return {"k_values": k_values, "details": details}


def edit_channel_name(c: int, vocab_size: int) -> str:
    V = vocab_size
    if c < V:
        return f"ins(token={c})"
    elif c < 2 * V:
        return f"sub(token={c - V})"
    else:
        return "del"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze oracle generation failures")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--targets", type=str, required=True)
    parser.add_argument("--products", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--deduplicate", type=int, default=0,
                        help="Deduplicate stride for --products only "
                             "(targets should already match predictions)")
    parser.add_argument("--output", type=str, default="",
                        help="Output file (default: <predictions_dir>/error_analysis.txt)")
    args = parser.parse_args()

    token2id, vocab_size = load_vocab(args.vocab_file)
    id2token = {v: k for k, v in token2id.items()}

    with open(args.predictions) as f:
        predictions = [line.strip() for line in f]
    with open(args.targets) as f:
        targets = [line.strip() for line in f]
    with open(args.products) as f:
        products = [line.strip() for line in f]

    if args.deduplicate > 0:
        products = products[::args.deduplicate]
    # targets_subset.txt is already deduplicated — do NOT re-dedup

    n_products = len(targets)
    n_preds = len(predictions)

    assert n_preds == n_products * args.n_samples, \
        f"Predictions ({n_preds}) != products ({n_products}) x n_samples ({args.n_samples})"
    assert len(products) == n_products, \
        f"Products ({len(products)}) != targets ({n_products}) after dedup"

    print(f"Loaded: {n_products} products, {n_preds} predictions "
          f"({args.n_samples} samples x {n_products} products)")

    # --- Per-product analysis ---
    failures = []
    total_edits_list = []
    failure_edits_list = []
    all_missed_k = []
    missed_positions = []  # (normalized_pos, edit_type)
    invalid_count = 0
    valid_wrong_count = 0

    for i in range(n_products):
        tgt_str = targets[i]
        prod_str = products[i]
        top1_str = predictions[i * args.n_samples]

        # Tokenize (no BOS — raw token comparison)
        tgt_ids = tokenize(tgt_str, token2id)
        prod_ids = tokenize(prod_str, token2id)
        top1_ids = tokenize(top1_str, token2id)

        total_dist, _, _, _ = compute_edit_breakdown(prod_ids, tgt_ids)
        total_edits_list.append(total_dist)

        if top1_str == tgt_str:
            continue

        # Failure case
        is_valid = smiles_valid(top1_str)
        if not is_valid:
            invalid_count += 1
        else:
            valid_wrong_count += 1

        rem_dist, rem_ins, rem_del, rem_sub = compute_edit_breakdown(
            top1_ids, tgt_ids)
        failure_edits_list.append(rem_dist)

        # K-value analysis
        k_info = compute_missed_k_values(top1_ids, tgt_ids, vocab_size)
        all_missed_k.extend(k_info["k_values"])

        # Position analysis (normalized by target length)
        tgt_len = len(tgt_ids)
        for d in k_info["details"]:
            xp = d["x_pos"]
            # Approximate normalized position
            norm_pos = min(xp / max(tgt_len, 1), 1.0)
            missed_positions.append((norm_pos, d["channel"]))

        failures.append({
            "idx": i,
            "product": prod_str,
            "target": tgt_str,
            "top1": top1_str,
            "valid": is_valid,
            "total_edits": total_dist,
            "remaining_edits": rem_dist,
            "remaining_ins": rem_ins,
            "remaining_del": rem_del,
            "remaining_sub": rem_sub,
            "k_info": k_info,
        })

    # --- Report ---
    lines = []
    def emit(s):
        lines.append(s)
        print(s)

    emit("=" * 60)
    emit("ORACLE ERROR ANALYSIS")
    emit("=" * 60)
    emit(f"Predictions: {args.predictions}")
    emit(f"Total products: {n_products}")
    emit(f"Failures (top-1): {len(failures)} ({len(failures)/n_products*100:.1f}%)")
    emit("")

    # 1. Failure categories
    emit("--- Failure Categories ---")
    emit(f"Invalid SMILES:   {invalid_count:4d} ({invalid_count/max(len(failures),1)*100:.1f}% of failures)")
    emit(f"Valid-but-Wrong:  {valid_wrong_count:4d} ({valid_wrong_count/max(len(failures),1)*100:.1f}% of failures)")
    emit("")

    # 2. Edit distance distribution for valid-but-wrong
    emit("--- Valid-but-Wrong: Edit Distance Distribution ---")
    vw_dists = [f["remaining_edits"] for f in failures if f["valid"]]
    dist_counter = Counter(vw_dists)
    for d in sorted(dist_counter):
        emit(f"  Edit distance = {d}: {dist_counter[d]} ({dist_counter[d]/max(len(vw_dists),1)*100:.1f}%)")
    if not vw_dists:
        emit("  (none)")
    emit("")

    # 3. Edit distance for ALL failures (including invalid)
    emit("--- All Failures: Edit Distance Distribution ---")
    all_dists = [f["remaining_edits"] for f in failures]
    all_counter = Counter(all_dists)
    for d in sorted(all_counter):
        emit(f"  Edit distance = {d}: {all_counter[d]} ({all_counter[d]/max(len(failures),1)*100:.1f}%)")
    emit(f"  Mean: {sum(all_dists)/max(len(all_dists),1):.1f}")
    emit("")

    # 4. Missed edit types
    emit("--- Missed Edit Types (aggregated over all failures) ---")
    total_ins = sum(f["remaining_ins"] for f in failures)
    total_del = sum(f["remaining_del"] for f in failures)
    total_sub = sum(f["remaining_sub"] for f in failures)
    total_missed = total_ins + total_del + total_sub
    if total_missed > 0:
        emit(f"  Insert:     {total_ins:4d} ({total_ins/total_missed*100:.1f}%)")
        emit(f"  Delete:     {total_del:4d} ({total_del/total_missed*100:.1f}%)")
        emit(f"  Substitute: {total_sub:4d} ({total_sub/total_missed*100:.1f}%)")
    emit("")

    # 5. K-value distribution
    emit("--- K-value Distribution for Missed Edits ---")
    if all_missed_k:
        k_counter = Counter(all_missed_k)
        for k in sorted(k_counter):
            emit(f"  K={k}: {k_counter[k]} ({k_counter[k]/len(all_missed_k)*100:.1f}%)")
        emit(f"  K=1 fraction: {k_counter.get(1,0)/len(all_missed_k)*100:.1f}%")
        emit(f"  K>1 fraction: {(len(all_missed_k)-k_counter.get(1,0))/len(all_missed_k)*100:.1f}%")
        emit(f"  Mean K: {sum(all_missed_k)/len(all_missed_k):.2f}")
    else:
        emit("  (no missed edits found — all failures are exact matches?)")
    emit("")

    # 6. Position distribution
    emit("--- Missed Edit Position Distribution (normalized) ---")
    if missed_positions:
        buckets = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for lo, hi in buckets:
            count = sum(1 for p, _ in missed_positions if lo <= p < hi)
            emit(f"  [{lo:.1f}-{hi:.1f}): {count:4d} ({count/len(missed_positions)*100:.1f}%)")
    emit("")

    # 7. Total edits needed vs completed
    emit("--- Completion Analysis ---")
    avg_total = sum(total_edits_list) / max(len(total_edits_list), 1)
    emit(f"  Mean edits needed (product→target): {avg_total:.1f}")
    if failures:
        avg_remaining = sum(f["remaining_edits"] for f in failures) / len(failures)
        emit(f"  Mean remaining edits (on failures): {avg_remaining:.1f}")
        emit(f"  Mean completion rate (on failures): {(1 - avg_remaining/max(avg_total,1))*100:.1f}%")
    emit("")

    # 8. Length correlation
    emit("--- Sequence Length Analysis ---")
    tgt_lens = [len(tokenize(t, token2id)) for t in targets]
    fail_lens = [len(tokenize(f["target"], token2id)) for f in failures]
    avg_tgt_len = sum(tgt_lens) / max(len(tgt_lens), 1)
    avg_fail_len = sum(fail_lens) / max(len(fail_lens), 1) if fail_lens else 0
    emit(f"  Mean target length (all):    {avg_tgt_len:.1f}")
    emit(f"  Mean target length (failed): {avg_fail_len:.1f}")
    emit("")

    # 9. Per-failure details (top 20)
    emit("--- Per-Failure Details (first 30) ---")
    for f in failures[:30]:
        status = "INVALID" if not f["valid"] else "VALID_WRONG"
        emit(f"  [{f['idx']:4d}] {status:12s} | "
             f"total_edits={f['total_edits']:2d} | "
             f"missed={f['remaining_edits']:2d} "
             f"(ins={f['remaining_ins']}, del={f['remaining_del']}, sub={f['remaining_sub']}) | "
             f"K_vals={f['k_info']['k_values']}")
        emit(f"         product: {f['product'][:80]}")
        emit(f"         target:  {f['target'][:80]}")
        emit(f"         top-1:   {f['top1'][:80]}")
    if len(failures) > 30:
        emit(f"  ... and {len(failures)-30} more failures")

    emit("")
    emit("=" * 60)

    # Save to file
    out_path = args.output
    if not out_path:
        pred_dir = os.path.dirname(args.predictions)
        out_path = os.path.join(pred_dir, "error_analysis.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
