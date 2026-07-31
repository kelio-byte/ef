#!/usr/bin/env python
"""Experiment 1: Oracle Greedy — single-edit greedy search with oracle rates.

Uses the theoretically optimal edit rates (from known target x_1) to drive
greedy single-edit search.  This establishes the upper bound for single-edit
search: if oracle+greedy works well, the search framework is viable and the
problem is model rate quality; if not, single-edit discretisation itself is
flawed.
"""

import argparse
import math
import os
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from edit_flows.data.dataset import load_vocab
from edit_flows.sampling.beam import sample_greedy_single_edit
from edit_flows.sampling.time_policy import FixedTimePolicy
from edit_flows.sampling.oracle import compute_oracle_model_output
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN


class OracleModel(nn.Module):
    """Model wrapper that replaces learned rates with oracle (optimal) rates.

    Forwards to ``compute_oracle_model_output`` which dynamically aligns x_t
    with the known target x_1 and computes theoretically optimal edit rates.
    """

    def __init__(self, x_1: Tensor, scheduler, vocab_size: int,
                 pad_token: int = PAD_TOKEN, bos_token: int = BOS_TOKEN):
        super().__init__()
        self.register_buffer("x_1", x_1)  # (B, L_1) — targets, PAD-padded
        self._scheduler = scheduler
        self._vocab_size = vocab_size
        self._pad_token = pad_token
        self._bos_token = bos_token
        # Minimal embedding so beam.py can infer vocab_size.
        self.token_embedding = nn.Embedding(vocab_size, 1)

    def forward(self, tokens, time_step, padding_mask, origin_mask=None):
        B = tokens.shape[0]
        # x_1 is (B_all, L_1).  The batch order matches — beam.py never
        # shuffles samples within a batch.
        x_1_batch = self.x_1[:B]
        log_rates, log_ins_probs, log_sub_probs, _edit_dists = \
            compute_oracle_model_output(
                tokens, x_1_batch, time_step, self._scheduler,
                self._vocab_size, pad_token=self._pad_token,
                bos_token=self._bos_token,
            )
        return log_rates, log_ins_probs, log_sub_probs


def tokenize_smiles(smiles: str, token2id: dict) -> list:
    tokens = smiles.strip().split()
    unk_id = token2id.get("<unk>", 3)
    return [token2id.get(t, unk_id) for t in tokens]


def _ids_to_str(ids: list, id2token: dict) -> str:
    return " ".join(id2token[tid] for tid in ids
                    if tid not in (PAD_TOKEN, BOS_TOKEN))


def _make_batch(ids_list: list[list[int]], pad_token: int,
                bos_token: int = BOS_TOKEN) -> Tensor:
    B = len(ids_list)
    max_len = max(len(ids) for ids in ids_list)
    x = torch.full((B, max_len + 1), pad_token, dtype=torch.long)
    x[:, 0] = bos_token
    for i, ids in enumerate(ids_list):
        x[i, 1:1 + len(ids)] = torch.tensor(ids, dtype=torch.long)
    return x


def main():
    parser = argparse.ArgumentParser(
        description="Oracle Greedy: single-edit greedy with oracle rates")
    parser.add_argument("--data_dir", type=str,
                        default="analysis_subsets/USPTO_50K_PtoR_aug20_#global#/"
                                "test_dedup_seed42_1000")
    parser.add_argument("--vocab_file", type=str,
                        default="/data6/duanbh/desktop/retrosynthesis/dataset/"
                                "USPTO_50K_PtoR_aug20_#global#/example.vocab.src")
    parser.add_argument("--max_edits", type=int, default=20)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scheduler", type=str, default="cubic",
                        choices=["cubic", "linear"])
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of samples per product (greedy is "
                             "deterministic, so >1 just replicates)")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load vocab.
    token2id, model_vocab = load_vocab(args.vocab_file)
    id2token = {v: k for k, v in token2id.items()}

    # Load data.
    src_path = os.path.join(args.data_dir, "src-test.txt")
    tgt_path = os.path.join(args.data_dir, "tgt-test.txt")
    with open(src_path) as f:
        products = [line.strip() for line in f]
    with open(tgt_path) as f:
        targets = [line.strip() for line in f]
    print(f"Loaded {len(products)} products, {len(targets)} targets")

    product_ids = [tokenize_smiles(s, token2id) for s in products]
    target_ids = [tokenize_smiles(s, token2id) for s in targets]

    scheduler = CubicScheduler() if args.scheduler == "cubic" else LinearScheduler()

    # Build x_1 batch (all targets).
    x_1_full = _make_batch(target_ids, PAD_TOKEN, BOS_TOKEN).to(device)

    n_products = len(products)
    n_batches = math.ceil(n_products / args.batch_size)

    pred_file = os.path.join(args.output_dir, "predictions.txt")
    with open(pred_file, "w") as f_out:
        for batch_idx in tqdm(range(n_batches), desc="Oracle Greedy"):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, n_products)
            batch_products = product_ids[start:end]

            x_0 = _make_batch(batch_products, PAD_TOKEN, BOS_TOKEN)

            # Per-batch OracleModel with correct x_1 slice.
            batch_targets = target_ids[start:end]
            x_1_batch = _make_batch(batch_targets, PAD_TOKEN, BOS_TOKEN).to(device)
            oracle_model = OracleModel(
                x_1_batch, scheduler, model_vocab,
                pad_token=PAD_TOKEN, bos_token=BOS_TOKEN,
            ).to(device)

            results = sample_greedy_single_edit(
                oracle_model, x_0, scheduler,
                FixedTimePolicy(scheduler=scheduler, time_const=0.5),
                max_edits=args.max_edits,
                max_seq_len=args.max_seq_len,
                use_rate_reparam=False,  # oracle outputs raw real rates
                use_origin_mask=False,
                k_ins_token=4, k_sub_token=4, k_edit_expand=16,
                stop_u_tot_base=0.1,   # prevent noise edits after sequence complete
            )

            results = results.cpu()
            B = end - start
            for i in range(B):
                for _ in range(args.n_samples):
                    line = _ids_to_str(results[i].tolist(), id2token)
                    f_out.write(line + "\n")

    print(f"Done. Predictions saved to: {pred_file}")


if __name__ == "__main__":
    main()
