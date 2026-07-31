import json
from typing import Dict, List, Sequence

import torch
from torch import Tensor

from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN, UNK_TOKEN


def parse_time_grid(time_grid: str) -> List[float]:
    return [float(item.strip()) for item in time_grid.split(",") if item.strip()]


def load_parallel_texts(
    products_file: str,
    targets_file: str,
    deduplicate: int = 0,
    max_lines: int = 0,
) -> tuple[List[str], List[str]]:
    with open(products_file) as f:
        products = [line.rstrip("\n") for line in f]
    with open(targets_file) as f:
        targets = [line.rstrip("\n") for line in f]

    if len(products) != len(targets):
        raise ValueError(
            f"Product/target count mismatch: {len(products)} vs {len(targets)}",
        )

    if deduplicate > 0:
        products = products[::deduplicate]
        targets = targets[::deduplicate]

    if max_lines > 0:
        products = products[:max_lines]
        targets = targets[:max_lines]

    return products, targets


def tokenize_smiles(smiles: str, token2id: Dict[str, int]) -> List[int]:
    unk_id = token2id.get("<UNK>", UNK_TOKEN)
    return [token2id.get(tok, unk_id) for tok in smiles.strip().split()]


def build_model_batch(
    product_ids: Sequence[Sequence[int]],
    target_ids: Sequence[Sequence[int]],
    pad_token: int = PAD_TOKEN,
    bos_token: int = BOS_TOKEN,
) -> tuple[Tensor, Tensor]:
    batch_size = len(product_ids)
    max_src = max(len(ids) for ids in product_ids) if batch_size > 0 else 0
    max_tgt = max(len(ids) for ids in target_ids) if batch_size > 0 else 0

    x_0 = torch.full((batch_size, max_src + 1), pad_token, dtype=torch.long)
    x_1 = torch.full((batch_size, max_tgt + 1), pad_token, dtype=torch.long)
    x_0[:, 0] = bos_token
    x_1[:, 0] = bos_token

    for i, (src_ids, tgt_ids) in enumerate(zip(product_ids, target_ids)):
        if src_ids:
            x_0[i, 1:1 + len(src_ids)] = torch.tensor(src_ids, dtype=torch.long)
        if tgt_ids:
            x_1[i, 1:1 + len(tgt_ids)] = torch.tensor(tgt_ids, dtype=torch.long)

    return x_0, x_1


def decode_sequence(ids: Sequence[int], id2token: Dict[int, str]) -> str:
    return " ".join(
        id2token[token_id] for token_id in ids
        if token_id not in (PAD_TOKEN, BOS_TOKEN)
    )


def compute_average_precision(
    scores: Sequence[float],
    labels: Sequence[bool],
) -> float:
    ranked = sorted(
        zip(scores, labels),
        key=lambda item: item[0],
        reverse=True,
    )
    n_pos = sum(1 for label in labels if label)
    if n_pos == 0:
        return 0.0

    hit_count = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            hit_count += 1
            precision_sum += hit_count / rank
    return precision_sum / n_pos


def _tensor_to_bool_list(mask: Tensor) -> List[bool]:
    return [bool(x) for x in mask.tolist()]


def extract_position_labels(
    log_rates: Tensor,
    x_tokens: Tensor,
    rate_threshold: float = 1e-6,
    exclude_bos: bool = True,
) -> tuple[List[float], List[bool]]:
    rates = torch.exp(log_rates)
    if rates.dim() == 2:
        pos_scores = rates.sum(dim=-1)
        pos_labels = pos_scores > rate_threshold
        pos_pad = x_tokens == PAD_TOKEN
        pos_labels[pos_pad] = False
        pos_scores[pos_pad] = float("-inf")
        if exclude_bos and x_tokens.numel() > 0:
            pos_labels[0] = False
            pos_scores[0] = float("-inf")
        return pos_scores.tolist(), _tensor_to_bool_list(pos_labels)

    pos_scores = rates.sum(dim=-1)
    pos_labels = pos_scores > rate_threshold
    pos_pad = x_tokens == PAD_TOKEN
    pos_labels[pos_pad] = False
    pos_scores[pos_pad] = float("-inf")
    if exclude_bos and x_tokens.numel() > 0:
        pos_labels[:, 0] = False
        pos_scores[:, 0] = float("-inf")
    return pos_scores.tolist(), _tensor_to_bool_list(pos_labels)


def extract_oracle_event_set(
    log_rates: Tensor,
    log_ins_probs: Tensor,
    log_sub_probs: Tensor,
    x_tokens: Tensor,
    rate_threshold: float = 1e-6,
    exclude_bos: bool = True,
) -> dict:
    rates = torch.exp(log_rates)
    if rates.dim() == 2:
        ins_rates = rates[:, 0]
        sub_rates = rates[:, 1]
        del_rates = rates[:, 2]
    else:
        ins_rates = rates[..., 0]
        sub_rates = rates[..., 1]
        del_rates = rates[..., 2]

    pos_mask = (ins_rates + sub_rates + del_rates) > rate_threshold
    ins_mask = ins_rates > rate_threshold
    sub_mask = sub_rates > rate_threshold
    del_mask = del_rates > rate_threshold

    pad_mask = x_tokens == PAD_TOKEN
    pos_mask[pad_mask] = False
    ins_mask[pad_mask] = False
    sub_mask[pad_mask] = False
    del_mask[pad_mask] = False

    if exclude_bos and x_tokens.numel() > 0:
        if pos_mask.dim() == 1:
            pos_mask[0] = False
            ins_mask[0] = False
            sub_mask[0] = False
            del_mask[0] = False
        else:
            pos_mask[:, 0] = False
            ins_mask[:, 0] = False
            sub_mask[:, 0] = False
            del_mask[:, 0] = False

    return {
        "pos_mask": pos_mask,
        "ins_mask": ins_mask,
        "sub_mask": sub_mask,
        "del_mask": del_mask,
        "type_argmax": rates.argmax(dim=-1),
        "ins_token": log_ins_probs.argmax(dim=-1),
        "sub_token": log_sub_probs.argmax(dim=-1),
    }


def compute_reaction_edit_distance(
    pred_ids: Sequence[int],
    target_ids: Sequence[int],
    pad_token: int = PAD_TOKEN,
) -> int:
    from edit_flows.sampling.oracle import _align_pair

    pred = torch.tensor(list(pred_ids), dtype=torch.long)
    tgt = torch.tensor(list(target_ids), dtype=torch.long)
    pred = pred[pred != pad_token]
    tgt = tgt[tgt != pad_token]
    _, _, dist = _align_pair(pred, tgt)
    return int(dist)


def compute_exact_match_flags(
    predictions: Sequence[str],
    targets: Sequence[str],
) -> List[bool]:
    return [pred == target for pred, target in zip(predictions, targets)]


def dump_json(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
