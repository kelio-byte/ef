from itertools import zip_longest
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN, GAP_TOKEN, UNK_TOKEN

SPECIAL_TOKENS = {"<PAD>": PAD_TOKEN, "<BOS>": BOS_TOKEN, "<GAP>": GAP_TOKEN, "<UNK>": UNK_TOKEN}


def load_vocab(vocab_path: str) -> Tuple[Dict[str, int], int]:
    token2id = dict(SPECIAL_TOKENS)
    with open(vocab_path) as f:
        for i, line in enumerate(f):
            token = line.strip().split()[0]
            token2id[token] = i + 4
    model_vocab = len(token2id)
    return token2id, model_vocab


class RetroDataset(Dataset):
    def __init__(self, src_path: str, tgt_path: str, token2id: Dict[str, int]):
        unk_id = token2id["<UNK>"]
        self.pairs: List[Tuple[List[int], List[int]]] = []
        with open(src_path) as f_src, open(tgt_path) as f_tgt:
            for line_no, (src_line, tgt_line) in enumerate(
                zip_longest(f_src, f_tgt), start=1,
            ):
                if src_line is None or tgt_line is None:
                    raise ValueError(
                        "source/target line-count mismatch at line "
                        f"{line_no}: {src_path} vs {tgt_path}"
                    )
                src_ids = [token2id.get(t, unk_id) for t in src_line.strip().split()]
                tgt_ids = [token2id.get(t, unk_id) for t in tgt_line.strip().split()]
                self.pairs.append((src_ids, tgt_ids))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.pairs[idx]


class PreAlignedDataset(Dataset):
    def __init__(self, z0_path: str, z1_path: str, token2id: Dict[str, int]):
        unk_id = token2id["<UNK>"]
        self.pairs: List[Tuple[List[int], List[int]]] = []
        with open(z0_path) as f0, open(z1_path) as f1:
            for line_no, (z0_line, z1_line) in enumerate(
                zip_longest(f0, f1), start=1,
            ):
                if z0_line is None or z1_line is None:
                    raise ValueError(
                        "aligned source/target line-count mismatch at line "
                        f"{line_no}: {z0_path} vs {z1_path}"
                    )
                z0_ids = [token2id.get(t, unk_id) for t in z0_line.strip().split()]
                z1_ids = [token2id.get(t, unk_id) for t in z1_line.strip().split()]
                if len(z0_ids) != len(z1_ids):
                    raise ValueError(
                        "aligned pair length mismatch at line "
                        f"{line_no}: {len(z0_ids)} != {len(z1_ids)} "
                        f"({z0_path} vs {z1_path})"
                    )
                self.pairs.append((z0_ids, z1_ids))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.pairs[idx]


def collate_fn(
    batch: List[Tuple[List[int], List[int]]],
    pad_token: int = PAD_TOKEN,
) -> Tuple[Tensor, Tensor]:
    x0_list, x1_list = zip(*batch)
    max_src = max(len(ids) for ids in x0_list)
    max_tgt = max(len(ids) for ids in x1_list)

    x0 = torch.full((len(batch), max_src), pad_token, dtype=torch.long)
    x1 = torch.full((len(batch), max_tgt), pad_token, dtype=torch.long)

    for i, (ids_src, ids_tgt) in enumerate(batch):
        x0[i, :len(ids_src)] = torch.tensor(ids_src, dtype=torch.long)
        x1[i, :len(ids_tgt)] = torch.tensor(ids_tgt, dtype=torch.long)

    return x0, x1
