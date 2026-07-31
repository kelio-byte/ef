from typing import List, Tuple

import torch
from torch import Tensor

from edit_flows.core.coupling import Coupling


class DatasetCoupling(Coupling):
    def __init__(
        self,
        pairs: List[Tuple[List[int], List[int]]],
        pad_token: int = 0,
    ):
        self.pairs = pairs
        self.pad_token = pad_token
        self._index = 0

    def set_batch_indices(self, indices: List[int]) -> None:
        self._batch_indices = indices

    def sample(self, x1: Tensor) -> Tuple[Tensor, Tensor]:
        batch_pairs = [self.pairs[i] for i in self._batch_indices]
        x0_list, x1_list = zip(*batch_pairs)

        max_src = max(len(ids) for ids in x0_list)
        max_tgt = max(len(ids) for ids in x1_list)

        batch_size = len(batch_pairs)
        device = x1.device
        x0 = torch.full((batch_size, max_src), self.pad_token, dtype=torch.long, device=device)
        x1_out = torch.full((batch_size, max_tgt), self.pad_token, dtype=torch.long, device=device)

        for i, (ids_src, ids_tgt) in enumerate(batch_pairs):
            x0[i, :len(ids_src)] = torch.tensor(ids_src, dtype=torch.long, device=device)
            x1_out[i, :len(ids_tgt)] = torch.tensor(ids_tgt, dtype=torch.long, device=device)

        return x0, x1_out
