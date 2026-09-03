"""Guidance-data utilities for product-conditioned DGM adapters.

The first guidance dataset is intentionally a small, explicit ``.pt``
artifact rather than a new training data format.  Each record keeps the
product, an intermediate Edit Flows state, its time, the sampled terminal
state, reward, and provenance seeds.  The base Edit Flows checkpoint remains
frozen and the input split must be performed by product before records are
generated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from edit_flows.core.alignment import opt_align_xs_to_zs
from edit_flows.core.scheduler import KappaScheduler
from edit_flows.core.z_space import rm_gap_tokens, sample_cond_zt
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


GUIDANCE_DATA_SCHEMA_VERSION = 1


def _validate_bos_states(states: Tensor, name: str) -> None:
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError(
            f"{name} must be a non-empty [batch, length] tensor, got "
            f"{tuple(states.shape)}"
        )
    if (states[:, 0] != BOS_TOKEN).any():
        raise ValueError(f"{name} must contain BOS_TOKEN in column 0")


def _pad_aligned_pair(z0: Tensor, z1: Tensor) -> tuple[Tensor, Tensor]:
    max_len = max(z0.shape[1], z1.shape[1])
    if z0.shape[1] != max_len:
        padded = torch.full(
            (z0.shape[0], max_len), PAD_TOKEN,
            dtype=z0.dtype, device=z0.device,
        )
        padded[:, :z0.shape[1]] = z0
        z0 = padded
    if z1.shape[1] != max_len:
        padded = torch.full(
            (z1.shape[0], max_len), PAD_TOKEN,
            dtype=z1.dtype, device=z1.device,
        )
        padded[:, :z1.shape[1]] = z1
        z1 = padded
    return z0, z1


@torch.no_grad()
def sample_intermediate_states(
    product_states: Tensor,
    terminal_states: Tensor,
    time_steps: Tensor | Sequence[float] | float,
    *,
    vocab_size: int,
    scheduler: KappaScheduler,
    align_fn: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] =
        opt_align_xs_to_zs,
) -> Tensor:
    """Sample aligned intermediate ``x_t`` states for guidance records.

    Both inputs include BOS in column zero and use PAD for storage padding.
    Alignment is performed in CPU/GPU tensor space using the same optimal
    alignment and conditional interpolation used by Edit Flows training.  The
    returned states have gaps removed, so they can be fed directly to the
    action-level guidance model.
    """
    _validate_bos_states(product_states, "product_states")
    _validate_bos_states(terminal_states, "terminal_states")
    if product_states.shape[0] != terminal_states.shape[0]:
        raise ValueError("product_states and terminal_states batch sizes differ")
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")

    batch_size = product_states.shape[0]
    if isinstance(time_steps, Tensor):
        times = time_steps.to(device=product_states.device, dtype=torch.float32)
    elif isinstance(time_steps, (float, int)):
        times = torch.full(
            (batch_size, 1), float(time_steps),
            dtype=torch.float32, device=product_states.device,
        )
    else:
        times = torch.tensor(
            list(time_steps), dtype=torch.float32, device=product_states.device,
        )
    if times.ndim == 0:
        times = times.expand(batch_size).reshape(batch_size, 1)
    elif times.ndim == 1 and times.shape[0] == batch_size:
        times = times.unsqueeze(1)
    elif times.ndim == 2 and times.shape == (batch_size, 1):
        pass
    else:
        raise ValueError(
            "time_steps must be scalar, [batch], or [batch, 1], got "
            f"{tuple(times.shape)}"
        )
    if not torch.isfinite(times).all() or (times < 0).any() or (times > 1).any():
        raise ValueError("time_steps must be finite values in [0, 1]")

    product_states = product_states.to(dtype=torch.long)
    terminal_states = terminal_states.to(
        device=product_states.device, dtype=torch.long,
    )
    z0, z1 = align_fn(product_states, terminal_states)
    z0, z1 = _pad_aligned_pair(z0, z1)
    z_t = sample_cond_zt(
        z0, z1, times, vocab_size, scheduler,
    )
    x_t, _, _, _ = rm_gap_tokens(z_t)
    return x_t


def make_guidance_record(
    *,
    product_tokens: Sequence[int],
    state_tokens: Sequence[int],
    terminal_tokens: Sequence[int],
    time_step: float,
    reward: float,
    source_index: int,
    sample_index: int,
    time_index: int,
    sample_seed: int,
    coupling_seed: int,
    transition_tokens: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Create a serializable, validated guidance record."""
    if not 0.0 <= float(time_step) <= 1.0:
        raise ValueError(f"time_step must be in [0, 1], got {time_step}")
    if not torch.isfinite(torch.tensor(float(reward))):
        raise ValueError(f"reward must be finite, got {reward}")
    if source_index < 0 or sample_index < 0 or time_index < 0:
        raise ValueError("record indices must be non-negative")
    if sample_seed < 0 or coupling_seed < 0:
        raise ValueError("record seeds must be non-negative")
    product = [int(token) for token in product_tokens]
    state = [int(token) for token in state_tokens]
    terminal = [int(token) for token in terminal_tokens]
    transition = (
        [int(token) for token in transition_tokens]
        if transition_tokens is not None else None
    )
    if not product or product[0] != BOS_TOKEN:
        raise ValueError("product_tokens must begin with BOS_TOKEN")
    if not state or state[0] != BOS_TOKEN:
        raise ValueError("state_tokens must begin with BOS_TOKEN")
    if not terminal or terminal[0] != BOS_TOKEN:
        raise ValueError("terminal_tokens must begin with BOS_TOKEN")
    if transition is not None and (not transition or transition[0] != BOS_TOKEN):
        raise ValueError("transition_tokens must begin with BOS_TOKEN")
    record = {
        "product_tokens": product,
        "state_tokens": state,
        "terminal_tokens": terminal,
        "time": float(time_step),
        "reward": float(reward),
        "source_index": int(source_index),
        "sample_index": int(sample_index),
        "time_index": int(time_index),
        "sample_seed": int(sample_seed),
        "coupling_seed": int(coupling_seed),
    }
    if transition is not None:
        record["transition_tokens"] = transition
    return record


def save_guidance_dataset(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save guidance records and provenance metadata to a CPU ``.pt`` file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": GUIDANCE_DATA_SCHEMA_VERSION,
        "records": [dict(record) for record in records],
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, destination)


def load_guidance_dataset(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate a guidance dataset saved by this module."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("guidance dataset payload must be a dictionary")
    if payload.get("schema_version") != GUIDANCE_DATA_SCHEMA_VERSION:
        raise ValueError(
            "unsupported guidance dataset schema: "
            f"{payload.get('schema_version')}"
        )
    records = payload.get("records")
    metadata = payload.get("metadata", {})
    if not isinstance(records, list) or not isinstance(metadata, dict):
        raise ValueError("guidance dataset has invalid records/metadata fields")
    return [dict(record) for record in records], dict(metadata)


def _pad_record_sequences(records: Sequence[Mapping[str, Any]], key: str) -> Tensor:
    sequences = [list(record[key]) for record in records]
    if not sequences:
        raise ValueError("cannot collate an empty guidance batch")
    max_len = max(len(sequence) for sequence in sequences)
    result = torch.full((len(sequences), max_len), PAD_TOKEN, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        result[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return result


def collate_guidance_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
    """Pad records into tensors accepted by ``ProductConditionedGuidance``."""
    if not records:
        raise ValueError("cannot collate an empty guidance batch")
    product_tokens = _pad_record_sequences(records, "product_tokens")
    state_tokens = _pad_record_sequences(records, "state_tokens")
    terminal_tokens = _pad_record_sequences(records, "terminal_tokens")
    has_transition = ["transition_tokens" in record for record in records]
    if any(has_transition) and not all(has_transition):
        raise ValueError(
            "transition_tokens must be present in either every record or none"
        )
    batch = {
        "product_tokens": product_tokens,
        "state_tokens": state_tokens,
        "terminal_tokens": terminal_tokens,
        "time": torch.tensor(
            [float(record["time"]) for record in records], dtype=torch.float32,
        ),
        "reward": torch.tensor(
            [float(record["reward"]) for record in records], dtype=torch.float32,
        ),
        "source_index": torch.tensor(
            [int(record["source_index"]) for record in records], dtype=torch.long,
        ),
        "sample_index": torch.tensor(
            [int(record["sample_index"]) for record in records], dtype=torch.long,
        ),
        "time_index": torch.tensor(
            [int(record["time_index"]) for record in records], dtype=torch.long,
        ),
    }
    if all(has_transition):
        transition_tokens = _pad_record_sequences(records, "transition_tokens")
        if (transition_tokens[:, 0] != BOS_TOKEN).any():
            raise ValueError("transition_tokens must begin with BOS_TOKEN")
        batch["transition_tokens"] = transition_tokens
    return batch


class GuidanceDataset(Dataset):
    """Torch Dataset wrapper around the explicit guidance record artifact."""

    def __init__(self, path: str | Path) -> None:
        self.records, self.metadata = load_guidance_dataset(path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class ProductGroupBatchSampler(Sampler[list[int]]):
    """Yield batches that contain complete product groups.

    Guidance records are stored one record per sampled terminal.  For the
    multi-terminal data used by pairwise guidance, all records with one
    ``source_index`` must be visible in the same batch.  This sampler groups
    record indices before shuffling, so a group can never be split across two
    batches.  The default DataLoader path remains unchanged; callers opt into
    this sampler through ``batch_sampler``.
    """

    def __init__(
        self,
        data_source,
        *,
        batch_size: int,
        group_size: int = 4,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1 or group_size < 1:
            raise ValueError("batch_size and group_size must be positive")
        if batch_size % group_size:
            raise ValueError(
                "batch_size must be divisible by group_size, got "
                f"batch_size={batch_size}, group_size={group_size}"
            )
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.batch_size = int(batch_size)
        self.group_size = int(group_size)
        self.groups_per_batch = self.batch_size // self.group_size
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        groups: dict[int, list[int]] = {}
        for index in range(len(data_source)):
            record = data_source[index]
            try:
                source_index = int(record["source_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "every guidance record must contain an integer source_index"
                ) from exc
            groups.setdefault(source_index, []).append(index)
        invalid = {
            source_index: len(indices)
            for source_index, indices in groups.items()
            if len(indices) != self.group_size
        }
        if invalid:
            preview = list(sorted(invalid.items()))[:5]
            raise ValueError(
                "all product groups must have exactly group_size records; "
                f"invalid groups (first five)={preview}"
            )
        self._groups = [groups[key] for key in sorted(groups)]

    @property
    def group_count(self) -> int:
        """Number of complete product groups available to the sampler."""
        return len(self._groups)

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic epoch offset used when shuffling groups."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self):
        if self.shuffle and len(self._groups) > 1:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(self._groups), generator=generator).tolist()
        else:
            order = list(range(len(self._groups)))

        group_limit = len(order)
        if self.drop_last:
            group_limit = (
                group_limit // self.groups_per_batch
            ) * self.groups_per_batch
        for start in range(0, group_limit, self.groups_per_batch):
            selected = order[start:start + self.groups_per_batch]
            if not selected:
                continue
            batch: list[int] = []
            for group_index in selected:
                batch.extend(self._groups[group_index])
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self._groups) // self.groups_per_batch
        return (
            len(self._groups) + self.groups_per_batch - 1
        ) // self.groups_per_batch
