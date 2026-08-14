#!/usr/bin/env python
"""Training script for Edit Flows on retrosynthesis data."""

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import glob
import json
import os
import random
import re
import sys
import shutil
import time
import yaml
import numpy as np
import torch
import torch.distributed as dist
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler

from edit_flows.data.dataset import (
    RetroDataset, PreAlignedDataset, load_vocab, collate_fn,
)
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.training.trainer import (
    prepare_batch, train_step, evaluate_step,
)
from edit_flows.training.schedulers import NoamScheduler
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.core.alignment import (
    opt_align_xs_to_zs, naive_align_xs_to_zs, shifted_align_xs_to_zs,
    identity_align_xs_to_zs,
)


@dataclass(frozen=True)
class DistributedContext:
    """Runtime topology supplied by ``torchrun`` (or the single-process default)."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str | None = None

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def initialize_distributed(device_arg: str) -> DistributedContext:
    """Initialize one-process-per-GPU DDP when launched through ``torchrun``.

    A normal ``python scripts/train_retro.py ...`` invocation retains the
    existing single-process behavior.  ``torchrun`` supplies ``RANK``,
    ``WORLD_SIZE`` and ``LOCAL_RANK``; when ``WORLD_SIZE > 1`` we use NCCL for
    CUDA and Gloo for a CPU-only smoke test.
    """
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"RANK={rank} is outside [0, {world_size})")

    requested_device = torch.device(device_arg)
    if world_size == 1:
        if requested_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but torch.cuda.is_available() is false"
                )
            device_index = requested_device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            torch.cuda.set_device(device_index)
            requested_device = torch.device("cuda", device_index)
        return DistributedContext(
            rank=0,
            world_size=1,
            local_rank=0,
            device=requested_device,
        )

    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DDP was launched for CUDA but torch.cuda.is_available() is false"
            )
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} has no visible CUDA device; "
                f"visible devices={torch.cuda.device_count()}"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = requested_device
        backend = "gloo"

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable in this PyTorch build")
    dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        backend=backend,
    )


def destroy_distributed(context: DistributedContext) -> None:
    if context.is_distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(context: DistributedContext) -> None:
    if context.is_distributed:
        dist.barrier()


def broadcast_from_main(value, context: DistributedContext):
    """Broadcast a small Python object from rank 0 without affecting single GPU."""
    if not context.is_distributed:
        return value
    values = [value if context.is_main_process else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def rank_zero_print(context: DistributedContext, *args, **kwargs) -> None:
    if context.is_main_process:
        print(*args, **kwargs)


class Tee:
    """Mirror training output to the terminal and a timestamped log file.

    ``print`` commonly calls ``write`` twice (once for the message and once
    for the newline), so the timestamp is added at logical line boundaries
    instead of once per ``write`` call.  The minute-level format is compact
    and matches the experiment notes: ``[MM/DD/HH/MM]``.
    """

    def __init__(self, filepath: str):
        self.file = open(filepath, "a", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self._line_start = True

    def write(self, text: str):
        if not text:
            return

        chunks = []
        for char in text:
            if self._line_start and char not in "\r\n":
                chunks.append(f"[{datetime.now():%m/%d/%H/%M}] ")
                self._line_start = False
            chunks.append(char)
            if char in "\r\n":
                self._line_start = True

        rendered = "".join(chunks)
        self.file.write(rendered)
        self.stdout.write(rendered)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, *args):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()


def extract_dataset_name(data_dir: str) -> str:
    return os.path.basename(data_dir.rstrip("/"))


def seed_everything(seed: int, *, cuda_device: torch.device | None = None) -> None:
    """Seed all local RNGs without touching peer CUDA devices in DDP."""
    random.seed(seed)
    np.random.seed(seed)
    # ``torch.manual_seed`` also calls ``cuda.manual_seed_all``.  DDP ranks
    # must not initialize or seed peer GPUs, so seed the CPU generator
    # directly and then seed only this rank's CUDA device below.
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_available() and cuda_device is None:
        torch.cuda.manual_seed_all(seed)
    elif cuda_device is not None and cuda_device.type == "cuda":
        torch.cuda.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python in DataLoader workers from PyTorch's worker seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EpochRandomSampler(Sampler[int]):
    """Deterministic per-epoch permutation with a resumable batch offset.

    ``DataLoader(shuffle=True)`` consumes a whole new permutation when its
    iterator is created.  Restoring only its generator therefore cannot
    resume from the middle of an epoch: a checkpoint would silently skip the
    remainder of the old permutation.  This sampler derives each permutation
    from ``seed + epoch`` and lets the training loop reconstruct the exact
    starting offset from ``completed_steps``.  Prefetching workers can request
    ahead, but a restart still begins at the last *consumed* batch.
    """

    def __init__(self, data_source, seed: int):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0
        self.start_index = 0

    def set_position(self, epoch: int, start_index: int = 0) -> None:
        if epoch < 0 or start_index < 0:
            raise ValueError("epoch and start_index must be non-negative")
        self.epoch = int(epoch)
        self.start_index = int(start_index)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_source), generator=generator).tolist()
        return iter(indices[self.start_index:])

    def __len__(self) -> int:
        return max(0, len(self.data_source) - self.start_index)


class DistributedEpochRandomSampler(Sampler[int]):
    """Rank-sharded deterministic sampler with an exact local resume offset.

    ``DistributedSampler`` normally pads validation shards and does not expose
    a mid-epoch offset.  Padding would bias validation metrics, and dropping a
    partial epoch on resume would break the existing checkpoint guarantee.
    This sampler instead takes a shared ``seed + epoch`` permutation, assigns
    every ``world_size``-th item to each rank, and slices only that rank's
    already-sharded sequence at ``start_index``.

    For training, ``drop_last=True`` first removes the small tail needed to
    make shards equally sized.  ``DataLoader(drop_last=True)`` then removes a
    final incomplete *per-rank* batch, which is exactly equivalent to dropping
    the incomplete global batch.
    """

    def __init__(
        self,
        data_source,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
        shuffle: bool,
        drop_last: bool,
    ):
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank={rank} is outside [0, {num_replicas})")
        self.data_source = data_source
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.start_index = 0

    def set_position(self, epoch: int, start_index: int = 0) -> None:
        if epoch < 0 or start_index < 0:
            raise ValueError("epoch and start_index must be non-negative")
        self.epoch = int(epoch)
        self.start_index = int(start_index)

    def _local_sample_count(self) -> int:
        total = len(self.data_source)
        if self.drop_last:
            return total // self.num_replicas
        return max(0, (total - self.rank + self.num_replicas - 1) // self.num_replicas)

    def __iter__(self):
        total = len(self.data_source)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(total, generator=generator).tolist()
        else:
            indices = list(range(total))

        if self.drop_last:
            indices = indices[:(total // self.num_replicas) * self.num_replicas]
        rank_indices = indices[self.rank::self.num_replicas]
        return iter(rank_indices[self.start_index:])

    def __len__(self) -> int:
        return max(0, self._local_sample_count() - self.start_index)


def _build_split_loader(
    data_dir: str,
    split: str,
    token2id: dict,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
    generator: torch.Generator | None,
    align_name: str,
    seed: int | None = None,
    distributed: DistributedContext | None = None,
    pin_memory: bool | None = None,
):
    """Build a loader while failing fast on incomplete aligned/raw pairs."""
    split_dir = os.path.join(data_dir, split)
    aligned_src = os.path.join(split_dir, f"{split}_aligned_src.txt")
    aligned_tgt = os.path.join(split_dir, f"{split}_aligned_tgt.txt")
    raw_src = os.path.join(split_dir, f"src-{split}.txt")
    raw_tgt = os.path.join(split_dir, f"tgt-{split}.txt")

    aligned_exists = (os.path.exists(aligned_src), os.path.exists(aligned_tgt))
    raw_exists = (os.path.exists(raw_src), os.path.exists(raw_tgt))
    if aligned_exists[0] != aligned_exists[1]:
        raise FileNotFoundError(
            f"Incomplete pre-aligned {split} files: {aligned_src}, {aligned_tgt}"
        )
    if raw_exists[0] != raw_exists[1]:
        raise FileNotFoundError(
            f"Incomplete raw {split} files: {raw_src}, {raw_tgt}"
        )

    if all(aligned_exists):
        dataset = PreAlignedDataset(aligned_src, aligned_tgt, token2id)
        align_fn = identity_align_xs_to_zs
        source_kind = "pre-aligned"
    elif all(raw_exists):
        dataset = RetroDataset(raw_src, raw_tgt, token2id)
        align_fn = {
            "opt": opt_align_xs_to_zs,
            "naive": naive_align_xs_to_zs,
            "shifted": shifted_align_xs_to_zs,
        }[align_name]
        source_kind = "raw"
    else:
        raise FileNotFoundError(
            f"No usable {split} pair found in {split_dir}; expected "
            f"{aligned_src}/{aligned_tgt} or {raw_src}/{raw_tgt}"
        )

    if distributed is not None and distributed.is_distributed:
        if shuffle and seed is None:
            raise ValueError(
                "DDP training requires retro.seed so every rank uses the same "
                "per-epoch permutation"
            )
        sampler = DistributedEpochRandomSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            seed=0 if seed is None else seed,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    else:
        sampler = EpochRandomSampler(dataset, seed) if shuffle and seed is not None else None
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )
    return dataset, loader, align_fn, source_kind, sampler


def capture_rng_state(
    train_generator: torch.Generator | None = None,
    *,
    cuda_device: torch.device | None = None,
) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and cuda_device is None:
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif cuda_device is not None and cuda_device.type == "cuda":
        # In DDP each process owns one GPU.  Capturing every visible device
        # would initialize contexts for its peers and consume their memory.
        state["cuda_device_rng"] = torch.cuda.get_rng_state(cuda_device)
    if train_generator is not None:
        state["train_loader"] = train_generator.get_state()
    return state


def _as_cpu_rng_state(value: torch.Tensor) -> torch.Tensor:
    """Normalize a serialized RNG state for PyTorch restore APIs.

    Checkpoints are loaded with ``map_location=device``.  When ``device`` is
    CUDA, that remaps the CPU ByteTensors returned by the RNG APIs as well.
    Both CPU and CUDA RNG restore functions expect their state tensor on CPU.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"RNG state must be a tensor, got {type(value)!r}")
    return value.detach().to(device="cpu", dtype=torch.uint8)


def restore_rng_state(
    state: dict,
    train_generator: torch.Generator | None = None,
    *,
    cuda_device: torch.device | None = None,
) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_as_cpu_rng_state(state["torch"]))
    if torch.cuda.is_available() and cuda_device is None:
        if "cuda_device_rng" in state:
            raise ValueError("checkpoint has per-device CUDA RNG state but no CUDA device")
        if "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])
    elif cuda_device is not None and cuda_device.type == "cuda":
        if "cuda_device_rng" in state:
            torch.cuda.set_rng_state(
                _as_cpu_rng_state(state["cuda_device_rng"]), cuda_device,
            )
        elif "cuda" in state:
            # Old single-process checkpoints contain a list of all visible
            # device RNG states.  In DDP restore only this rank's device so a
            # process never creates a CUDA context on a peer GPU.
            cuda_states = state["cuda"]
            if isinstance(cuda_states, (list, tuple)):
                index = cuda_device.index
                if index is None:
                    index = torch.cuda.current_device()
                if index >= len(cuda_states):
                    raise ValueError(
                        "checkpoint CUDA RNG state has fewer devices than "
                        f"requested device index {index}"
                    )
                torch.cuda.set_rng_state(
                    _as_cpu_rng_state(cuda_states[index]), cuda_device,
                )
            else:
                torch.cuda.set_rng_state(
                    _as_cpu_rng_state(cuda_states), cuda_device,
                )
    if train_generator is not None and "train_loader" in state:
        train_generator.set_state(_as_cpu_rng_state(state["train_loader"]))


def log_metrics(writer, prefix: str, metrics: dict, step: int) -> None:
    if writer is None:
        return
    for key in ("loss", "u_tot", "u_ins", "u_del", "u_sub"):
        if key in metrics:
            writer.add_scalar(f"{prefix}/{key}", metrics[key], step)
    for key in ("t_mean", "kappa_mean", "rate_scale_mean", "rate_scale_max"):
        if key in metrics:
            writer.add_scalar(f"{prefix}/schedule/{key}", metrics[key], step)
    if "u_tot" in metrics and metrics["u_tot"] > 0.0:
        for key, name in (("u_ins", "insert"), ("u_sub", "substitute"),
                          ("u_del", "delete")):
            if key in metrics:
                writer.add_scalar(
                    f"{prefix}/lambda_fraction/{name}",
                    metrics[key] / metrics["u_tot"], step,
                )
    for key, name in (("lambda_total", "total"), ("lambda_ins", "insert"),
                      ("lambda_sub", "substitute"), ("lambda_del", "delete")):
        if key in metrics:
            writer.add_scalar(f"{prefix}/lambda/{name}", metrics[key], step)


def reduce_mean_metrics(metrics: dict, context: DistributedContext) -> dict:
    """Average scalar train diagnostics across DDP ranks for rank-0 logging."""
    if not context.is_distributed:
        return metrics
    keys = tuple(metrics)
    values = torch.tensor(
        [float(metrics[key]) for key in keys],
        device=context.device,
        dtype=torch.float64,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= context.world_size
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist())}


def gather_rng_states(
    train_generator: torch.Generator | None,
    context: DistributedContext,
) -> list[dict]:
    """Collect rank-local RNG states only at a checkpoint boundary."""
    local_state = capture_rng_state(train_generator, cuda_device=context.device)
    if not context.is_distributed:
        return [local_state]
    states: list[dict | None] = [None] * context.world_size
    dist.all_gather_object(states, local_state)
    return [state for state in states if state is not None]


def gradient_diagnostics(model) -> dict:
    """Return inexpensive gradient/parameter health diagnostics.

    The scan is intentionally separate from ``train_step`` and is called only
    at the configured monitoring interval.  This keeps the historical
    training path unchanged while making pilot failures (NaN/Inf or exploding
    gradients) explicit in both the terminal log and the JSONL trace.
    """
    grad_sq_sum = 0.0
    grad_max_abs = 0.0
    nonfinite_grad_values = 0
    gradient_tensors = 0
    nonfinite_parameter_values = 0
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            nonfinite_parameter_values += int(
                (~torch.isfinite(parameter)).sum().item()
            )
        if parameter.grad is None:
            continue
        gradient_tensors += 1
        grad = parameter.grad.detach()
        finite = torch.isfinite(grad)
        nonfinite_grad_values += int((~finite).sum().item())
        if finite.any():
            finite_grad = grad[finite]
            grad_max_abs = max(grad_max_abs, float(finite_grad.abs().max().item()))
            grad_sq_sum += float((finite_grad.float() ** 2).sum().item())
    return {
        "grad_norm": grad_sq_sum ** 0.5,
        "grad_max_abs": grad_max_abs,
        "gradient_tensors": gradient_tensors,
        "nonfinite_grad_values": nonfinite_grad_values,
        "nonfinite_parameter_values": nonfinite_parameter_values,
    }


def _metrics_are_finite(metrics: dict) -> bool:
    """Check scalar train/validation metrics before they enter a trace."""
    return all(
        isinstance(value, (int, float)) and np.isfinite(value)
        for value in metrics.values()
    )


def validation_due(
    completed_steps: int,
    validation_start_step: int,
    validation_interval: int,
) -> bool:
    """Return whether validation should run after this optimizer update."""
    if validation_interval <= 0 or validation_start_step < 0:
        return False
    return (
        completed_steps >= validation_start_step
        and completed_steps % validation_interval == 0
    )


def evaluate_model(
    model,
    loader,
    scheduler,
    align_fn,
    model_vocab: int,
    device,
    max_batches: int | None,
    cfg: dict,
    distributed: DistributedContext | None = None,
) -> dict:
    """Evaluate the same objective used by train_step on a validation prefix."""
    was_training = model.training
    model.eval()
    metric_names = (
        "loss", "u_tot", "u_ins", "u_del", "u_sub",
        "lambda_total", "lambda_ins", "lambda_del", "lambda_sub",
        "t_mean", "kappa_mean", "rate_scale_mean", "rate_scale_max",
    )
    totals = {key: 0.0 for key in metric_names}
    total_examples = 0
    with torch.no_grad():
        for batch_index, (x_0, x_1) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = prepare_batch(
                x_0, x_1, scheduler, align_fn,
                model_vocab_size=model_vocab,
                use_origin_mask=cfg.get("use_origin_mask", False),
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            metrics = evaluate_step(
                model, batch, scheduler,
                use_rate_reparam=cfg.get("use_rate_reparam", False),
                clamp_kappa=cfg.get("clamp_kappa", False),
                clamp_max=cfg.get("clamp_max", 50.0),
                time_input=cfg.get("time_input", "t"),
            )
            batch_examples = x_0.shape[0]
            total_examples += batch_examples
            for key in totals:
                totals[key] += metrics[key] * batch_examples
    if distributed is not None and distributed.is_distributed:
        values = torch.tensor(
            [totals[key] for key in metric_names] + [float(total_examples)],
            device=distributed.device,
            dtype=torch.float64,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        total_examples = int(values[-1].item())
        totals = {
            key: float(value)
            for key, value in zip(metric_names, values[:-1].cpu().tolist())
        }
    if was_training:
        model.train()
    if total_examples == 0:
        raise RuntimeError("Validation loader produced no examples")
    return {key: value / total_examples for key, value in totals.items()}


def prune_checkpoints(save_dir: str, keep: int) -> None:
    ckpts = sorted(
        glob.glob(os.path.join(save_dir, "checkpoint_step*.pt")),
        key=lambda p: int(re.search(r"step(\d+)", p).group(1)),
    )
    while len(ckpts) > keep:
        old = ckpts.pop(0)
        os.remove(old)
        print(f"Pruned old checkpoint: {os.path.basename(old)}")


def save_checkpoint(
    save_dir: str,
    completed_steps: int,
    model,
    optimizer,
    lr_scheduler,
    cfg,
    real_vocab_size,
    model_vocab,
    keep: int,
    *,
    train_generator: torch.Generator | None = None,
    train_position: dict | None = None,
    filename: str | None = None,
    best_val_loss: float | None = None,
    rng_state: dict | None = None,
    rng_state_by_rank: list[dict] | None = None,
    training_topology: dict | None = None,
) -> str:
    ckpt_name = filename or f"checkpoint_step{completed_steps}.pt"
    ckpt_path = os.path.join(save_dir, ckpt_name)
    unwrapped_model = model.module if isinstance(model, DistributedDataParallel) else model
    state = {
        # Save the unwrapped model so one-card sampling and a later DDP resume
        # both retain the historical checkpoint key format (no ``module.``).
        "model_state_dict": unwrapped_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict(),
        # ``step`` is retained for compatibility; both fields now mean the
        # number of completed optimizer updates, so resume never repeats one.
        "step": completed_steps,
        "completed_steps": completed_steps,
        "config": cfg,
        "real_vocab_size": real_vocab_size,
        "model_vocab": model_vocab,
        "rng_state": (
            capture_rng_state(train_generator)
            if rng_state is None else rng_state
        ),
    }
    if train_position is not None:
        state["train_position"] = dict(train_position)
    if best_val_loss is not None:
        state["best_val_loss"] = best_val_loss
    if rng_state_by_rank is not None:
        state["rng_state_by_rank"] = list(rng_state_by_rank)
    if training_topology is not None:
        state["training_topology"] = dict(training_topology)
    torch.save(state, ckpt_path)
    if filename is None:
        prune_checkpoints(save_dir, keep)
    return ckpt_path


def load_model_state(model, state_dict: dict) -> None:
    """Load historical single-GPU and accidental ``module.``-prefixed states."""
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)


def run_training(args, context: DistributedContext) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    cfg = config["retro"]
    device = context.device

    seed = cfg.get("seed")
    if seed is not None:
        seed = int(seed)
        seed_everything(seed + context.rank, cuda_device=device)
        rank_zero_print(
            context,
            f"Seed: {seed} (rank-local seeds: {seed}..{seed + context.world_size - 1})",
        )
    else:
        rank_zero_print(context, "Seed: not configured (legacy non-deterministic mode)")

    data_dir = cfg["data_dir"]
    dataset_name = extract_dataset_name(data_dir)
    vocab_path = os.path.join(data_dir, cfg["vocab_file"])
    token2id, model_vocab = load_vocab(vocab_path)
    real_vocab_size = model_vocab - 4
    per_rank_batch_size = int(cfg["batch_size"])
    if per_rank_batch_size <= 0:
        raise ValueError("retro.batch_size must be positive")
    effective_global_batch_size = per_rank_batch_size * context.world_size
    training_topology = {
        "world_size": context.world_size,
        "batch_size_per_rank": per_rank_batch_size,
        "effective_global_batch_size": effective_global_batch_size,
        "backend": context.backend or "single_process",
    }

    keep_checkpoints = (
        args.keep_checkpoints
        if args.keep_checkpoints is not None
        else cfg.get("keep_checkpoints", 10)
    )
    num_workers = int(cfg.get("num_workers", 2))
    tensorboard_cfg = cfg.get("tensorboard", {}) or {}
    validation_interval = int(tensorboard_cfg.get(
        "validation_interval", cfg.get("validation_interval", 0),
    ))
    validation_start_step = int(tensorboard_cfg.get(
        "validation_start_step", cfg.get("validation_start_step", 0),
    ))
    if validation_interval < 0 or validation_start_step < 0:
        raise ValueError(
            "validation_interval and validation_start_step must be non-negative"
        )
    need_validation = (
        validation_interval > 0
        and validation_start_step <= int(cfg["total_steps"])
    )
    train_generator = None
    val_generator = None
    if seed is not None:
        train_generator = torch.Generator()
        train_generator.manual_seed(seed + context.rank)
        val_generator = torch.Generator()
        val_generator.manual_seed(seed + context.world_size + context.rank)

    if context.is_main_process:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.save_dir:
            proposed_save_dir = os.path.join(args.save_dir, dataset_name, timestamp)
        elif args.checkpoint:
            proposed_save_dir = os.path.dirname(args.checkpoint)
        else:
            proposed_save_dir = os.path.join("checkpoints", dataset_name, timestamp)
    else:
        proposed_save_dir = None
    save_dir = broadcast_from_main(proposed_save_dir, context)
    if context.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
    distributed_barrier(context)

    log_path = os.path.join(save_dir, "train.log")
    log_context = Tee(log_path) if context.is_main_process else nullcontext()
    with log_context:
        config_dst = os.path.join(save_dir, "config.yaml")
        if context.is_main_process and not os.path.exists(config_dst):
            shutil.copy(args.config, config_dst)
        distributed_barrier(context)
        rank_zero_print(context, f"Config saved to {config_dst}")

        rank_zero_print(context, f"Checkpoint dir: {save_dir}")
        rank_zero_print(context, f"Dataset: {dataset_name}")
        rank_zero_print(context, f"Vocab: {real_vocab_size} real tokens, {model_vocab} model tokens")

        (
            train_dataset, train_loader, align_fn, train_source_kind,
            train_sampler,
        ) = (
            _build_split_loader(
                data_dir=data_dir,
                split="train",
                token2id=token2id,
                batch_size=per_rank_batch_size,
                num_workers=num_workers,
                shuffle=True,
                drop_last=True,
                generator=train_generator,
                align_name=cfg["align_fn"],
                seed=seed,
                distributed=context,
                pin_memory=device.type == "cuda",
            )
        )

        val_dataset = val_loader = val_align_fn = val_source_kind = val_sampler = None
        val_dir = os.path.join(data_dir, "val")
        val_paths = [
            os.path.join(val_dir, "val_aligned_src.txt"),
            os.path.join(val_dir, "val_aligned_tgt.txt"),
            os.path.join(val_dir, "src-val.txt"),
            os.path.join(val_dir, "tgt-val.txt"),
        ]
        if need_validation and any(os.path.exists(path) for path in val_paths):
            (
                val_dataset, val_loader, val_align_fn, val_source_kind,
                val_sampler,
            ) = (
                _build_split_loader(
                    data_dir=data_dir,
                    split="val",
                    token2id=token2id,
                    batch_size=int(cfg.get("val_batch_size", per_rank_batch_size)),
                    num_workers=num_workers,
                    shuffle=False,
                    drop_last=False,
                    generator=val_generator,
                    align_name=cfg["align_fn"],
                    seed=None,
                    distributed=context,
                    pin_memory=device.type == "cuda",
                )
            )

        base_model = EditFlowsTransformer(
            vocab_size=model_vocab,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            dim_feedforward=cfg["dim_feedforward"],
            max_seq_len=cfg["max_seq_len"],
            dropout=cfg["dropout"],
            attention_dropout=cfg["attention_dropout"],
            activation=cfg["activation"],
            pos_encoding_scale=cfg["pos_encoding_scale"],
            use_origin_mask=cfg.get("use_origin_mask", False),
        ).to(device)

        resume_checkpoint = None
        if args.checkpoint:
            resume_checkpoint = torch.load(
                args.checkpoint, map_location=device, weights_only=False,
            )
            load_model_state(base_model, resume_checkpoint["model_state_dict"])

        if context.is_distributed:
            if device.type == "cuda":
                model = DistributedDataParallel(
                    base_model,
                    device_ids=[context.local_rank],
                    output_device=context.local_rank,
                    broadcast_buffers=False,
                )
            else:
                model = DistributedDataParallel(base_model, broadcast_buffers=False)
        else:
            model = base_model

        # The scheduler is stepped immediately before each optimizer update.
        # Starting at lr=0 prevents an accidental unscheduled first update.
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.0, betas=(0.9, 0.998), eps=1e-8,
        )

        lr_scheduler = NoamScheduler(
            optimizer,
            d_model=cfg["hidden_dim"],
            warmup_steps=cfg["warmup_steps"],
            factor=cfg["learning_rate_factor"],
        )

        kappa_scheduler = CubicScheduler() if cfg["scheduler"] == "cubic" else LinearScheduler()

        rank_zero_print(context, f"Train source: {train_source_kind}")
        rank_zero_print(
            context,
            f"Train: {len(train_dataset):,} pairs, {len(train_loader):,} "
            "batches/epoch/rank",
        )
        if val_loader is not None:
            rank_zero_print(
                context,
                f"Validation: {len(val_dataset):,} pairs, "
                f"source={val_source_kind}, batches/rank={len(val_loader):,}"
            )
        else:
            rank_zero_print(context, "Validation: unavailable")
        rank_zero_print(context, f"DataLoader workers/rank: {num_workers}")
        if train_sampler is not None:
            sampler_name = (
                "deterministic rank-sharded permutation (resumable)"
                if context.is_distributed
                else "deterministic per-epoch permutation (resumable)"
            )
            rank_zero_print(context, f"Train sampler: {sampler_name}")
        else:
            rank_zero_print(context, "Train sampler: DataLoader default shuffle")
        rank_zero_print(
            context,
            f"Topology: world_size={context.world_size}, backend="
            f"{training_topology['backend']}, batch/rank={per_rank_batch_size}, "
            f"effective global batch={effective_global_batch_size}",
        )
        rank_zero_print(context, f"Rate reparam: {cfg.get('use_rate_reparam', False)}")
        rank_zero_print(context, f"Time input: {cfg.get('time_input', 't')}")
        rank_zero_print(
            context,
            f"Clamp kappa: {cfg.get('clamp_kappa', False)} "
            f"(max={cfg.get('clamp_max', 50.0)})",
        )
        rank_zero_print(context, f"Origin mask: {cfg.get('use_origin_mask', False)}")

        start_step = 0
        best_val_loss = float("inf")
        checkpoint_position = None
        if resume_checkpoint is not None:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            lr_scheduler.load_state_dict(resume_checkpoint.get("lr_scheduler_state", {}))
            start_step = int(
                resume_checkpoint.get("completed_steps", resume_checkpoint.get("step", 0))
            )
            best_val_loss = float(resume_checkpoint.get("best_val_loss", float("inf")))
            checkpoint_position = resume_checkpoint.get("train_position")

            saved_topology = resume_checkpoint.get("training_topology") or {}
            saved_global_batch = saved_topology.get("effective_global_batch_size")
            if (
                saved_global_batch is not None
                and int(saved_global_batch) != effective_global_batch_size
            ):
                raise ValueError(
                    "checkpoint effective global batch size "
                    f"({saved_global_batch}) differs from this run "
                    f"({effective_global_batch_size}); start a new run instead"
                )
            if (
                saved_topology.get("world_size") is not None
                and int(saved_topology["world_size"]) != context.world_size
            ):
                rank_zero_print(
                    context,
                    "WARNING: resuming with a different world_size; effective "
                    "batch matches, but this is not bitwise-identical continuation.",
                )

            rng_states = resume_checkpoint.get("rng_state_by_rank")
            if isinstance(rng_states, list) and len(rng_states) == context.world_size:
                restore_rng_state(
                    rng_states[context.rank], train_generator, cuda_device=device,
                )
                rank_zero_print(context, "Restored rank-local RNG/DataLoader states")
            elif context.world_size == 1 and isinstance(rng_states, list) and rng_states:
                restore_rng_state(rng_states[0], train_generator, cuda_device=device)
                rank_zero_print(
                    context,
                    "WARNING: resumed rank-0 RNG from a multi-rank checkpoint; "
                    "continuation is not bitwise identical.",
                )
            elif "rng_state" in resume_checkpoint:
                if context.is_distributed and context.rank != 0:
                    rank_zero_print(
                        context,
                        "WARNING: checkpoint has only single-process RNG state; "
                        "rank 1+ keep deterministic rank-local seeds.",
                    )
                else:
                    restore_rng_state(
                        resume_checkpoint["rng_state"],
                        train_generator,
                        cuda_device=device,
                    )
                    rank_zero_print(context, "Restored Python/NumPy/PyTorch/DataLoader RNG state")
            else:
                rank_zero_print(
                    context,
                    "WARNING: checkpoint has no RNG state; reproducibility is limited",
                )
            rank_zero_print(context, f"Resumed from step {start_step}")

        rank_zero_print(context, f"Keep: {keep_checkpoints} latest checkpoints")
        rank_zero_print(context, f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        rank_zero_print(context, f"Device: {device}")
        rank_zero_print(context, f"Total steps: {cfg['total_steps']}")
        rank_zero_print(context, f"Log: {log_path}")
        rank_zero_print(context, "=" * 55)

        total_steps = cfg["total_steps"]
        if start_step > total_steps:
            raise ValueError(
                f"checkpoint completed_steps={start_step} exceeds "
                f"configured total_steps={total_steps}"
            )

        writer = None
        if context.is_main_process and tensorboard_cfg.get("enabled", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError(
                    "TensorBoard is enabled but unavailable; install "
                    "tensorboard with `pip install tensorboard`"
                ) from exc
            tensorboard_dir = tensorboard_cfg.get("log_dir", "tensorboard")
            if not os.path.isabs(tensorboard_dir):
                tensorboard_dir = os.path.join(save_dir, tensorboard_dir)
            writer = SummaryWriter(
                log_dir=tensorboard_dir,
                flush_secs=int(tensorboard_cfg.get("flush_secs", 30)),
            )
            writer.add_text(
                "run/config",
                yaml.safe_dump(config, sort_keys=False),
                start_step,
            )
            rank_zero_print(context, f"TensorBoard: {tensorboard_dir}")

        log_interval = int(tensorboard_cfg.get(
            "log_interval", cfg.get("log_interval", 100),
        ))
        validation_batches = tensorboard_cfg.get(
            "validation_batches", cfg.get("validation_batches", 100),
        )
        if validation_batches is not None:
            validation_batches = int(validation_batches)
        checkpoint_interval = int(cfg.get("checkpoint_interval", 10000))
        save_best = bool(cfg.get("save_best_checkpoint", True))

        monitoring_cfg = cfg.get("monitoring", {}) or {}
        monitoring_enabled = bool(monitoring_cfg.get("enabled", False))
        monitor_interval = int(monitoring_cfg.get("interval", log_interval))
        if monitoring_enabled and monitor_interval <= 0:
            raise ValueError("monitoring.interval must be positive when enabled")
        monitor_file = None
        monitor_path = None
        if monitoring_enabled and context.is_main_process:
            monitor_filename = str(
                monitoring_cfg.get("jsonl", "training_monitor.jsonl")
            )
            monitor_path = (
                monitor_filename
                if os.path.isabs(monitor_filename)
                else os.path.join(save_dir, monitor_filename)
            )
            monitor_file = open(monitor_path, "a", buffering=1)
            rank_zero_print(context, f"Training monitor: {monitor_path}")

        train_batches_per_epoch = len(train_loader)
        if train_batches_per_epoch <= 0:
            raise RuntimeError(
                "Training loader has no complete batches; reduce batch_size or "
                "provide more training examples"
            )
        if (
            train_sampler is not None
            and checkpoint_position is not None
            and int(checkpoint_position.get("batches_per_epoch", train_batches_per_epoch))
            != train_batches_per_epoch
        ):
            raise ValueError(
                "checkpoint was created with a different effective batch size; "
                "start a new run instead of silently changing the data order"
            )
        train_epoch = start_step // train_batches_per_epoch
        train_batch_offset = start_step % train_batches_per_epoch
        if train_sampler is not None:
            train_sampler.set_position(
                train_epoch, train_batch_offset * per_rank_batch_size,
            )

        def checkpoint_now(completed_steps: int, *, filename: str | None = None) -> str:
            """Synchronously save one portable checkpoint from rank 0."""
            rng_states = gather_rng_states(train_generator, context)
            checkpoint_path = None
            if context.is_main_process:
                checkpoint_path = save_checkpoint(
                    save_dir, completed_steps, model, optimizer, lr_scheduler,
                    cfg, real_vocab_size, model_vocab, keep_checkpoints,
                    train_generator=train_generator,
                    train_position={
                        "epoch": train_epoch,
                        "batch_offset": train_batch_offset,
                        "batches_per_epoch": train_batches_per_epoch,
                    },
                    filename=filename,
                    best_val_loss=(
                        best_val_loss if best_val_loss < float("inf") else None
                    ),
                    rng_state=rng_states[0],
                    rng_state_by_rank=rng_states,
                    training_topology=training_topology,
                )
            checkpoint_path = broadcast_from_main(checkpoint_path, context)
            distributed_barrier(context)
            return checkpoint_path

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        monitor_started_at = time.perf_counter()
        monitor_window_started_at = monitor_started_at
        monitor_records = 0
        monitor_anomalies = 0
        model.train()
        train_iter = iter(train_loader)
        try:
            for step in range(start_step, total_steps):
                # Noam step n is the learning rate used by optimizer update n.
                current_lr = lr_scheduler.step()
                try:
                    x_0, x_1 = next(train_iter)
                except StopIteration:
                    if train_sampler is not None:
                        train_sampler.set_position(
                            train_epoch,
                            train_batch_offset * per_rank_batch_size,
                        )
                    train_iter = iter(train_loader)
                    x_0, x_1 = next(train_iter)

                batch = prepare_batch(
                    x_0, x_1, kappa_scheduler, align_fn,
                    model_vocab_size=model_vocab,
                    use_origin_mask=cfg.get("use_origin_mask", False),
                )
                batch = {
                    key: value.to(device, non_blocking=device.type == "cuda")
                    for key, value in batch.items()
                }

                metrics = train_step(
                    model, batch, kappa_scheduler, optimizer,
                    max_grad_norm=cfg["max_grad_norm"],
                    use_rate_reparam=cfg.get("use_rate_reparam", False),
                    clamp_kappa=cfg.get("clamp_kappa", False),
                    clamp_max=cfg.get("clamp_max", 50.0),
                    time_input=cfg.get("time_input", "t"),
                )
                completed_steps = step + 1
                if train_sampler is not None:
                    train_batch_offset += 1
                    if train_batch_offset >= train_batches_per_epoch:
                        train_epoch += 1
                        train_batch_offset = 0
                        train_sampler.set_position(train_epoch, 0)

                log_due = log_interval > 0 and completed_steps % log_interval == 0
                monitor_due = (
                    monitoring_enabled and completed_steps % monitor_interval == 0
                )
                reported_metrics = (
                    reduce_mean_metrics(metrics, context)
                    if log_due or monitor_due else metrics
                )
                if log_due and context.is_main_process:
                    log_metrics(writer, "train", reported_metrics, completed_steps)
                    if writer is not None:
                        writer.add_scalar(
                            "train/learning_rate", current_lr, completed_steps,
                        )
                    rank_zero_print(
                        context,
                        f"step {completed_steps:>8}/{total_steps} | "
                        f"loss: {reported_metrics['loss']:.4f} | "
                        f"lr: {current_lr:.2e} | "
                        f"u_tot: {reported_metrics['u_tot']:6.2f} | "
                        f"ins: {reported_metrics['u_ins']:6.2f} | "
                        f"del: {reported_metrics['u_del']:6.2f} | "
                        f"sub: {reported_metrics['u_sub']:6.2f}",
                    )

                if (
                    val_loader is not None
                    and validation_due(
                        completed_steps,
                        validation_start_step,
                        validation_interval,
                    )
                ):
                    rng_before_val = capture_rng_state(
                        train_generator, cuda_device=device,
                    )
                    val_metrics = evaluate_model(
                        model, val_loader, kappa_scheduler, val_align_fn,
                        model_vocab, device, validation_batches, cfg, context,
                    )
                    restore_rng_state(
                        rng_before_val, train_generator, cuda_device=device,
                    )
                    if context.is_main_process:
                        log_metrics(writer, "validation", val_metrics, completed_steps)
                        rank_zero_print(
                            context,
                            f"validation step {completed_steps:>8} | "
                            f"loss: {val_metrics['loss']:.4f} | "
                            f"u_tot: {val_metrics['u_tot']:6.2f} | "
                            f"ins: {val_metrics['u_ins']:6.2f} | "
                            f"del: {val_metrics['u_del']:6.2f} | "
                            f"sub: {val_metrics['u_sub']:6.2f}",
                        )
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        if save_best:
                            best_path = checkpoint_now(
                                completed_steps, filename="checkpoint_best.pt",
                            )
                            rank_zero_print(
                                context, f"--- Best checkpoint saved: {best_path}",
                            )

                if checkpoint_interval > 0 and completed_steps % checkpoint_interval == 0:
                    path = checkpoint_now(completed_steps)
                    rank_zero_print(context, f"--- Checkpoint saved: {path}")

                if monitor_due and context.is_main_process:
                    if device.type == "cuda":
                        # CUDA work is asynchronous; synchronize only at the
                        # monitoring boundary so the window speed is real.
                        torch.cuda.synchronize(device)
                    now = time.perf_counter()
                    window_seconds = now - monitor_window_started_at
                    total_seconds = now - monitor_started_at
                    grad_info = gradient_diagnostics(model)
                    record = {
                        "event": "train_metrics",
                        "step": completed_steps,
                        "learning_rate": float(current_lr),
                        "metrics": {
                            key: float(value) for key, value in reported_metrics.items()
                        },
                        "gradient": grad_info,
                        "timing": {
                            "window_seconds": window_seconds,
                            "total_seconds": total_seconds,
                            "steps_per_second": (
                                monitor_interval / window_seconds
                                if window_seconds > 0 else 0.0
                            ),
                            "milliseconds_per_step": (
                                1000.0 * window_seconds / monitor_interval
                                if window_seconds > 0 else 0.0
                            ),
                        },
                    }
                    if device.type == "cuda":
                        record["memory"] = {
                            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                        }
                    if not _metrics_are_finite(reported_metrics):
                        record["anomaly"] = "nonfinite_metrics"
                    if grad_info["nonfinite_grad_values"]:
                        record["anomaly"] = "nonfinite_gradients"
                    if grad_info["nonfinite_parameter_values"]:
                        record["anomaly"] = "nonfinite_parameters"
                    if "anomaly" in record:
                        monitor_anomalies += 1
                    if monitor_file is not None:
                        monitor_file.write(json.dumps(record, sort_keys=True) + "\n")
                    monitor_records += 1
                    monitor_window_started_at = now
                    if "anomaly" in record:
                        raise FloatingPointError(
                            f"Non-finite training state at step {completed_steps}: "
                            f"{record['anomaly']}"
                        )

                if writer is not None and completed_steps % int(
                    tensorboard_cfg.get("flush_interval", 500)
                ) == 0:
                    writer.flush()

            final_path = checkpoint_now(total_steps)
            if monitoring_enabled and context.is_main_process:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                finished_at = time.perf_counter()
                summary = {
                    "schema_version": 1,
                    "status": "completed",
                    "completed_steps": int(total_steps),
                    "elapsed_seconds": finished_at - monitor_started_at,
                    "steps_per_second": (
                        total_steps / (finished_at - monitor_started_at)
                        if finished_at > monitor_started_at else 0.0
                    ),
                    "monitor_interval": monitor_interval,
                    "monitor_records": monitor_records,
                    "monitor_anomalies": monitor_anomalies,
                    "real_vocab_size": int(real_vocab_size),
                    "model_vocab": int(model_vocab),
                    "model_parameters": int(sum(p.numel() for p in model.parameters())),
                    "checkpoint": final_path,
                    "training_topology": training_topology,
                }
                if device.type == "cuda":
                    summary["gpu"] = torch.cuda.get_device_name(device)
                    summary["peak_memory"] = {
                        "allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    }
                summary_path = os.path.join(save_dir, "training_summary.json")
                with open(summary_path, "w") as summary_file:
                    json.dump(summary, summary_file, indent=2, sort_keys=True)
                    summary_file.write("\n")
                rank_zero_print(context, f"Training summary: {summary_path}")
            rank_zero_print(context, "=" * 55)
            rank_zero_print(context, f"Training complete. Final model saved to {final_path}")
        finally:
            if writer is not None:
                writer.close()
            if monitor_file is not None:
                monitor_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Edit Flows for retrosynthesis")
    parser.add_argument("--config", type=str, default="configs/retro.yaml")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Single-process device. Under torchrun with --device cuda, each "
            "rank is automatically pinned to its LOCAL_RANK GPU."
        ),
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint (.pt path)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override save directory")
    parser.add_argument("--keep_checkpoints", type=int, default=None,
                        help="Max checkpoints to keep (default 10)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = initialize_distributed(args.device)
    try:
        run_training(args, context)
    finally:
        destroy_distributed(context)


if __name__ == "__main__":
    main()
