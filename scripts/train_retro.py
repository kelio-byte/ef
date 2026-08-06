#!/usr/bin/env python
"""Training script for Edit Flows on retrosynthesis data."""

import argparse
import glob
import os
import random
import re
import sys
import shutil
import yaml
import numpy as np
import torch
from datetime import datetime
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


class Tee:
    def __init__(self, filepath: str):
        self.file = open(filepath, "a", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, text: str):
        self.file.write(text)
        self.stdout.write(text)

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


def seed_everything(seed: int) -> None:
    """Seed all RNGs used by the training process."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    sampler = EpochRandomSampler(dataset, seed) if shuffle and seed is not None else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )
    return dataset, loader, align_fn, source_kind, sampler


def capture_rng_state(train_generator: torch.Generator | None = None) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if train_generator is not None:
        state["train_loader"] = train_generator.get_state()
    return state


def restore_rng_state(
    state: dict,
    train_generator: torch.Generator | None = None,
) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if train_generator is not None and "train_loader" in state:
        train_generator.set_state(state["train_loader"])


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


def evaluate_model(
    model,
    loader,
    scheduler,
    align_fn,
    model_vocab: int,
    device,
    max_batches: int | None,
    cfg: dict,
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
) -> str:
    ckpt_name = filename or f"checkpoint_step{completed_steps}.pt"
    ckpt_path = os.path.join(save_dir, ckpt_name)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict(),
        # ``step`` is retained for compatibility; both fields now mean the
        # number of completed optimizer updates, so resume never repeats one.
        "step": completed_steps,
        "completed_steps": completed_steps,
        "config": cfg,
        "real_vocab_size": real_vocab_size,
        "model_vocab": model_vocab,
        "rng_state": capture_rng_state(train_generator),
    }
    if train_position is not None:
        state["train_position"] = dict(train_position)
    if best_val_loss is not None:
        state["best_val_loss"] = best_val_loss
    torch.save(state, ckpt_path)
    if filename is None:
        prune_checkpoints(save_dir, keep)
    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Train Edit Flows for retrosynthesis")
    parser.add_argument("--config", type=str, default="configs/retro.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint (.pt path)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override save directory")
    parser.add_argument("--keep_checkpoints", type=int, default=None,
                        help="Max checkpoints to keep (default 10)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cfg = config["retro"]
    device = torch.device(args.device)

    seed = cfg.get("seed")
    if seed is not None:
        seed = int(seed)
        seed_everything(seed)
        print(f"Seed: {seed}")
    else:
        print("Seed: not configured (legacy non-deterministic mode)")

    data_dir = cfg["data_dir"]
    dataset_name = extract_dataset_name(data_dir)
    vocab_path = os.path.join(data_dir, cfg["vocab_file"])
    token2id, model_vocab = load_vocab(vocab_path)
    real_vocab_size = model_vocab - 4

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
    need_validation = validation_interval > 0
    train_generator = None
    val_generator = None
    if seed is not None:
        train_generator = torch.Generator()
        train_generator.manual_seed(seed)
        val_generator = torch.Generator()
        val_generator.manual_seed(seed + 1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.save_dir:
        save_dir = os.path.join(args.save_dir, dataset_name, timestamp)
    elif args.checkpoint:
        save_dir = os.path.dirname(args.checkpoint)
    else:
        save_dir = os.path.join("checkpoints", dataset_name, timestamp)

    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "train.log")
    with Tee(log_path):
        config_dst = os.path.join(save_dir, "config.yaml")
        if not os.path.exists(config_dst):
            shutil.copy(args.config, config_dst)
        print(f"Config saved to {config_dst}")

        print(f"Checkpoint dir: {save_dir}")
        print(f"Dataset: {dataset_name}")
        print(f"Vocab: {real_vocab_size} real tokens, {model_vocab} model tokens")

        (
            train_dataset, train_loader, align_fn, train_source_kind,
            train_sampler,
        ) = (
            _build_split_loader(
                data_dir=data_dir,
                split="train",
                token2id=token2id,
                batch_size=cfg["batch_size"],
                num_workers=num_workers,
                shuffle=True,
                drop_last=True,
                generator=train_generator,
                align_name=cfg["align_fn"],
                seed=seed,
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
                    batch_size=cfg.get("val_batch_size", cfg["batch_size"]),
                    num_workers=num_workers,
                    shuffle=False,
                    drop_last=False,
                    generator=val_generator,
                    align_name=cfg["align_fn"],
                    seed=None,
                )
            )

        model = EditFlowsTransformer(
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

        print(f"Train source: {train_source_kind}")
        print(f"Train: {len(train_dataset):,} pairs, {len(train_loader):,} batches/epoch")
        if val_loader is not None:
            print(
                f"Validation: {len(val_dataset):,} pairs, "
                f"source={val_source_kind}, batches={len(val_loader):,}"
            )
        else:
            print("Validation: unavailable")
        print(f"DataLoader workers: {num_workers}")
        if train_sampler is not None:
            print("Train sampler: deterministic per-epoch permutation (resumable)")
        else:
            print("Train sampler: DataLoader default shuffle")
        print(f"Rate reparam: {cfg.get('use_rate_reparam', False)}")
        print(f"Time input: {cfg.get('time_input', 't')}")
        print(f"Clamp kappa: {cfg.get('clamp_kappa', False)} (max={cfg.get('clamp_max', 50.0)})")
        print(f"Origin mask: {cfg.get('use_origin_mask', False)}")

        start_step = 0
        best_val_loss = float("inf")
        checkpoint_position = None
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            lr_scheduler.load_state_dict(ckpt.get("lr_scheduler_state", {}))
            start_step = int(ckpt.get("completed_steps", ckpt.get("step", 0)))
            best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
            checkpoint_position = ckpt.get("train_position")
            if "rng_state" in ckpt:
                restore_rng_state(ckpt["rng_state"], train_generator)
                print("Restored Python/NumPy/PyTorch/DataLoader RNG state")
            else:
                print("WARNING: checkpoint has no RNG state; reproducibility is limited")
            print(f"Resumed from step {start_step}")

        print(f"Keep: {keep_checkpoints} latest checkpoints")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Device: {device}")
        print(f"Total steps: {cfg['total_steps']}")
        print(f"Log: {log_path}")
        print("=" * 55)

        total_steps = cfg["total_steps"]
        if start_step > total_steps:
            raise ValueError(
                f"checkpoint completed_steps={start_step} exceeds "
                f"configured total_steps={total_steps}"
            )

        writer = None
        if tensorboard_cfg.get("enabled", False):
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
            print(f"TensorBoard: {tensorboard_dir}")

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
                train_epoch, train_batch_offset * int(cfg["batch_size"]),
            )

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
                            train_batch_offset * int(cfg["batch_size"]),
                        )
                    train_iter = iter(train_loader)
                    x_0, x_1 = next(train_iter)

                batch = prepare_batch(
                    x_0, x_1, kappa_scheduler, align_fn,
                    model_vocab_size=model_vocab,
                    use_origin_mask=cfg.get("use_origin_mask", False),
                )
                batch = {key: value.to(device) for key, value in batch.items()}

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

                if log_interval > 0 and completed_steps % log_interval == 0:
                    log_metrics(writer, "train", metrics, completed_steps)
                    if writer is not None:
                        writer.add_scalar(
                            "train/learning_rate", current_lr, completed_steps,
                        )
                    print(
                        f"step {completed_steps:>8}/{total_steps} | "
                        f"loss: {metrics['loss']:.4f} | "
                        f"lr: {current_lr:.2e} | "
                        f"u_tot: {metrics['u_tot']:6.2f} | "
                        f"ins: {metrics['u_ins']:6.2f} | "
                        f"del: {metrics['u_del']:6.2f} | "
                        f"sub: {metrics['u_sub']:6.2f}"
                    )

                if (
                    val_loader is not None
                    and validation_interval > 0
                    and completed_steps % validation_interval == 0
                ):
                    rng_before_val = capture_rng_state(train_generator)
                    val_metrics = evaluate_model(
                        model, val_loader, kappa_scheduler, val_align_fn,
                        model_vocab, device, validation_batches, cfg,
                    )
                    restore_rng_state(rng_before_val, train_generator)
                    log_metrics(writer, "validation", val_metrics, completed_steps)
                    print(
                        f"validation step {completed_steps:>8} | "
                        f"loss: {val_metrics['loss']:.4f} | "
                        f"u_tot: {val_metrics['u_tot']:6.2f} | "
                        f"ins: {val_metrics['u_ins']:6.2f} | "
                        f"del: {val_metrics['u_del']:6.2f} | "
                        f"sub: {val_metrics['u_sub']:6.2f}"
                    )
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        if save_best:
                            best_path = save_checkpoint(
                                save_dir, completed_steps, model, optimizer,
                                lr_scheduler, cfg, real_vocab_size, model_vocab,
                                keep_checkpoints, train_generator=train_generator,
                                train_position={
                                    "epoch": train_epoch,
                                    "batch_offset": train_batch_offset,
                                    "batches_per_epoch": train_batches_per_epoch,
                                },
                                filename="checkpoint_best.pt",
                                best_val_loss=best_val_loss,
                            )
                            print(f"--- Best checkpoint saved: {best_path}")

                if checkpoint_interval > 0 and completed_steps % checkpoint_interval == 0:
                    path = save_checkpoint(
                        save_dir, completed_steps, model, optimizer,
                        lr_scheduler, cfg, real_vocab_size, model_vocab,
                        keep_checkpoints, train_generator=train_generator,
                        train_position={
                            "epoch": train_epoch,
                            "batch_offset": train_batch_offset,
                            "batches_per_epoch": train_batches_per_epoch,
                        },
                        best_val_loss=(best_val_loss if best_val_loss < float("inf") else None),
                    )
                    print(f"--- Checkpoint saved: {path}")

                if writer is not None and completed_steps % int(
                    tensorboard_cfg.get("flush_interval", 500)
                ) == 0:
                    writer.flush()

            final_path = save_checkpoint(
                save_dir, total_steps, model, optimizer, lr_scheduler,
                cfg, real_vocab_size, model_vocab, keep_checkpoints,
                train_generator=train_generator,
                train_position={
                    "epoch": train_epoch,
                    "batch_offset": train_batch_offset,
                    "batches_per_epoch": train_batches_per_epoch,
                },
                best_val_loss=(best_val_loss if best_val_loss < float("inf") else None),
            )
            print("=" * 55)
            print(f"Training complete. Final model saved to {final_path}")
        finally:
            if writer is not None:
                writer.close()


if __name__ == "__main__":
    main()
