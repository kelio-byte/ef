"""Checkpoint I/O that preserves an existing file when serialization fails."""

import os
from pathlib import Path
import tempfile

import torch


def average_model_checkpoints(paths, weights=None, *, save_optimizer=False):
    """Average trusted local checkpoints after checking experimental identity.

    Configuration, vocabulary metadata, tensor keys/shapes/dtypes must match.
    Integer buffers are copied only when equal; they are not trainable weights.
    Optimizer state, if requested for legacy use, is copied from the first input
    and is NOT an optimizer trajectory corresponding to the averaged model.
    """
    if not paths:
        raise ValueError("No checkpoints selected")
    if weights is None:
        weights = torch.ones(len(paths), dtype=torch.float32)
    weights = torch.as_tensor(weights, dtype=torch.float32, device="cpu")
    if (weights.ndim != 1 or weights.numel() != len(paths)
            or not torch.isfinite(weights).all() or (weights < 0).any()
            or not torch.isfinite(weights.sum()) or weights.sum() <= 0):
        raise ValueError("Weights must be finite, non-negative, and have positive sum")
    weights = weights / weights.sum()
    output = None
    averaged = {}
    dtypes = {}
    for index, path in enumerate(paths):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        vocab = {key: checkpoint[key] for key in ("real_vocab_size", "model_vocab")
                 if key in checkpoint}
        if output is None:
            output = {"config": checkpoint["config"], "step": -1, **vocab}
            reference_vocab = vocab
            dtypes = {key: value.dtype for key, value in state.items()}
            for key, value in state.items():
                if value.is_complex():
                    raise ValueError(f"Unsupported complex checkpoint tensor: {key}")
                averaged[key] = (value.float() * weights[index]
                                 if value.is_floating_point() else value.clone())
            if save_optimizer:
                for key in ("optimizer_state_dict", "lr_scheduler_state"):
                    if key in checkpoint:
                        output[key] = checkpoint[key]
                        output["step"] = checkpoint.get("step", -1)
        else:
            if checkpoint["config"] != output["config"] or vocab != reference_vocab:
                raise ValueError(f"Checkpoint configuration/vocabulary mismatch: {path}")
            if state.keys() != averaged.keys():
                raise ValueError(f"Checkpoint parameter keys mismatch: {path}")
            for key, value in state.items():
                if value.shape != averaged[key].shape or value.dtype != dtypes[key]:
                    raise ValueError(f"Checkpoint tensor shape/dtype mismatch: {path}: {key}")
                if value.is_floating_point():
                    averaged[key].add_(value.float() * weights[index])
                elif not torch.equal(averaged[key], value):
                    raise ValueError(f"Non-floating checkpoint buffer mismatch: {path}: {key}")
        del checkpoint, state
    output["model_state_dict"] = {
        key: value.to(dtypes[key]) for key, value in averaged.items()
    }
    output["averaging"] = {"sources": [str(path) for path in paths],
                           "weights": weights.tolist()}
    return output


def atomic_torch_save(state, path: str | os.PathLike) -> None:
    """Serialize beside the destination, then atomically replace it.

    A same-directory temporary file keeps the rename on one filesystem.
    This protects readers and an existing best checkpoint from partial writes;
    it does not promise recovery from disk failure or concurrent writers.
    """
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
