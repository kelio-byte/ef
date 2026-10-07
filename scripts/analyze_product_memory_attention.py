#!/usr/bin/env python
"""Capture and visualize product-memory cross-attention during Euler sampling.

The product-memory module is queried after selected state-transformer layers.
This script records the *post-softmax, per-head* cross-attention probabilities
at a small set of actual Euler states.  It does not alter the normal sampling
path unless the model is inside its diagnostic capture context.

Example
-------
PYTHONPATH=. /root/miniconda3/envs/ef/bin/python \
  scripts/analyze_product_memory_attention.py \
  --checkpoint /path/to/checkpoint_step500000.pt \
  --products-file datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/evaluation_v2/dev_unique1000_aug20/src.txt \
  --targets-file datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/evaluation_v2/dev_unique1000_aug20/tgt.txt \
  --vocab-file datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/example.vocab.src \
  --output-dir experiments/product_memory_attention/results/pm500k_dev32 \
  --augmentation 20 --num-products 32 --num-trajectories 3 \
  --snapshot-steps 0,25,50,75,99 \
  --device cuda --seed 42

The output directory contains ``attention_records.pt`` (the reusable dense
per-head weights), CSV summaries, a Markdown report and PNG figures.  The
attention weights are descriptive diagnostics rather than causal attribution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from edit_flows.analysis.first_step import build_model_batch, tokenize_smiles
from edit_flows.core.scheduler import CubicScheduler, KappaScheduler, LinearScheduler
from edit_flows.data.dataset import load_vocab
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.sampling.euler import sample_euler
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


ATTENTION_DEFINITION = (
    "Per-head post-softmax cross-attention probabilities from a dynamic x_t "
    "query token to an immutable encoded product-memory token. Padding keys "
    "are masked before softmax and are excluded from exported tensors."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture product-memory cross-attention at selected Euler steps and "
            "write reusable records plus diagnostic figures."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--products-file", type=Path, required=True)
    parser.add_argument("--targets-file", type=Path)
    parser.add_argument("--vocab-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--augmentation",
        type=int,
        default=20,
        help="Rows per reaction in the input file; one selected augmentation is used per reaction.",
    )
    parser.add_argument("--augmentation-index", type=int, default=0)
    parser.add_argument("--start-reaction", type=int, default=0)
    parser.add_argument(
        "--num-products",
        type=int,
        default=32,
        help="Number of independent reactions to sample and aggregate.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=1,
        help="Independent stochastic Euler trajectories to retain per selected reaction.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Euler steps; defaults to n_sampling_steps in the checkpoint config.",
    )
    parser.add_argument(
        "--snapshot-steps",
        default="0,25,50,75,99",
        help="Comma-separated zero-based model-call indices to retain.",
    )
    parser.add_argument(
        "--case-snapshot-step",
        type=int,
        default=None,
        help="Snapshot used for case heatmaps; defaults to the middle requested step.",
    )
    parser.add_argument(
        "--case-trajectory-index",
        type=int,
        default=0,
        help="Which trajectory to show for each preselected case reaction.",
    )
    parser.add_argument("--max-case-plots", type=int, default=3)
    parser.add_argument("--key-position-bins", type=int, default=24)
    parser.add_argument("--scheduler", choices=("cubic", "linear"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
        help="torch float32 matmul precision; match the inference protocol when relevant.",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: Path, *, sha256: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    result: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if sha256:
        result["sha256"] = _sha256_file(resolved)
    return result


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout)
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.load(path, map_location="cpu")


def _checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint does not contain a dictionary config")
    if isinstance(config.get("retro"), dict):
        config = config["retro"]
    return config


def _build_scheduler(name: str) -> KappaScheduler:
    if name == "cubic":
        return CubicScheduler()
    if name == "linear":
        return LinearScheduler()
    raise ValueError(f"Unsupported scheduler: {name}")


def _parse_snapshot_steps(raw: str, n_steps: int) -> list[int]:
    pieces = [item.strip() for item in raw.split(",") if item.strip()]
    if not pieces:
        raise ValueError("snapshot-steps must contain at least one index")
    values: list[int] = []
    for piece in pieces:
        try:
            value = int(piece)
        except ValueError as exc:
            raise ValueError(f"Invalid snapshot step {piece!r}") from exc
        if value < 0 or value >= n_steps:
            raise ValueError(
                f"snapshot step {value} must be in [0, {n_steps - 1}]",
            )
        if value not in values:
            values.append(value)
    return sorted(values)


def _read_lines(path: Path) -> list[str]:
    with path.open() as handle:
        return [line.rstrip("\n") for line in handle]


def _select_rows(
    products: list[str],
    targets: list[str] | None,
    *,
    augmentation: int,
    augmentation_index: int,
    start_reaction: int,
    num_products: int,
) -> tuple[list[str], list[str] | None, list[dict[str, int]]]:
    if augmentation < 1:
        raise ValueError("augmentation must be positive")
    if not 0 <= augmentation_index < augmentation:
        raise ValueError(
            f"augmentation-index must be in [0, {augmentation - 1}]",
        )
    if start_reaction < 0:
        raise ValueError("start-reaction must be non-negative")
    if num_products < 1:
        raise ValueError("num-products must be positive")
    if len(products) % augmentation:
        raise ValueError(
            f"products file has {len(products)} rows, not divisible by "
            f"augmentation={augmentation}",
        )
    if targets is not None and len(targets) != len(products):
        raise ValueError(
            f"products/targets row mismatch: {len(products)} vs {len(targets)}",
        )

    available_reactions = len(products) // augmentation
    stop_reaction = start_reaction + num_products
    if stop_reaction > available_reactions:
        raise ValueError(
            f"requested reactions [{start_reaction}, {stop_reaction}) exceed "
            f"available reaction count {available_reactions}",
        )

    input_rows = [
        reaction_index * augmentation + augmentation_index
        for reaction_index in range(start_reaction, stop_reaction)
    ]
    selected_products = [products[index] for index in input_rows]
    selected_targets = (
        [targets[index] for index in input_rows] if targets is not None else None
    )
    row_metadata = [
        {
            "input_row": input_row,
            "reaction_index": start_reaction + offset,
            "augmentation_index": augmentation_index,
        }
        for offset, input_row in enumerate(input_rows)
    ]
    return selected_products, selected_targets, row_metadata


class AttentionCollector:
    """Retain selected per-head weights while an ordinary Euler rollout runs."""

    def __init__(
        self,
        *,
        snapshot_steps: Iterable[int],
        fusion_layers: Iterable[int],
        input_metadata: list[dict[str, Any]],
        products: list[str],
        targets: list[str] | None,
    ):
        self.snapshot_steps = set(snapshot_steps)
        self.fusion_layers = tuple(sorted(int(layer) for layer in fusion_layers))
        if not self.fusion_layers:
            raise ValueError("attention capture requires at least one fusion layer")
        self.input_metadata = input_metadata
        self.products = products
        self.targets = targets
        self.records: list[dict[str, Any]] = []
        self.forward_count = 0
        self._active_capture = False
        self._active_time: float | None = None

    def __call__(self, **payload: Any) -> None:
        layer_index = int(payload["layer_index"])
        if layer_index not in self.fusion_layers:
            raise RuntimeError(f"unexpected fusion layer {layer_index}")

        time_step = payload["time_step"]
        if not torch.allclose(time_step, time_step[:1]):
            raise RuntimeError("attention analysis expects one shared Euler time per batch")
        time_value = float(time_step[0, 0].detach().cpu().item())

        if layer_index == self.fusion_layers[0]:
            if self._active_time is not None:
                raise RuntimeError("received a new first fusion layer before closing a forward")
            self._active_capture = self.forward_count in self.snapshot_steps
            self._active_time = time_value
        elif self._active_time is None:
            raise RuntimeError("received a later fusion layer before the first one")
        elif not math.isclose(time_value, self._active_time, rel_tol=0.0, abs_tol=1e-7):
            raise RuntimeError("fusion layers disagreed about their model time")

        if self._active_capture:
            self._record_layer(
                layer_index=layer_index,
                attention_weights=payload["attention_weights"],
                state_tokens=payload["state_tokens"],
                state_padding_mask=payload["state_padding_mask"],
                product_memory_padding_mask=payload[
                    "product_memory_padding_mask"
                ],
                time_value=time_value,
            )

        if layer_index == self.fusion_layers[-1]:
            self.forward_count += 1
            self._active_capture = False
            self._active_time = None

    def _record_layer(
        self,
        *,
        layer_index: int,
        attention_weights: torch.Tensor,
        state_tokens: torch.Tensor,
        state_padding_mask: torch.Tensor,
        product_memory_padding_mask: torch.Tensor,
        time_value: float,
    ) -> None:
        if attention_weights.ndim != 4:
            raise RuntimeError(
                "expected attention weights [batch, heads, query, memory], got "
                f"{tuple(attention_weights.shape)}",
            )
        batch_size = attention_weights.shape[0]
        if batch_size != len(self.input_metadata):
            raise RuntimeError(
                f"capture batch has {batch_size} rows, expected "
                f"{len(self.input_metadata)}",
            )
        if state_tokens.shape != state_padding_mask.shape:
            raise RuntimeError("state tokens/padding mask shape mismatch")
        if product_memory_padding_mask.shape[0] != batch_size:
            raise RuntimeError("product-memory mask batch mismatch")

        for batch_index in range(batch_size):
            state_length = int((~state_padding_mask[batch_index]).sum().item())
            memory_length = int(
                (~product_memory_padding_mask[batch_index]).sum().item(),
            )
            if state_length < 1 or memory_length < 1:
                raise RuntimeError("attention capture encountered an empty sequence")
            weights = attention_weights[
                batch_index, :, :state_length, :memory_length
            ].detach().to(device="cpu", dtype=torch.float32).clone()
            if not torch.isfinite(weights).all():
                raise RuntimeError("attention capture contains non-finite values")
            expected = torch.ones_like(weights.sum(dim=-1))
            if not torch.allclose(
                weights.sum(dim=-1), expected, atol=2e-5, rtol=2e-5,
            ):
                raise RuntimeError("attention rows are not normalized")

            record = {
                **self.input_metadata[batch_index],
                "layer": layer_index,
                "snapshot_step": self.forward_count,
                "time": time_value,
                "product": self.products[batch_index],
                "target": (
                    self.targets[batch_index] if self.targets is not None else None
                ),
                "state_tokens": state_tokens[
                    batch_index, :state_length
                ].detach().cpu().clone(),
                "memory_tokens": torch.empty(0, dtype=torch.long),
                "attention_weights": weights,
            }
            # The immutable product sequence is not necessarily equal to a
            # dynamic state after the first edit.  Recover its token IDs from
            # the callback's memory mask only after retaining a separate copy
            # supplied at construction time below.
            self.records.append(record)

    def attach_memory_tokens(self, product_tokens: torch.Tensor) -> None:
        """Populate immutable memory token IDs after the batch has been built."""
        if product_tokens.shape[0] != len(self.input_metadata):
            raise ValueError("product token batch does not match capture batch")
        row_lookup = {
            (metadata["input_row"], metadata["trajectory_index"]): index
            for index, metadata in enumerate(self.input_metadata)
        }
        for record in self.records:
            batch_index = row_lookup[
                (record["input_row"], record["trajectory_index"])
            ]
            memory_length = record["attention_weights"].shape[-1]
            record["memory_tokens"] = product_tokens[
                batch_index, :memory_length
            ].detach().cpu().clone()


def _token_label(token_id: int, id2token: dict[int, str]) -> str:
    if token_id == BOS_TOKEN:
        return "BOS"
    if token_id == PAD_TOKEN:
        return "PAD"
    return id2token.get(token_id, f"<{token_id}>")


def _record_metrics(record: dict[str, Any]) -> dict[str, Any]:
    weights = record["attention_weights"].numpy()
    _, query_length, memory_length = weights.shape
    entropy = -np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0)), axis=-1)
    normalized_entropy = entropy / math.log(memory_length) if memory_length > 1 else entropy * 0.0
    diagonal_length = min(query_length, memory_length)
    diagonal = weights[:, np.arange(diagonal_length), np.arange(diagonal_length)]
    return {
        "input_row": record["input_row"],
        "reaction_index": record["reaction_index"],
        "augmentation_index": record["augmentation_index"],
        "trajectory_index": record["trajectory_index"],
        "layer": record["layer"],
        "snapshot_step": record["snapshot_step"],
        "time": record["time"],
        "num_heads": int(weights.shape[0]),
        "state_tokens": query_length,
        "memory_tokens": memory_length,
        "normalized_entropy": float(normalized_entropy.mean()),
        "top1_mass": float(weights.max(axis=-1).mean()),
        "memory_bos_mass": float(weights[:, :, 0].mean()),
        "same_index_mass": float(diagonal.mean()),
    }


def _head_metrics(record: dict[str, Any]) -> list[dict[str, Any]]:
    weights = record["attention_weights"].numpy()
    _, query_length, memory_length = weights.shape
    entropy = -np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0)), axis=-1)
    normalized_entropy = entropy / math.log(memory_length) if memory_length > 1 else entropy * 0.0
    diagonal_length = min(query_length, memory_length)
    diagonal = weights[:, np.arange(diagonal_length), np.arange(diagonal_length)]
    result = []
    for head in range(weights.shape[0]):
        result.append({
            "input_row": record["input_row"],
            "trajectory_index": record["trajectory_index"],
            "layer": record["layer"],
            "snapshot_step": record["snapshot_step"],
            "time": record["time"],
            "head": head,
            "normalized_entropy": float(normalized_entropy[head].mean()),
            "top1_mass": float(weights[head].max(axis=-1).mean()),
            "memory_bos_mass": float(weights[head, :, 0].mean()),
            "same_index_mass": float(diagonal[head].mean()),
        })
    return result


def _mean_std_sem(values: list[float]) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    sem = std / math.sqrt(len(values)) if values else float("nan")
    return mean, std, sem


def _aggregate_rows(
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
    *,
    cluster_field: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate metrics, optionally using a clustered experimental unit.

    Multiple stochastic Euler trajectories of the same product are repeated
    measurements, rather than independent reactions.  When ``cluster_field``
    is supplied, metrics are therefore first averaged within that field and
    the reported standard deviation/SEM is computed across clusters.
    """
    metrics = ("normalized_entropy", "top1_mass", "memory_bos_mass", "same_index_mass")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    aggregated: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        output = {field: value for field, value in zip(group_fields, key)}
        output["raw_n"] = len(group)
        clusters: dict[Any, list[dict[str, Any]]] | None = None
        if cluster_field is not None:
            clusters = defaultdict(list)
            for row in group:
                clusters[row[cluster_field]].append(row)
            output["n"] = len(clusters)
        else:
            output["n"] = len(group)
        output["time"] = float(np.mean([row["time"] for row in group]))
        for metric in metrics:
            values = (
                [
                    float(np.mean([float(row[metric]) for row in cluster_rows]))
                    for cluster_rows in clusters.values()
                ]
                if clusters is not None
                else [float(row[metric]) for row in group]
            )
            mean, std, sem = _mean_std_sem(values)
            output[f"{metric}_mean"] = mean
            output[f"{metric}_std"] = std
            output[f"{metric}_sem"] = sem
        aggregated.append(output)
    return aggregated


def _key_profile(weights: torch.Tensor, bins: int) -> np.ndarray:
    mean_by_key = weights.numpy().mean(axis=(0, 1))
    memory_length = len(mean_by_key)
    profile = np.zeros(bins, dtype=np.float64)
    for key_index, mass in enumerate(mean_by_key):
        bin_index = min(bins - 1, int(key_index * bins / memory_length))
        profile[bin_index] += mass
    return profile


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_global_dynamics(
    output_dir: Path,
    aggregate: list[dict[str, Any]],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("normalized_entropy", "Mean normalized entropy", "0 = focused, 1 = uniform"),
        ("top1_mass", "Mean top-1 attention mass", "largest memory weight"),
        ("memory_bos_mass", "Mean memory-BOS mass", "attention to BOS key"),
    )
    layers = sorted({int(row["layer"]) for row in aggregate})
    figure, axes = plt.subplots(1, len(metrics), figsize=(16, 4.5), sharex=True)
    for axis, (metric, title, ylabel) in zip(axes, metrics):
        for layer in layers:
            rows = [row for row in aggregate if int(row["layer"]) == layer]
            rows.sort(key=lambda row: int(row["snapshot_step"]))
            times = np.array([row["time"] for row in rows])
            values = np.array([row[f"{metric}_mean"] for row in rows])
            sems = np.array([row[f"{metric}_sem"] for row in rows])
            axis.plot(times, values, marker="o", label=f"fusion after L{layer}")
            axis.fill_between(times, values - sems, values + sems, alpha=0.18)
        axis.set_title(title)
        axis.set_xlabel("Euler time t")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("Product-memory attention dynamics across Euler sampling", y=1.02)
    figure.tight_layout()
    filename = "global_attention_dynamics.png"
    figure.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return filename


def _plot_key_profiles(
    output_dir: Path,
    profiles: dict[tuple[int, int], np.ndarray],
    time_by_step: dict[int, float],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted({layer for layer, _ in profiles})
    steps = sorted({step for _, step in profiles})
    color_max = max(float(profile.max()) for profile in profiles.values())
    figure, axes = plt.subplots(
        len(layers), len(steps),
        figsize=(max(12, 2.75 * len(steps)), max(4.5, 2.0 * len(layers))),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    image = None
    for row_index, layer in enumerate(layers):
        for column_index, step in enumerate(steps):
            axis = axes[row_index, column_index]
            profile = profiles[(layer, step)]
            image = axis.imshow(
                profile[np.newaxis, :],
                aspect="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=color_max,
            )
            axis.set_yticks([])
            axis.set_title(f"L{layer}, step {step}\nt={time_by_step[step]:.3f}")
    assert image is not None
    figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    figure.suptitle("Where attention mass lands in the immutable product memory", y=0.99)
    figure.supxlabel("normalized memory position bin", y=0.06)
    figure.supylabel("attention mass", x=0.02)
    # ``tight_layout`` cannot account for the shared colorbar.  Reserve its
    # space explicitly so that the exported figure is warning-free.
    figure.subplots_adjust(
        left=0.07,
        right=0.90,
        bottom=0.20,
        top=0.80,
        wspace=0.25,
        hspace=0.55,
    )
    filename = "global_memory_position_mass.png"
    figure.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return filename


def _plot_head_summary(
    output_dir: Path,
    head_aggregate: list[dict[str, Any]],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted({int(row["layer"]) for row in head_aggregate})
    steps = sorted({int(row["snapshot_step"]) for row in head_aggregate})
    num_heads = max(int(row["head"]) for row in head_aggregate) + 1
    metrics = ("normalized_entropy", "top1_mass")
    figure, axes = plt.subplots(
        len(layers), len(metrics),
        figsize=(10, max(3.0, 2.5 * len(layers))),
        squeeze=False,
    )
    for layer_index, layer in enumerate(layers):
        layer_rows = [row for row in head_aggregate if int(row["layer"]) == layer]
        for metric_index, metric in enumerate(metrics):
            matrix = np.full((num_heads, len(steps)), np.nan)
            for row in layer_rows:
                matrix[int(row["head"]), steps.index(int(row["snapshot_step"]))] = row[
                    f"{metric}_mean"
                ]
            axis = axes[layer_index, metric_index]
            image = axis.imshow(matrix, aspect="auto", cmap="magma")
            axis.set_title(f"Fusion after L{layer}: {metric.replace('_', ' ')}")
            axis.set_xlabel("snapshot step")
            axis.set_xticks(range(len(steps)), steps)
            axis.set_ylabel("head")
            axis.set_yticks(range(num_heads))
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    filename = "head_attention_summary.png"
    figure.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return filename


def _plot_case_heatmaps(
    output_dir: Path,
    records: list[dict[str, Any]],
    *,
    id2token: dict[int, str],
    snapshot_step: int,
    trajectory_index: int,
    max_case_plots: int,
) -> list[str]:
    if max_case_plots <= 0:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted({int(record["layer"]) for record in records})
    input_rows = sorted({
        int(record["input_row"])
        for record in records
        if int(record["trajectory_index"]) == trajectory_index
    })[:max_case_plots]
    filenames: list[str] = []
    for input_row in input_rows:
        selected = {
            int(record["layer"]): record
            for record in records
            if int(record["input_row"]) == input_row
            and int(record["trajectory_index"]) == trajectory_index
            and int(record["snapshot_step"]) == snapshot_step
        }
        if set(selected) != set(layers):
            continue
        max_memory = max(record["attention_weights"].shape[-1] for record in selected.values())
        max_state = max(record["attention_weights"].shape[-2] for record in selected.values())
        figure, axes = plt.subplots(
            len(layers), 1,
            figsize=(max(11, 0.32 * max_memory + 4), max(4, 0.25 * max_state * len(layers) + 2)),
            squeeze=False,
        )
        image = None
        for axis, layer in zip(axes[:, 0], layers):
            record = selected[layer]
            weights = record["attention_weights"].numpy().mean(axis=0)
            image = axis.imshow(weights, aspect="auto", interpolation="nearest", cmap="magma")
            state_labels = [
                f"{index}:{_token_label(int(token), id2token)}"
                for index, token in enumerate(record["state_tokens"].tolist())
            ]
            memory_labels = [
                f"{index}:{_token_label(int(token), id2token)}"
                for index, token in enumerate(record["memory_tokens"].tolist())
            ]
            axis.set_xticks(range(len(memory_labels)), memory_labels, rotation=90, fontsize=6)
            axis.set_yticks(range(len(state_labels)), state_labels, fontsize=6)
            axis.set_xlabel("immutable product-memory key")
            axis.set_ylabel("dynamic x_t query")
            axis.set_title(
                f"input row {input_row}, fusion after L{layer}, "
                f"trajectory {trajectory_index}, step {snapshot_step}, "
                f"t={record['time']:.3f} (mean over heads)",
            )
        assert image is not None
        figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02, label="attention probability")
        # The colorbar occupies a separate axes, which is incompatible with
        # ``tight_layout``.  Explicit margins keep labels and the colorbar
        # visible without the warning.
        figure.subplots_adjust(
            left=0.11,
            right=0.91,
            bottom=0.18,
            top=0.93,
            hspace=0.40,
        )
        filename = (
            f"case_input{input_row:06d}_trajectory{trajectory_index:02d}_"
            f"step{snapshot_step:03d}.png"
        )
        figure.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)
        filenames.append(filename)
    return filenames


def _write_report(
    output_dir: Path,
    *,
    metadata: dict[str, Any],
    aggregate: list[dict[str, Any]],
    figures: list[str],
) -> None:
    lines = [
        "# Product-memory 交叉注意力分析",
        "",
        "## 采集对象",
        "",
        "- 导出的是逐 head、softmax 后的交叉注意力概率；padding memory key "
        "在 softmax 前被遮蔽，并已从记录中移除。",
        f"- Checkpoint：`{metadata['checkpoint']['path']}`",
        f"- 反应数：{metadata['sampling']['num_reactions']}；每个反应的随机轨迹数："
        f"{metadata['sampling']['num_trajectories']}；增强序号 "
        f"{metadata['sampling']['augmentation_index']}",
        f"- Euler：{metadata['sampling']['scheduler']}，"
        f"{metadata['sampling']['n_steps']} 步，seed "
        f"{metadata['sampling']['seed']}",
        f"- 融合层：{metadata['model']['fusion_after_layers']}",
        "",
        "## 汇总注意力统计",
        "",
        "每条轨迹先对有效动态 query token 和 head 取平均；同一反应的多条随机"
        "轨迹再平均，最后跨反应计算均值和 SEM。因此表中的 `n` 是独立反应数，"
        "`raw_n` 是保留的轨迹记录数。 "
        "`same-index mass` 只是序列位置参照；编辑改变长度后，它不是化学对齐指标。",
        "",
        "| 融合后层 | Euler 步 | t | 归一化熵 | 最大权重 | Memory-BOS 权重 | 同索引权重 | 反应 n | 轨迹 raw_n |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            "| L{layer} | {step} | {time:.3f} | {entropy:.4f} ± {entropy_sem:.4f} "
            "| {top1:.4f} ± {top1_sem:.4f} | {bos:.4f} ± {bos_sem:.4f} "
            "| {diagonal:.4f} ± {diagonal_sem:.4f} | {n} | {raw_n} |".format(
                layer=int(row["layer"]),
                step=int(row["snapshot_step"]),
                time=float(row["time"]),
                entropy=float(row["normalized_entropy_mean"]),
                entropy_sem=float(row["normalized_entropy_sem"]),
                top1=float(row["top1_mass_mean"]),
                top1_sem=float(row["top1_mass_sem"]),
                bos=float(row["memory_bos_mass_mean"]),
                bos_sem=float(row["memory_bos_mass_sem"]),
                diagonal=float(row["same_index_mass_mean"]),
                diagonal_sem=float(row["same_index_mass_sem"]),
                n=int(row["n"]),
                raw_n=int(row["raw_n"]),
            )
        )
    lines.extend([
        "",
        "## 图",
        "",
    ])
    lines.extend(f"- [{figure}]({figure})" for figure in figures)
    lines.extend([
        "",
        "## 解读边界",
        "",
        "这些权重描述模型计算隐藏状态时读取了哪些 product-memory 位置，不能单独证明"
        "某个位置导致了某次编辑。因果验证应在相同产品、时间和随机种子下遮蔽或替换"
        "选定 memory 位置，再测量编辑 rate 和 Top-K 行为的变化。",
        "",
    ])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _build_model(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    *,
    vocab_size: int,
    device: torch.device,
) -> EditFlowsTransformer:
    if not bool(config.get("use_product_memory", False)):
        raise ValueError("checkpoint config does not enable product memory")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict) or not any(
        key.startswith("product_memory_fusion_layers.") for key in state_dict
    ):
        raise ValueError("checkpoint does not contain product-memory fusion weights")
    required = (
        "hidden_dim", "num_layers", "num_heads", "dim_feedforward",
        "max_seq_len", "dropout",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"checkpoint config is missing model fields: {missing}")
    model = EditFlowsTransformer(
        vocab_size=vocab_size,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        dim_feedforward=int(config["dim_feedforward"]),
        max_seq_len=int(config["max_seq_len"]),
        dropout=float(config["dropout"]),
        attention_dropout=float(config.get("attention_dropout", config["dropout"])),
        activation=str(config.get("activation", "relu")),
        pos_encoding_scale=bool(config.get("pos_encoding_scale", True)),
        use_origin_mask=bool(config.get("use_origin_mask", False)),
        use_product_memory=True,
        product_memory_encoder_layers=int(config.get("product_memory_encoder_layers", 0)),
        product_memory_fusion_after_layers=config.get("product_memory_fusion_after_layers"),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    args = _parse_args()
    for path in (args.checkpoint, args.products_file, args.vocab_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.targets_file is not None and not args.targets_file.is_file():
        raise FileNotFoundError(args.targets_file)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}; choose a new directory",
        )
    if args.key_position_bins < 2:
        raise ValueError("key-position-bins must be at least 2")
    if args.max_case_plots < 0:
        raise ValueError("max-case-plots must be non-negative")
    if args.num_trajectories < 1:
        raise ValueError("num-trajectories must be positive")
    if not 0 <= args.case_trajectory_index < args.num_trajectories:
        raise ValueError(
            "case-trajectory-index must be in "
            f"[0, {args.num_trajectories - 1}]",
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_float32_matmul_precision(args.matmul_precision)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = _load_checkpoint(args.checkpoint)
    config = _checkpoint_config(checkpoint)
    token2id, vocab_size = load_vocab(str(args.vocab_file))
    checkpoint_vocab_size = int(checkpoint.get("model_vocab", vocab_size))
    if checkpoint_vocab_size != vocab_size:
        raise ValueError(
            f"checkpoint vocabulary size {checkpoint_vocab_size} does not match "
            f"vocabulary size {vocab_size}",
        )
    model = _build_model(
        checkpoint, config, vocab_size=vocab_size, device=device,
    )
    del checkpoint

    n_steps = int(args.n_steps or config.get("n_sampling_steps", 100))
    if n_steps < 1:
        raise ValueError("n-steps must be positive")
    snapshot_steps = _parse_snapshot_steps(args.snapshot_steps, n_steps)
    case_snapshot_step = (
        args.case_snapshot_step
        if args.case_snapshot_step is not None
        else snapshot_steps[len(snapshot_steps) // 2]
    )
    if case_snapshot_step not in snapshot_steps:
        raise ValueError("case-snapshot-step must be one of snapshot-steps")
    scheduler_name = args.scheduler or str(
        config.get("sample_scheduler", config.get("scheduler", "cubic")),
    )
    scheduler = _build_scheduler(scheduler_name)
    train_scheduler = _build_scheduler(str(config.get("scheduler", scheduler_name)))

    products = _read_lines(args.products_file)
    targets = _read_lines(args.targets_file) if args.targets_file is not None else None
    selected_products, selected_targets, selected_row_metadata = _select_rows(
        products,
        targets,
        augmentation=args.augmentation,
        augmentation_index=args.augmentation_index,
        start_reaction=args.start_reaction,
        num_products=args.num_products,
    )
    input_metadata = [
        {**metadata, "trajectory_index": trajectory_index}
        for metadata, product in zip(selected_row_metadata, selected_products)
        for trajectory_index in range(args.num_trajectories)
    ]
    trajectory_products = [
        product for product in selected_products for _ in range(args.num_trajectories)
    ]
    trajectory_targets = (
        [target for target in selected_targets for _ in range(args.num_trajectories)]
        if selected_targets is not None else None
    )
    product_ids = [tokenize_smiles(product, token2id) for product in trajectory_products]
    if any(not ids for ids in product_ids):
        bad_rows = [
            item["input_row"] for item, ids in zip(input_metadata, product_ids) if not ids
        ]
        raise ValueError(f"empty tokenized products at input rows {bad_rows[:10]}")
    x_0, _ = build_model_batch(product_ids, product_ids, pad_token=PAD_TOKEN)
    if x_0.shape[1] > model.max_seq_len:
        raise ValueError(
            f"selected product batch needs {x_0.shape[1]} tokens including BOS, "
            f"but model max_seq_len is {model.max_seq_len}",
        )
    x_0 = x_0.to(device)

    collector = AttentionCollector(
        snapshot_steps=snapshot_steps,
        fusion_layers=model.product_memory_fusion_after_layers,
        input_metadata=input_metadata,
        products=trajectory_products,
        targets=trajectory_targets,
    )
    with model.capture_product_memory_attention(collector):
        final_states, _ = sample_euler(
            model,
            x_0,
            scheduler,
            n_steps=n_steps,
            max_seq_len=model.max_seq_len,
            use_rate_reparam=bool(config.get("use_rate_reparam", False)),
            clamp_kappa=bool(config.get("clamp_kappa", False)),
            clamp_max=float(config.get("clamp_max", 50.0)),
            time_input=str(config.get("time_input", "t")),
            train_scheduler=train_scheduler,
            use_origin_mask=bool(config.get("use_origin_mask", False)),
        )
    collector.attach_memory_tokens(x_0.detach().cpu())

    expected_records = (
        len(input_metadata) * len(snapshot_steps) * len(model.product_memory_fusion_after_layers)
    )
    if len(collector.records) != expected_records:
        captured_steps = sorted({record["snapshot_step"] for record in collector.records})
        raise RuntimeError(
            f"captured {len(collector.records)} records, expected {expected_records}; "
            f"seen snapshot steps: {captured_steps}; model calls: {collector.forward_count}",
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    id2token = {token_id: token for token, token_id in token2id.items()}
    summaries = [_record_metrics(record) for record in collector.records]
    head_summaries = [
        head_row for record in collector.records for head_row in _head_metrics(record)
    ]
    aggregate = _aggregate_rows(
        summaries,
        ("layer", "snapshot_step"),
        cluster_field="input_row",
    )
    head_aggregate = _aggregate_rows(
        head_summaries,
        ("layer", "snapshot_step", "head"),
        cluster_field="input_row",
    )
    _write_csv(args.output_dir / "attention_summary.csv", summaries)
    _write_csv(args.output_dir / "attention_aggregate.csv", aggregate)
    _write_csv(args.output_dir / "attention_head_aggregate.csv", head_aggregate)

    profiles: dict[tuple[int, int], np.ndarray] = {}
    profile_counts: dict[tuple[int, int], int] = defaultdict(int)
    for record in collector.records:
        key = (int(record["layer"]), int(record["snapshot_step"]))
        profile = _key_profile(record["attention_weights"], args.key_position_bins)
        if key not in profiles:
            profiles[key] = np.zeros_like(profile)
        profiles[key] += profile
        profile_counts[key] += 1
    for key in profiles:
        profiles[key] /= profile_counts[key]
    time_by_step = {
        int(row["snapshot_step"]): float(row["time"])
        for row in aggregate
    }

    figures = [
        _plot_global_dynamics(args.output_dir, aggregate),
        _plot_key_profiles(args.output_dir, profiles, time_by_step),
        _plot_head_summary(args.output_dir, head_aggregate),
    ]
    figures.extend(_plot_case_heatmaps(
        args.output_dir,
        collector.records,
        id2token=id2token,
        snapshot_step=case_snapshot_step,
        trajectory_index=args.case_trajectory_index,
        max_case_plots=args.max_case_plots,
    ))

    final_state_tokens = []
    for metadata, state in zip(input_metadata, final_states.cpu()):
        valid = state[state != PAD_TOKEN].tolist()
        final_state_tokens.append({
            **metadata,
            "final_state": " ".join(_token_label(int(token), id2token) for token in valid),
        })
    metadata: dict[str, Any] = {
        "format": "product_memory_attention_v1",
        "attention_definition": ATTENTION_DEFINITION,
        "checkpoint": _file_metadata(args.checkpoint, sha256=True),
        "inputs": {
            "products": _file_metadata(args.products_file, sha256=True),
            "targets": (
                _file_metadata(args.targets_file, sha256=True)
                if args.targets_file is not None else None
            ),
            "vocab": _file_metadata(args.vocab_file, sha256=True),
            "selected_rows": selected_row_metadata,
            "capture_rows": input_metadata,
        },
        "model": {
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "fusion_after_layers": list(model.product_memory_fusion_after_layers),
            "product_memory_encoder_layers": int(
                config["product_memory_encoder_layers"],
            ),
            "num_heads": int(config["num_heads"]),
        },
        "sampling": {
            "scheduler": scheduler_name,
            "train_scheduler": str(config.get("scheduler", scheduler_name)),
            "time_input": str(config.get("time_input", "t")),
            "n_steps": n_steps,
            "snapshot_steps": snapshot_steps,
            "snapshot_times": time_by_step,
            "seed": args.seed,
            "num_reactions": len(selected_row_metadata),
            "num_trajectories": args.num_trajectories,
            "captured_trajectories": len(input_metadata),
            "augmentation": args.augmentation,
            "augmentation_index": args.augmentation_index,
            "start_reaction": args.start_reaction,
            "matmul_precision": args.matmul_precision,
        },
        "capture": {
            "records": len(collector.records),
            "model_forward_calls": collector.forward_count,
            "case_snapshot_step": case_snapshot_step,
            "case_trajectory_index": args.case_trajectory_index,
            "key_position_bins": args.key_position_bins,
            "summary_unit": "reaction (trajectory-averaged)",
        },
        "artifacts": {
            "records": "attention_records.pt",
            "per_record_summary": "attention_summary.csv",
            "aggregate_summary": "attention_aggregate.csv",
            "head_summary": "attention_head_aggregate.csv",
            "report": "report.md",
            "figures": figures,
        },
        "final_states": final_state_tokens,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "device": str(device),
            "git": _git_state(),
        },
    }
    torch.save({
        "format": metadata["format"],
        "attention_definition": ATTENTION_DEFINITION,
        "records": collector.records,
    }, args.output_dir / "attention_records.pt")
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_report(
        args.output_dir,
        metadata=metadata,
        aggregate=aggregate,
        figures=figures,
    )

    print(f"Captured {len(collector.records)} attention records.")
    print(f"Output: {args.output_dir.resolve()}")
    print("Figures:", ", ".join(figures))


if __name__ == "__main__":
    main()
