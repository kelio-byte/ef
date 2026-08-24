#!/usr/bin/env python
"""Sampling script for Edit Flows retrosynthesis.

Products are processed in GPU batches.  Each product produces consecutive
outputs: ``n_samples`` for Euler or ``n_runs * n_branches`` for Euler-Beam.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import subprocess
import time
import torch
from torch import Tensor
from tqdm import tqdm

from edit_flows.data.dataset import load_vocab
from edit_flows.guidance.model import ProductConditionedGuidance
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.sampling.euler import sample_euler
from edit_flows.sampling.euler_beam import _mix_child_seed, sample_euler_beam
from edit_flows.sampling.structured_diversification import (
    sample_structured_diversification,
)
from edit_flows.sampling.structured_diversification_v2 import (
    sample_delayed_structured_diversification,
)
from edit_flows.sampling.beam import sample_greedy_single_edit, sample_beam_single_edit
from edit_flows.sampling.time_policy import (
    DepthTimePolicy, FixedTimePolicy, RatioTimePolicy, KappaTimePolicy,
)
from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN, UNK_TOKEN


def tokenize_smiles(smiles: str, token2id: dict) -> list:
    tokens = smiles.strip().split()
    unk_id = token2id.get("<unk>", UNK_TOKEN)
    return [token2id.get(t, unk_id) for t in tokens]


def _apply_sampling_seed(seed: int, device: torch.device) -> None:
    """Apply the CLI seed to ordinary stochastic samplers as well as CUDA."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _load_guidance_model(
    checkpoint_path: str,
    device: torch.device,
    expected_vocab_size: int,
) -> ProductConditionedGuidance:
    """Load an independent action-level guidance adapter checkpoint."""
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False,
        )
    except TypeError:  # pragma: no cover - older torch fallback
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    vocab_size = int(config.get("model_vocab", 0))
    if vocab_size != expected_vocab_size:
        raise ValueError(
            "guidance/model vocabulary mismatch: guidance checkpoint has "
            f"{vocab_size}, base checkpoint has {expected_vocab_size}"
        )
    model = ProductConditionedGuidance(
        vocab_size=vocab_size,
        hidden_dim=int(config.get("hidden_dim", 256)),
        product_layers=int(config.get("product_layers", 2)),
        state_layers=int(config.get("state_layers", 4)),
        num_heads=int(config.get("num_heads", 8)),
        dim_feedforward=int(config.get("dim_feedforward", 1024)),
        max_seq_len=int(config.get("max_seq_len", 256)),
        dropout=float(config.get("dropout", 0.1)),
        attention_dropout=float(config.get("attention_dropout", 0.1)),
        activation=config.get("activation", "relu"),
        pos_encoding_scale=not bool(config.get("no_pos_encoding_scale", False)),
        pad_token=PAD_TOKEN,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _ids_to_str(ids: list, id2token: dict) -> str:
    return " ".join(id2token[tid] for tid in ids
                    if tid not in (PAD_TOKEN, BOS_TOKEN))


def _make_batch(product_ids: list[list[int]], n_samples: int,
                pad_token: int, bos_token: int = BOS_TOKEN) -> Tensor:
    B = len(product_ids)
    max_len = max(len(ids) for ids in product_ids)
    x_0 = torch.full((B, max_len + 1), pad_token, dtype=torch.long)
    x_0[:, 0] = bos_token
    for i, ids in enumerate(product_ids):
        x_0[i, 1:1 + len(ids)] = torch.tensor(ids, dtype=torch.long)
    return x_0.repeat_interleave(n_samples, dim=0)


def _load_center_bias_sidecar(
    sidecar_dir: str,
    products_file: str,
) -> tuple[dict, list[dict]]:
    directory = os.path.abspath(sidecar_dir)
    metadata_path = os.path.join(directory, "metadata.json")
    scores_path = os.path.join(directory, "scores.jsonl")
    with open(metadata_path) as handle:
        metadata = json.load(handle)
    expected_sha256 = metadata["files"]["m500_products"]["sha256"]
    actual_sha256 = _sha256_file(products_file)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "center sidecar/input product SHA256 mismatch: "
            f"{actual_sha256} != {expected_sha256}"
        )
    records = []
    with open(scores_path) as handle:
        for line_number, line in enumerate(handle):
            record = json.loads(line)
            if record.get("status") != "ok":
                raise ValueError(
                    "center sidecar contains a failed row at "
                    f"line {line_number + 1}: {record.get('error')}"
                )
            if record.get("input_row_index") != line_number:
                raise ValueError(
                    "center sidecar rows are not contiguous input order"
                )
            records.append(record)
    if len(records) != int(metadata["input_row_count"]):
        raise ValueError(
            "center sidecar row count differs from metadata: "
            f"{len(records)} != {metadata['input_row_count']}"
        )
    if _sha256_file(scores_path) != metadata["files"]["scores"]["sha256"]:
        raise ValueError("center sidecar scores SHA256 differs from metadata")
    return metadata, records


def _make_center_bias_batch(
    product_ids: list[list[int]],
    sidecar_records: list[dict],
    *,
    n_samples: int,
    source: str,
    global_start: int,
    guided_trajectories: int | None = None,
) -> tuple[Tensor, list[dict], Tensor]:
    if len(product_ids) != len(sidecar_records):
        raise ValueError("center sidecar batch does not match product batch")
    if guided_trajectories is None:
        guided_trajectories = n_samples
    if not 0 <= guided_trajectories <= n_samples:
        raise ValueError(
            "guided_trajectories must be between 0 and n_samples inclusive"
        )
    component_field = f"{source}_components"
    max_length = max(len(ids) for ids in product_ids) + 1
    scores = torch.zeros(
        len(product_ids) * n_samples, max_length, 3, dtype=torch.float32
    )
    bias_enabled = torch.zeros(
        len(product_ids) * n_samples, dtype=torch.bool
    )
    row_metadata = []
    output_row = 0
    for product_offset, (ids, record) in enumerate(
        zip(product_ids, sidecar_records)
    ):
        components = record.get(component_field, [])[:3]
        if not components:
            raise ValueError(
                f"center sidecar row {record['input_row_index']} has no "
                f"{source} component"
            )
        for trajectory_index in range(n_samples):
            component = components[trajectory_index % len(components)]
            component_scores = torch.tensor(
                component["position_scores"], dtype=torch.float32
            )
            expected_length = len(ids) + 1
            if component_scores.shape != (expected_length, 3):
                raise ValueError(
                    "center score shape differs from tokenized product: "
                    f"{tuple(component_scores.shape)} != "
                    f"({expected_length}, 3)"
                )
            scores[output_row, :expected_length] = component_scores
            is_guided = trajectory_index < guided_trajectories
            bias_enabled[output_row] = is_guided
            row_metadata.append(
                {
                    "global_input_row": global_start + product_offset,
                    "input_row_index": record["input_row_index"],
                    "reaction_position": record["reaction_position"],
                    "augmentation_index": record["augmentation_index"],
                    "trajectory_index": trajectory_index,
                    # In RC1.5 every row retains true-center scores for
                    # diagnostics, while only this fixed subset actually
                    # reweights its first-event rates.  The remaining rows
                    # are ordinary Euler trajectories.
                    "trajectory_role": (
                        "center_guided" if is_guided else "ordinary_euler"
                    ),
                    "center_source": source,
                    "component_id": component["component_id"],
                    "component_atom_maps": component["atom_maps"],
                    "pseudo_relaxation": component.get("relaxation"),
                }
            )
            output_row += 1
    return scores, row_metadata, bias_enabled


def _make_euler_beam_sample_seeds(
    base_seed: int,
    global_start: int,
    n_products: int,
    n_runs: int,
) -> list[int]:
    """构造不依赖 n_branches 和 batch 划分的 product/run seeds。"""
    return [
        _mix_child_seed(base_seed, global_start + i, r + 1)
        for i in range(n_products)
        for r in range(n_runs)
    ]


def _make_grouped_euler_beam_branch_seeds(
    base_seed: int,
    global_start: int,
    n_products: int,
    n_seed_groups: int,
    branches_per_group: int,
) -> list[list[int]]:
    """Recreate virtual run/branch streams inside one global branch pool."""
    if n_seed_groups < 1 or branches_per_group < 1:
        raise ValueError("seed groups and branches per group must be >= 1")
    return [
        [
            _mix_child_seed(base_seed, global_start + product_index, group + 1)
            + branch_index
            for group in range(n_seed_groups)
            for branch_index in range(branches_per_group)
        ]
        for product_index in range(n_products)
    ]


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_metadata(path: str, include_sha256: bool = False) -> dict:
    absolute_path = os.path.abspath(path)
    stat = os.stat(absolute_path)
    metadata = {
        "path": absolute_path,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        metadata["sha256"] = _sha256_file(absolute_path)
    return metadata


def _infer_augmentation(*values: str | None) -> tuple[int | None, str | None]:
    """Infer augmentation only from an explicit ``augN`` path component."""
    matches = []
    for value in values:
        if not value:
            continue
        match = re.search(r"(?:^|[/_\-])aug(?:mentation)?[_\-]?(\d+)", value,
                          flags=re.IGNORECASE)
        if match:
            matches.append((int(match.group(1)), value))
    unique = {number for number, _ in matches}
    if len(unique) == 1:
        number = next(iter(unique))
        source = next(value for candidate, value in matches
                      if candidate == number)
        return number, source
    return None, None


def _git_state() -> dict:
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


def _outputs_per_product(args) -> int:
    if args.sampler == "euler_beam":
        return args.n_runs * args.n_branches
    if args.sampler == "structured_diversification":
        return args.structured_n_trajectories
    if args.sampler == "structured_diversification_v2":
        return args.structured_v2_k_mode * args.structured_v2_k_completion
    return args.n_samples


def _center_trajectory_count(args) -> int:
    """Number of independent paths eligible for a first-event bias."""
    return args.n_runs if args.sampler == "euler_beam" else args.n_samples


def _is_frozen_r9k1m2(args) -> bool:
    """Return whether the CLI exactly describes the reported R9K1M2 setup."""
    return (
        args.sampler == "euler_beam"
        and args.n_runs == 9
        and args.n_branches == 1
        and args.n_children == 2
        and args.euler_beam_score_mode == "full_probability"
        and args.euler_beam_child_policy == "stochastic_noop"
        and args.euler_beam_changed_state_bonus == 0.5
        and args.euler_beam_q_temperature == 1.0
        and not args.euler_beam_first_edit_diversity
        and not args.euler_beam_share_identical_forwards
        and args.euler_beam_initial_seed_groups is None
    )


def _euler_beam_output_row_indices(
    product_index: int,
    n_runs: int,
    n_branches: int,
) -> list[int]:
    """Return rank-major/run-minor rows for one product.

    ``sample_euler_beam`` returns K rows for each input run in run-major
    order.  The prediction file instead places every run winner first, then
    every rank-2 branch, so adding branch tails does not demote later run
    winners in cross-augmentation local ranking.
    """
    input_start = product_index * n_runs
    return [
        (input_start + run_index) * n_branches + branch_rank
        for branch_rank in range(n_branches)
        for run_index in range(n_runs)
    ]


def _select_products(
    products: list[str],
    start_product: int,
    max_products: int | None,
    augmentation: int | None,
) -> tuple[list[str], int]:
    """Select a reproducible, augmentation-aligned input interval."""
    if start_product < 0:
        raise ValueError(
            f"start_product must be >= 0, got {start_product}"
        )
    if max_products is not None and max_products <= 0:
        raise ValueError(
            f"max_products must be > 0, got {max_products}"
        )
    end_product = (
        len(products)
        if max_products is None else start_product + max_products
    )
    if start_product >= len(products) or end_product > len(products):
        raise ValueError(
            "requested product interval is outside the input file: "
            f"[{start_product}, {end_product}) for {len(products)} lines"
        )
    if augmentation is not None and (
        start_product % augmentation != 0
        or (end_product - start_product) % augmentation != 0
    ):
        raise ValueError(
            "product interval must preserve complete augmentation blocks: "
            f"start={start_product}, count={end_product - start_product}, "
            f"augmentation={augmentation}"
        )
    return products[start_product:end_product], end_product


def _build_sampling_metadata(
    args,
    cfg: dict,
    *,
    prediction_path: str,
    source_product_count: int,
    selection_start_product: int,
    product_count: int,
    output_line_count: int,
    n_sampling_steps: int,
    sample_scheduler_name: str,
    train_scheduler_name: str,
    use_origin_mask: bool,
    elapsed_seconds: float,
    use_product_memory: bool = False,
    peak_cuda_allocated_bytes: int | None = None,
    peak_cuda_reserved_bytes: int | None = None,
    euler_beam_profile: dict | None = None,
    euler_beam_stats: dict | None = None,
    structured_stats: dict | None = None,
    structured_diagnostics_path: str | None = None,
    center_bias_diagnostics_path: str | None = None,
) -> dict:
    # Augmentation describes the actual input layout, so it must not be
    # inferred from the checkpoint's training data directory.  A single
    # --product has no augmentation layout to infer.
    augmentation, augmentation_source = _infer_augmentation(
        args.products_file,
    )
    input_metadata = {
        "kind": "products_file" if args.products_file else "single_product",
        "product_count": product_count,
        "source_product_count": source_product_count,
        "selection_start_product": selection_start_product,
        "selection_end_product_exclusive": (
            selection_start_product + product_count
        ),
    }
    if args.products_file:
        input_metadata.update(_path_metadata(
            args.products_file, include_sha256=True,
        ))

    sampling = {
        "n_steps": n_sampling_steps,
        "sample_scheduler": sample_scheduler_name,
        "train_scheduler": train_scheduler_name,
        "seed": args.seed,
        "seed_applied_to_sampler": args.sampler in (
                            "euler", "euler_beam", "structured_diversification",
                            "structured_diversification_v2",
        ),
    }
    if args.sampler == "euler_beam":
        sampling.update({
            "n_branches": args.n_branches,
            "n_children": args.n_children,
            "n_runs": args.n_runs,
            "final_branches_per_run": args.n_branches,
            "output_order": "branch-rank-major, run-minor",
            "initial_seed_groups": args.euler_beam_initial_seed_groups,
            "score_mode": args.euler_beam_score_mode,
            "changed_state_bonus": args.euler_beam_changed_state_bonus,
            "matmul_precision": args.euler_beam_matmul_precision,
            "child_policy": args.euler_beam_child_policy,
            "q_temperature": args.euler_beam_q_temperature,
            "first_edit_diversity": getattr(
                args, "euler_beam_first_edit_diversity", False,
            ),
            "share_identical_forwards": (
                args.euler_beam_share_identical_forwards
            ),
            "seed_scope": (
                "grouped virtual-run/branch streams"
                if args.euler_beam_initial_seed_groups is not None
                else "stable product/run streams"
            ),
        })
    elif args.sampler == "structured_diversification":
        sampling.update({
            "n_trajectories": args.structured_n_trajectories,
            "direction_selection": "descending_log_lambda",
            "token_selection": args.structured_token_selection,
            "continuation": "ordinary_euler_m1_from_first_step",
            "cross_trajectory_competition": False,
            "seed_scope": "global torch RNG",
        })
        if structured_diagnostics_path is not None:
            sampling["diagnostics"] = _path_metadata(
                structured_diagnostics_path, include_sha256=True,
            )
    elif args.sampler == "structured_diversification_v2":
        sampling.update({
            "k_mode": args.structured_v2_k_mode,
            "k_completion": args.structured_v2_k_completion,
            "mode_pool_size": args.structured_v2_mode_pool_size,
            "trigger": "first_sampled_edit_event_or_final_step",
            "mode_selection": "top1_anchor_plus_weighted_top_pool_without_replacement",
            "token_selection": "top_q_completion",
            "continuation": "ordinary_euler_m1_stateless_seeded",
            "cross_trajectory_competition": False,
            "seed_scope": "stable product/trajectory streams",
        })
        if structured_diagnostics_path is not None:
            sampling["diagnostics"] = _path_metadata(
                structured_diagnostics_path, include_sha256=True,
            )
    else:
        sampling["n_samples"] = args.n_samples
        if getattr(args, "guidance_checkpoint", None):
            sampling.update({
                "guidance_checkpoint": _path_metadata(
                    args.guidance_checkpoint, include_sha256=True,
                ),
                "guidance_beta": args.guidance_beta,
                "guidance_mode": "action_rate_normalized",
                "guidance_rate_normalization": getattr(
                    args, "guidance_rate_normalization", "per_position",
                ),
            })
        sampling["seed_scope"] = (
            "global torch RNG"
            if args.sampler == "euler" else "sampler-specific"
        )

    if getattr(args, "first_event_center_sidecar", None):
        center_trajectory_count = _center_trajectory_count(args)
        guided_count = (
            args.first_event_center_guided_trajectories
            if args.first_event_center_guided_trajectories is not None
            else center_trajectory_count
        )
        sampling.update({
            "first_event_center_sidecar": _path_metadata(
                os.path.join(
                    args.first_event_center_sidecar, "metadata.json"
                ),
                include_sha256=True,
            ),
            "first_event_center_source": args.first_event_center_source,
            "first_event_center_max_multiplier": (
                args.first_event_center_max_multiplier
            ),
            "first_event_center_guided_trajectories": guided_count,
            "first_event_center_ordinary_trajectories": (
                center_trajectory_count - guided_count
            ),
            "first_event_center_trajectory_assignment": (
                "trajectory indices [0, guided_count) are center-guided; "
                "the remainder are ordinary sampler trajectories"
            ),
            "first_event_center_diagnostic_detail": getattr(
                args, "first_event_center_diagnostics_resolved", "full",
            ),
            "first_event_position_only": True,
            "per_mode_total_hazard_preserved": True,
            "continuation": (
                "ordinary Euler after each trajectory's first non-noop step"
                if args.sampler == "euler"
                else "ordinary R9K1M2 after each selected lineage's first "
                "non-noop step"
            ),
        })
        if args.sampler == "euler_beam":
            sampling["first_event_center_beam_semantics"] = (
                "each of the nine independent R9 runs receives its own "
                "first-event bias; child selection remains the frozen K1M2 "
                "full-probability rule"
            )
        if center_bias_diagnostics_path is not None:
            sampling["first_event_center_diagnostics"] = _path_metadata(
                center_bias_diagnostics_path,
                include_sha256=True,
            )

    runtime = {
        "elapsed_seconds": elapsed_seconds,
        "batch_size": args.batch_size,
        "device": args.device,
    }
    if peak_cuda_allocated_bytes is not None:
        runtime["peak_cuda_allocated_bytes"] = peak_cuda_allocated_bytes
        runtime["peak_cuda_reserved_bytes"] = peak_cuda_reserved_bytes
    if euler_beam_profile is not None:
        runtime["euler_beam_profile"] = euler_beam_profile
    if euler_beam_stats is not None:
        runtime["euler_beam_stats"] = euler_beam_stats
    if structured_stats is not None:
        runtime["structured_diversification_stats"] = structured_stats

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "layout": (
            "input-product-major, branch-rank-major, run-minor"
            if args.sampler == "euler_beam"
            else "input-product-major, output-minor"
        ),
        "sampler": args.sampler,
        "augmentation": augmentation,
        "augmentation_inferred_from": augmentation_source,
        "checkpoint": _path_metadata(args.checkpoint),
        "input": input_metadata,
        "product_count": product_count,
        "output_beam_size": _outputs_per_product(args),
        "output_line_count": output_line_count,
        "output_sha256": _sha256_file(prediction_path),
        "sampling": sampling,
        "runtime": runtime,
        "model": {
            "configured_use_origin_mask": cfg.get("use_origin_mask", False),
            "effective_use_origin_mask": use_origin_mask,
            "configured_use_product_memory": cfg.get(
                "use_product_memory", False,
            ),
            "effective_use_product_memory": use_product_memory,
            "product_memory_encoder_layers": (
                cfg.get("product_memory_encoder_layers")
                if use_product_memory else None
            ),
            "product_memory_fusion_after_layers": (
                cfg.get("product_memory_fusion_after_layers")
                if use_product_memory else None
            ),
            "product_memory_sampling_cache": (
                "encode_x0_once_per_input_row_then_repeat_per_trajectory"
                if use_product_memory else None
            ),
        },
        "git": _git_state(),
    }


def main():
    parser = argparse.ArgumentParser(description="Sample Edit Flows retrosynthesis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--product", type=str, default=None,
                        help="Product SMILES string (tokenized, space-separated)")
    parser.add_argument("--products_file", type=str, default=None,
                        help="File with one tokenized product SMILES per line")
    parser.add_argument("--start_product", type=int, default=0,
                        help="0-based products_file line offset")
    parser.add_argument("--max_products", type=int, default=None,
                        help="Optional number of products_file lines to sample")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data dir (for vocab)")
    parser.add_argument("--vocab_file", type=str, default=None,
                        help="Override vocab file path")
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of independent samples per product")
    parser.add_argument(
        "--guidance_checkpoint", type=str, default=None,
        help=(
            "Optional action-level DGM guidance checkpoint; currently only "
            "supported with --sampler euler"
        ),
    )
    parser.add_argument(
        "--guidance_beta", type=float, default=1.0,
        help="Exponent applied to guidance weights (0 gives baseline identity)",
    )
    parser.add_argument(
        "--guidance_rate_normalization",
        choices=("per_position", "per_sample"),
        default="per_position",
        help=(
            "Preserve edit rate at each position (legacy) or across each "
            "sample while allowing guidance to move rate between positions"
        ),
    )
    parser.add_argument(
        "--first_event_center_sidecar",
        type=str,
        default=None,
        help=(
            "Directory containing metadata.json and scores.jsonl for the "
            "oracle-only RC1 first-event position-bias experiment"
        ),
    )
    parser.add_argument(
        "--first_event_center_source",
        choices=("oracle", "pseudo"),
        default="oracle",
        help="Use true oracle centers or same-product pseudo centers",
    )
    parser.add_argument(
        "--first_event_center_max_multiplier",
        type=float,
        default=3.0,
        help="Maximum first-event position multiplier at center score 1",
    )
    parser.add_argument(
        "--first_event_center_guided_trajectories",
        type=int,
        default=None,
        help=(
            "Fixed number of leading independent trajectories/runs whose "
            "first event uses the center position bias; default: all"
        ),
    )
    parser.add_argument(
        "--first_event_center_diagnostics",
        choices=("auto", "full", "summary"),
        default="auto",
        help=(
            "First-event diagnostic detail: auto keeps full records for "
            "ordinary Euler and compact counts for Euler-Beam"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32,
                        help="GPU batch size (number of products per batch)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save predictions")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (used as base_seed for euler_beam)")
    parser.add_argument("--scheduler", type=str, default=None,
                        choices=["cubic", "linear"],
                        help="Override sampling scheduler (default: use config sample_scheduler or training scheduler)")
    parser.add_argument("--sampler", type=str, default="euler",
                        choices=[
                            "euler", "euler_beam",
                            "structured_diversification",
                            "structured_diversification_v2",
                            "greedy_edit", "beam_edit",
                        ],
                        help="Sampling algorithm (default: euler)")
    parser.add_argument("--n_branches", type=int, default=5,
                        help="并行分支数 (euler_beam)")
    parser.add_argument("--n_children", type=int, default=1,
                        help="每个父分支每步生成的后继数 (euler_beam)")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="每个产物独立运行次数 (euler_beam, 等价于 Euler 的 --n_samples)")
    parser.add_argument(
        "--structured_n_trajectories", type=int, default=9,
        help="Structured sampler trajectories per product",
    )
    parser.add_argument(
        "--structured_token_selection", type=str, default="argmax",
        choices=["argmax", "sample"],
        help="INS/SUB token selection for structured first edits",
    )
    parser.add_argument(
        "--structured_v2_k_mode", type=int, default=3,
        help="Delayed structured v2 high-probability modes",
    )
    parser.add_argument(
        "--structured_v2_k_completion", type=int, default=3,
        help="Delayed structured v2 Q completions per mode",
    )
    parser.add_argument(
        "--structured_v2_mode_pool_size", type=int, default=6,
        help="Candidate mode pool used for delayed v2 alternatives",
    )
    parser.add_argument(
        "--euler_beam_initial_seed_groups",
        type=int,
        default=None,
        help=(
            "Optional virtual-run seed groups inside one global branch pool; "
            "requires n_runs=1 and n_branches divisible by this value"
        ),
    )
    parser.add_argument("--euler_beam_score_mode", type=str,
                        default="full_probability",
                        choices=["full_probability", "legacy_triggered_reverse"],
                        help="Euler-Beam scoring mode; legacy is only for ablation")
    parser.add_argument("--euler_beam_changed_state_bonus", type=float,
                        default=0.0,
                        help="Fixed search bonus for states changed from the product")
    parser.add_argument(
        "--euler_beam_q_temperature", type=float, default=1.0,
        help=(
            "Temperature for insert/substitute token posterior; 1.0 keeps "
            "the checkpoint distribution"
        ),
    )
    parser.add_argument("--euler_beam_matmul_precision", type=str,
                        default="highest", choices=["highest", "high"],
                        help=("Float32 matmul precision for CUDA Euler-Beam; "
                              "'high' enables TF32 on supported GPUs"))
    parser.add_argument("--euler_beam_child_policy", type=str,
                        default="stochastic",
                        choices=["stochastic", "stochastic_noop"],
                        help="Euler-Beam child proposal policy")
    parser.add_argument(
        "--euler_beam_first_edit_diversity",
        action="store_true",
        default=False,
        help=(
            "Within each beam, reserve branches for distinct first real "
            "edits; no extra Transformer forwards or output slots"
        ),
    )
    parser.add_argument(
        "--euler_beam_profile",
        action="store_true",
        default=False,
        help=(
            "Synchronize CUDA between Euler-Beam stages and record a "
            "timing breakdown; use only for short profiling runs"
        ),
    )
    parser.add_argument(
        "--euler_beam_share_identical_forwards",
        action="store_true",
        default=False,
        help=(
            "Share deterministic model forwards for exact duplicate states "
            "inside each product's protected run group"
        ),
    )
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam size for beam_edit sampler")
    parser.add_argument("--max_edits", type=int, default=20,
                        help="Max edit steps for greedy/beam samplers")
    parser.add_argument("--time_policy", type=str, default="depth",
                        choices=["depth", "fixed", "ratio", "kappa"],
                        help="Time policy for greedy/beam: depth, fixed, ratio, kappa")
    parser.add_argument("--time_const", type=float, default=0.5,
                        help="Fixed t value when time_policy=fixed")
    parser.add_argument("--k_ins_token", type=int, default=4,
                        help="Top-k insert tokens per position")
    parser.add_argument("--k_sub_token", type=int, default=4,
                        help="Top-k substitute tokens per position")
    parser.add_argument("--k_edit_expand", type=int, default=16,
                        help="Global top-k edit candidates per step")
    parser.add_argument("--stop_u_tot_base", type=float, default=-1.0,
                        help="Stop threshold on executable edit mass in scoring-rate space (< 0 disables)")
    parser.add_argument("--explicit_stop", action="store_true", default=False,
                        help="Treat STOP as an explicit candidate action alongside edits")
    parser.add_argument("--kappa_mode", type=str, default="ratio",
                        choices=["ratio", "frozen_hazard", "poisson"],
                        help="Kappa update mode when explicit_stop=True")
    parser.add_argument("--p_stop_mode", type=str, default="absolute",
                        choices=["absolute", "normalized"],
                        help="p_stop formula: e^{-U} (absolute) or e^{-U/U_init} (normalized)")
    parser.add_argument("--fh_warmup_steps", type=int, default=0,
                        help="Warmup steps using depth kappa before frozen-hazard kappa kicks in")
    args = parser.parse_args()

    if args.euler_beam_profile and args.sampler != "euler_beam":
        raise ValueError("euler_beam_profile requires --sampler euler_beam")
    if (
        args.euler_beam_first_edit_diversity
        and args.sampler != "euler_beam"
    ):
        raise ValueError(
            "euler_beam_first_edit_diversity requires --sampler euler_beam"
        )
    if args.structured_n_trajectories < 1:
        raise ValueError("structured_n_trajectories must be >= 1")
    if args.structured_token_selection not in {"argmax", "sample"}:
        raise ValueError(
            "structured_token_selection must be 'argmax' or 'sample'"
        )
    if args.structured_v2_k_mode < 1:
        raise ValueError("structured_v2_k_mode must be >= 1")
    if args.structured_v2_k_completion < 1:
        raise ValueError("structured_v2_k_completion must be >= 1")
    if args.structured_v2_mode_pool_size < args.structured_v2_k_mode:
        raise ValueError(
            "structured_v2_mode_pool_size must be >= structured_v2_k_mode"
        )
    if args.sampler == "euler_beam":
        if args.euler_beam_initial_seed_groups is not None:
            if args.n_runs != 1:
                raise ValueError(
                    "euler_beam_initial_seed_groups requires n_runs=1"
                )
            if args.euler_beam_initial_seed_groups < 1:
                raise ValueError(
                    "euler_beam_initial_seed_groups must be >= 1"
                )
            if args.n_branches % args.euler_beam_initial_seed_groups != 0:
                raise ValueError(
                    "n_branches must be divisible by "
                    "euler_beam_initial_seed_groups"
                )
    if args.guidance_checkpoint and args.sampler != "euler":
        raise ValueError(
            "--guidance_checkpoint is currently supported only with "
            "--sampler euler"
        )
    if args.first_event_center_sidecar:
        if args.sampler not in {"euler", "euler_beam"}:
            raise ValueError(
                "first_event_center_sidecar requires --sampler euler or "
                "the frozen R9K1M2 Euler-Beam layout"
            )
        if args.products_file is None or args.product is not None:
            raise ValueError(
                "first_event_center_sidecar requires --products_file"
            )
        if args.sampler == "euler" and args.n_samples != 9:
            raise ValueError(
                "RC1 Euler is frozen to --n_samples 9 with center sidecars"
            )
        if args.sampler == "euler_beam" and not _is_frozen_r9k1m2(args):
            raise ValueError(
                "Euler-Beam center bias is frozen to R9K1M2: "
                "n_runs=9, n_branches=1, n_children=2, "
                "score_mode=full_probability, stochastic_noop, "
                "changed_state_bonus=0.5, q_temperature=1.0, "
                "without first-edit diversity or forward sharing"
            )
        if args.guidance_checkpoint:
            raise ValueError(
                "center first-event bias and learned guidance cannot be "
                "combined in RC1"
            )
        center_trajectory_count = _center_trajectory_count(args)
        if (
            args.first_event_center_guided_trajectories is not None
            and not 0 <= args.first_event_center_guided_trajectories
            <= center_trajectory_count
        ):
            raise ValueError(
                "first_event_center_guided_trajectories must be between 0 "
                "and the number of independent trajectories inclusive"
            )
        if args.first_event_center_diagnostics == "auto":
            args.first_event_center_diagnostics_resolved = (
                "full" if args.sampler == "euler" else "summary"
            )
        else:
            args.first_event_center_diagnostics_resolved = (
                args.first_event_center_diagnostics
            )
    elif args.first_event_center_guided_trajectories is not None:
        raise ValueError(
            "first_event_center_guided_trajectories requires "
            "first_event_center_sidecar"
        )
    elif args.first_event_center_diagnostics != "auto":
        raise ValueError(
            "first_event_center_diagnostics requires "
            "first_event_center_sidecar"
        )
    if (
        args.first_event_center_max_multiplier < 1
        or not torch.isfinite(
            torch.tensor(args.first_event_center_max_multiplier)
        )
    ):
        raise ValueError(
            "first_event_center_max_multiplier must be finite and >= 1"
        )
    if args.guidance_beta < 0 or not torch.isfinite(torch.tensor(args.guidance_beta)):
        raise ValueError("guidance_beta must be finite and non-negative")

    device = torch.device(args.device)
    if args.sampler == "euler_beam" and device.type == "cuda":
        torch.set_float32_matmul_precision(
            args.euler_beam_matmul_precision
        )

    # PyTorch 2.6+ defaults ``weights_only=True`` for ``torch.load``.  Our
    # trusted training checkpoints intentionally contain the model config and
    # vocabulary metadata (including NumPy values), so they must be loaded as
    # the complete checkpoint object.  Keep a fallback for older PyTorch
    # versions where the ``weights_only`` keyword is unavailable.
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model_vocab = ckpt.get("model_vocab")

    if args.vocab_file:
        vocab_path = args.vocab_file
    elif args.data_dir:
        vocab_path = os.path.join(args.data_dir, cfg.get("vocab_file", "example.vocab.src"))
    else:
        vocab_path = os.path.join(cfg["data_dir"], cfg.get("vocab_file", "example.vocab.src"))

    token2id, _ = load_vocab(vocab_path)
    if model_vocab is None:
        model_vocab = len(token2id)

    id2token = {v: k for k, v in token2id.items()}

    use_origin_mask = cfg.get("use_origin_mask", False)
    has_origin_embed = any("origin_embedding" in k for k in ckpt["model_state_dict"])
    if use_origin_mask and not has_origin_embed:
        print("WARNING: config has use_origin_mask=True but checkpoint lacks "
              "origin_embedding weights. Falling back to use_origin_mask=False.")
        use_origin_mask = False

    use_product_memory = bool(cfg.get("use_product_memory", False))
    has_product_memory = any(
        key.startswith("product_memory_encoder_layers.")
        or key.startswith("product_memory_fusion_layers.")
        for key in ckpt["model_state_dict"]
    )
    if use_product_memory and not has_product_memory:
        raise ValueError(
            "checkpoint config has use_product_memory=True but its state "
            "dict lacks product-memory weights"
        )
    if not use_product_memory and has_product_memory:
        raise ValueError(
            "checkpoint state dict contains product-memory weights but its "
            "config has use_product_memory=False"
        )
    if use_product_memory and args.sampler != "euler":
        raise ValueError(
            "product-memory checkpoints currently support only --sampler "
            "euler; beam/structured samplers need cache-aware branch "
            "bookkeeping before they can be compared fairly"
        )

    model = EditFlowsTransformer(
        vocab_size=model_vocab,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        dim_feedforward=cfg["dim_feedforward"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
        attention_dropout=cfg.get("attention_dropout", cfg["dropout"]),
        activation=cfg.get("activation", "relu"),
        pos_encoding_scale=cfg.get("pos_encoding_scale", True),
        use_origin_mask=use_origin_mask,
        use_product_memory=use_product_memory,
        product_memory_encoder_layers=cfg.get(
            "product_memory_encoder_layers", 0,
        ),
        product_memory_fusion_after_layers=cfg.get(
            "product_memory_fusion_after_layers",
        ),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    guidance_model = None
    if args.guidance_checkpoint:
        guidance_model = _load_guidance_model(
            args.guidance_checkpoint, device, model_vocab,
        )
    _apply_sampling_seed(args.seed, device)

    if args.scheduler:
        sample_scheduler_name = args.scheduler
    else:
        sample_scheduler_name = cfg.get("sample_scheduler", cfg.get("scheduler", "cubic"))
    train_scheduler_name = cfg.get("scheduler", "cubic")
    kappa_scheduler = CubicScheduler() if sample_scheduler_name == "cubic" else LinearScheduler()
    train_scheduler = CubicScheduler() if train_scheduler_name == "cubic" else LinearScheduler()
    time_input = cfg.get("time_input", "t")
    clamp_kappa = cfg.get("clamp_kappa", False)
    clamp_max = cfg.get("clamp_max", 50.0)
    n_sampling_steps = args.n_steps or cfg.get("n_sampling_steps", 100)

    if args.product and args.products_file:
        raise ValueError("Provide only one of --product or --products_file")
    if args.product:
        if args.start_product != 0 or args.max_products is not None:
            raise ValueError(
                "start_product/max_products require --products_file"
            )
        products = [args.product]
    elif args.products_file:
        with open(args.products_file) as f:
            products = [line.strip() for line in f]
    else:
        raise ValueError("Provide --product or --products_file")

    source_product_count = len(products)
    input_augmentation, _ = _infer_augmentation(args.products_file)
    products, selection_end_product = _select_products(
        products,
        start_product=args.start_product,
        max_products=args.max_products,
        augmentation=input_augmentation,
    )
    n_products = len(products)
    if args.start_product or args.max_products is not None:
        print(
            "Selected product interval: "
            f"[{args.start_product}, {selection_end_product}) from "
            f"{source_product_count} input lines"
        )
    product_ids = [tokenize_smiles(s, token2id) for s in products]
    outputs_per_product = _outputs_per_product(args)
    center_sidecar_metadata = None
    selected_center_records = None
    center_bias_stats = None
    guided_center_trajectories = None
    center_trajectory_count = None
    if args.first_event_center_sidecar:
        center_sidecar_metadata, all_center_records = (
            _load_center_bias_sidecar(
                args.first_event_center_sidecar,
                args.products_file,
            )
        )
        if selection_end_product > len(all_center_records):
            raise ValueError(
                "selected product interval exceeds center sidecar rows: "
                f"{selection_end_product} > {len(all_center_records)}"
            )
        selected_center_records = all_center_records[
            args.start_product:selection_end_product
        ]
        guided_center_trajectories = (
            args.first_event_center_guided_trajectories
            if args.first_event_center_guided_trajectories is not None
            else _center_trajectory_count(args)
        )
        center_trajectory_count = _center_trajectory_count(args)
        center_bias_stats = {
            "schema_version": 3,
            "sampler": args.sampler,
            "center_source": args.first_event_center_source,
            "max_multiplier": args.first_event_center_max_multiplier,
            "guided_trajectories_per_product": guided_center_trajectories,
            "ordinary_euler_trajectories_per_product": (
                center_trajectory_count - guided_center_trajectories
            ),
            "independent_trajectories_per_product": center_trajectory_count,
            "diagnostic_detail": (
                args.first_event_center_diagnostics_resolved
            ),
            "sidecar_metadata": center_sidecar_metadata,
            "records": [],
        }

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        pred_file = os.path.join(args.output_dir, "predictions.txt")
        f_out = open(pred_file, "w")
        print(
            f"Sampling {n_products} products x {outputs_per_product} "
            "outputs"
        )
        print(f"Batch size: {args.batch_size}")
        print(f"Predictions will be saved to: {pred_file}")
    else:
        f_out = None

    batch_size = args.batch_size
    n_batches = math.ceil(n_products / batch_size)

    use_greedy_beam = args.sampler in ("greedy_edit", "beam_edit")
    print(f"Sampler: {args.sampler}")
    euler_beam_profile = {} if args.euler_beam_profile else None
    euler_beam_stats = {} if args.sampler == "euler_beam" else None
    structured_stats = (
        {} if args.sampler in {
            "structured_diversification",
            "structured_diversification_v2",
        } else None
    )
    structured_action_records: list[dict] = []
    if use_greedy_beam:
        # Build time policy.
        if args.time_policy == "depth":
            time_policy = DepthTimePolicy(scheduler=kappa_scheduler)
        elif args.time_policy == "fixed":
            time_policy = FixedTimePolicy(scheduler=kappa_scheduler, time_const=args.time_const)
        elif args.time_policy == "ratio":
            time_policy = RatioTimePolicy(scheduler=kappa_scheduler)
        elif args.time_policy == "kappa":
            time_policy = KappaTimePolicy(scheduler=kappa_scheduler)
        else:
            raise ValueError(f"Unknown time_policy: {args.time_policy}")
        print(f"  max_edits={args.max_edits}, time_policy={args.time_policy}, "
              f"k_ins={args.k_ins_token}, k_sub={args.k_sub_token}, "
              f"k_edit_expand={args.k_edit_expand}")
    if args.sampler == "beam_edit":
        print(f"  beam_size={args.beam_size}")
    if args.sampler == "euler_beam":
        print(f"  n_branches={args.n_branches}, "
              f"n_children={args.n_children}, n_runs={args.n_runs}, "
              f"outputs_per_product={outputs_per_product}, "
              f"matmul_precision={args.euler_beam_matmul_precision}, "
              f"child_policy={args.euler_beam_child_policy}, "
              f"first_edit_diversity={args.euler_beam_first_edit_diversity}, "
              f"q_temperature={args.euler_beam_q_temperature}")
    if args.sampler == "structured_diversification_v2":
        print(
            f"  k_mode={args.structured_v2_k_mode}, "
            f"k_completion={args.structured_v2_k_completion}, "
            f"mode_pool_size={args.structured_v2_mode_pool_size}"
        )
    if args.guidance_checkpoint:
        print(
            f"  guidance_checkpoint={args.guidance_checkpoint}, "
            f"guidance_beta={args.guidance_beta}"
        )
    if selected_center_records is not None:
        print(
            "  first_event_center_source="
            f"{args.first_event_center_source}, multiplier="
            f"{args.first_event_center_max_multiplier}, guided="
            f"{guided_center_trajectories}/{center_trajectory_count}, ordinary="
            f"{center_trajectory_count - guided_center_trajectories}, detail="
            f"{args.first_event_center_diagnostics_resolved}"
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sampling_started_at = time.perf_counter()
    written_predictions = 0
    try:
        for batch_idx in tqdm(range(n_batches), desc="Batches"):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_products)
            batch_products = product_ids[start:end]

            if use_greedy_beam:
                n_rep = 1
            elif args.sampler == "euler_beam":
                n_rep = args.n_runs
            else:
                n_rep = args.n_samples

            # Product memory encodes each immutable input only once.  Rows are
            # product-major both here and in _make_batch, so repeat_interleave
            # aligns each cache row with its n_rep Euler trajectories.
            product_memory = None
            product_memory_padding_mask = None
            if use_product_memory:
                x_0_unique = _make_batch(batch_products, 1, PAD_TOKEN).to(device)
                x_0 = x_0_unique.repeat_interleave(n_rep, dim=0)
                product_memory_padding_mask = x_0_unique == PAD_TOKEN
                with torch.no_grad():
                    product_memory = model.encode_product(
                        x_0_unique, product_memory_padding_mask,
                    ).repeat_interleave(n_rep, dim=0)
                product_memory_padding_mask = (
                    product_memory_padding_mask.repeat_interleave(n_rep, dim=0)
                )
            else:
                x_0 = _make_batch(batch_products, n_rep, PAD_TOKEN).to(device)

            center_scores = None
            center_row_metadata = None
            center_bias_enabled = None
            if selected_center_records is not None:
                (
                    center_scores,
                    center_row_metadata,
                    center_bias_enabled,
                ) = _make_center_bias_batch(
                    batch_products,
                    selected_center_records[start:end],
                    n_samples=n_rep,
                    source=args.first_event_center_source,
                    global_start=args.start_product + start,
                    guided_trajectories=guided_center_trajectories,
                )

            if args.sampler == "euler":
                results, _ = sample_euler(
                    model, x_0, kappa_scheduler,
                    n_steps=n_sampling_steps,
                    max_seq_len=cfg["max_seq_len"],
                    use_rate_reparam=cfg.get("use_rate_reparam", False),
                    clamp_kappa=clamp_kappa,
                    clamp_max=clamp_max,
                    time_input=time_input,
                    train_scheduler=train_scheduler,
                    use_origin_mask=use_origin_mask,
                    product_memory=product_memory,
                    product_memory_padding_mask=product_memory_padding_mask,
                    guidance_model=guidance_model,
                    guidance_product=x_0 if guidance_model is not None else None,
                    guidance_beta=args.guidance_beta,
                    guidance_rate_normalization=args.guidance_rate_normalization,
                    first_event_position_scores=center_scores,
                    first_event_position_bias_enabled=center_bias_enabled,
                    first_event_bias_max_multiplier=(
                        args.first_event_center_max_multiplier
                    ),
                    first_event_bias_stats=center_bias_stats,
                    first_event_row_metadata=center_row_metadata,
                )
            elif args.sampler == "euler_beam":
                B_prod = end - start
                # _make_batch uses repeat_interleave, so rows are product-major:
                # P0R0, P0R1, ..., P1R0, P1R1, ...
                initial_branch_seeds = None
                if args.euler_beam_initial_seed_groups is not None:
                    branches_per_group = (
                        args.n_branches
                        // args.euler_beam_initial_seed_groups
                    )
                    initial_branch_seeds = (
                        _make_grouped_euler_beam_branch_seeds(
                            args.seed,
                            args.start_product + start,
                            B_prod,
                            args.euler_beam_initial_seed_groups,
                            branches_per_group,
                        )
                    )
                    sample_seeds = None
                else:
                    sample_seeds = _make_euler_beam_sample_seeds(
                        args.seed, args.start_product + start,
                        B_prod, args.n_runs,
                    )
                results = sample_euler_beam(
                    model, x_0, kappa_scheduler,
                    n_branches=args.n_branches,
                    n_children=args.n_children,
                    n_steps=n_sampling_steps,
                    max_seq_len=cfg["max_seq_len"],
                    use_rate_reparam=cfg.get("use_rate_reparam", False),
                    clamp_kappa=clamp_kappa,
                    clamp_max=clamp_max,
                    time_input=time_input,
                    train_scheduler=train_scheduler,
                    use_origin_mask=use_origin_mask,
                    sample_seeds=sample_seeds,
                    score_mode=args.euler_beam_score_mode,
                    changed_state_bonus=args.euler_beam_changed_state_bonus,
                    child_policy=args.euler_beam_child_policy,
                    q_temperature=args.euler_beam_q_temperature,
                    first_edit_diversity=(
                        args.euler_beam_first_edit_diversity
                    ),
                    profile=euler_beam_profile,
                    profile_sample_group_size=args.n_runs,
                    share_identical_forwards=(
                        args.euler_beam_share_identical_forwards
                    ),
                    initial_branch_seeds=initial_branch_seeds,
                    sampling_stats=euler_beam_stats,
                    first_event_position_scores=center_scores,
                    first_event_position_bias_enabled=center_bias_enabled,
                    first_event_bias_max_multiplier=(
                        args.first_event_center_max_multiplier
                    ),
                    first_event_bias_stats=center_bias_stats,
                    first_event_row_metadata=center_row_metadata,
                    first_event_bias_record_events=(
                        getattr(
                            args,
                            "first_event_center_diagnostics_resolved",
                            None,
                        ) == "full"
                    ),
                )
            elif args.sampler == "structured_diversification":
                B_prod = end - start
                # Structured sampler expands each product internally after
                # selecting its distinct first-edit directions.
                x_structured = _make_batch(
                    batch_products, 1, PAD_TOKEN,
                ).to(device)
                results, _ = sample_structured_diversification(
                    model,
                    x_structured,
                    kappa_scheduler,
                    n_trajectories=args.structured_n_trajectories,
                    n_steps=n_sampling_steps,
                    max_seq_len=cfg["max_seq_len"],
                    use_rate_reparam=cfg.get("use_rate_reparam", False),
                    clamp_kappa=clamp_kappa,
                    clamp_max=clamp_max,
                    time_input=time_input,
                    train_scheduler=train_scheduler,
                    token_selection=args.structured_token_selection,
                    product_indices=[
                        args.start_product + start + product_index
                        for product_index in range(B_prod)
                    ],
                    action_records=structured_action_records,
                    sampling_stats=structured_stats,
                )
            elif args.sampler == "structured_diversification_v2":
                B_prod = end - start
                x_structured = _make_batch(
                    batch_products, 1, PAD_TOKEN,
                ).to(device)
                results, _ = sample_delayed_structured_diversification(
                    model,
                    x_structured,
                    kappa_scheduler,
                    k_mode=args.structured_v2_k_mode,
                    k_completion=args.structured_v2_k_completion,
                    n_steps=n_sampling_steps,
                    max_seq_len=cfg["max_seq_len"],
                    use_rate_reparam=cfg.get("use_rate_reparam", False),
                    clamp_kappa=clamp_kappa,
                    clamp_max=clamp_max,
                    time_input=time_input,
                    train_scheduler=train_scheduler,
                    event_prob_mode="poisson",
                    use_origin_mask=use_origin_mask,
                    mode_pool_size=args.structured_v2_mode_pool_size,
                    base_seed=args.seed,
                    product_indices=[
                        args.start_product + start + product_index
                        for product_index in range(B_prod)
                    ],
                    action_records=structured_action_records,
                    sampling_stats=structured_stats,
                )
            elif args.sampler == "greedy_edit":
                results = sample_greedy_single_edit(
                    model, x_0, kappa_scheduler,
                    time_policy,
                    max_edits=args.max_edits,
                    max_seq_len=cfg["max_seq_len"],
                    use_rate_reparam=cfg.get("use_rate_reparam", False),
                    clamp_kappa=clamp_kappa,
                    clamp_max=clamp_max,
                    time_input=time_input,
                    train_scheduler=train_scheduler,
                    use_origin_mask=use_origin_mask,
                    k_ins_token=args.k_ins_token,
                    k_sub_token=args.k_sub_token,
                    k_edit_expand=args.k_edit_expand,
                    stop_u_tot_base=args.stop_u_tot_base,
                    explicit_stop=args.explicit_stop,
                    kappa_mode=args.kappa_mode,
                    p_stop_mode=args.p_stop_mode,
                    fh_warmup_steps=args.fh_warmup_steps,
                )
            elif args.sampler == "beam_edit":
                results = sample_beam_single_edit(
                    model, x_0, kappa_scheduler,
                    time_policy,
                    beam_size=args.beam_size,
                    max_edits=args.max_edits,
                    max_seq_len=cfg["max_seq_len"],
                    use_rate_reparam=cfg.get("use_rate_reparam", False),
                    clamp_kappa=clamp_kappa,
                    clamp_max=clamp_max,
                    time_input=time_input,
                    train_scheduler=train_scheduler,
                    use_origin_mask=use_origin_mask,
                    k_ins_token=args.k_ins_token,
                    k_sub_token=args.k_sub_token,
                    k_edit_expand=args.k_edit_expand,
                    stop_u_tot_base=args.stop_u_tot_base,
                )
            else:
                raise ValueError(f"Unknown sampler: {args.sampler}")

            results = results.cpu()
            B = end - start
            for i in range(B):
                if args.sampler == "euler_beam":
                    row_indices = _euler_beam_output_row_indices(
                        i, args.n_runs, args.n_branches,
                    )
                else:
                    n_out = (
                        1 if use_greedy_beam else _outputs_per_product(args)
                    )
                    row_indices = [
                        i if use_greedy_beam else i * n_out + s
                        for s in range(n_out)
                    ]
                for row_idx in row_indices:
                    row = results[row_idx]
                    line = _ids_to_str(row.tolist(), id2token)
                    if f_out:
                        f_out.write(line + "\n")
                        written_predictions += 1
                    else:
                        print(line)
    finally:
        if f_out:
            f_out.close()

    if args.output_dir:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - sampling_started_at
        peak_cuda_allocated_bytes = (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else None
        )
        peak_cuda_reserved_bytes = (
            torch.cuda.max_memory_reserved(device)
            if device.type == "cuda" else None
        )
        expected_predictions = n_products * outputs_per_product
        if written_predictions != expected_predictions:
            raise RuntimeError(
                "sampler output count mismatch: expected "
                f"{expected_predictions}, wrote {written_predictions}"
            )
        structured_diagnostics_path = None
        if args.sampler in {
            "structured_diversification",
            "structured_diversification_v2",
        }:
            structured_diagnostics_path = os.path.join(
                args.output_dir, "structured_diagnostics.json",
            )
            with open(structured_diagnostics_path, "w") as f:
                json.dump(
                    {
                        "schema_version": 1,
                        "sampler": args.sampler,
                        "records": structured_action_records,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
                f.write("\n")
        center_bias_diagnostics_path = None
        if center_bias_stats is not None:
            center_bias_diagnostics_path = os.path.join(
                args.output_dir, "center_bias_diagnostics.json"
            )
            if center_bias_stats.get("summary_from_final_lineages", False):
                # Euler-Beam full-test mode deliberately avoids a potentially
                # huge per-trajectory JSON list.  The sampler has already
                # accumulated the lineage-level counts needed to verify that
                # B1 was active and hazard-preserving.
                center_bias_stats["summary"] = {
                    "action_count_histogram": {},
                    "center_score_histogram": {},
                    "mode_counts": {},
                    "first_event_trajectory_role_counts": (
                        center_bias_stats.get(
                            "first_event_trajectory_role_counts", {}
                        )
                    ),
                    "reweighted_first_event_trajectory_role_counts": (
                        center_bias_stats.get(
                            "reweighted_first_event_trajectory_role_counts",
                            {},
                        )
                    ),
                    "detail_omitted": True,
                }
            else:
                action_count_histogram = {}
                center_score_histogram = {}
                mode_counts = {}
                first_event_role_counts = {}
                reweighted_event_role_counts = {}
                for record in center_bias_stats["records"]:
                    role = str(record.get("row_metadata", {}).get(
                        "trajectory_role", "unspecified"
                    ))
                    first_event_role_counts[role] = (
                        first_event_role_counts.get(role, 0) + 1
                    )
                    if record.get("position_bias_reweighted", False):
                        reweighted_event_role_counts[role] = (
                            reweighted_event_role_counts.get(role, 0) + 1
                        )
                    action_count = str(record["action_count"])
                    action_count_histogram[action_count] = (
                        action_count_histogram.get(action_count, 0) + 1
                    )
                    for action in record["actions"]:
                        score = str(action["center_score"])
                        center_score_histogram[score] = (
                            center_score_histogram.get(score, 0) + 1
                        )
                        mode = action["mode"]
                        mode_counts[mode] = mode_counts.get(mode, 0) + 1
                center_bias_stats["summary"] = {
                    "action_count_histogram": action_count_histogram,
                    "center_score_histogram": center_score_histogram,
                    "mode_counts": mode_counts,
                    "first_event_trajectory_role_counts": first_event_role_counts,
                    "reweighted_first_event_trajectory_role_counts": (
                        reweighted_event_role_counts
                    ),
                }
            with open(center_bias_diagnostics_path, "w") as f:
                json.dump(center_bias_stats, f, indent=2, sort_keys=True)
                f.write("\n")
        metadata = _build_sampling_metadata(
            args,
            cfg,
            prediction_path=pred_file,
            source_product_count=source_product_count,
            selection_start_product=args.start_product,
            product_count=n_products,
            output_line_count=written_predictions,
            n_sampling_steps=n_sampling_steps,
            sample_scheduler_name=sample_scheduler_name,
            train_scheduler_name=train_scheduler_name,
            use_origin_mask=use_origin_mask,
            use_product_memory=use_product_memory,
            elapsed_seconds=elapsed_seconds,
            peak_cuda_allocated_bytes=peak_cuda_allocated_bytes,
            peak_cuda_reserved_bytes=peak_cuda_reserved_bytes,
            euler_beam_profile=euler_beam_profile,
            euler_beam_stats=euler_beam_stats,
            structured_stats=structured_stats,
            structured_diagnostics_path=structured_diagnostics_path,
            center_bias_diagnostics_path=center_bias_diagnostics_path,
        )
        metadata_path = os.path.join(
            args.output_dir, "sampling_metadata.json",
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")
        if args.sampler == "euler_beam":
            print(f"Done. Total predictions: "
                  f"{n_products * outputs_per_product} "
                  f"(n_branches={args.n_branches}, "
                  f"n_children={args.n_children}, n_runs={args.n_runs}, "
                  f"outputs_per_product={outputs_per_product})")
        elif args.sampler == "structured_diversification":
            print(f"Done. Total predictions: "
                  f"{n_products * outputs_per_product} "
                  f"(n_trajectories={args.structured_n_trajectories})")
        elif args.sampler == "structured_diversification_v2":
            print(f"Done. Total predictions: "
                  f"{n_products * outputs_per_product} "
                  f"(k_mode={args.structured_v2_k_mode}, "
                  f"k_completion={args.structured_v2_k_completion})")
        else:
            print(f"Done. Total predictions: {n_products * args.n_samples}")
        print(f"Saved to: {pred_file}")
        print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
