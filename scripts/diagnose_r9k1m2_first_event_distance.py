#!/usr/bin/env python3
"""Stream a first-event distance diagnostic for frozen R9K1M2 sampling.

This is deliberately a diagnostic-only entry point.  It runs the same
R9K1M2 sampler used for the reaction-center oracle experiment, but consumes
each selected lineage's first non-empty edit immediately.  It therefore
reports whether that first event makes the initial M500 token sequence closer
to its paired target without materializing roughly one million event dicts in
RAM or JSON.

``--max-multiplier 1`` is the neutral B0-trace control: it supplies the
sidecar solely so first events can be observed, while preserving the sampler's
log-rates bit-for-bit.  ``--max-multiplier 3`` is oracle B1.  The two runs can
then be compared by ``scripts/compare_first_event_distance.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch
from torch import Tensor
from tqdm import tqdm

from edit_flows.core.scheduler import CubicScheduler, LinearScheduler
from edit_flows.data.dataset import load_vocab
from edit_flows.models.transformer import EditFlowsTransformer
from edit_flows.sampling.euler_beam import _mix_child_seed, sample_euler_beam
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN
try:
    from scripts.sample_retro import (
        _apply_sampling_seed,
        _load_center_bias_sidecar,
        _make_batch,
        _make_center_bias_batch,
        _select_products,
        tokenize_smiles,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from sample_retro import (
        _apply_sampling_seed,
        _load_center_bias_sidecar,
        _make_batch,
        _make_center_bias_batch,
        _select_products,
        tokenize_smiles,
    )


R9_N_RUNS = 9
R9_N_BRANCHES = 1
R9_N_CHILDREN = 2


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_lines(path: Path) -> list[str]:
    with path.open() as handle:
        return [line.rstrip("\n") for line in handle]


def _token_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Levenshtein distance after removing the BOS position.

    This is intentionally the same token-space diagnostic used by
    ``analyze_stage1_rc1.py``.  It does not claim chemical equivalence or
    choose one unique edit ordering.
    """
    left_values = left[1:] if left and int(left[0]) == BOS_TOKEN else left
    right_values = right[1:] if right and int(right[0]) == BOS_TOKEN else right
    if len(left_values) < len(right_values):
        left_values, right_values = right_values, left_values
    previous = list(range(len(right_values) + 1))
    for left_index, left_token in enumerate(left_values, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right_values, start=1):
            deletion = previous[right_index] + 1
            insertion = current[-1] + 1
            substitution = previous[right_index - 1] + (
                left_token != right_token
            )
            current.append(min(deletion, insertion, substitution))
        previous = current
    return int(previous[-1])


def _apply_first_event(
    source_ids: Sequence[int],
    actions: Sequence[Mapping],
    *,
    max_seq_len: int,
) -> tuple[list[int], int]:
    """Replay a sampled first event with Edit Flows' simultaneous-op rules."""
    by_position: dict[int, dict[str, int | None]] = {}
    for action in actions:
        mode = str(action["mode"]).lower()
        if mode not in {"ins", "sub", "del"}:
            raise ValueError(f"unsupported action mode: {mode!r}")
        position = int(action["position"])
        if not 0 <= position < len(source_ids):
            raise ValueError(
                f"action position {position} outside source length "
                f"{len(source_ids)}"
            )
        token = action.get("token_id")
        if mode != "del" and token is None:
            raise ValueError(f"{mode} action lacks token_id")
        by_position.setdefault(position, {})[mode] = (
            None if token is None else int(token)
        )

    output: list[int] = []
    effective_action_count = 0
    for position, source_token in enumerate(source_ids):
        event = by_position.get(position)
        if event is None:
            output.append(int(source_token))
            continue

        has_ins = "ins" in event
        has_sub = "sub" in event
        has_del = "del" in event
        if has_ins and has_del:
            # ``apply_ins_del_operations`` treats simultaneous INS+DEL as a
            # replacement.  A simultaneous SUB is overwritten.
            output.append(int(event["ins"]))
            effective_action_count += 1
            continue
        if has_del:
            effective_action_count += 1
            continue

        current = int(event["sub"]) if has_sub else int(source_token)
        output.append(current)
        if has_sub:
            effective_action_count += 1
        if has_ins:
            output.append(int(event["ins"]))
            effective_action_count += 1
    return output[:max_seq_len], effective_action_count


def _percent(value: int | float, total: int | float) -> float:
    return 100.0 * float(value) / float(total) if total else 0.0


def _center_bucket(score: float) -> str:
    if score >= 0.999999:
        return "1.0"
    if score >= 0.499999:
        return "0.5"
    return "0.0"


class _FirstEventDistanceCollector:
    """Aggregate selected-lineage first events without retaining records."""

    def __init__(
        self,
        source_ids: Sequence[Sequence[int]],
        target_ids: Sequence[Sequence[int]],
        before_distances: Sequence[int],
        *,
        global_start: int,
        max_seq_len: int,
    ) -> None:
        if not (
            len(source_ids) == len(target_ids) == len(before_distances)
        ):
            raise ValueError("source/target/before lengths differ")
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.before_distances = before_distances
        self.global_start = global_start
        self.max_seq_len = max_seq_len
        self.progress: Counter[str] = Counter()
        self.mode_counts: Counter[str] = Counter()
        self.max_score_counts: Counter[str] = Counter()
        self.event_count = 0
        self.sum_distance_delta = 0
        self.sum_effective_action_count = 0
        self.sum_step_index = 0
        self.reweighted_event_count = 0
        self.guided_event_count = 0

    def consume(self, record: dict) -> None:
        metadata = record.get("row_metadata") or {}
        global_row = int(metadata["global_input_row"])
        row_index = global_row - self.global_start
        if not 0 <= row_index < len(self.source_ids):
            raise ValueError(
                f"first-event global row {global_row} is outside selected "
                f"interval [{self.global_start}, "
                f"{self.global_start + len(self.source_ids)})"
            )

        actions = record.get("actions") or []
        if not actions:
            raise ValueError("a recorded first event has no actions")
        after_ids, effective_action_count = _apply_first_event(
            self.source_ids[row_index], actions, max_seq_len=self.max_seq_len
        )
        before = int(self.before_distances[row_index])
        after = _token_distance(after_ids, self.target_ids[row_index])
        delta = before - after
        if delta > 0:
            self.progress["closer"] += 1
        elif delta == 0:
            self.progress["unchanged"] += 1
        else:
            self.progress["farther"] += 1

        self.event_count += 1
        self.sum_distance_delta += delta
        self.sum_effective_action_count += effective_action_count
        self.sum_step_index += int(record["first_event_step_idx"])
        self.reweighted_event_count += int(
            bool(record.get("position_bias_reweighted", False))
        )
        self.guided_event_count += int(
            bool(record.get("position_bias_enabled", False))
        )

        scores = [float(action["center_score"]) for action in actions]
        self.max_score_counts[_center_bucket(max(scores))] += 1
        for action in actions:
            self.mode_counts[str(action["mode"])] += 1

    def summary(self) -> dict:
        count = self.event_count
        return {
            "closer_count": int(self.progress["closer"]),
            "closer_percent": _percent(self.progress["closer"], count),
            "unchanged_count": int(self.progress["unchanged"]),
            "unchanged_percent": _percent(self.progress["unchanged"], count),
            "farther_count": int(self.progress["farther"]),
            "farther_percent": _percent(self.progress["farther"], count),
            "mean_distance_improvement": (
                float(self.sum_distance_delta) / count if count else 0.0
            ),
            "mean_effective_action_count": (
                float(self.sum_effective_action_count) / count if count else 0.0
            ),
            "mean_first_event_step_index": (
                float(self.sum_step_index) / count if count else 0.0
            ),
            "guided_event_count": int(self.guided_event_count),
            "reweighted_event_count": int(self.reweighted_event_count),
            "event_max_center_score_histogram": dict(
                sorted(self.max_score_counts.items())
            ),
            "action_mode_counts": dict(sorted(self.mode_counts.items())),
        }


def _load_checkpoint_model(
    checkpoint_path: Path,
    vocab_path: Path,
    device: torch.device,
) -> tuple[EditFlowsTransformer, dict, dict[str, int]]:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:  # pragma: no cover - compatibility with older torch
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    token2id, _ = load_vocab(str(vocab_path))
    model_vocab = checkpoint.get("model_vocab", len(token2id))

    use_origin_mask = bool(config.get("use_origin_mask", False))
    has_origin_embed = any(
        "origin_embedding" in key for key in checkpoint["model_state_dict"]
    )
    if use_origin_mask and not has_origin_embed:
        use_origin_mask = False
    if use_origin_mask:
        raise ValueError(
            "R9K1M2 diagnostic does not support use_origin_mask=True"
        )
    if bool(config.get("use_product_memory", False)):
        raise ValueError(
            "R9K1M2 diagnostic only supports the non-product-memory "
            "SPE-M500 checkpoint"
        )

    model = EditFlowsTransformer(
        vocab_size=model_vocab,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dim_feedforward=config["dim_feedforward"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
        attention_dropout=config.get("attention_dropout", config["dropout"]),
        activation=config.get("activation", "relu"),
        pos_encoding_scale=config.get("pos_encoding_scale", True),
        use_origin_mask=False,
        use_product_memory=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, token2id


def _sample_seeds(
    base_seed: int,
    global_start: int,
    n_products: int,
) -> list[int]:
    return [
        _mix_child_seed(base_seed, global_start + product_index, run_index + 1)
        for product_index in range(n_products)
        for run_index in range(R9_N_RUNS)
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--products_file", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--vocab_file", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--max_multiplier", type=float, required=True)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scheduler", choices=("cubic", "linear"), default="cubic"
    )
    parser.add_argument("--start_product", type=int, default=0)
    parser.add_argument("--max_products", type=int)
    parser.add_argument(
        "--assert_neutral",
        action="store_true",
        help=(
            "For a multiplier-1 B0 trace, sample the first batch once "
            "without any center input and require bitwise-identical outputs."
        ),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if not math.isfinite(args.max_multiplier) or args.max_multiplier < 1:
        raise ValueError("max_multiplier must be finite and >= 1")
    if args.assert_neutral and args.max_multiplier != 1.0:
        raise ValueError("assert_neutral is only valid with max_multiplier=1")
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite existing output: {args.output_json}")

    required_files = (
        args.checkpoint,
        args.products_file,
        args.targets,
        args.vocab_file,
        args.sidecar / "metadata.json",
        args.sidecar / "scores.jsonl",
    )
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"missing required file: {path}")
    if not args.data_dir.is_dir():
        raise NotADirectoryError(f"missing data directory: {args.data_dir}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)

    all_products = _read_lines(args.products_file)
    all_targets = _read_lines(args.targets)
    if len(all_products) != len(all_targets):
        raise ValueError(
            "products/targets line-count mismatch: "
            f"{len(all_products)} != {len(all_targets)}"
        )
    if len(all_products) % 20:
        raise ValueError(
            "this R9 diagnostic requires complete aug20 blocks; got "
            f"{len(all_products)} product rows"
        )
    products, selection_end = _select_products(
        all_products,
        start_product=args.start_product,
        max_products=args.max_products,
        augmentation=20,
    )
    targets = all_targets[args.start_product:selection_end]
    if len(products) != len(targets):  # defensive; selection was validated above
        raise AssertionError("selected product/target lengths differ")

    sidecar_metadata, all_center_records = _load_center_bias_sidecar(
        str(args.sidecar), str(args.products_file)
    )
    if selection_end > len(all_center_records):
        raise ValueError(
            "selected interval exceeds center sidecar rows: "
            f"{selection_end} > {len(all_center_records)}"
        )
    selected_center_records = all_center_records[
        args.start_product:selection_end
    ]

    model, config, token2id = _load_checkpoint_model(
        args.checkpoint, args.vocab_file, device
    )
    _apply_sampling_seed(args.seed, device)
    scheduler = CubicScheduler() if args.scheduler == "cubic" else LinearScheduler()
    train_scheduler_name = config.get("scheduler", "cubic")
    train_scheduler = (
        CubicScheduler() if train_scheduler_name == "cubic" else LinearScheduler()
    )
    if train_scheduler_name not in {"cubic", "linear"}:
        raise ValueError(
            f"unsupported checkpoint training scheduler: {train_scheduler_name!r}"
        )

    product_ids = [tokenize_smiles(value, token2id) for value in products]
    source_ids = [[BOS_TOKEN, *row] for row in product_ids]
    target_ids = [
        [BOS_TOKEN, *tokenize_smiles(value, token2id)] for value in targets
    ]
    before_distances = [
        _token_distance(source, target)
        for source, target in zip(source_ids, target_ids)
    ]
    collector = _FirstEventDistanceCollector(
        source_ids,
        target_ids,
        before_distances,
        global_start=args.start_product,
        max_seq_len=int(config["max_seq_len"]),
    )
    center_bias_stats: dict = {
        "schema_version": 4,
        "sampler": "euler_beam",
        "center_source": "oracle",
        "max_multiplier": float(args.max_multiplier),
        "guided_trajectories_per_product": R9_N_RUNS,
        "ordinary_euler_trajectories_per_product": 0,
        "independent_trajectories_per_product": R9_N_RUNS,
        "diagnostic_detail": "streaming_distance",
        "sidecar_metadata": sidecar_metadata,
        "records": [],
    }

    total_products = len(products)
    total_trajectories = total_products * R9_N_RUNS
    neutral_verified = None
    started = time.perf_counter()
    n_batches = (total_products + args.batch_size - 1) // args.batch_size
    for batch_index in tqdm(range(n_batches), desc=f"{args.condition} batches"):
        start = batch_index * args.batch_size
        end = min(start + args.batch_size, total_products)
        batch_product_ids = product_ids[start:end]
        x_0 = _make_batch(batch_product_ids, R9_N_RUNS, PAD_TOKEN).to(device)
        (
            center_scores,
            center_row_metadata,
            center_bias_enabled,
        ) = _make_center_bias_batch(
            batch_product_ids,
            selected_center_records[start:end],
            n_samples=R9_N_RUNS,
            source="oracle",
            global_start=args.start_product + start,
        )
        batch_seeds = _sample_seeds(
            args.seed, args.start_product + start, end - start
        )
        common = {
            "n_branches": R9_N_BRANCHES,
            "n_children": R9_N_CHILDREN,
            "n_steps": args.n_steps,
            "max_seq_len": int(config["max_seq_len"]),
            "use_rate_reparam": bool(config.get("use_rate_reparam", False)),
            "clamp_kappa": bool(config.get("clamp_kappa", False)),
            "clamp_max": float(config.get("clamp_max", 50.0)),
            "time_input": config.get("time_input", "t"),
            "train_scheduler": train_scheduler,
            "use_origin_mask": False,
            "sample_seeds": batch_seeds,
            "score_mode": "full_probability",
            "changed_state_bonus": 0.5,
            "child_policy": "stochastic_noop",
            "q_temperature": 1.0,
            "profile_sample_group_size": R9_N_RUNS,
        }
        if batch_index == 0 and args.assert_neutral:
            plain = sample_euler_beam(model, x_0, scheduler, **common)

        results = sample_euler_beam(
            model,
            x_0,
            scheduler,
            first_event_position_scores=center_scores,
            first_event_position_bias_enabled=center_bias_enabled,
            first_event_bias_max_multiplier=args.max_multiplier,
            first_event_bias_stats=center_bias_stats,
            first_event_row_metadata=center_row_metadata,
            first_event_bias_record_events=True,
            first_event_record_sink=collector.consume,
            **common,
        )
        expected_batch_outputs = (end - start) * R9_N_RUNS
        if results.shape[0] != expected_batch_outputs:
            raise RuntimeError(
                f"unexpected batch output count: {results.shape[0]} != "
                f"{expected_batch_outputs}"
            )
        if batch_index == 0 and args.assert_neutral:
            if not torch.equal(plain, results):
                raise RuntimeError(
                    "multiplier=1 trace changed the first batch; refusing "
                    "to label it as an ordinary R9K1M2 baseline"
                )
            neutral_verified = True

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    first_event_count = int(center_bias_stats.get("first_event_count", 0))
    no_event_count = int(center_bias_stats.get("no_event_count", 0))
    if collector.event_count != first_event_count:
        raise RuntimeError(
            "streamed event count differs from sampler stats: "
            f"{collector.event_count} != {first_event_count}"
        )
    if first_event_count + no_event_count != total_trajectories:
        raise RuntimeError(
            "first/no-event count differs from expected R9 trajectories: "
            f"{first_event_count} + {no_event_count} != {total_trajectories}"
        )
    if center_bias_stats.get("records"):
        raise RuntimeError("streaming diagnostic unexpectedly retained records")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": args.condition,
        "interpretation": (
            "first non-empty Euler step; all sampled actions in that step "
            "are replayed together and compared with the paired target in "
            "M500 token Levenshtein distance"
        ),
        "protocol": {
            "sampler": "R9K1M2",
            "n_runs": R9_N_RUNS,
            "n_branches": R9_N_BRANCHES,
            "n_children": R9_N_CHILDREN,
            "score_mode": "full_probability",
            "child_policy": "stochastic_noop",
            "changed_state_bonus": 0.5,
            "q_temperature": 1.0,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "scheduler": args.scheduler,
            "max_multiplier": float(args.max_multiplier),
            "first_event_position_only": True,
            "center_source": "oracle",
        },
        "input": {
            "data_dir": str(args.data_dir.resolve()),
            "products_file": str(args.products_file.resolve()),
            "products_sha256": _sha256_file(args.products_file),
            "targets_file": str(args.targets.resolve()),
            "targets_sha256": _sha256_file(args.targets),
            "sidecar": str(args.sidecar.resolve()),
            "sidecar_scores_sha256": _sha256_file(args.sidecar / "scores.jsonl"),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256_file(args.checkpoint),
            "selection_start_product": args.start_product,
            "selection_end_product_exclusive": selection_end,
            "product_views": total_products,
            "reactions": total_products // 20,
        },
        "trajectory_counts": {
            "expected": total_trajectories,
            "first_event": first_event_count,
            "first_event_percent": _percent(first_event_count, total_trajectories),
            "no_event": no_event_count,
            "no_event_percent": _percent(no_event_count, total_trajectories),
        },
        "first_event_distance": collector.summary(),
        "sampler_sanity": {
            "max_hazard_relative_error": float(
                center_bias_stats.get("max_hazard_relative_error", 0.0)
            ),
            "neutral_first_batch_bitwise_verified": neutral_verified,
            "streamed_record_count": collector.event_count,
            "retained_record_count": len(center_bias_stats["records"]),
        },
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda" else None
            ),
            "peak_cuda_reserved_bytes": (
                torch.cuda.max_memory_reserved(device)
                if device.type == "cuda" else None
            ),
        },
    }
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    distance = summary["first_event_distance"]
    print(f"Saved: {args.output_json}")
    print(
        f"{args.condition}: events={first_event_count}/{total_trajectories} | "
        f"closer={distance['closer_percent']:.3f}% | "
        f"unchanged={distance['unchanged_percent']:.3f}% | "
        f"farther={distance['farther_percent']:.3f}% | "
        f"mean_improvement={distance['mean_distance_improvement']:.5f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
