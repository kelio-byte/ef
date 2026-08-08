#!/usr/bin/env python
"""Generate retrosynthesis predictions and score them through one safe CLI.

The sampling and scoring algorithms remain in ``sample_retro.py`` and
``score_#global#.py``.  This entry point shares their dataset/output settings,
derives the scoring layout from sampling metadata, and prevents accidental
overwrites by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_SCRIPT = SCRIPT_DIR / "sample_retro.py"
SCORE_SCRIPT = SCRIPT_DIR / "score_#global#.py"


def _add_value(command: list[str], name: str, value: Any) -> None:
    if value is not None:
        command.extend((name, str(value)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample retrosynthesis predictions and immediately score Top-1 "
            "through Top-N accuracy with validated layout metadata."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("shared data and output")
    data.add_argument("--checkpoint", required=True)
    data.add_argument("--products_file", required=True)
    data.add_argument("--targets", required=True)
    data.add_argument("--output_dir", required=True)
    data.add_argument("--augmentation", type=int, default=20)
    data.add_argument("--start_product", type=int, default=0)
    data.add_argument("--max_products", type=int, default=None)
    data.add_argument("--data_dir", default=None)
    data.add_argument("--vocab_file", default=None)
    data.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow sampling to replace predictions/metadata in output_dir",
    )
    data.add_argument(
        "--score_only",
        action="store_true",
        help="Skip sampling and score an existing predictions/metadata pair",
    )
    data.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the commands without executing them",
    )

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument(
        "--sampler",
        choices=["euler", "euler_beam", "greedy_edit", "beam_edit"],
        default="euler_beam",
    )
    sampling.add_argument("--n_steps", type=int, default=100)
    sampling.add_argument("--n_samples", type=int, default=3)
    sampling.add_argument("--batch_size", type=int, default=64)
    sampling.add_argument("--device", default="cuda")
    sampling.add_argument("--seed", type=int, default=42)
    sampling.add_argument("--scheduler", choices=["cubic", "linear"])
    sampling.add_argument(
        "--guidance_checkpoint",
        default=None,
        help="Optional action-level guidance checkpoint (ordinary Euler only)",
    )
    sampling.add_argument(
        "--guidance_beta",
        type=float,
        default=1.0,
        help="Exponent applied to action guidance weights",
    )
    sampling.add_argument("--n_branches", type=int, default=3)
    sampling.add_argument("--n_children", type=int, default=2)
    sampling.add_argument("--n_runs", type=int, default=3)
    sampling.add_argument(
        "--euler_beam_initial_seed_groups", type=int, default=None,
    )
    sampling.add_argument(
        "--euler_beam_score_mode",
        choices=["full_probability", "legacy_triggered_reverse"],
        default="full_probability",
    )
    sampling.add_argument(
        "--euler_beam_changed_state_bonus", type=float, default=0.5,
    )
    sampling.add_argument(
        "--euler_beam_q_temperature", type=float, default=1.0,
    )
    sampling.add_argument(
        "--euler_beam_matmul_precision",
        choices=["highest", "high"],
        default="high",
    )
    sampling.add_argument(
        "--euler_beam_child_policy",
        choices=["stochastic", "stochastic_noop"],
        default="stochastic_noop",
    )
    sampling.add_argument("--euler_beam_profile", action="store_true")
    sampling.add_argument(
        "--euler_beam_share_identical_forwards", action="store_true",
    )

    edit = parser.add_argument_group("greedy/beam-edit sampling")
    edit.add_argument("--edit_beam_size", type=int, default=5)
    edit.add_argument("--max_edits", type=int, default=20)
    edit.add_argument(
        "--time_policy",
        choices=["depth", "fixed", "ratio", "kappa"],
        default="depth",
    )
    edit.add_argument("--time_const", type=float, default=0.5)
    edit.add_argument("--k_ins_token", type=int, default=4)
    edit.add_argument("--k_sub_token", type=int, default=4)
    edit.add_argument("--k_edit_expand", type=int, default=16)
    edit.add_argument("--stop_u_tot_base", type=float, default=-1.0)
    edit.add_argument("--explicit_stop", action="store_true")
    edit.add_argument(
        "--kappa_mode",
        choices=["ratio", "frozen_hazard", "poisson"],
        default="ratio",
    )
    edit.add_argument(
        "--p_stop_mode", choices=["absolute", "normalized"],
        default="absolute",
    )
    edit.add_argument("--fh_warmup_steps", type=int, default=0)

    scoring = parser.add_argument_group("scoring")
    scoring.add_argument("--n_best", type=int, default=10)
    scoring.add_argument("--score_alpha", type=float, default=1.0)
    scoring.add_argument(
        "--aggregation_mode",
        choices=["legacy_best_rank", "rrf", "frequency_first", "hybrid"],
        default="legacy_best_rank",
    )
    scoring.add_argument("--process_number", type=int, default=16)
    scoring.add_argument("--sources", default="")
    scoring.add_argument("--synthon", action="store_true")
    scoring.add_argument("--detailed", action="store_true")
    scoring.add_argument("--raw", action="store_true")
    scoring.add_argument("--save_file", default="")
    scoring.add_argument("--save_accurate_indices", default="")
    scoring.add_argument(
        "--no_diagnostics",
        action="store_true",
        help="Disable the standard coverage/invalid/diversity diagnostics",
    )
    scoring.add_argument(
        "--diagnostics_json",
        default=None,
        help="Defaults to OUTPUT_DIR/diagnostics.json",
    )
    return parser


def build_sample_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(SAMPLE_SCRIPT),
        "--checkpoint", args.checkpoint,
        "--products_file", args.products_file,
        "--output_dir", args.output_dir,
        "--start_product", str(args.start_product),
        "--n_steps", str(args.n_steps),
        "--n_samples", str(args.n_samples),
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--seed", str(args.seed),
        "--sampler", args.sampler,
        "--n_branches", str(args.n_branches),
        "--n_children", str(args.n_children),
        "--n_runs", str(args.n_runs),
        "--euler_beam_score_mode", args.euler_beam_score_mode,
        "--euler_beam_changed_state_bonus",
        str(args.euler_beam_changed_state_bonus),
        "--euler_beam_q_temperature",
        str(args.euler_beam_q_temperature),
        "--euler_beam_matmul_precision",
        args.euler_beam_matmul_precision,
        "--euler_beam_child_policy", args.euler_beam_child_policy,
        "--beam_size", str(args.edit_beam_size),
        "--max_edits", str(args.max_edits),
        "--time_policy", args.time_policy,
        "--time_const", str(args.time_const),
        "--k_ins_token", str(args.k_ins_token),
        "--k_sub_token", str(args.k_sub_token),
        "--k_edit_expand", str(args.k_edit_expand),
        "--stop_u_tot_base", str(args.stop_u_tot_base),
        "--kappa_mode", args.kappa_mode,
        "--p_stop_mode", args.p_stop_mode,
        "--fh_warmup_steps", str(args.fh_warmup_steps),
    ]
    _add_value(command, "--max_products", args.max_products)
    _add_value(command, "--data_dir", args.data_dir)
    _add_value(command, "--vocab_file", args.vocab_file)
    _add_value(command, "--scheduler", args.scheduler)
    if args.guidance_checkpoint is not None:
        command.extend((
            "--guidance_checkpoint", args.guidance_checkpoint,
            "--guidance_beta", str(args.guidance_beta),
        ))
    _add_value(
        command,
        "--euler_beam_initial_seed_groups",
        args.euler_beam_initial_seed_groups,
    )
    if args.euler_beam_profile:
        command.append("--euler_beam_profile")
    if args.euler_beam_share_identical_forwards:
        command.append("--euler_beam_share_identical_forwards")
    if args.explicit_stop:
        command.append("--explicit_stop")
    return command


def derive_score_layout(
    metadata: dict[str, Any], augmentation: int,
) -> tuple[int, int, int]:
    """Return ``(beam_size, reaction_count, target_offset)``."""
    if augmentation <= 0:
        raise ValueError("augmentation must be positive")
    metadata_augmentation = metadata.get("augmentation")
    if (metadata_augmentation is not None
            and metadata_augmentation != augmentation):
        raise ValueError(
            "augmentation does not match sampling metadata: "
            f"{augmentation} != {metadata_augmentation}"
        )

    beam_size = int(metadata["output_beam_size"])
    product_count = int(metadata["product_count"])
    start_product = int(
        metadata.get("input", {}).get("selection_start_product", 0)
    )
    if beam_size <= 0:
        raise ValueError("metadata output_beam_size must be positive")
    if product_count <= 0 or product_count % augmentation:
        raise ValueError(
            "sampled product_count must contain complete augmentation "
            f"blocks: {product_count} % {augmentation} != 0"
        )
    if start_product < 0 or start_product % augmentation:
        raise ValueError(
            "selection_start_product must start at an augmentation block: "
            f"{start_product} % {augmentation} != 0"
        )
    return (
        beam_size,
        product_count // augmentation,
        start_product // augmentation,
    )


def build_score_command(
    args: argparse.Namespace,
    *,
    beam_size: int,
    reaction_count: int,
    target_offset: int,
) -> list[str]:
    diagnostics_json = (
        args.diagnostics_json
        if args.diagnostics_json is not None
        else str(Path(args.output_dir) / "diagnostics.json")
    )
    command = [
        sys.executable,
        str(SCORE_SCRIPT),
        "--predictions", str(Path(args.output_dir) / "predictions.txt"),
        "--targets", args.targets,
        "--augmentation", str(args.augmentation),
        "--beam_size", str(beam_size),
        "--n_best", str(args.n_best),
        "--length", str(reaction_count),
        "--target_offset", str(target_offset),
        "--process_number", str(args.process_number),
        "--score_alpha", str(args.score_alpha),
        "--aggregation_mode", args.aggregation_mode,
    ]
    _add_value(command, "--sources", args.sources or None)
    _add_value(command, "--save_file", args.save_file or None)
    _add_value(
        command, "--save_accurate_indices",
        args.save_accurate_indices or None,
    )
    if args.synthon:
        command.append("--synthon")
    if args.detailed:
        command.append("--detailed")
    if args.raw:
        command.append("--raw")
    if not args.no_diagnostics:
        command.extend(("--diagnostics", "--diagnostics_json", diagnostics_json))
    return command


def _load_metadata(output_dir: Path) -> dict[str, Any]:
    metadata_path = output_dir / "sampling_metadata.json"
    prediction_path = output_dir / "predictions.txt"
    if not prediction_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "expected both predictions.txt and sampling_metadata.json in "
            f"{output_dir}"
        )
    with metadata_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_requested_interval(args: argparse.Namespace) -> None:
    if args.augmentation <= 0:
        raise ValueError("augmentation must be positive")
    if args.start_product < 0:
        raise ValueError("start_product must be non-negative")
    if args.max_products is not None and args.max_products <= 0:
        raise ValueError("max_products must be positive")
    if args.start_product % args.augmentation:
        raise ValueError(
            "start_product must align with a complete augmentation block"
        )
    if (args.max_products is not None
            and args.max_products % args.augmentation):
        raise ValueError(
            "max_products must contain complete augmentation blocks"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_requested_interval(args)
    output_dir = Path(args.output_dir)
    prediction_path = output_dir / "predictions.txt"
    metadata_path = output_dir / "sampling_metadata.json"

    if not args.score_only and not args.overwrite:
        existing = [path for path in (prediction_path, metadata_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing evaluation artifacts; use "
                f"--overwrite explicitly: {', '.join(map(str, existing))}"
            )

    sample_command = build_sample_command(args)
    if not args.score_only:
        print("Sampling command:", shlex.join(sample_command), flush=True)
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(sample_command, check=True)

    if args.dry_run and not args.score_only:
        if args.max_products is None:
            print(
                "Scoring command depends on generated metadata; provide "
                "--max_products to preview it during --dry_run."
            )
            return 0
        beam_size = (
            args.n_runs * args.n_branches
            if args.sampler == "euler_beam" else args.n_samples
        )
        reaction_count = args.max_products // args.augmentation
        target_offset = args.start_product // args.augmentation
    else:
        metadata = _load_metadata(output_dir)
        beam_size, reaction_count, target_offset = derive_score_layout(
            metadata, args.augmentation,
        )

    score_command = build_score_command(
        args,
        beam_size=beam_size,
        reaction_count=reaction_count,
        target_offset=target_offset,
    )
    print("Scoring command:", shlex.join(score_command), flush=True)
    if not args.dry_run:
        subprocess.run(score_command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
