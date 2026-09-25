"""Train from scratch, then evaluate a checkpoint on the entire test split."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=ROOT / "training_runs")
    parser.add_argument(
        "--evaluate-step",
        type=int,
        default=500000,
        help="Checkpoint to test after training (official method: 500000)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-products",
        type=int,
        help="Smoke tests only; omit to score the complete test split",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["retro"]
    total_steps = int(config["total_steps"])
    interval = int(config["checkpoint_interval"])
    keep = int(config["keep_checkpoints"])
    step = args.evaluate_step
    if not 0 < step <= total_steps:
        raise ValueError("--evaluate-step must be between 1 and total_steps")
    if interval < 1 or keep < 1:
        raise ValueError("checkpoint_interval and keep_checkpoints must be positive")
    if step != total_steps and step % interval:
        raise ValueError("--evaluate-step must be a checkpoint interval or total_steps")
    newer = (total_steps // interval) - (step // interval)
    if total_steps % interval and step != total_steps:
        newer += 1  # train.py also saves the final checkpoint
    if newer >= keep:
        raise ValueError(
            f"step {step} would be pruned before testing; raise keep_checkpoints above {newer}"
        )
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.max_products is not None and (
        args.max_products <= 0 or args.max_products % 20
    ):
        raise ValueError("--max-products must be positive and divisible by 20")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root.resolve() / f"full_{run_id}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), child_env.get("PYTHONPATH", "")) if part
    )
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Training from scratch to step {total_steps}", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train.py"),
            "--config",
            str(config_path),
            "--device",
            args.device,
            "--save_dir",
            str(run_dir),
        ],
        cwd=ROOT,
        env=child_env,
        check=True,
    )

    checkpoints = list(run_dir.rglob(f"checkpoint_step{step}.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected one freshly trained step-{step} checkpoint in {run_dir}; "
            f"found {len(checkpoints)}"
        )
    checkpoint = checkpoints[0]
    test_dir = run_dir / f"test_step{step}"
    print(f"Evaluating full test from {checkpoint}", flush=True)
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate.py"),
        "--split",
        "test",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(test_dir),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
    ]
    if args.max_products is not None:
        print(f"Smoke-test limit: {args.max_products} products", flush=True)
        command.extend(["--max-products", str(args.max_products)])
    subprocess.run(command, cwd=ROOT, env=child_env, check=True)
    print(f"Completed. Metrics: {test_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
