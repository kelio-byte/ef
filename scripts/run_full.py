"""用途：按配置从零训练，并用指定训练步数的 checkpoint 测试。
输入：训练配置、设备及可选的测试范围参数。
输出：训练文件、全量 test 预测、元数据和评测指标。
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    """作用：解析一键训练评测参数。输入：命令行参数。输出：包含配置、设备和输出路径的参数对象。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=ROOT / "training_run")
    parser.add_argument(
        "--run-name",
        help="Directory name under output-root (default: config-name_MM-DD)",
    )
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
    """作用：串接从零训练和指定 checkpoint 的完整测试。输入：命令行参数及仓库配置。输出：训练文件和 test 评测结果。"""
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

    run_name = args.run_name or f"{config_path.stem}_{datetime.now():%m-%d}"
    if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("--run-name must be one directory name")
    run_dir = args.output_root.resolve() / run_name
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SystemExit(
            f"Run directory already exists: {run_dir}. "
            "Choose --run-name to keep the previous run intact."
        ) from exc
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
            "--run_dir",
            str(run_dir),
        ],
        cwd=ROOT,
        env=child_env,
        check=True,
    )

    checkpoint = run_dir / f"checkpoint_step{step}.pt"
    if not checkpoint.is_file():
        raise RuntimeError(
            f"Freshly trained step-{step} checkpoint was not saved at {checkpoint}"
        )
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
