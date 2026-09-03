"""End-to-end CPU DDP regression test for the retrosynthesis training entrypoint."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_aligned_split(data_dir: Path, split: str, source: list[str], target: list[str]) -> None:
    split_dir = data_dir / split
    split_dir.mkdir(parents=True)
    (split_dir / f"{split}_aligned_src.txt").write_text("\n".join(source) + "\n")
    (split_dir / f"{split}_aligned_tgt.txt").write_text("\n".join(target) + "\n")


@pytest.mark.parametrize(
    ("device", "nproc", "backend"),
    [
        pytest.param(
            "cpu", 2, "gloo",
            marks=pytest.mark.skipif(
                not torch.distributed.is_available()
                or not torch.distributed.is_gloo_available(),
                reason="PyTorch Gloo distributed backend is unavailable",
            ),
        ),
        pytest.param(
            "cuda", 1, "single_process",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable",
            ),
        ),
    ],
)
def test_train_retro_distributed_cpu_and_single_gpu_write_portable_checkpoints(
    tmp_path, device, nproc, backend,
):
    """Exercise DDP sharding on CPU and the real one-GPU training path."""
    data_dir = tmp_path / "tiny_retro"
    data_dir.mkdir()
    (data_dir / "example.vocab.src").write_text("A\nB\nC\n")
    _write_aligned_split(
        data_dir,
        "train",
        ["A", "B", "C", "A B", "B C", "C A", "A C", "B A"],
        ["B", "C", "A", "B A", "C B", "A C", "C A", "A B"],
    )
    _write_aligned_split(
        data_dir,
        "val",
        ["A", "B", "C", "A B", "B C"],
        ["B", "C", "A", "B A", "C B"],
    )

    config_path = tmp_path / f"{device}.yaml"
    config_path.write_text(textwrap.dedent(f"""\
        retro:
          vocab_size: null
          hidden_dim: 8
          num_layers: 1
          num_heads: 2
          dim_feedforward: 16
          max_seq_len: 16
          dropout: 0.0
          attention_dropout: 0.0
          activation: relu
          pos_encoding_scale: true
          batch_size: 2
          val_batch_size: 2
          total_steps: 2
          learning_rate_factor: 1.0
          warmup_steps: 2
          max_grad_norm: 0.0
          scheduler: cubic
          sample_scheduler: cubic
          align_fn: opt
          n_sampling_steps: 2
          use_rate_reparam: false
          time_input: t
          clamp_kappa: false
          clamp_max: 50.0
          use_origin_mask: false
          data_dir: {data_dir.as_posix()}
          vocab_file: example.vocab.src
          seed: 123
          num_workers: 0
          checkpoint_interval: 1
          keep_checkpoints: 2
          save_best_checkpoint: true
          tensorboard:
            enabled: false
            log_interval: 1
            validation_start_step: 1
            validation_interval: 1
            validation_batches: null
            flush_interval: 1
          monitoring:
            enabled: false
            interval: 1
        """))

    output_dir = tmp_path / "checkpoints"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(REPO_ROOT)]
    )
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    command = [sys.executable]
    if nproc > 1:
        command.extend([
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc}",
        ])
    command.extend([
        "scripts/train_retro.py",
        "--config", str(config_path),
        "--device", device,
        "--save_dir", str(output_dir),
    ])
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout

    checkpoints = list(output_dir.rglob("checkpoint_step2.pt"))
    assert len(checkpoints) == 1
    checkpoint_path = checkpoints[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["completed_steps"] == 2
    assert checkpoint["training_topology"] == {
        "world_size": nproc,
        "batch_size_per_rank": 2,
        "effective_global_batch_size": 2 * nproc,
        "backend": backend,
    }
    assert len(checkpoint["rng_state_by_rank"]) == nproc
    if device == "cuda":
        assert "cuda_device_rng" in checkpoint["rng_state_by_rank"][0]
    assert all(not key.startswith("module.") for key in checkpoint["model_state_dict"])
    assert checkpoint_path.with_name("checkpoint_best.pt").is_file()
    assert checkpoint_path.with_name("train.log").is_file()

    if device == "cpu":
        # A DDP checkpoint must resume using the same rank-local data offset
        # and RNG-state list rather than reusing rank 0's stream everywhere.
        config_path.write_text(
            config_path.read_text().replace("total_steps: 2", "total_steps: 4")
        )
        resume_command = command + ["--checkpoint", str(checkpoint_path)]
        resumed = subprocess.run(
            resume_command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=90,
        )
        assert resumed.returncode == 0, resumed.stdout
        resumed_checkpoints = list(output_dir.rglob("checkpoint_step4.pt"))
        assert len(resumed_checkpoints) == 1
        resumed_checkpoint = resumed_checkpoints[0]
        resumed_state = torch.load(
            resumed_checkpoint, map_location="cpu", weights_only=False,
        )
        assert resumed_state["completed_steps"] == 4
        assert len(resumed_state["rng_state_by_rank"]) == 2
