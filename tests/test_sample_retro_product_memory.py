import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from edit_flows.models.transformer import EditFlowsTransformer


def test_sample_retro_loads_and_runs_a_product_memory_checkpoint(tmp_path):
    """Exercise the public sampling CLI, including its per-product cache."""
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vocab = data_dir / "example.vocab.src"
    vocab.write_text("C\nO\n")
    products = tmp_path / "products.txt"
    products.write_text("C O\n")

    config = {
        "data_dir": str(data_dir),
        "vocab_file": "example.vocab.src",
        "hidden_dim": 16,
        "num_layers": 2,
        "num_heads": 4,
        "dim_feedforward": 32,
        "max_seq_len": 12,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "activation": "relu",
        "pos_encoding_scale": True,
        "use_origin_mask": False,
        "use_product_memory": True,
        "product_memory_encoder_layers": 1,
        "product_memory_fusion_after_layers": [1, 2],
        "scheduler": "cubic",
        "sample_scheduler": "cubic",
    }
    model = EditFlowsTransformer(vocab_size=6, **{
        key: config[key]
        for key in (
            "hidden_dim", "num_layers", "num_heads", "dim_feedforward",
            "max_seq_len", "dropout", "attention_dropout", "activation",
            "pos_encoding_scale", "use_origin_mask", "use_product_memory",
            "product_memory_encoder_layers",
            "product_memory_fusion_after_layers",
        )
    })
    checkpoint = tmp_path / "product_memory.pt"
    torch.save({
        "config": config,
        "model_vocab": 6,
        "model_state_dict": model.state_dict(),
    }, checkpoint)

    output_dir = tmp_path / "out"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/sample_retro.py",
            "--checkpoint", str(checkpoint),
            "--products_file", str(products),
            "--vocab_file", str(vocab),
            "--output_dir", str(output_dir),
            "--sampler", "euler",
            "--n_samples", "2",
            "--n_steps", "2",
            "--batch_size", "1",
            "--device", "cpu",
            "--seed", "42",
        ],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Done. Total predictions: 2" in completed.stdout
    assert len((output_dir / "predictions.txt").read_text().splitlines()) == 2

    metadata = json.loads((output_dir / "sampling_metadata.json").read_text())
    assert metadata["model"] == {
        "configured_use_origin_mask": False,
        "effective_use_origin_mask": False,
        "configured_use_product_memory": True,
        "effective_use_product_memory": True,
        "product_memory_encoder_layers": 1,
        "product_memory_fusion_after_layers": [1, 2],
        "product_memory_sampling_cache": (
            "encode_x0_once_per_input_row_then_repeat_per_trajectory"
        ),
    }
