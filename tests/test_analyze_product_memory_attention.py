import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from edit_flows.models.transformer import EditFlowsTransformer


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable",
    )),
])
def test_product_memory_attention_cli_exports_records_and_figures(tmp_path, device):
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vocab = data_dir / "example.vocab.src"
    vocab.write_text("C\nO\nN\n")
    products = data_dir / "products.txt"
    products.write_text("C O\nN C\n")
    targets = data_dir / "targets.txt"
    targets.write_text("O C\nC N\n")

    config = {
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
        "n_sampling_steps": 3,
    }
    model = EditFlowsTransformer(vocab_size=7, **{
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
        "model_vocab": 7,
        "model_state_dict": model.state_dict(),
    }, checkpoint)

    output_dir = tmp_path / "attention"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)
    environment["OMP_NUM_THREADS"] = "1"
    environment["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_product_memory_attention.py",
            "--checkpoint", str(checkpoint),
            "--products-file", str(products),
            "--targets-file", str(targets),
            "--vocab-file", str(vocab),
            "--output-dir", str(output_dir),
            "--augmentation", "1",
            "--num-products", "2",
            "--num-trajectories", "2",
            "--n-steps", "3",
            "--snapshot-steps", "0,1,2",
            "--max-case-plots", "1",
            "--key-position-bins", "4",
            "--device", device,
            "--seed", "42",
        ],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "Captured 24 attention records." in completed.stdout

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["attention_definition"].startswith("Per-head post-softmax")
    assert metadata["capture"]["records"] == 24
    assert metadata["sampling"]["snapshot_steps"] == [0, 1, 2]
    assert metadata["sampling"]["num_reactions"] == 2
    assert metadata["sampling"]["captured_trajectories"] == 4
    assert metadata["capture"]["summary_unit"] == "reaction (trajectory-averaged)"
    assert metadata["model"]["fusion_after_layers"] == [1, 2]

    try:
        captures = torch.load(
            output_dir / "attention_records.pt",
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        captures = torch.load(output_dir / "attention_records.pt", map_location="cpu")
    assert len(captures["records"]) == 24
    record = captures["records"][0]
    assert record["attention_weights"].shape[:2] == (4, 3)
    assert torch.allclose(
        record["attention_weights"].sum(dim=-1),
        torch.ones_like(record["attention_weights"].sum(dim=-1)),
    )
    assert torch.equal(record["memory_tokens"], record["state_tokens"])

    aggregate = list(csv.DictReader((output_dir / "attention_aggregate.csv").open()))
    assert len(aggregate) == 6
    assert {row["n"] for row in aggregate} == {"2"}
    assert {row["raw_n"] for row in aggregate} == {"4"}

    for filename in (
        "attention_summary.csv",
        "attention_aggregate.csv",
        "attention_head_aggregate.csv",
        "global_attention_dynamics.png",
        "global_memory_position_mass.png",
        "head_attention_summary.png",
        "case_input000000_trajectory00_step001.png",
        "report.md",
    ):
        assert (output_dir / filename).is_file()
        assert (output_dir / filename).stat().st_size > 0
