"""Checkpoint loading and the frozen product-major, run-major sampling layout."""

from pathlib import Path
import hashlib
import json
import time
import torch
from tqdm import tqdm
from .data.dataset import load_vocab
from .models.transformer import EditFlowsTransformer
from .core.scheduler import CubicScheduler
from .sampling.r9 import sample_r9
from .sampling.r9_helpers import _mix_child_seed
from .utils.tokens import PAD_TOKEN, BOS_TOKEN, UNK_TOKEN

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/uspto50k_m500"
CHECKPOINT = ROOT / "checkpoints/product_memory_m500_step500000.pt"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint, vocab, device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    required = {
        "use_product_memory": True,
        "use_rate_reparam": False,
        "time_input": "t",
        "scheduler": "cubic",
    }
    for key, value in required.items():
        if cfg.get(key) != value:
            raise ValueError(f"Unsupported checkpoint setting {key}: {cfg.get(key)!r}")
    token2id, vocab_size = load_vocab(str(vocab))
    if ckpt["model_vocab"] != vocab_size:
        raise ValueError("Checkpoint and vocabulary sizes differ")
    # Validate the official token order as well as its size.
    manifest = ROOT / "assets.json"
    if manifest.exists():
        expected = json.loads(manifest.read_text())["files"][
            "data/uspto50k_m500/example.vocab.src"
        ]["sha256"]
        if sha256(vocab) != expected:
            raise ValueError(
                "Vocabulary token order differs from the frozen vocabulary"
            )
    names = (
        "hidden_dim",
        "num_layers",
        "num_heads",
        "dim_feedforward",
        "max_seq_len",
        "dropout",
        "attention_dropout",
        "activation",
        "pos_encoding_scale",
        "use_product_memory",
        "product_memory_encoder_layers",
        "product_memory_fusion_after_layers",
    )
    model = EditFlowsTransformer(
        vocab_size=vocab_size, **{k: cfg[k] for k in names}
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, cfg, token2id


def make_batch(products, device):
    batch = torch.full(
        (len(products), max(map(len, products)) + 1), PAD_TOKEN, dtype=torch.long
    )
    batch[:, 0] = BOS_TOKEN
    for i, ids in enumerate(products):
        batch[i, 1 : len(ids) + 1] = torch.tensor(ids, dtype=torch.long)
    return batch.to(device)


def predict(
    products_file,
    output_dir,
    checkpoint=CHECKPOINT,
    vocab=DATA / "example.vocab.src",
    protocol="r9",
    batch_size=32,
    device="cuda",
    max_products=None,
):
    device = torch.device(device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_file = output / "predictions.txt"
    if prediction_file.exists():
        raise FileExistsError(f"Refusing to overwrite {prediction_file}")
    if batch_size != 32:
        raise ValueError("Frozen reproduction batch_size is 32")
    if protocol != "r9":
        raise ValueError("Only the formal R9K1M2 protocol is supported")
    torch.set_float32_matmul_precision("high")
    model, cfg, token2id = load_model(checkpoint, vocab, device)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    products = Path(products_file).read_text().splitlines()
    if max_products is not None:
        if max_products < 1 or max_products > len(products):
            raise ValueError("max_products outside input bounds")
        products = products[:max_products]
    if not products:
        raise ValueError("Empty input file")
    product_ids = [
        [token2id.get(t, UNK_TOKEN) for t in p.strip().split()] for p in products
    ]
    id2token = {i: t for t, i in token2id.items()}
    scheduler = CubicScheduler()
    started = time.perf_counter()
    with prediction_file.open("w") as out:
        for start in tqdm(range(0, len(products), batch_size), desc="Sampling"):
            batch = product_ids[start : start + batch_size]
            x_unique = make_batch(batch, device)
            x_0 = x_unique.repeat_interleave(9, dim=0)
            mask = x_unique == PAD_TOKEN
            with torch.no_grad():
                memory = model.encode_product(x_unique, mask).repeat_interleave(
                    9, dim=0
                )
            mask = mask.repeat_interleave(9, dim=0)
            kwargs = dict(
                product_memory=memory,
                product_memory_padding_mask=mask,
                n_steps=100,
                max_seq_len=cfg["max_seq_len"],
            )
            seeds = [
                _mix_child_seed(42, start + i, r + 1)
                for i in range(len(batch))
                for r in range(9)
            ]
            result = sample_r9(model, x_0, scheduler, seeds, **kwargs)
            for row in result.cpu().tolist():
                out.write(
                    " ".join(
                        id2token.get(i, "<UNK>")
                        for i in row
                        if i not in (PAD_TOKEN, BOS_TOKEN)
                    )
                    + "\n"
                )
    metadata = {
        "protocol": protocol,
        "seed": 42,
        "n_steps": 100,
        "n_runs": 9,
        "outputs_per_product": 9,
        "n_products": len(products),
        "augmentation": 20,
        "batch_size": batch_size,
        "scheduler": "cubic",
        "matmul_precision": torch.get_float32_matmul_precision(),
        "checkpoint_sha256": sha256(checkpoint),
        "products_sha256": sha256(products_file),
        "vocab_sha256": sha256(vocab),
        "predictions_sha256": sha256(prediction_file),
        "seconds": time.perf_counter() - started,
        "torch": str(torch.__version__),
    }
    if protocol == "r9":
        metadata.update(
            n_branches=1,
            n_children=2,
            score_mode="full_probability",
            changed_state_bonus=0.5,
            child_policy="stochastic_noop",
        )
    (output / "sampling_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return prediction_file
