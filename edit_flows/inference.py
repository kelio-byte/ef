"""用途：加载模型 checkpoint，并按 K=1、M 可配置分支策略生成候选。
输入：产品 token 序列、checkpoint、词表和采样参数。
输出：候选预测文件及采样元数据。
"""

from pathlib import Path
import hashlib
import json
import time
import torch
from tqdm import tqdm
from .data.dataset import load_vocab
from .models.transformer import EditFlowsTransformer
from .core.scheduler import CubicScheduler
from .sampling.branch_sampler import sample_branches
from .sampling.branch_sampler_helpers import _mix_child_seed
from .utils.tokens import PAD_TOKEN, BOS_TOKEN, UNK_TOKEN

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/uspto50k_m500"
CHECKPOINT = ROOT / "saved_checkpoints/product_memory_m500_step500000.pt"


def sha256(path):
    """作用：计算文件校验和。输入：文件路径。输出：SHA-256 十六进制字符串。"""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint, vocab, device):
    """作用：校验并载入模型权重。输入：checkpoint、词表路径和设备。输出：模型、配置及 token 映射。"""
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
    """作用：将 token 列表转换为补 PAD 的模型输入。输入：一批 token 编号列表和设备。输出：含 BOS 的张量批次。"""
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
    n_runs=9,
    batch_size=32,
    device="cuda",
    max_products=None,
    n_children=2,
):
    """作用：按 K=1、M 子候选分支策略采样并写出预测。输入：产品文件、模型资产和采样选项。输出：预测路径，并写入元数据。

    n_runs 控制独立运行数；n_children 控制每步采样的子候选数 M。
    """
    if not isinstance(n_runs, int) or isinstance(n_runs, bool) or n_runs < 1:
        raise ValueError("n_runs must be a positive integer")
    if (
        not isinstance(n_children, int)
        or isinstance(n_children, bool)
        or n_children < 1
    ):
        raise ValueError("n_children must be a positive integer")
    device = torch.device(device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_file = output / "predictions.txt"
    if prediction_file.exists():
        raise FileExistsError(f"Refusing to overwrite {prediction_file}")
    if batch_size != 32:
        raise ValueError("Frozen reproduction batch_size is 32")
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
    changed_state_bonus = 0.5
    started = time.perf_counter()
    with prediction_file.open("w") as out:
        for start in tqdm(range(0, len(products), batch_size), desc="Sampling"):
            batch = product_ids[start : start + batch_size]
            x_unique = make_batch(batch, device)
            x_0 = x_unique.repeat_interleave(n_runs, dim=0)
            mask = x_unique == PAD_TOKEN
            with torch.no_grad():
                memory = model.encode_product(x_unique, mask).repeat_interleave(
                    n_runs, dim=0
                )
            mask = mask.repeat_interleave(n_runs, dim=0)
            kwargs = dict(
                product_memory=memory,
                product_memory_padding_mask=mask,
                n_steps=100,
                max_seq_len=cfg["max_seq_len"],
                n_children=n_children,
                changed_state_bonus=changed_state_bonus,
            )
            seeds = [
                _mix_child_seed(42, start + i, r + 1)
                for i in range(len(batch))
                for r in range(n_runs)
            ]
            result = sample_branches(model, x_0, scheduler, seeds, **kwargs)
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
        "sampling_method": "k1m_state_count",
        "seed": 42,
        "n_steps": 100,
        "n_runs": n_runs,
        "outputs_per_product": n_runs,
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
    metadata.update(
        n_branches=1,
        n_children=n_children,
        score_mode="full_probability",
        changed_state_bonus=changed_state_bonus,
        child_selection="log_occurrence_count_plus_changed_state_bonus",
        child_tie_break="lowest_child_seed",
    )
    (output / "sampling_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return prediction_file
