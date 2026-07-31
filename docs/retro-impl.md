# Edit Flows 用于化学逆合成：实现文档

## 1. 概述

基于 [Edit Flows](https://arxiv.org/abs/2506.09018)（CTMC + 编辑操作的非自回归序列生成框架），将化学逆合成任务（产物 SMILES → 反应物 SMILES）从 OpenNMT 自回归 seq2seq 迁移到编辑生成范式。

**核心思路**：Copy Product 耦合——以产物 SMILES 作为流起点 x₀，目标反应物 SMILES 作为 x₁，模型学习如何通过插入/删除/替换操作从产物编辑出反应物。模型为单个双向 Transformer，无需 encoder-decoder。

## 2. 文件结构

```
edit-flows/
├── edit_flows/
│   ├── data/                     # 新增：数据模块
│   │   ├── __init__.py           # 导出
│   │   ├── dataset.py            # RetroDataset, load_vocab, collate_fn
│   │   └── coupling.py           # DatasetCoupling
│   ├── models/
│   │   └── transformer.py        # 重写：Pre-Norm + ReLU + Sinusoidal PE
│   ├── training/
│   │   ├── trainer.py            # 修改：max_grad_norm 可配置
│   │   ├── schedulers.py         # 新增：NoamScheduler
│   │   └── __init__.py           # 导出 NoamScheduler
│   └── __init__.py               # 导出新模块
├── configs/
│   └── retro.yaml                # 新增：逆合成训练超参数
├── scripts/
│   ├── train_retro.py            # 新增：训练入口
│   └── sample_retro.py           # 新增：采样入口
└── docs/
    ├── retro-todo.md             # 迁移方案设计文档
    └── retro-impl.md             # 本文档
```

## 3. 模型架构

### 3.1 EditFlowsTransformer（重写）

单双向 Transformer encoder，从 PyTorch 默认 `TransformerEncoderLayer`（Post-Norm + GELU）切换为对齐 OpenNMT R-SMILES 项目的 Pre-Norm 风格。

```
Input: x_t (token IDs, B×L) + t (time, B×1)

Token Embedding → *√d_model
    + Time Embedding (Sinusoidal → 2-layer SiLU MLP, B×1→B×L×H)
    + Position Encoding (Sinusoidal, fixed, B×L×H)
    = x (B×L×H)

× N layers:
    PreNormEncoderLayer:
        x = x + Dropout( MultiHeadAttn( LayerNorm(x) ) )       ← self-attn
        x = x + Dropout2( Linear2( Dropout1( ReLU( Linear1( LayerNorm(x) ) ) ) ) )  ← FFN

Final LayerNorm

→ rates_out (B×L×3)       ← λ_ins, λ_sub, λ_del (softplus)
→ ins_logits_out (B×L×V)  ← Q_ins (softmax)
→ sub_logits_out (B×L×V)  ← Q_sub (softmax)
```

### 3.2 与原始实现的差异

| 组件 | 原 Edit Flows demo | 新 Retro 实现 | 依据 |
|------|-------------------|--------------|------|
| Norm 位置 | Post-Norm | **Pre-Norm** | OpenNMT |
| 激活函数 | GELU | **ReLU** | OpenNMT |
| 位置编码 | Learnable Embedding | **固定 Sinusoidal** | OpenNMT |
| PE 缩放 | 无 | **emb × √d_model** | OpenNMT |
| Attention Dropout | N/A (同 dropout) | **独立 0.3** | OpenNMT |
| 参数初始化 | Xavier(gain=0.1) | **Xavier(gain=1.0)** | OpenNMT |
| d_model | 512 | **256** | OpenNMT |
| d_ff | 2048 | 2048 | 一致 |
| heads | 8 | 8 | 一致 |
| num_layers | 8 | **10** | enc(6)+dec(6)-2 |
| dropout | 0.1 | **0.3** | OpenNMT from scratch |

参数总量：~13.5M（USPTO-50K 词表 72 时）。词表增大到 ~260 时约 ~14M。

## 4. 训练配置

### 4.1 超参数 (`configs/retro.yaml`)

```yaml
retro:
  # Model
  hidden_dim: 256
  num_layers: 10
  num_heads: 8
  dim_feedforward: 2048
  max_seq_len: 256
  dropout: 0.3
  attention_dropout: 0.3
  activation: relu
  pos_encoding_scale: true

  # Training
  batch_size: 128
  total_steps: 600000
  learning_rate_factor: 1.0       # Noam 缩放因子
  warmup_steps: 8000
  max_grad_norm: 0.0              # 0 = 不裁剪

  # Edit Flows
  scheduler: cubic                # κ_t = t³
  align_fn: opt                   # Levenshtein DP
  n_sampling_steps: 100

  # Data (vocab_size 由 vocab 文件自动推断)
  data_dir: /data6/duanbh/desktop/retrosynthesis/dataset/USPTO_50K_PtoR_aug20
  vocab_file: example.vocab.src
```

### 4.2 双调度器

| 调度器 | 作用 | 公式 |
|--------|------|------|
| **NoamScheduler** | 控制 Adam 学习率 | `lr = factor × d⁻⁰·⁵ × min(step⁻⁰·⁵, step × warmup⁻¹·⁵)` |
| **CubicScheduler** | 控制 CTMC 时间 κ_t | `κ_t = t³` |

两者独立运作：Noam 控制梯度更新步长，Cubic 控制 Bregman loss 中的 `κ̇/(1-κ)` 时间权重。

峰值学习率约 7×10⁻⁴（d=256, warmup=8000 时自动确定）。

### 4.3 数据流与 Token 约定

**特殊 Token**：

| Token | ID | 说明 |
|-------|-----|------|
| `PAD_TOKEN` | 0 | 序列填充，模型可见 |
| `BOS_TOKEN` | 1 | 序列起始标记 |
| `GAP_TOKEN` | 2 | Z 空间对齐标记，**模型不可见** |
| `UNK_TOKEN` | 3 | 未知 token，词表外 token 自动映射 |
| 真实 token | 4, 5, ..., V+3 | 从 `example.vocab.src` 加载 |

词汇表从 OpenNMT 的 `example.vocab.src` 加载，自动添加 4 个特殊 token（PAD/BOS/GAP/UNK）。真实 token ID 从 4 开始偏移。模型词表大小 = 真实 token 数 + 4。

`RetroDataset` 对词表外 token 自动映射到 `UNK_TOKEN`，避免验证集/测试集中的未知 token 导致 KeyError。

**数据流**：

```
src-train.txt / tgt-train.txt
    ↓ RetroDataset (token→ID, 未知→UNK)
    ↓ DataLoader + collate_fn (PAD to batch max)
    ↓ prepare_batch: x_0=产物, x_1=反应物
      ├── opt_align_xs_to_zs: strip PAD → Levenshtein DP → pad to uniform
      ├── add BOS prefix, sample t, sample z_t
      └── rm_gap → x_t (no GAP tokens)
    ↓ train_step (forward → ux_cat → fill_gap → Bregman loss)
```

`prepare_batch` 接受 `model_vocab_size`（完整词表大小，含特殊 token），不再需要调用者手动计算 `real_vocab_size + 3`。`opt_align_xs_to_zs` 内部先 strip PAD 再进行 DP 对齐，对齐结果再 PAD 到batch统一长度。不同数据集的词表大小不同（USPTO-50K: 68, FlowER: ~257 等），一套代码适配。

## 5. 使用方法

### 5.1 训练

```bash
cd /data3/duanbh/desktop/edit-flows

# 默认配置训练（USPTO-50K）
PYTHONPATH=. python scripts/train_retro.py --config configs/retro.yaml --device cuda

# 指定其他数据集
PYTHONPATH=. python scripts/train_retro.py --config configs/retro.yaml --device cuda \
    --save_dir ./checkpoints

# 断点续训
PYTHONPATH=. python scripts/train_retro.py --config configs/retro.yaml --device cuda \
    --checkpoint checkpoints/USPTO_50K_PtoR_aug20/2026-05-23_14-30-00/checkpoint_step50000.pt

# 调整保留 checkpoint 数量
PYTHONPATH=. python scripts/train_retro.py --config configs/retro.yaml --device cuda \
    --keep_checkpoints 20
```

### 5.2 CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `configs/retro.yaml` | 训练配置 YAML |
| `--device` | `cpu` | 设备 (`cuda` / `cpu`) |
| `--checkpoint` | `None` | 断点续训的 .pt 路径 |
| `--save_dir` | `None` | 覆盖保存目录（默认 `./checkpoints`） |
| `--keep_checkpoints` | `None` | 保留 checkpoint 数量（默认 10） |

### 5.3 采样

```bash
# 单条产物采样
PYTHONPATH=. python scripts/sample_retro.py \
    --checkpoint checkpoints/.../checkpoint_step600000.pt \
    --product "C O C ( = O ) [C@H] ( C C C C N ) N C ( = O ) N c 1 c c ..." \
    --n_steps 100 --device cuda

# 批量文件采样
PYTHONPATH=. python scripts/sample_retro.py \
    --checkpoint checkpoints/.../checkpoint_step600000.pt \
    --products_file test_products.txt \
    --n_steps 100 --device cuda
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | 必填 | 模型 .pt 路径 |
| `--product` | `None` | 单条 tokenized 产物 SMILES |
| `--products_file` | `None` | 每行一条产物 SMILES 的文件 |
| `--data_dir` | `None` | 词表目录（覆盖 checkpoint 中记录） |
| `--vocab_file` | `None` | 词表文件路径（覆盖默认） |
| `--n_steps` | `100` | Euler 采样步数 |
| `--device` | `cpu` | 设备 |

### 5.4 Checkpoint 目录结构

```
checkpoints/
└── <dataset_name>/
    └── <YYYY-MM-DD_HH-MM-SS>/
        ├── config.yaml                   ← 训练配置副本
        ├── train.log                     ← 完整控制台日志
        ├── checkpoint_step10000.pt
        ├── checkpoint_step20000.pt
        └── ...                           ← 最多保留最新 10 个（可配）
```

### 5.5 切换数据集

修改 `configs/retro.yaml` 中的 `data_dir` 即可：

```yaml
retro:
  data_dir: /data6/duanbh/desktop/retrosynthesis/dataset/USPTO_ZJU_chain_PtoR_aug20_#center#
  vocab_file: example.vocab.src
```

词表大小、模型输出头维度自动适配，无需改代码或调整其他超参。

## 6. 训练监控

每 100 步输出一行到控制台（同时写入 `train.log`）：

```
step     100/600000 | loss: 3.2456 | lr: 7.00e-05 | u_tot:  45.12 | ins:  12.34 | del:  15.67 | sub:  17.11
step     200/600000 | loss: 2.9812 | lr: 1.40e-04 | u_tot:  42.87 | ins:  11.89 | del:  14.92 | sub:  16.06
```

指标含义：

| 指标 | 含义 |
|------|------|
| `loss` | Bregman 散度（u_tot - CE_term），越低越好 |
| `lr` | 当前学习率（Noam warmup → decay） |
| `u_tot` | 总编辑速率之和，应稳定下降 |
| `ins/del/sub` | 插入/删除/替换的平均速率，反映模型偏好 |

## 7. 设计决策 FAQ

### 为什么不需要 Encoder？

Copy Product 耦合使产物信息就在 x_t 中——混合路径 `(1-κ_t)×产物 + κ_t×反应物` 天然携带了产物信息。模型通过双向 self-attention 能看到 x_t 中所有 token，不需要额外的 encoder 来编码产物。

### 为什么用 Sinusoidal 位置编码？

对齐 OpenNMT。相比 learnable embedding：(1) 可外推到训练时未见过的长度，(2) SMILES token 的位置语义更依赖相对位置。R-SMILES 的 root-alignment 保证了产物和反应物共享公共前缀，对齐质量好。

### 为什么取消梯度裁剪？

对齐 OpenNMT（`max_grad_norm=0.0`）。Bregman loss 中的 `κ̇/(1-κ)` 被 clamp 到最大 50，梯度理论上可控。如果出现 loss spike/NAN，可改 `max_grad_norm: 1.0` 重新训练。

### 为什么需要 UNK token？

OpenNMT 的 `example.vocab.src` 仅从训练集构建，验证集/测试集可能包含训练集未出现的 token（如 USPTO-50K 验证集中的 `p`）。添加 UNK_TOKEN 并在 `RetroDataset` 中用 `token2id.get(t, unk_id)` 进行自动映射，避免 KeyError。UNK token 在模型词表中但不会作为目标出现（x_1 不含 UNK），模型通常学会将其删除或替换。
