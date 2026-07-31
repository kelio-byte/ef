# Edit Flows 用于化学逆合成：迁移方案

## 1. 任务总览

将化学逆合成任务（产物 → 反应物）从原有的 OpenNMT 自回归 Transformer seq2seq 框架迁移到 Edit Flows 非自回归编辑生成框架。

**核心思路**：以 Copy Product 为耦合先验（x₀ = 产物 SMILES，x₁ = 反应物 SMILES），利用 Edit Flows 的 CTMC 编辑操作（插入/删除/替换）从产物逐步编辑出反应物。模型只需一个双向 Transformer encoder，不需要 encoder-decoder 结构。

**待解决的核心问题**：
- 模型架构：参考 OpenNMT 原项目的架构细节，切换到 Pre-Norm Transformer
- 训练配置：参考 OpenNMT 的 Noam 调度、Adam 参数等
- 数据适配：从 OpenNMT 格式（src-train.txt / tgt-train.txt）构建 Edit Flows 的 Dataset + Coupling

---

## 2. 原项目情况 (OpenNMT R-SMILES)

### 2.1 任务与数据

| 项目 | 详情 |
|------|------|
| 任务 | 逆合成预测：产物 SMILES → 反应物 SMILES |
| 数据 | USPTO-50K，20x 数据增强（不同 root atom），约 80 万条训练数据 |
| 序列长度 | 产物平均 ~45 token，反应物平均 ~50 token，最大 ~162 |
| 词表大小 | **因数据集而异**（62 ~ 257 个 SMILES token），无 BPE/subword |
| 数据格式 | OpenNMT 格式：`src-train.txt` / `tgt-train.txt` + `example.vocab.src`（空格分隔的 token） |
| 多数据集 | 原项目有 20+ 个数据集变体（USPTO-50K / USPTO-ZJU / FlowER 等），词表各不相同 |

### 2.2 模型架构（OpenNMT Transformer Encoder-Decoder）

```
Encoder (6 layers):                 Decoder (6 layers):
  Embedding (d=256)                   Embedding (d=256)
  + Sinusoidal PositionEncoding       + Sinusoidal PositionEncoding
  × 6:                                × 6:
    Pre-Norm LayerNorm                  Pre-Norm LayerNorm
    → MultiHead Self-Attn (8 heads)    → MultiHead Self-Attn (8 heads, causal)
    → Dropout + Residual               → Dropout + Residual
    → Pre-Norm FFN:                    → Pre-Norm LayerNorm
      LayerNorm                        → MultiHead Cross-Attn (8 heads)
      → Linear(256→2048)               → Dropout + Residual
      → ReLU                           → Pre-Norm FFN:
      → Dropout                         LayerNorm
      → Linear(2048→256)               → Linear(256→2048)
      → Dropout + Residual             → ReLU
  Final LayerNorm                      → Dropout
                                       → Linear(2048→256)
                                       → Dropout + Residual
                                     Final LayerNorm
                                     → Linear(256→vocab_size)
```

**关键细节**：
- **Pre-Norm**：每个子层前做 LayerNorm（非 Post-Norm）
- **激活函数**：ReLU（非 GELU）
- **位置编码**：固定 Sinusoidal（非 learnable），`emb * sqrt(d_model)` 缩放
- **参数初始化**：Xavier uniform（gain=1.0）
- **Q/K/V 投影**：三个独立 `Linear(d_model, d_model)`
- **Attention dropout**：0.3（训练 from scratch 设置）
- **总参数量**：encoder ~4.7M + decoder ~10.3M ≈ 15M

### 2.3 训练配置（from scratch）

| 参数 | 值 |
|------|-----|
| 优化器 | Adam (β₁=0.9, β₂=0.998) |
| 学习率调度 | Noam：`lr = d^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))` |
| 学习率 scale factor | 1.0 |
| warmup 步数 | 8000 |
| 峰值学习率 | ~7×10⁻⁴（d=256, warmup=8000） |
| 总步数 | 600,000 |
| batch size | 48 sentences |
| dropout | 0.3 |
| attention dropout | 0.3 |
| 梯度裁剪 | 无 (max_grad_norm=0.0) |
| label smoothing | 0.0 |

---

## 3. 实现修改计划

### 3.1 数据模块（新增）

- 创建 `edit_flows/data/` 目录
- `dataset.py`：读取 `src-train.txt` / `tgt-train.txt`，加载 vocab 文件，token → ID 映射
- 返回 `(x_0=产物_token_ids, x_1=反应物_token_ids)` 对（均为 `List[int]`，长度可变）
- `coupling.py` 中新增 `DatasetCoupling`：直接返回 dataset 中的 (x₀, x₁) 对，不做随机生成
- Edit Flows 词表 = 原词表 token + 3 个特殊 token（PAD=0, BOS=1, GAP=2）。原数据 token ID 需要 +3 偏移

### 3.1.1 词表加载（适配多数据集）

不同数据集词表大小差异显著：

| 数据集系列 | 词表大小示例 |
|-----------|------------|
| USPTO_50K_PtoR | 62 ~ 77 token |
| USPTO_ZJU | 62 ~ 88 token |
| FlowER | 214 ~ 257 token |

统一处理流程：

1. **读取 `example.vocab.src`**：每行格式为 `token\tcount`，按出现顺序赋予 ID（0, 1, 2, ...）
2. **构建 `token2id` 映射**，预留前 3 个位置给特殊 token
3. **计算 `model_vocab_size = len(token2id) + 3`**（PAD + BOS + GAP + 真实 token）
4. **传给模型构造函数**，其他参数（hidden_dim 等）不变

```python
# 伪代码
token2id = {"<PAD>": 0, "<BOS>": 1, "<GAP>": 2}
with open(vocab_path) as f:
    for i, line in enumerate(f):
        token = line.strip().split()[0]  # 第一列是 token
        token2id[token] = i + 3

vocab_size = len(token2id)  # 即 model_vocab_size
```

这样同一个模型代码、同一套超参数可以跑所有数据集，只需换数据路径和 vocab 路径。`hidden_dim=256` 对 65 和 250 的词表都适用——输出头 `Linear(256, vocab_size)` 会自动适配。

### 3.2 模型修改

- 修改 `edit_flows/models/transformer.py`
- 从 PyTorch 默认 `nn.TransformerEncoderLayer`（Post-Norm + GELU）切换到自实现的 Pre-Norm + ReLU
- 实现 `PreNormEncoderLayer`：

```
x_norm = LayerNorm(x)
attn_out = MultiHeadAttn(x_norm, x_norm, x_norm)
x = x + Dropout(attn_out)
x_norm = LayerNorm(x)
ffn_out = Linear_down(ReLU(Dropout(Linear_up(x_norm))))
x = x + Dropout(ffn_out)
```

- 位置编码从 learnable Embedding 改为固定 Sinusoidal + `sqrt(d_model)` 缩放
- 时间嵌入保留（SinusoidalTimeEmbedding + MLP），注入方式不变（加到 token+pos embedding 上）

### 3.3 训练模块适配

- `prepare_batch` / `train_step`：基本不变——x₀/x₁ 来自 DatasetCoupling 而非随机生成
- 对齐策略：用 `opt_align_xs_to_zs`（Levenshtein DP），R-SMILES 保证了产物和反应物共享公共前缀，对齐质量好
- 需要新增一个 PyTorch DataLoader wrapper，处理变长序列的 padding 和 batching
- 新增 `scripts/train_retro.py`：整合数据加载 + 模型构建 + 训练循环

### 3.4 采样/评估模块

- `sample_euler` 基本不变——但 x₀ 不再是空序列，需要传入产物 token IDs
- 采样完成后需要 token ID → SMILES token → SMILES string 的还原
- 评估脚本：计算 top-k exact match accuracy（与 `score.py` 对齐的指标）

---

## 4. 网络与训练参数确定

### 4.1 模型参数

| 参数 | 建议值 | 决策依据 |
|------|--------|---------|
| `vocab_size` | **动态** (len(vocab) + 3) | 从 `example.vocab.src` 读取，PAD/BOS/GAP 占前 3 位；USPTO-50K 典型值 71，FlowER 典型值 ~260 |
| `hidden_dim` | 256 | 对齐 OpenNMT d_model；对 65~260 的词表范围均适用 |
| `num_layers` | 10 | OpenNMT enc(6)+dec(6)=12，去掉 cross-attn 开销，取 10 |
| `num_heads` | 8 | 对齐 OpenNMT；256/8=32 per head |
| `dim_feedforward` | 2048 | 对齐 OpenNMT (8× expansion) |
| `max_seq_len` | 256 | 覆盖最大长度 162 + 对齐膨胀余量 |
| `dropout` | 0.3 | 对齐 OpenNMT train from scratch |
| `attention_dropout` | 0.3 | 对齐 OpenNMT |
| `norm_style` | **Pre-Norm** | 对齐 OpenNMT；训练更稳定 |
| `activation` | **ReLU** | 对齐 OpenNMT FFN |
| `pos_encoding` | **Sinusoidal 固定** | 对齐 OpenNMT；支持长度外推 |
| `pos_encoding_scale` | **`emb * sqrt(d)`** | 对齐 OpenNMT |
| `time_embedding` | Sinusoidal + 2-layer MLP (SiLU) | Edit Flows 现有设计，不改 |

**参数量估算**：~12M（10 层, d=256, 以 USPTO-50K 词表 71 为例）。词表从 71 到 260 对总参数量影响很小（embedding + 三个输出头各增加约 0.05M）。比 Edit Flows demo (~17M) 小，比 OpenNMT (~15M) 也略小——对 80 万条数据合理。

### 4.2 训练参数

| 参数 | 建议值 | 决策依据 |
|------|--------|---------|
| 优化器 | Adam (β₁=0.9, β₂=0.998) | 对齐 OpenNMT |
| 学习率调度 | **Noam** | 对齐 OpenNMT |
| 学习率 scale factor | 1.0 | 对齐 OpenNMT |
| warmup_steps | 8000 | 对齐 OpenNMT |
| 峰值学习率 | ~7×10⁻⁴ | 由 Noam 公式自动确定 |
| 总步数 | **600,000** | 对齐 OpenNMT from scratch；约 96 epoch |
| batch_size | 128 | 序列短（~50 token），可用大 batch |
| 梯度裁剪 | **无 (max_grad_norm=0.0)** | 初值对齐 OpenNMT |
| κ_t 调度器 | CubicScheduler (κ_t = t³) | Edit Flows 论文默认 |
| 耦合 | DatasetCoupling (x₀=产物, x₁=反应物) | Copy Product 先验 |
| 对齐 | opt_align_xs_to_zs (Levenshtein DP) | 最优编辑距离对齐 |

### 4.3 Noam 与 Cubic 双调度器说明

两个调度器独立工作，互不冲突：

- **Noam**：控制优化器学习率，作用于梯度更新的步长
- **CubicScheduler (κ_t = t³)**：控制 CTMC 的时间进程，决定 x_t 中来自产物的信息和来自反应物的信息的混合比例，以及 Bregman loss 中的 `κ̇/(1-κ)` 权重

### 4.4 关键差异摘要

| 方面 | 原 OpenNMT | 新 Edit Flows | 原因 |
|------|-----------|---------------|------|
| 架构 | Encoder-Decoder (15M) | 单 Encoder (12M) | Copy Product 不需要 encoder |
| 生成 | 自回归 Beam Search | Euler 迭代精炼 | Edit Flows 的核心优势 |
| Norm | Pre-Norm | **改为 Pre-Norm** | 对齐 OpenNMT，训练稳定 |
| 激活 | ReLU | **改为 ReLU** | 对齐 OpenNMT |
| 位置编码 | Sinusoidal | **改为 Sinusoidal** | 对齐 OpenNMT |
| 词表 | 68 (共享) | 71 (+PAD/BOS/GAP) | Edit Flows 框架需要 |

---

## 5. 后续可能调整方向

由于网络和训练参数均为参考两个项目推测而来，训练过程中可能需要以下调整：

### 5.1 Loss 相关

| 现象 | 可能原因 | 调整方案 |
|------|---------|---------|
| **Loss spike / NaN** | Bregman loss 中 `κ̇/(1-κ)` 在 t→1 时发散（被 clamp 到 50 仍可能很大） | 加回 `max_grad_norm=1.0` |
| **Loss 下降太慢** | 学习率太低；Noam 峰值仅 ~7e-4 | 提高 lr scale factor 到 2.0，或改用固定 lr (如 1e-3) + linear decay |
| **u_tot 项主导、CE 项不降** | 编辑操作学不到；对齐失败 | 检查产物-反应物对齐质量；尝试 `shifted_align` 或 `naive_align` |
| **只有 Insert、无 Delete/Sub 学不到** | 产物和反应物对齐后主要是 GAP→token（插入），模型退化 | 检查编辑类型分布；若 Insert 占 >80%，Copy Product 可能太接近目标，考虑 ExtendedCoupling（随机插入噪声） |

### 5.2 模型相关

| 现象 | 可能原因 | 调整方案 |
|------|---------|---------|
| **训练不稳定、loss 震荡** | Pre-Norm + Noam warmup 不够 | 增大 warmup 到 16000 |
| **验证集指标不收敛** | 模型容量不够 | 增大 `hidden_dim` 到 512 或 `num_layers` 到 12 |
| **过拟合（train loss 很低但 val 差）** | 80 万条对 12M 参数太小 | 增大 dropout 到 0.4；减小 batch_size 增加噪声 |
| **生成序列质量差** | 采样步数不够 | 增大 `n_sampling_steps` 到 200+ |

### 5.3 生成/评估相关

| 现象 | 可能原因 | 调整方案 |
|------|---------|---------|
| **生成的 SMILES 不合法** | 模型没学到 SMILES 语法约束 | 在采样时加 validity filter；或考虑在 loss 中加入语法约束 |
| **Top-1 准确率远低于 OpenNMT** | Edit Flows 的随机性；非自回归 vs 自回归的固有差距 | 增加 n_best（多次采样取最优）；加 Classifier-Free Guidance |
| **根原子选择敏感** | 与 R-SMILES 不同，Edit Flows 对产物根原子的选择可能更敏感 | 测试时保留原始 20x 增强策略，多次采样取频率最高者 |

### 5.4 其他可尝试的改进

- **CFG (Classifier-Free Guidance)**：论文提到能显著提升质量。在 `sample_euler` 中分别计算 conditioned/unconditioned rates 后组合。实现量约 50 行
- **ExtendedCoupling**：如果 Copy Product 太简单（编辑距离太小），在反应物中随机插入噪声 token，增加编辑多样性
- **Gradient accumulation**：如果显存不够 128 batch，用 accum_count=2 或 4 等效
- **大词表适配**：若以后用到 BPE 或更大化学词表（>1000），需同步增大 `hidden_dim` 到 512+（当前 d=256 / V≈260, dim_per_head=32 在输出头可能成为瓶颈）

---

## 6. 实现文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `edit_flows/data/__init__.py` | 新增 | 数据模块导出 |
| `edit_flows/data/dataset.py` | 新增 | RetroDataset: 读取 src/tgt 文件, token→ID |
| `edit_flows/data/coupling.py` | 新增 | DatasetCoupling: 从 dataset 返回 (x₀, x₁) |
| `edit_flows/models/transformer.py` | **修改** | 切换 Pre-Norm, ReLU, Sinusoidal PE |
| `scripts/train_retro.py` | 新增 | 逆合成训练入口 |
| `scripts/sample_retro.py` | 新增 | 逆合成采样入口 |
| `configs/retro.yaml` | 新增 | 逆合成训练超参数配置（不含 vocab_size，由数据模块自动推断） |
| `tests/` | 视情况 | 新模块的单元测试 |
