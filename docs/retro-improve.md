# Edit Flows 逆合成训练优化：GPU 利用率分析与修复

## 问题描述

训练脚本 `scripts/train_retro.py` 运行时 GPU 利用率偏低（~60%），且频繁剧烈波动到 ~10% 后恢复。

## 根因分析

GPU 利用率低的**本质不是 GPU 算不动，而是 CPU 侧的数据预处理把 GPU 喂不饱**。

### 耗时分布（优化前，单步，batch_size=128）

```
GPU forward+backward:   ~30ms  ← 模型只有 13.5M 参数，序列 ~50 token
CPU prepare_batch:     ~100-200ms  ← 占绝对主导
  ├── DP 对齐 ×128:    ~80-150ms   (Levenshtein 编辑距离, Python 双重循环)
  ├── GPU→CPU sync:    ~10-30ms    (.item() 逐条调用触发 cudaMemcpy)
  ├── sample_cond_zt:  ~380ms      (F.one_hot 构建 128×98×72 全量 tensor)
  └── BOS for-loop:    ~5ms        (逐条 .item() GPU sync)
```

GPU 每步只工作 ~30ms 就要等 CPU 100-200ms，利用率自然在 10-60% 间波动。

### 三大瓶颈

1. **`sample_cond_zt` 用 `F.one_hot` 构建 (B, L, V) 全张量**（~380ms）
   - 混合路径 `p_t = (1-κ)·δ(z₀) + κ·δ(z₁)` 每个位置只有 2 个非零项
   - 不需要 (B, L, V) 的 one-hot，一次 Bernoulli 采样即可

2. **`_align_pair` DP 对齐逐条 Python 循环**（~80-150ms）
   - batch_size=128，每次 `.cpu().numpy()` 触发 GPU→CPU 传输
   - Python 双重循环跑 Levenshtein DP

3. **`prepare_batch` / `rm_gap_tokens` 中逐条 for 循环**（~15-35ms）
   - 每次 `.item()` 调用触发 GPU sync
   - 128 次独立的 GPU→CPU 小数据传输

---

## 优化方案

### 优化 1：`sample_cond_zt` Bernoulli 重写

**原理**：条件概率路径 `p_t(z_t|z₀,z₁) = (1-κᵗ)·δ(z₀) + κᵗ·δ(z₁)` 是两次 one-hot 的混合，等价于：以概率 κᵗ 取 z₁，概率 (1-κᵗ) 取 z₀。无需构建完整 (B, L, V) 张量。

```python
# 之前 (~380ms)
p0 = F.one_hot(z_0, vocab_size).float()     # (B, L, V)
p1 = F.one_hot(z_1, vocab_size).float()     # (B, L, V)
pt = (1 - kappa_t) * p0 + kappa_t * p1       # (B, L, V) 大张量运算
z_t = torch.multinomial(pt, 1)              # 从 (B*L, V) 多项式采样

# 之后 (~0.3ms)
kappa_t = kappa_fn(t)
rand = torch.rand_like(z_0, dtype=torch.float)
pick_z1 = rand < kappa_t                     # (B, L) 布尔掩码
z_t = torch.where(pick_z1, z_1, z_0)         # 逐位置 Bernoulli 采样
```

**效果**：~380ms → ~0.3ms，**~1000x 加速**。

### 优化 2：对齐预计算

Copy Product 耦合下，(产物, 反应物) 的 DP 对齐是确定性的——训练时不随 t 改变。因此可以一次性离线计算。

**设计要点**：
- 直接使用字符串 token 进行 Levenshtein DP 对齐（无需 vocab），`<GAP>` 作为对齐占位符
- `multiprocessing.Pool` 并行处理，8 workers 下 ~11.7k pairs/s
- 输出为两个纯文本文件（空格分隔的 token），人类可读、跨词表复用

```bash
PYTHONPATH=. python scripts/precompute_alignments.py \
    --data_dir /data6/duanbh/desktop/retrosynthesis/dataset/USPTO_50K_PtoR_aug20 \
    --num_workers 8
```

生成文件（以 train 为例）：

| 文件 | 大小 | 内容 |
|------|------|------|
| `train/train_aligned_src.txt` | 103 MB | 800k 对齐后产物序列 |
| `train/train_aligned_tgt.txt` | 89 MB | 800k 对齐后反应物序列 |
| **合计** | **192 MB** | vs 旧 `.pt` 格式 1.1 GB（**5.7x 缩小**） |

旧 `.pt` 格式膨胀的原因：160 万个独立 `torch.Tensor` 对象的 pickle 序列化开销（占总大小 ~40%），加上 int64 存储浪费（词表仅 ~72 token）。文本格式避免了这些开销。

训练时，`train_retro.py` 自动检测 `train_aligned_src.txt` + `train_aligned_tgt.txt` 是否存在：
- 存在 → 使用 `PreAlignedDataset`（加载时在线 tokenize）+ `identity_align_xs_to_zs`（跳过 DP）
- 不存在 → 回退到 `RetroDataset` + `opt_align_xs_to_zs`（打印警告）

**效果**：DP 对齐从每步 ~80-150ms 降为 0ms（完全消除）。

### 优化 3：`rm_gap_tokens` 向量化

```python
# 之前：逐条 for b in range(B) → strip PAD → strip GAP → repad
for b in range(batch_size):
    zb = z[b]
    zb_no_pad = zb[zb != PAD_TOKEN]
    zb_no_gap = zb_no_pad[zb_no_pad != GAP_TOKEN]
    z_no_gap.append(zb_no_gap)

# 之后：valid mask → cumsum → 一次 scatter
valid = (z != PAD_TOKEN) & (z != GAP_TOKEN)
dest_col = valid.long().cumsum(dim=1) - 1
row_idx = torch.arange(B).unsqueeze(1).expand(-1, L)
x[row_idx[valid], dest_col[valid]] = z[valid]
```

### 优化 4：`prepare_batch` BOS 插入向量化

```python
# 之前：for b in range(B): sl = int(seq_lens[b].item()); ...
for b in range(batch_size):
    sl = int(seq_lens[b].item())
    z_0_padded[b, 1:sl + 1] = z_0[b, :sl]

# 之后：mask 索引一次完成
cols = torch.arange(L_z).unsqueeze(0).expand(B, -1)
copy_0 = (cols < seq_lens_0.unsqueeze(1))[:, :max_sl]
z_0_padded[:, 1:][copy_0] = z_0[:, :max_sl][copy_0]
```

消除了 ~128 次 `.item()` GPU sync。

### 优化 5：`prepare_batch` 在 CPU 侧执行

```python
# 之前：数据先上 GPU，对齐又从 GPU 拷回 CPU 做 DP
x_0, x_1 = x_0.to(device), x_1.to(device)  # GPU
batch = prepare_batch(x_0, x_1, ...)         # 又拷回 CPU！

# 之后：CPU 上完成全部预处理，再统一上 GPU
batch = prepare_batch(x_0, x_1, ...)         # CPU
batch = {k: v.to(device) for k, v in batch.items()}  # 一次批量传输
```

### 优化 6：DataLoader 配置

- `num_workers`: 0 → 2（并行加载数据）
- `pin_memory`: 新增 `True`（加速 CPU→GPU 传输）

### 辅助修复

| 修复 | 文件 | 说明 |
|------|------|------|
| `UNK_TOKEN = 3` | `utils/tokens.py` | 补充缺失的 token 定义 |
| `train_step` 梯度裁剪 | `training/trainer.py` | 从硬编码 `max_norm=1.0` 改为读取 `max_grad_norm` 配置（0.0 = 跳过裁剪） |
| `prepare_batch` 参数重命名 | `training/trainer.py` | `vocab_size` → `model_vocab_size`，与 `train_retro.py` 对齐 |

---

## 优化效果汇总

| 阶段 | `prepare_batch` 耗时 | 变化 |
|------|---------------------|------|
| 原始 (DP 对齐 + GPU sync + one-hot) | ~100-500ms | — |
| 预计算 + 向量化 + one-hot | ~409ms | 消除 DP，但 one-hot 仍是瓶颈 |
| **最终版本 (全部优化)** | **~4.5ms** | **~50-100x 加速** |

单步总耗时从 ~130-230ms 降为 ~35ms（GPU 计算 ~30ms + CPU 预处理 ~5ms），GPU 利用率预期从 10-60% 波动稳定到 80-95%。

### 涉及的文件

| 文件 | 改动 |
|------|------|
| `edit_flows/core/z_space.py` | `rm_gap_tokens` 向量化；`sample_cond_zt` Bernoulli 重写 |
| `edit_flows/training/trainer.py` | `prepare_batch` BOS 插入向量化；`train_step` 梯度裁剪修复 |
| `edit_flows/core/alignment.py` | 新增 `identity_align_xs_to_zs` |
| `edit_flows/core/__init__.py` | 导出 identity_align |
| `edit_flows/__init__.py` | 导出 identity_align |
| `edit_flows/data/dataset.py` | 新增 `PreAlignedDataset`（读文本对齐文件，在线 tokenize，复用 `collate_fn`）；移除 `collate_tensors` |
| `edit_flows/data/__init__.py` | 导出新类；移除 `collate_tensors` |
| `edit_flows/utils/tokens.py` | 新增 `UNK_TOKEN = 3` |
| `edit_flows/utils/__init__.py` | 导出 UNK_TOKEN |
| `scripts/precompute_alignments.py` | **新文件**：字符串级 Levenshtein DP，multiprocessing 并行，输出纯文本 |
| `scripts/train_retro.py` | 预计算数据自动检测（新格式）、CPU 侧预处理、DataLoader 优化 |
| `tests/*.py` | 适配参数名变更 |
