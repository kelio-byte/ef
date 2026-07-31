# Edit Flows 逆合成评测：实现文档

## 1. 概述

Edit Flows 的生成是随机采样（Euler-Maruyama），与自回归 seq2seq 的 beam search 有本质区别：每次采样独立、等权、无序。原有的评测脚本依赖 beam 位置加权，因此需要适配。

### 设计原则

- **`sample_retro.py`** 专注采样，输出原始结果（token 序列去空格），不做任何后处理。支持 GPU 批量化并行采样。
- **`score.py` / `score_#global#.py`** 负责 canonicalize + 排名 + 评测，`#global#` 格式的 `inverse_global_align` 在此阶段处理
- **`eval_retro.py`** 将采样和评测串联，一键完成

### 数据流

```
test/src-test.txt (产物, tokenized SMILES)
    │
    ▼  tokenize → batch (batch_size 个产物 × n_samples 次复制) → GPU 并行
sample_retro.py  ──►  predictions.txt  (raw SMILES, n_samples × n_products 行)
    │
    ▼
score.py / score_#global#.py
    │  ├── inverse_global_align  (#global# 专用: R-SMILES → 标准 SMILES)
    │  ├── RDKit canonicalize + 去重
    │  ├── compute_rank (--edit_flows: 等权)
    │  └── Top-K accuracy vs ground truth
```

## 2. 新增/修改文件

```
scripts/
├── sample_retro.py              # 修改：新增批量采样能力
├── score.py                     # 新增：标准数据集评测
├── score_#global#.py            # 新增：#global# 数据集评测
├── eval_retro.py                # 新增：一键采样+评测
└── preprocessing/               # 从原项目复制
    ├── global_align.py          # inverse_global_align()
    └── bipartite.py             # glue_two_parts() (仅 --inv_bipartite 模式)
```

## 3. sample_retro.py

### 批量化并行采样

采样脚本将产物按 `batch_size` 分组，每组在 GPU 上一次 forward 并行处理，充分利用 GPU 算力。

```
全部产物 tokenize → product_ids
    │
    ▼ 按 batch_size 分组
Batch 0: [p0, ..., p_{B-1}]  →  PAD 到统一长度  →  repeat_interleave(n_samples)
Batch 1: [pB, ..., p_{2B-1}] →  PAD 到统一长度  →  repeat_interleave(n_samples)
...
    │
    ▼ 每个 batch 一次 sample_euler() 调用
results (B × n_samples, L_out)  →  逐行写出（每产物 n_samples 连续行）
```

`repeat_interleave(n_samples, dim=0)` 将每个产物复制 n_samples 份且相邻排列：
```
[p0, p0, ..., p0 (n_samples), p1, p1, ..., p1 (n_samples), ...]
```
从而保证输出中每个产物有 n_samples 个连续行，与 score 脚本的 `beam_size × augmentation` 格式兼容。

### 效率对比

| | 优化前 (batch_size=1) | 优化后 (batch_size=32) |
|---|---|---|
| 每次 forward 的 batch | 1 | `32 × n_samples`（如 n_samples=10 → 320） |
| GPU 利用率 | ~5-15% | 预期 80-95% |
| n=5000 产物的 forward 次数 | `5000 × n_samples` | `ceil(5000/32) × 1`（157 次 batch） |

参考了训练时的优化经验（`docs/retro-improve.md`）：一次大批量 GPU forward 替代多次逐条 forward，消除 CPU-GPU 同步开销。

### 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_samples` | 1 | 每个产物的独立采样次数 |
| `--batch_size` | 32 | GPU 批大小（每 batch 包含的产物数量） |
| `--output_dir` | None | 输出目录，结果写入 `<output_dir>/predictions.txt` |

### 输出格式

每行一个去空格的 SMILES（token 直接拼接），共 `n_samples × len(products)` 行。第 i 个产物的 n_samples 个采样结果占据连续的 n_samples 行。

### 使用示例

```bash
# 单条产物多次采样
PYTHONPATH=. python scripts/sample_retro.py \
    --checkpoint checkpoints/.../checkpoint_step1150000.pt \
    --product "C ( N ) C C C ..." \
    --n_samples 5 --n_steps 100 --device cuda

# 批量文件采样（GPU 批处理）
PYTHONPATH=. python scripts/sample_retro.py \
    --checkpoint checkpoints/.../checkpoint_step1150000.pt \
    --products_file /path/to/test/src-test.txt \
    --n_samples 10 --batch_size 32 --n_steps 100 \
    --output_dir ./results/my_run --device cuda
```

## 4. score.py / score_#global#.py

### 两者差异

| | score.py | score_#global#.py |
|---|---|---|
| 适用数据 | 普通 PtoR | `#global#` 格式 (R-SMILES) |
| canonic. 前处理 | 无（默认） | 固定 `inverse_global_align()` |
| 额外依赖 | 无 | `preprocessing/global_align.py` |

### `--edit_flows` 模式

新增 `--edit_flows` flag，切换 `compute_rank()` 行为：

| | 原版 (beam search) | Edit Flows |
|---|---|---|
| 候选顺序 | 有（beam likelihood 排序） | 无（独立等权） |
| 排名权重 | `1/(α×k+1)` 位置加权 | 所有候选等权 `1.0` |
| 跨 augmentation | 累加权重 | 仅去重 |
| hit 判定 | rank 排序后取 top-K | 去重后 set membership |

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-predictions` | 必填 | 采样结果文件 |
| `-targets` | 必填 | ground truth 文件 |
| `-beam_size` | 10 | 对应 `n_samples` |
| `-augmentation` | 1 | Edit Flows 固定为 1 |
| `-n_best` | 10 | 报告 Top-1 ~ Top-N 准确率 |
| `-edit_flows` | False | **启用 Edit Flows 等权排名** |
| `-detailed` | False | 按 chirality/ring 等细分准确率 |
| `-sources` | "" | detailed 模式需要的产物文件 |
| `-process_number` | CPU 核数 | canonicalize 并行数 |
| `-save_file` | "" | 保存排名结果到文件 |

### 使用示例

```bash
# 标准数据集评测
python scripts/score.py \
    -predictions ./results/my_run/predictions.txt \
    -targets /path/to/test/tgt-test.txt \
    -beam_size 10 -augmentation 1 --edit_flows

# #global# 数据集评测
python scripts/score_#global#.py \
    -predictions ./results/my_run/predictions.txt \
    -targets /path/to/test/tgt-test.txt \
    -beam_size 10 -augmentation 1 --edit_flows

# 详细评测（按原子数差异、手性、开环/关环细分）
python scripts/score.py \
    -predictions ./results/my_run/predictions.txt \
    -targets /path/to/test/tgt-test.txt \
    -sources /path/to/test/src-test.txt \
    -beam_size 10 -augmentation 1 --edit_flows --detailed
```

### 输出指标

```
Top-1  Acc:xx.xxx%, MaxFrag xx.xxx%,  Invalid SMILES:xx.xxx%, Sorted Invalid SMILES:xx.xxx%
Top-3  Acc:...
Top-5  Acc:...
Top-10 Acc:...
Top-20 Acc:...
Top-50 Acc:...
Unique Rates:xx.xxx%
```

| 指标 | 含义 |
|------|------|
| Top-K Acc | 去重后的前 K 个唯一预测中命中 ground truth 的比例 |
| MaxFrag | 最大片段命中率（反应物是多组分时，最大片段匹配即算） |
| Invalid SMILES | 采样结果中无效 SMILES 的比例 |
| Unique Rates | 唯一有效预测数 / beam_size |

detailed 模式额外输出：按 chirality（手性有无）、ring opening/formation（开环/关环）、atom differ size（产物→反应物原子数差异）分组的 Top-K 准确率。

## 5. eval_retro.py — 一键评测

### augmentation 自动检测

`eval_retro.py` 从数据集目录名自动解析 augmentation 因子（如 `aug20` → 20），并传递给 score 脚本。也可通过 `--augmentation` 手动覆盖。

测试集如 `USPTO_50K_PtoR_aug20` 包含 20 个增强变体（100,140 = 5,007 × 20 行），score 脚本通过 `augmentation=20` 正确地将变体分组到同一产品下，跨变体去重后计算 Top-K 准确率。

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | 必填 | checkpoint .pt 路径 |
| `--n_samples` | 10 | 每产物独立采样数 |
| `--n_steps` | 100 | Euler 采样步数 |
| `--batch_size` | 32 | GPU 批大小（传递给 sample_retro.py） |
| `--device` | cuda | 设备 |
| `--output_dir` | `<ckpt_dir>/eval` | 输出目录 |
| `--n_best` | 10 | Top-N 准确率 |
| `--detailed` | False | 启用详细评测 |
| `--data_dir` | 从 checkpoint 读取 | 覆盖数据集路径 |
| `--test_src` | `test/src-test.txt` | 测试产物文件（相对 data_dir） |
| `--test_tgt` | `test/tgt-test.txt` | 测试目标文件（相对 data_dir） |
| `--augmentation` | 自动检测 | 测试集增强因子（从 `augXX` 解析，默认 1） |

### 自动逻辑

1. 从 checkpoint 读取 `data_dir`
2. 检测 `#global#` → 选择对应 score 脚本
3. 运行 `sample_retro.py` 采样
4. 运行 `score.py` / `score_#global#.py` 评测
5. 结果写入 `eval/predictions.txt` + `eval/eval.log`

### 使用示例

```bash
# 一键评测（自动检测数据集类型）
PYTHONPATH=. python scripts/eval_retro.py \
    --checkpoint checkpoints/USPTO_50K_PtoR_aug20_#global#/2026-05-23_16-01-07/checkpoint_step1150000.pt \
    --n_samples 10 --batch_size 32 --device cuda

# 在另一个数据集上评测（跨数据集）
PYTHONPATH=. python scripts/eval_retro.py \
    --checkpoint checkpoints/USPTO_50K_PtoR_aug20_#global#/.../checkpoint_step1150000.pt \
    --data_dir /data6/.../USPTO_50K_PtoR_aug20 \
    --n_samples 10 --batch_size 32 --device cuda

# 详细评测
PYTHONPATH=. python scripts/eval_retro.py \
    --checkpoint checkpoints/.../checkpoint_step1150000.pt \
    --n_samples 10 --batch_size 32 --device cuda --detailed
```

## 6. 设计决策 FAQ

### 为什么 augmentation 固定为 1？

原项目的 augmentation 是对同一输入做多种 SMILES 变体（shuffle 等），Edit Flows 的随机采样本身就提供了多样性，n_samples 次独立采样即可。如需真正的 augmentation 支持，可在 sample_retro 层面增加输入变换。

### 为什么 inverse_global_align 放在 score 脚本而非 sample 脚本？

采样脚本输出原始结果更通用：方便检查模型原始输出、调试 R-SMILES 格式问题、复用不同后处理策略。评测时的 canonicalize 链统一处理格式转换。

### Edit Flows 的 Top-K 准确率如何计算？

n_samples 个独立采样 → canonicalize → 去重 → 取前 K 个唯一预测 → 检查是否包含 ground truth。所有预测等权，无 beam 位置偏好。
