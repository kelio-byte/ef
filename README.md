# Edit Flows：基于离散编辑流的逆合成预测

本项目研究如何把序列生成建模为连续时间马尔可夫链（Continuous-Time Markov
Chain, CTMC）上的编辑过程，并将其用于单步逆合成预测。给定产物 SMILES，模型通过
插入（insert）、删除（delete）和替换（substitute）逐步生成反应物 SMILES。

仓库目前处于研究迭代阶段。当前工作的重点不是修改训练模型，而是在固定 checkpoint
和测试协议下改进 `Euler-Beam` 采样：同时保证方法正确、Top-k 准确率可靠，并降低
RTX 3090 上的推理时间。

## 1. 方法概览

### 1.1 Edit Flows

对产物序列 $x_0$ 和目标反应物序列 $x_1$ 做编辑对齐，得到等长的增广序列
$z_0,z_1$。增广空间包含特殊 gap token：

$$
\mathcal{Z}=(\mathcal{T}\cup\{\varepsilon\})^N.
$$

每个对齐位置对应一种编辑：

- $z_0^i=\varepsilon,z_1^i=c$：插入 token $c$；
- $z_0^i=c,z_1^i=\varepsilon$：删除 token $c$；
- $z_0^i=c_1,z_1^i=c_2$：将 $c_1$ 替换为 $c_2$。

模型接收当前序列和时间，预测每个有效位置的三类编辑速率，以及插入、替换 token
的条件分布：

```text
model(x_t, t)
  -> lambda_insert, lambda_substitute, lambda_delete
  -> Q_insert(token), Q_substitute(token)
```

采样器再将速率转换为一个 Euler 时间步内的事件概率。例如，速率为 $\lambda$、步长
为 $h$ 时，事件发生概率为：

$$
p(\text{event})=1-\exp(-h\lambda).
$$

同一步可以在多个位置触发编辑，并不是每步最多编辑一次。

### 1.2 对齐、训练和采样流程

```text
tokenized product/target
        |
        v
Levenshtein alignment -> z0, z1 with <GAP>
        |
        v
sample intermediate z_t -> remove gaps -> x_t
        |
        v
Transformer predicts edit rates and token distributions
        |
        v
Euler / Euler-Beam / single-edit greedy or beam sampler
        |
        v
canonicalize and aggregate 20 test augmentations -> Top-k accuracy
```

训练配置位于 `configs/retro.yaml`。训练脚本优先读取预先对齐的数据；不存在时会退回
在线动态规划对齐，但速度明显更慢。

## 2. 采样器

项目主要提供以下采样方式：

| 采样器 | 入口 | 核心行为 | 每个产物的输出数 |
|---|---|---|---:|
| Euler | `sample_euler()` | 多位置随机编辑，轨迹互相独立 | `n_samples` |
| Euler-Beam | `sample_euler_beam()` | 每个状态产生 M 个后继，合并、排序并保留 K 个状态 | `n_runs` |
| Greedy edit | `sample_greedy_single_edit()` | 每步选择一个最高分编辑 | 1 |
| Beam edit | `sample_beam_single_edit()` | 单编辑候选展开与 beam 剪枝 | 1 |

### 2.1 Euler

`n_samples=N` 表示对同一个输入运行 N 条独立随机轨迹。每条轨迹都完整执行
`n_steps` 个 Euler 时间步，轨迹之间不合并、不排序、不剪枝。

在每个时间步，插入过程和删除/替换竞争过程会在所有有效位置上采样，因此一次时间步
可能不发生编辑，也可能同时发生多次编辑。

### 2.2 Euler-Beam

Euler-Beam 的三个关键规模参数是：

- `n_branches=K`：每个 run 最多保留的内部状态数；
- `n_children=M`：每个父状态在每一步产生的随机后继数；
- `n_runs=R`：对每个产物独立执行 R 次搜索，最终输出 R 条预测。

单步搜索结构为：

```text
最多 K 个父状态
    -> 一次批量模型 forward
    -> 每个父状态采样 M 个后继（最多 K*M 个候选）
    -> 批量应用编辑并计算一步分数
    -> 相同 token 状态合并概率质量
    -> 排序、剪枝，最多保留 K 个状态
```

当 `M=1` 时，每个父状态仍只有一个随机后继，候选竞争空间有限；当前正式研究配置使用
`M=2`。分支和 child 的随机流由稳定 seed 派生，不依赖 batch 划分或 K 的取值。

Euler-Beam 支持两种 child policy：

- `stochastic`：所有 child 都按标准 Euler 转移随机采样；
- `stochastic_noop`：仅适用于 `M=2`。child 0 保持随机采样，child 1 只在
  `t≈0.9` 的一个时间步作为 no-op anchor，避免该步破坏已有高质量状态。

`stochastic_noop` 是启发式搜索 proposal，不应解释为无偏的 CTMC 状态概率。

## 3. 当前 Euler-Beam 状态

固定 checkpoint 为 `checkpoint_step600000.pt`。已确认该 checkpoint：

- `use_origin_mask: false`；
- 不包含 `origin_embedding` 权重；
- 当前采样优化不依赖 origin mask。

截至 2026-08-01，在 50 个原始反应、20 倍测试增强的 tiny benchmark 上，当前推荐
研究配置为：

```text
K=3, M=2, R=3, n_steps=100
score_mode=full_probability
changed_state_bonus=0.5
child_policy=stochastic_noop
float32_matmul_precision=high (RTX 3090 TF32)
```

其完整结果为：

| 指标 | 当前推荐配置 |
|---|---:|
| 采样时间 | 约 122.6 秒 |
| Top-1 | 60% |
| Top-2 | 64% |
| Top-3 | 70% |
| Invalid SMILES（rank 1/2/3） | 12.5% / 14.5% / 13.4% |

这些结果仅用于固定 tiny benchmark 上的版本比较，不等同于完整 USPTO-50K 测试集
结论。历史恢复版本曾得到 58%/68%/76%，但它采用旧 seed 和旧路径评分语义，不能与
当前结果作严格的单变量比较。详细实验历史见
[`new_docs/euler_beam_optimization_plan.md`](new_docs/euler_beam_optimization_plan.md)。

## 4. 环境与安装

基础包要求 Python 3.10 或更高版本。项目元数据声明了 PyTorch、NumPy、PyYAML 和
tqdm：

```bash
python -m pip install -e ".[dev]"
```

逆合成评分还依赖 RDKit 和 pandas，但它们目前没有写入 `pyproject.toml`。建议在
conda 环境中安装：

```bash
conda install -c conda-forge rdkit pandas
```

在 CUDA 设备上运行前，请安装与本机 CUDA/驱动匹配的 PyTorch。TF32 的性能结论来自
RTX 3090；其他 GPU 需要重新基准测试。

## 5. 数据和 checkpoint

逆合成数据目录应至少包含：

```text
datasets/USPTO_50K_PtoR_aug20_#global#/
├── example.vocab.src
├── train/
│   ├── src-train.txt
│   ├── tgt-train.txt
│   ├── train_aligned_src.txt      # 可选但推荐
│   └── train_aligned_tgt.txt      # 可选但推荐
├── val/
└── test/
    ├── src-test.txt
    ├── tgt-test.txt
    ├── src-test-tiny.txt
    └── tgt-test-tiny.txt
```

所有 SMILES 文件均为按空格分词的文本，每行一个序列。测试集若使用 20 倍增强，同一
原始反应的 20 条增强输入必须连续排列。

checkpoint 至少需要包含：

```text
model_state_dict
config
model_vocab（可选，可由词表推断）
```

采样脚本会优先使用 checkpoint 内的配置，并允许通过命令行覆盖数据目录、词表和采样
scheduler。

## 6. 快速开始

### 6.1 当前推荐 Euler-Beam 基准

```bash
python scripts/sample_retro.py \
    --checkpoint checkpoint_step600000.pt \
    --products_file "datasets/USPTO_50K_PtoR_aug20_#global#/test/src-test-tiny.txt" \
    --sampler euler_beam \
    --n_branches 3 \
    --n_children 2 \
    --n_runs 3 \
    --n_steps 100 \
    --batch_size 64 \
    --device cuda \
    --seed 42 \
    --euler_beam_score_mode full_probability \
    --euler_beam_changed_state_bonus 0.5 \
    --euler_beam_matmul_precision high \
    --euler_beam_child_policy stochastic_noop \
    --output_dir results/bench_beam/

python 'scripts/score_#global#.py' \
    --predictions results/bench_beam/predictions.txt \
    --targets "datasets/USPTO_50K_PtoR_aug20_#global#/test/tgt-test-tiny.txt" \
    --augmentation 20 \
    --beam_size 3 \
    --n_best 5
```

这里评分器的 `beam_size` 必须等于采样时的 `n_runs`，而不是内部的
`n_branches` 或 `n_children`。

### 6.2 Euler 对照实验

```bash
python scripts/sample_retro.py \
    --checkpoint checkpoint_step600000.pt \
    --products_file "datasets/USPTO_50K_PtoR_aug20_#global#/test/src-test-tiny.txt" \
    --sampler euler \
    --n_samples 3 \
    --n_steps 100 \
    --batch_size 16 \
    --device cuda \
    --seed 42 \
    --output_dir results/bench_euler/

python 'scripts/score_#global#.py' \
    --predictions results/bench_euler/predictions.txt \
    --targets "datasets/USPTO_50K_PtoR_aug20_#global#/test/tgt-test-tiny.txt" \
    --augmentation 20 \
    --beam_size 3 \
    --n_best 5
```

不要把 Euler 的评分路径误写为 `results/bench_beam/predictions.txt`。

### 6.3 预计算训练对齐

```bash
PYTHONPATH=. python scripts/precompute_alignments.py \
    --data_dir "datasets/USPTO_50K_PtoR_aug20_#global#" \
    --splits train \
    --num_workers 16
```

### 6.4 训练或恢复训练

```bash
python scripts/train_retro.py \
    --config configs/retro.yaml \
    --device cuda

python scripts/train_retro.py \
    --config configs/retro.yaml \
    --checkpoint path/to/checkpoint_stepN.pt \
    --device cuda
```

本轮 Euler-Beam 研究固定使用已有 checkpoint，不修改训练代码、数据集或模型权重。

## 7. 评分协议与已知限制

`scripts/score_#global#.py` 会先执行 global alignment 逆变换和 RDKit canonicalization，
再将候选按以下布局聚合：

```text
(原始反应, augmentation, run/rank)
```

评分器现在会严格检查 prediction、target、augmentation 和 beam size 的布局；默认
拒绝静默截断，仅在显式指定 `--length` 时允许评分完整的文件前缀。Top-N 可以报告到
`n_best`，而 input rank 不存在时不再把它解释成 RDKit invalid。

可用以下参数增加不改变排名语义的采样诊断：

```bash
python 'scripts/score_#global#.py' \
    ... \
    --diagnostics \
    --diagnostics_json results/bench_beam/diagnostics.json
```

诊断包括 Oracle-any、已覆盖但未进入 Top-3 的数量、每个 run 的命中/invalid/重复率、
真实唯一候选数和 run 两两 Jaccard overlap。旧 `Unique Rates` 为保持历史日志兼容继续
输出，但它基于 `rank[:n_best]`，不是真正的原始采样多样性，可能超过 100%。

当前仍保留一项重要的历史评分假设：聚合排序用一个很大的常数优先保证“任意
augmentation 中的最好局部排名”，跨增强出现频率只在最好局部排名相同时起主要作用。
在完成覆盖—排序归因前不改变该默认规则，避免破坏历史可比性。规划见
[`new_docs/euler_beam_next_stage_plan.md`](new_docs/euler_beam_next_stage_plan.md)。

## 8. 项目结构

```text
edit_flows/
├── core/       # scheduler、coupling、alignment、Z-space、rate scaling
├── data/       # 逆合成数据集和 batch 构造
├── models/     # Transformer 编辑速率模型
├── sampling/   # Euler、Euler-Beam、single-edit greedy/beam、编辑算子
├── training/   # loss、训练步骤、学习率 scheduler
└── utils/      # token 常量和通用工具

scripts/
├── train_retro.py              # 逆合成训练
├── sample_retro.py             # 统一采样入口
├── score_#global#.py           # #global# 数据评分
├── score.py                    # standard 数据评分
├── precompute_alignments.py    # 预计算 Levenshtein 对齐
└── visualize_*.py              # 首步与轨迹分析

configs/                         # 通用与逆合成配置
tests/                           # core/model/sampling/training 测试
docs/                            # 较早阶段的设计和实验记录
new_docs/                        # 当前 Euler-Beam 设计、计划与实验记录
```

恢复文件 `recover_euler_beam.py` 和 `recover_sample_retro.py` 仅用于保存历史 58% 版本，
不是当前实现入口。

## 9. 测试与研究约定

运行测试：

```bash
pytest -q
```

Euler-Beam 修改遵循以下约定：

- 优先保证概率、seed、输出布局和评分协议正确；
- 先做小规模短筛，再决定是否运行完整 tiny benchmark；
- 准确率实验固定 checkpoint、数据、seed 和评分参数；
- 性能比较固定硬件、precision、batch size 和输入规模；
- 任何启发式 proposal 都区分目标转移概率 $p$ 与实际 proposal $q$；
- 不用未经验证的新评分规则替代历史默认规则；
- 阶段性结果写入规划文档，并用范围明确的 Git commit 保存。

当前的首要问题是：Top-1 已达到 60%，但 Top-2/Top-3 增长慢。接下来将先判断正确
答案是没有被采到，还是已经存在于候选中但被聚合排序压低，再选择采样多样性或评分
聚合方向。
