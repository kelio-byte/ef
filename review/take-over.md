# 项目接手指南：Edit Flows 逆合成

> 写给接手这个项目的人。读完这篇你应该知道：项目在做什么、做到哪了、问题在哪、接下来该做什么。

⚠️ **重要提示**：当前仓库缺少以下文件，需要从原项目获取：
- **checkpoints/** — 训练好的模型权重（文档引用的 checkpoint：`USPTO_50K_PtoR_aug20_#global#/2026-06-08_17-20-39/checkpoint_step1680000.pt`，约 1.68M steps）
- **experiments/** — 上一个人的临时实验脚本，未提交到 git（不影响核心功能，所有采样功能已通过 `scripts/sample_retro.py` 暴露）

---

## 1. 一句话概括

用 **Edit Flows**（一种基于编辑操作的非自回归序列生成方法）做**化学逆合成**：输入产物 SMILES，通过插入/删除/替换三种编辑操作，逐步把产物"编辑"成反应物。

**核心公式**：`product --[insert/delete/substitute]--> reactant`

---

## 2. 当前状态速览

| 维度 | 状态 |
|------|------|
| **方法可行性** | ✅ Oracle（用真实速率）Top-1 = 97%，方法本身可行 |
| **模型瓶颈** | ❌ 模型学到的速率不够准，Euler 采样 Top-1 ≈ 43% |
| **Beam Search** | 🔄 单编辑贪心 Top-1 = **46.0%**（显式 STOP + Frozen-Hazard），比 Euler 好但远未到 oracle |
| **核心矛盾** | 模型 per-step edit ranking Top-1 = 78%，但多步累积后只剩 46%，差距来自时间 mismatch + 误差累积 |
| **代码质量** | ✅ 30 个单元测试通过，主体搜索语义正确 |

---

## 3. 项目文件结构速查

```
edit-flows/
├── edit_flows/           # 核心库
│   ├── core/             # 对齐、调度器(kappa/t)、Z空间映射、速率缩放
│   ├── data/             # 数据集加载、耦合采样
│   ├── models/           # Transformer 模型 (SimpleEditFlowsTransformer)
│   ├── sampling/         # ⭐ 当前主战场
│   │   ├── euler.py      #   原始 Euler 随机采样
│   │   ├── beam.py       #   单编辑 greedy/beam search + 显式 STOP
│   │   ├── ops.py        #   编辑操作底层实现 (apply_ins_del_operations)
│   │   ├── oracle.py     #   Oracle 采样（用真实速率）
│   │   └── time_policy.py#   时间推进策略 (RatioTimePolicy 等)
│   ├── training/         # 损失函数 (Bregman散度)、训练循环
│   └── analysis/         # 第一步分析、可视化
├── configs/              # YAML 配置文件
│   └── retro.yaml        #   主配置 (use_rate_reparam: true)
├── scripts/              # 训练/采样/评测脚本
│   ├── train_retro.py    #   训练入口
│   ├── sample_retro.py   #   采样入口 (支持 --sampler euler|greedy_edit|beam_edit)
│   ├── eval_retro.py     #   采样+评测一键脚本
│   └── edit_ranking_diag_v2.py  # edit ranking 诊断脚本
├── tests/                # 测试
│   └── sampling/test_beam.py  # 30 个 beam search 测试
├── dataset/              # 数据集（USPTO_50K 系列）
└── docs/                 # 文档（见下方文档地图）

注意：以下目录/文件**不在当前仓库中**，需从原项目获取：
  ├── checkpoints/        # 训练好的模型权重（约 1.68M steps）
  └── experiments/        # 上一个人的临时实验脚本（未提交）
```

---

## 4. 理论最简版

如果你只想看懂代码在干什么，知道这些就够了：

### 4.1 三个核心概念

1. **Z 空间（增广空间）**：训练时引入 GAP token，把不等长的 x₀（产物）和 x₁（反应物）对齐到等长，然后定义逐 token 的条件概率路径。模型在 X 空间（无 GAP）运行，损失在 Z 空间计算。

2. **Bregman 散度损失**：
   ```
   loss = u_tot - k(t) * CE_term
   ```
   - 第一项 `u_tot`：压制所有编辑速率，形成"最小编辑偏好"
   - 第二项 `CE_term`：只在正确编辑方向上拉高速率
   - `k(t) = κ'(t)/(1-κ(t))`：时间权重系数

3. **速率重参数化**（`use_rate_reparam: true`）：模型预测 base rate `u'`，采样时乘 `k(t)` 恢复真实速率。好处是解耦了"内容"和"时间尺度"。

### 4.2 关键公式

- **条件概率路径**：`z_t = z₁ with prob κ(t), z₀ with prob 1-κ(t)`，其中 κ(t) = t³（cubic scheduler）
- **速率 → 编辑概率**：`P(edit) = 1 - exp(-h · λ)`
- **单编辑 score**：`log p(e|x,t) = log u_real(e) - log u_tot_real`

### 4.3 必读代码（按顺序）

1. [trainer.py](edit_flows/training/trainer.py) — 理解训练 batch 怎么构造、loss 怎么算
2. [loss.py](edit_flows/training/loss.py) — Bregman 散度的具体实现
3. [transformer.py](edit_flows/models/transformer.py) — 模型架构（输出 rates + Q_ins + Q_sub）
4. [euler.py](edit_flows/sampling/euler.py) — Euler 采样流程
5. [beam.py](edit_flows/sampling/beam.py) — 当前主战场：单编辑 greedy/beam search

---

## 5. 项目演进时间线

```
正确率演进 (main 分支):
  22.2% → 40.6% → 43.5% → 45.1% → 46.8% → 50.7% → 54.8%
  (初始)  (fix)  (rate reparam)(linear κ)(origin mask)(no dropout)

Beam Search 分支 (当前):
  v1: 初版 greedy/beam → 0% (全是 bug)
  v2: 修 bug 后 → ~35% greedy
  v3: 时间 mismatch 修复 (RatioTimePolicy) → 44.5%
  v4: 显式 STOP (Frozen-Hazard) → 46.0%  ← 当前最佳
```

---

## 6. 文档地图

按阅读优先级排列：

### 第一层：必读（理解项目）

| 文档 | 内容 | 重要程度 |
|------|------|:--------:|
| **[edit-flows.md](docs/edit-flows.md)** | 完整理论：CTMC、Z空间、Bregman损失、Euler采样、FAQ | ⭐⭐⭐⭐⭐ |
| **[stage-1.md](docs/stage-1.md)** | 项目总览、正确率演进、核心判断、阅读顺序 | ⭐⭐⭐⭐⭐ |
| **本文档** | 接手指南 | ⭐⭐⭐⭐⭐ |

### 第二层：理解当前分支（beam-search）

| 文档 | 内容 | 重要程度 |
|------|------|:--------:|
| **[beam-search/todo1.md](docs/beam-search/todo1.md)** | Beam search 方案设计（5种方案 A-E、时间推进策略） | ⭐⭐⭐⭐ |
| **[beam-search/todo5.md](docs/beam-search/todo5.md)** | **显式 STOP 方案设计（当前最新方向）** | ⭐⭐⭐⭐⭐ |
| **[beam-search/exp5.md](docs/beam-search/exp5.md)** | 显式 STOP 实验：Phase 0 诊断 + Phase 1 实现 → 46.0% | ⭐⭐⭐⭐⭐ |
| **[beam-search/exp4.md](docs/beam-search/exp4.md)** | 时间 mismatch 验证 + RatioTimePolicy → 44.5% | ⭐⭐⭐ |
| **[beam-search/todo3.md](docs/beam-search/todo3.md)** | Beam 实现审查 + ranking 诊断 bug 修复 | ⭐⭐⭐ |

### 第三层：理解问题诊断

| 文档 | 内容 |
|------|------|
| **[oracle-analysis.md](docs/oracle-analysis.md)** | Oracle 实验：用真实速率替代模型 → 97%，方法可行 |
| **[loss-analysis.md](docs/loss-analysis.md)** | 损失分析：模型"总量级对了，但分布不够锐" |

### 第四层：按需查阅

| 文档 | 内容 |
|------|------|
| `beam-search/impl.md` | Beam search 初版实现总结 |
| `beam-search/todo2.md` | Beam bug 修复清单（5 个 correctness bug） |
| `beam-search/exp1~3.md` | 早期实验记录 |
| `beam-search/todo4.md` | 时间 mismatch 假说 |
| `retro-impl.md` | 逆合成实现细节 |
| `retro-improve.md` | 训练工程优化 |
| `rate-reparam-finish.md` | 速率重参数化实现 |
| `fix-1.md` / `fix-2.md` | 历史 bug 修复 |
| `first-step-analysis-*.md` | 第一步分析（在 `first-step-analysis` 分支） |
| `eval-finish.md` | 评测链路说明 |

---

## 7. 当前最核心的问题

### 问题一句话版

> **Per-step edit ranking 78% 正确，但多步累积后最终只有 46%。差距来自两方面：时间 mismatch + 模型 u_tot 语义未校准。**

### 问题拆解

| 子问题 | 现象 | 严重程度 |
|--------|------|:--------:|
| **时间 mismatch** | 搜索时 t 由步数决定，与编辑进度无关。快速收敛的样本上，还收到"早期"t 信号，模型被驱使继续编辑破坏已完成序列 | 🔴 已部分解决（RatioTimePolicy） |
| **停止逻辑不够 principled** | 当前 stop 靠外部阈值 `u_tot_base < 0.05`，与编辑排序不在同一概率空间 | 🟡 已部分解决（显式 STOP → +1.5pp） |
| **模型 u_tot 语义未校准** | 序列已正确时 u_tot 仍不降为 0（p_stop 中位数仅 0.073），模型未被训练过"序列正确时应输出极低 u_tot" | 🔴 核心瓶颈 |
| **Beam 收益极小** | Beam-5 仅比 greedy 高 ~2pp（统计不显著），时间 mismatch 对所有候选产生同向偏差 | 🟡 待时间问题解决后再评估 |
| **误差累积** | 0.78^5.5 ≈ 25%，实际 46% 说明 stop 提前退出救了部分样本 | 🟢 可接受 |

---

## 8. 你应该做什么

### 整体策略

```
当前瓶颈排序：模型速率校准 > 停止逻辑 > 时间推进 > Beam搜索
建议投入比例：  40%          30%       20%       10%
```

**核心思想**：不要急于调 beam 参数。先让模型在正确序列上输出更低的 u_tot，让 STOP 更自然地触发。

### Phase 1：立即可做（本周，成本极低）

#### 1.1 获取 checkpoint 并跑通 baseline

⚠️ **checkpoint 不在当前仓库中**，需要从上一个人那里获取。文档中引用的 checkpoint 路径为：
`checkpoints/USPTO_50K_PtoR_aug20_#global#/2026-06-08_17-20-39/checkpoint_step1680000.pt`

拿到 checkpoint 后，跑通最小验证：

```bash
# 确认模型能正常加载和采样（1 条产物，5 个样本，Euler 模式）
python scripts/sample_retro.py \
    --checkpoint <checkpoint路径> \
    --products_file <数据文件，每行一个tokenized SMILES> \
    --sampler euler \
    --n_samples 5 \
    --n_steps 100 \
    --device cpu

# 跑 greedy + 显式 STOP（当前最佳配置）
python scripts/sample_retro.py \
    --checkpoint <checkpoint路径> \
    --products_file <数据文件> \
    --sampler greedy_edit \
    --time_policy ratio \
    --explicit_stop \
    --kappa_mode frozen_hazard \
    --stop_u_tot_base 0.05 \
    --max_edits 20 \
    --device cpu

# 理解显式 STOP 的工作原理
# 核心文件：edit_flows/sampling/beam.py → sample_greedy_single_edit
```

**目标**：确认环境能跑通，理解 greedy + explicit_stop + FH 的完整流程。

#### 1.2 实现混合 STOP（预计 +1-2pp）

当前最佳 `fh_abs` = 46.0%。在显式 STOP 基础上叠加一个极小阈值兜底：

```
在 beam.py 的 greedy 循环中：
  if action is STOP:
      finished = True
  elif u_tot_base < 0.001:   # 极小阈值兜底
      finished = True
```

**为什么值得做**：exp5 分析显示，显式 STOP 在某些情况下会漏停（p_stop < p_e 但序列已对），极小阈值能兜住这些 case。成本极低，预计 10 行改动。

**参考文件**：[beam-search/exp5.md §13](docs/beam-search/exp5.md) P0 方向。

#### 1.3 扫 stop 混合阈值

扫 `stop_u_tot_base ∈ {0.001, 0.005, 0.01}`，看是否能在不引入过度早停的情况下提升 Top-1。

---

### Phase 2：训练侧改进（下周，需要重训）

这是真正可能拉开差距的方向。

#### 2.1 加"序列正确 → u_tot → 0"的训练信号

**问题**：模型从未被训练过"当序列已经等于 x₁ 时，输出 u_tot = 0"。训练时的 z_t 永远有正确的监督方向（需要编辑或不需要编辑），但模型从未见过"已经完美"的状态。

**方案**：在训练 batch 中混入一定比例的 `(x_t = x₁, t)` 样本，这些样本的 ground truth 所有编辑速率都为 0，loss 只保留第一项 `u_tot`（压制所有速率）。

**改动量**：在 [trainer.py](edit_flows/training/trainer.py) 的 batch 构造中增加一个 `p_correct` 概率。约 20 行改动。

**预期效果**：让模型学会"序列正确时 u_tot → 0"，从而让显式 STOP 的 p_stop 更可靠。

#### 2.2 尝试 sigmoid STOP（替代 e^{-U}）

当前 `p_stop = e^{-U}` 的问题：当 U=2 时 p_stop≈0.14，仍很小；当 U=0.5 时 p_stop≈0.61，才比较确定。这个映射可能过于激进。

**方案**：改为 `p_stop = sigmoid(-α(U - β))`，其中 α、β 从训练集统计或小规模 sweep 确定。

**改动量**：在 `beam.py` 中增加一种 `p_stop_mode="sigmoid"`。约 30 行改动。

---

### Phase 3：把显式 STOP 接入 Beam（需要 Phase 1/2 有进展后）

当前 beam search 的收益只有 ~2pp。等 greedy 被推到 50%+ 后，再评估 beam 能否放大收益。

#### 3.1 Beam + explicit_stop

当前 `beam.py` 中 `sample_beam_single_edit` 还没有接入显式 STOP。接入后：
- 每个 parent state 可展开出一个 STOP child
- STOP child 与 edit children 一起进入候选池，按累积 log_prob 排序
- Top-k 裁剪自然处理

#### 3.2 评估 beam 收益

若 beam-5 比 greedy 高 4-8pp（vs 之前的 ~2pp），说明方向正确。

---

### Phase 4：长期方向（需要较大改动）

| 方向 | 说明 | 优先级 |
|------|------|:------:|
| 训练侧数据增强 | 训练时混入非标准 (state, t) 对，减少时间依赖 | 低 |
| 解耦时间嵌入 | 时间嵌入仅加到 rate head，不贯穿主干 | 低 |
| 方案 C：Grid-Integrated First-Event | 多次 forward 沿 kappa 网格积分，更准确但更贵 | 低（除非 B 有瓶颈） |
| 重训更大模型 | 当前 ~17M 参数，论文用 280M+ | 低（先解决校准问题） |

---

## 9. 实验快速参考

### 需要从原项目获取的文件

| 需要什么 | 说明 |
|----------|------|
| **checkpoint** | `checkpoints/USPTO_50K_PtoR_aug20_#global#/2026-06-08_17-20-39/checkpoint_step1680000.pt`（1.68M steps） |
| **数据文件** | `dataset/` 下已有三个数据集目录，但需要确认是否有评测用的 `.jsonl` 文件 |
| **vocab 文件** | checkpoint 内含 `model_vocab`，无需单独获取 |

### 常用命令

```bash
# ===== 训练 =====
python scripts/train_retro.py --config configs/retro.yaml

# ===== 采样 =====

# Euler 采样（原始随机方式）
python scripts/sample_retro.py \
    --checkpoint <checkpoint路径> \
    --products_file <数据文件> \
    --sampler euler \
    --n_samples 5 --n_steps 100

# Greedy 单编辑（当前最佳配置：显式 STOP + Frozen-Hazard）
python scripts/sample_retro.py \
    --checkpoint <checkpoint路径> \
    --products_file <数据文件> \
    --sampler greedy_edit \
    --time_policy ratio \
    --explicit_stop \
    --kappa_mode frozen_hazard \
    --stop_u_tot_base 0.05 \
    --max_edits 20

# Greedy 单编辑（备选：RatioTimePolicy + 外部阈值，无需 explicit_stop）
python scripts/sample_retro.py \
    --checkpoint <checkpoint路径> \
    --products_file <数据文件> \
    --sampler greedy_edit \
    --time_policy ratio \
    --stop_u_tot_base 0.05 \
    --max_edits 20

# Beam 单编辑
python scripts/sample_retro.py \
    --checkpoint <checkpoint路径> \
    --products_file <数据文件> \
    --sampler beam_edit \
    --beam_size 5 \
    --time_policy ratio \
    --stop_u_tot_base 0.05 \
    --max_edits 20

# 扫参数时最有用的几个开关：
#   --sampler {euler, greedy_edit, beam_edit}
#   --time_policy {depth, fixed, ratio, kappa}
#   --explicit_stop              # 启用显式 STOP 动作
#   --kappa_mode {ratio, frozen_hazard, poisson}
#   --stop_u_tot_base <float>    # <0 表示禁用外部阈值
#   --time_const <float>         # time_policy=fixed 时的固定 t 值

# ===== 评测（需要 score.py） =====
# 对采样结果跑 standard 评测
python scripts/score.py --pred_file <predictions.txt>

# 对采样结果跑 #global# 评测
python scripts/score_#global#.py --pred_file <predictions.txt>

# ===== 诊断 =====

# Edit ranking 诊断（沿 oracle 轨迹测量模型排序能力）
python scripts/edit_ranking_diag_v2.py \
    --checkpoint_path <checkpoint路径> \
    --data_path <数据文件> \
    --max_samples 200

# Oracle greedy（用真实速率做单编辑贪心，验证理论上界）
python scripts/oracle_greedy.py \
    --data_path <数据文件> \
    --max_samples 200
```

### 关键 checkpoint

| Checkpoint | Step | 说明 |
|-----------|------|------|
| `2026-06-08_17-20-39/checkpoint_step1680000.pt` | 1.68M | 当前主 checkpoint，需从原项目获取 |

### 关键配置参数速查

| 参数 | 当前最佳值 | 含义 |
|------|:--------:|------|
| `sampler` | `greedy_edit` | 采样算法 |
| `time_policy` | `ratio` | 时间推进策略 |
| `explicit_stop` | `True` | 启用显式 STOP 动作 |
| `kappa_mode` | `frozen_hazard` | κ 更新公式（方案 B） |
| `stop_u_tot_base` | `0.05` | 外部阈值兜底 |
| `max_edits` | `20` | 最大编辑步数 |
| `k_ins_token` | `4` | 每位置插入 token top-k |
| `k_sub_token` | `4` | 每位置替换 token top-k |
| `k_edit_expand` | `16` | 全局编辑候选 top-k |

### 评测指标速查

| 指标 | 含义 | 当前最佳 |
|------|------|:--------:|
| Top-1 Acc | 生成的最优序列命中 target | 46.0% (greedy) / ~43% (Euler) |
| Invalid SMILES | 生成的序列不是合法化学式 | 4.0% (greedy) |
| Unique Rate | 生成的 N 个序列中去重比例 | — |
| Oracle Top-1 | 用真实速率生成 | 97.4% |

---

## 10. 踩坑指南

### ⚠️ 不要做的事

1. **不要直接调大 beam_size 期望收益** — 时间 mismatch 会让所有候选同步偏差，beam 收益天然受限
2. **不要重跑会覆盖已有结果的脚本** — 特别是 `checkpoints/*/eval/predictions.txt`
3. **不要在没跑通 200 条小规模实验前就上全量 1000/10000 条** — 浪费时间
4. **不要忽视 `use_origin_mask`** — 若 checkpoint 训练时用了 `use_origin_mask: true`，采样时也必须开启，否则语义不一致
5. **不要把 base rate 和 real rate 搞混** — 显式 STOP 的 U 应使用 base rate（与训练时模型原始输出一致）

### ✅ 最佳实践

1. **任何新想法先在 100-200 条上验证** — 几分钟跑完，快速迭代
2. **改采样逻辑先跑 `pytest tests/sampling/test_beam.py`** — 30 个测试，1 分钟内出结果
3. **对比实验控制单变量** — 一次只改一个因素
4. **记录每个实验的完整配置** — YAML 或 JSON，方便回溯

---

## 11. 核心文件速查（改代码时对照）

| 想改什么 | 去哪个文件 | 关键函数/类 |
|----------|-----------|------------|
| 显式 STOP 逻辑 | `sampling/beam.py` | `sample_greedy_single_edit()` |
| STOP 概率公式 | `sampling/beam.py` | `_update_fh_kappa()`, `_update_poisson_kappa()` |
| 候选编辑收集 | `sampling/beam.py` | `_collect_edit_candidates()` |
| 时间推进策略 | `sampling/time_policy.py` | `RatioTimePolicy`, `KappaTimePolicy` |
| Euler 采样 | `sampling/euler.py` | `sample_euler()` |
| 编辑操作应用 | `sampling/ops.py` | `apply_ins_del_operations()` |
| 训练 loss | `training/loss.py` | `bregman_loss()` |
| 训练 batch | `training/trainer.py` | `train_step()` |
| 模型架构 | `models/transformer.py` | `EditFlowsTransformer.forward()` |
| 速率缩放 | `core/rate_scale.py` | `apply_rate_parameterization()` |
| 调度器 (κ/t) | `core/scheduler.py` | `CubicScheduler`, `LinearScheduler` |

---

## 12. 一句话总结当前任务

> **让 greedy single-edit + 显式 STOP 的 Top-1 从 46% 突破 50%，最有希望的路径是：混合 STOP 兜底（短期） + 训练时教模型"序列正确时 u_tot→0"（中期）。**
