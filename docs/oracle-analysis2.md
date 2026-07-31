# Oracle 实验 Round 2：错误模式分析与轨迹分布分析

## 1. 背景与动机

前一轮 oracle 实验（[oracle-analysis.md](oracle-analysis.md)）已经证明：

- **方法可行**：Edit Flows + Copy Product 的 oracle Top-1 达 93%+
- **Cubic→Linear 大幅降低无效 SMILES**：Standard 上无效率从 22.6%→9.2%，#global# 上从 9.0%→2.6%

但仍有两个问题需要回答：

1. **Oracle 失败的那些 case 究竟差在哪里？** 是差了 1 个编辑还是全面崩溃？
2. **编辑距离 d(x_t, x_1) 随时间 t 的分布是否与理论 Binomial(L, 1-κ(t)) 一致？偏差有多大？**

本轮的两次实验分别回答这两个问题。

---

## 2. 实验一：Oracle 错误模式分析

### 2.1 分析方法

对 oracle 生成的 top-1 预测与 ground truth 做 Levenshtein 对齐，逐样本分析：

- 无效 SMILES vs 合法但错误
- 剩余编辑距离（差几个编辑）
- 漏掉的编辑类型（ins/del/sub）
- 漏掉编辑的 K 值（多少 Z 位置映射到同一个 X 位置编辑需求）
- 位置分布（序列前/中/后部）

### 2.2 脚本与使用

**脚本**：`scripts/oracle_error_analysis.py`

```bash
PYTHONPATH=. python scripts/oracle_error_analysis.py \
    --predictions train_subsets/eval/oracle_standard_linear/predictions.txt \
    --targets train_subsets/eval/oracle_standard_linear/targets_subset.txt \
    --products train_subsets/USPTO_50K_PtoR_aug20/test/src-test.txt \
    --vocab_file /data6/duanbh/desktop/retrosynthesis/dataset/USPTO_50K_PtoR_aug20/example.vocab.src \
    --deduplicate 20 --n_samples 10
```

**输出**：`{output_dir}/error_analysis.txt`

### 2.3 结果 (Linear Scheduler)

#### Standard 数据集

| 指标 | 数值 |
|------|------|
| First-sample 失败率 | 118/1000 (11.8%) |
| **Edit distance = 1** | **105 (89.0%)** |
| Edit distance = 2 | 11 (9.3%) |
| Edit distance ≥ 3 | 2 (1.7%) |
| **K=1 占比** | **133/133 (100.0%)** |
| Missed: Ins/Del/Sub | 56.4% / 12.0% / 31.6% |
| 失败样本平均 target 长度 | 54.5 (全量: 49.1) |

#### #global# 数据集

| 指标 | 数值 |
|------|------|
| First-sample 失败率 | 40/1000 (4.0%) |
| Invalid SMILES | 37.5% of failures |
| Valid-but-Wrong | 62.5% of failures |
| **Edit distance = 1** | **35 (87.5%)** |
| Edit distance = 2 | 5 (12.5%) |
| **K=1 占比** | **45/45 (100.0%)** |
| Missed: Ins/Del/Sub | 86.7% / 0.0% / 13.3% |
| 失败样本平均 target 长度 | 56.2 (全量: 51.5) |

### 2.4 关键发现

1. **100% 的 missed edits 都是 K=1**——完美验证了理论：K=1 的编辑有非零的"永不触发"概率（Linear 下 P≈0.92%，Cubic 下 P≈2.74%），而 K≥2 的编辑触发概率接近 100%

2. **~90% 的失败仅差 1 个编辑**——不是全面崩溃，而是所有编辑几乎都正确触发了，唯独一个位置漏了。这恰好符合 Poisson 过程"低概率独立事件"的期望

3. **Insert 是最易漏的编辑类型**——Standard 上占 56%，#global# 上占 87%。可能与 Insert 同时决定"位置"和"token"有关

4. **失败样本序列更长**（+5-6 tokens），编辑需求更多，面临的不触发风险更大

---

## 3. 实验二：编辑距离轨迹分布分析

### 3.1 分析方法

在 `sample_euler_oracle` 中记录每一步的 `d(x_t, x_1)`（通过 DP 对齐的 edit distance），然后：

- 将每个样本的完成率 `c(t) = 1 - d(t)/d(0)` 插值到公共 t-grid
- 与理论值 κ(t) 对比（Cubic: κ=t³, Linear: κ=t）
- 按初始编辑距离 L 分层分析
- 交叉比较 Cubic vs Linear scheduler

### 3.2 脚本与使用

**采样脚本**：`scripts/oracle_sample.py`（新增 `--record_trajectory` 参数）

```bash
PYTHONPATH=. python scripts/oracle_sample.py \
    --products_file train_subsets/USPTO_50K_PtoR_aug20/test/src-test.txt \
    --targets_file train_subsets/USPTO_50K_PtoR_aug20/test/tgt-test.txt \
    --vocab_file /data6/.../example.vocab.src \
    --output_dir train_subsets/eval/oracle_standard_cubic \
    --n_samples 10 --n_steps 100 --batch_size 32 \
    --deduplicate 20 --device cpu --scheduler cubic \
    --score_script scripts/score.py --record_trajectory
```

输出：`{output_dir}/trajectory.pt`（每 batch 的 t 和 d 序列）

**分析脚本**：`scripts/oracle_trajectory_analysis.py`

```bash
PYTHONPATH=. python scripts/oracle_trajectory_analysis.py \
    --traj_cubic_std train_subsets/eval/oracle_standard_cubic/trajectory.pt \
    --traj_linear_std train_subsets/eval/oracle_standard_linear/trajectory.pt \
    --traj_cubic_global train_subsets/eval/oracle_global_cubic/trajectory.pt \
    --traj_linear_global train_subsets/eval/oracle_global_linear/trajectory.pt
```

输出：console report + `train_subsets/eval/trajectory_comparison.csv`

### 3.3 代码修改

为支持轨迹记录，修改了以下文件：

| 文件 | 修改内容 |
|------|---------|
| `edit_flows/sampling/oracle.py` | `_align_pair()` 现在返回 `(aligned_0, aligned_1, edit_distance)`；`compute_oracle_model_output()` 额外返回 `edit_dists` 列表 |
| `edit_flows/sampling/euler.py` | `sample_euler_oracle()` 新增 `record_edit_distances` 参数，记录每步 (t, d)；新增 `_compute_batch_edit_dists()` 辅助函数 |
| `scripts/oracle_sample.py` | 新增 `--record_trajectory` 参数，按 batch 保存轨迹到 `trajectory.pt` |

### 3.4 结果：Final State 分布

#### 全量统计

| 数据集 | Scheduler | d=0 (完美) | d=1 | d≤1 | Mean Final d | Final Gap (1-c_mean) |
|--------|-----------|-----------|-----|-----|-------------|---------------------|
| Standard | Cubic | 70.7% | 22.1% | 92.8% | 0.384 | 0.030 |
| Standard | **Linear** | **88.8%** | **10.2%** | **99.0%** | **0.123** | **0.009** |
| #global# | Cubic | 84.8% | 13.1% | 97.8% | 0.178 | 0.030 |
| #global# | **Linear** | **95.1%** | **4.6%** | **99.8%** | **0.051** | **0.009** |

**Gap 减少约 70%**（两个数据集一致）。

#### 按初始编辑距离 L 分层

**Standard 数据集**：

| L 范围 | n | Cubic d=0 | Linear d=0 | 提升 |
|--------|---|-----------|------------|------|
| 1-5 (小) | 3170 | 92.2% | **97.8%** | +5.6pp |
| 6-15 (中) | 2930 | 73.2% | **91.1%** | +17.9pp |
| 16+ (大) | 3900 | 51.4% | **79.8%** | +28.4pp |

**#global# 数据集**：

| L 范围 | n | Cubic d=0 | Linear d=0 | 提升 |
|--------|---|-----------|------------|------|
| 1-5 (小) | 7070 | 91.5% | **97.5%** | +6.0pp |
| 6-15 (中) | 2050 | 74.8% | **92.2%** | +17.4pp |
| 16+ (大) | 880 | 53.5% | **83.0%** | +29.5pp |

**L 越大的样本，从 Linear 获益越多**——因为更多 K=1 编辑，Cubic 下累积不触发概率更高。

### 3.5 结果：Completion Rate 轨迹

#### Completion Rate vs κ(t)

```
       t     κ_cubic  c_cubic  Δ(c-κ)    κ_linear  c_linear  Δ(c-κ)
STD:
   0.200      0.008    0.008   -0.001      0.202     0.200   -0.002
   0.400      0.066    0.063   -0.003      0.404     0.403   -0.001
   0.600      0.212    0.203   -0.009      0.596     0.591   -0.005
   0.800      0.508    0.496   -0.013      0.798     0.795   -0.003
   0.900      0.727    0.710   -0.016      0.899     0.895   -0.004
   0.950      0.856    0.838   -0.018      0.949     0.945   -0.004
   0.990      0.970    0.952   -0.018      0.990     0.985   -0.005

GLOBAL:
   0.200      0.008    0.007   -0.001      0.202     0.200   -0.002
   0.400      0.066    0.061   -0.005      0.404     0.398   -0.007
   0.600      0.212    0.200   -0.012      0.596     0.589   -0.007
   0.800      0.508    0.493   -0.015      0.798     0.792   -0.006
   0.900      0.727    0.704   -0.022      0.899     0.894   -0.005
   0.950      0.856    0.836   -0.020      0.949     0.944   -0.005
   0.990      0.970    0.950   -0.020      0.990     0.985   -0.005
```

**核心观察**：

1. **Linear 几乎完美跟踪 κ(t)=t**——全程偏差仅 0.001-0.007，c_mean 基本与理论对角线重合
2. **Cubic 系统性落后于 κ(t)=t³**——偏差随 t 增长累积（t=0.2 时 -0.001，到 t=0.99 时 -0.018~-0.020）
3. 两个数据集的 Δ(c-κ) 模式高度一致

### 3.6 偏差来源分析

Cubic 下偏差随 t 增长，原因：
- Cubic 的 sched_coeff = 3t²/(1-t³)，在 t→0 时 →0，一半以上时间 sc<1
- 低 sc 阶段每步 hazard 贡献小，编辑触发慢
- 到高 t 时虽然有高 sc，但步数有限（clamped+adaptive），来不及追上

Linear 下偏差极小且不累积：
- Linear 的 sched_coeff = 1/(1-t)，从 t=0 起 ≥1，没有"低效期"
- 每一步都贡献足够的 hazard，编辑在整条轨迹上均匀触发

---

## 4. 生成文件汇总

### 评测结果

| 路径 | 内容 |
|------|------|
| `train_subsets/eval/oracle_standard_cubic/eval.log` | Standard + Cubic 评测 |
| `train_subsets/eval/oracle_standard_linear/eval.log` | Standard + Linear 评测 |
| `train_subsets/eval/oracle_global_cubic/eval.log` | #global# + Cubic 评测 |
| `train_subsets/eval/oracle_global_linear/eval.log` | #global# + Linear 评测 |

### 错误分析

| 路径 | 内容 |
|------|------|
| `train_subsets/eval/oracle_standard_linear/error_analysis.txt` | Standard + Linear 逐样本错误分析 |
| `train_subsets/eval/oracle_global_linear/error_analysis.txt` | #global# + Linear 逐样本错误分析 |

### 轨迹数据

| 路径 | 大小 | 内容 |
|------|------|------|
| `train_subsets/eval/oracle_standard_cubic/trajectory.pt` | 9.6M | Standard + Cubic, 32 batches × ~102 steps × 320 samples |
| `train_subsets/eval/oracle_standard_linear/trajectory.pt` | 9.6M | Standard + Linear, 同上 |
| `train_subsets/eval/oracle_global_cubic/trajectory.pt` | 9.6M | #global# + Cubic, 同上 |
| `train_subsets/eval/oracle_global_linear/trajectory.pt` | 9.6M | #global# + Linear, 同上 |
| `train_subsets/eval/trajectory_comparison.csv` | — | 四路交叉比较数据（t-grid, mean/std/p50） |

### 脚本

| 路径 | 用途 |
|------|------|
| `scripts/oracle_sample.py` | Oracle 采样（支持 `--scheduler`, `--record_trajectory`） |
| `scripts/oracle_error_analysis.py` | 预测 vs ground truth 错误模式分析 |
| `scripts/oracle_trajectory_analysis.py` | 编辑距离轨迹分析与 scheduler 对比 |

---

## 5. 总结

### 5.1 Oracle 剩余失败的根因

**100% 是 K=1 的单次编辑未触发。** 这是 Euler 离散化 + clamp 在有限 integrated hazard H 下的数学必然，不是 bug。

- Linear 下 P(K=1 不触发) ≈ 0.92%（H=4.69），Cubic 下 ≈ 2.74%（H=3.60）
- ~90% 的失败仅差 1 个编辑，~98-99% 差 ≤2 个
- Insert 是最容易漏的编辑类型

### 5.2 Scheduler 对比

**Linear 相对于 Cubic 的改进非常显著且具有跨数据集的一致性：**

| 指标 | Cubic | Linear | 改进 |
|------|-------|--------|------|
| Standard Top-1 | 92.6% | 97.4% | +4.8pp |
| #global# Top-1 | 92.7% | 98.1% | +5.4pp |
| Final gap (1-c_mean) | 0.030 | 0.009 | -70% |
| Gap std | 0.08-0.09 | 0.05 | ~-45% |

### 5.3 关键洞察

1. **不要仅看 n_steps 调参**——同样 100 步，换 scheduler 就能获得 ~70% 的 gap 减少，比盲目加步数高效得多
2. **#global# 格式天然更友好**——R-SMILES root-alignment 减少编辑数，两者叠加效果最佳（Linear + #global#: Top-1 98.1%, d≤1=99.8%）
3. **剩余 ~1% 的 gap 来自离散化的硬下界**——要进一步降低需要提高 n_steps 或 clamp 值，但这属于"治标"，当前瓶颈仍在模型侧
4. **后续模型改进方向**：模型不需要完美预测速率分布，只需要比当前更好——oracle 实验给出了可达上界（Top-1 97-98%），模型当前仅 22-41%，差距主要来自速率不够锐利
