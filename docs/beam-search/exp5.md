# Beam Search 实验 #5：显式 STOP 与 First-Event 概率化搜索

## Part 1: Phase 0 — STOP 诊断

### 1. 背景与目的

`todo5.md` 提出将 STOP 视为与 insert/substitute/delete 并列的显式动作。Phase 0 的目标是在不改动采样逻辑的前提下，在现有 RatioTimePolicy greedy 轨迹上测量 Poisson 概率框架的核心量级，判断方向是否可行。

### 2. 实验设置

| 项目 | 内容 |
|------|------|
| Checkpoint | `2026-06-08_17-20-39/checkpoint_step1680000.pt` |
| 数据 | `test_dedup_seed42_1000` 前 200 条 |
| 采样 | `greedy_edit`, `time_policy=ratio`, `stop_u_tot_base=0.05` |
| 测量 | 每步记录 U_global, U_topK, p_stop, p_other |

脚本：`experiments/exp8_stop_diag/run_stop_diag.py`

### 3. 核心指标定义

- `U_global`：所有合法编辑的 base-rate 总量（`_compute_executable_u_tot`）
- `U_topK`：top-K 候选编辑的 base-rate 之和
- `p_stop = e^(-U_global)`：Poisson 假设下"无需任何编辑"的概率
- `p_other = max(0, 1 - p_stop - Σ p_e)`：候选截断丢失的概率质量

### 4. 结果

#### 4.1 候选截断不破坏概率质量

| 指标 | 值 |
|------|:--:|
| Mean p_other | 0.086 |
| Median p_other | 0.022 |
| P90 | 0.259 |
| > 0.3 的比例 | 6.8% (102/1502) |

top-K 候选保留了绝大部分编辑概率，`k_edit_expand=16` 对概率化框架足够。

#### 4.2 p_stop 总体分布

| 指标 | 值 |
|------|:--:|
| Mean p_stop | 0.101 |
| Median p_stop | 0.006 |
| P90 | 0.363 |
| > 0.5 的比例 | 3.6% (54/1502) |

大多数编辑步上 p_stop 很小（模型认为"还需要编辑"），符合预期。

#### 4.3 p_stop 在序列已正确时的行为

30 个 `seq_correct` step 上：

| 指标 | 值 |
|------|:--:|
| Mean p_stop | 0.263 |
| Median p_stop | 0.073 |
| > 0.5 | 23.3% |
| > 0.1 | 40.0% |

p_stop 在正确序列上明显上升，但仍有 60% 的情况下低于 0.1——模型未被训练过"序列正确时应输出极低 u_tot"。这不影响显式 STOP 的相对比较逻辑。

#### 4.4 成功轨迹上的 U_global 动态

```
Sample 9 (ed=4):  U: 5.48 → 3.28 → 2.03 → 0.99 → 0.00
                  p_stop: 0.004 → 0.038 → 0.13 → 0.37 → 1.0

Sample 10 (ed=3): U: 3.35 → 2.06 → 1.00 → 0.00
                  p_stop: 0.035 → 0.128 → 0.37 → 1.0
```

U_global 单调递减，p_stop 在最后几步稳定上升。

#### 4.5 Type A 过度编辑的典型案例

Sample 2 (ed=1)，当前 ratio+threshold 花了 8 步才停：

```
Step 0: del → seq 已正确! U=0.38, p_stop=0.69
Step 1: ins (破坏) → U=0.37, p_stop=0.69
...
Step 7: u_tot_base < 0.05 → 外部阈值触发 stop
```

Step 1 时 p_stop=0.69，远大于任何单个 edit 的 p_e（约 0.05-0.10）。显式 STOP 框架下，STOP 会在第 1 步直接胜出，避免后续 6 步的破坏性编辑。

### 5. Phase 0 结论

1. **Poisson 概率框架在数值上可行。** p_other 很小（median 0.022），候选截断不破坏分布。
2. **显式 STOP 能直接解决 Type A 过度编辑。** STOP 作为候选动作后，p_stop 在正确序列上远大于单个 edit 的 p_e。
3. **主要挑战不在 p_stop 的绝对值，而在相对比较。** 即使 p_stop=0.1，只要 > 所有 p_e，STOP 就会正确触发。

---

## Part 2: Phase 1 — 实现：方案 A / B 的代码改动

### 6. 实现的三个方案

| 方案 | 来源 | κ' 公式 | p_stop 公式 |
|------|------|---------|------------|
| **A** (Poisson) | `todo5.md` §4.1 | `κ' = 1 - (1-κ)·max(0, (U-1)/U)` = `κ + (1-κ)/U` (U>1 时) | `e^{-U}` |
| **B** (Frozen-Hazard) | `todo5.md` §4.2 | `κ' = κ + (1-κ)·(1/U - e^{-U}/(1-e^{-U}))` | `e^{-U}` |
| **Ratio** (混合参考) | `exp4.md` RatioTimePolicy | `κ = 1 - u_prev/u_init` | `e^{-U}` |

三种方案共享相同的 p_stop / p_e 计算（Poisson 假设下的 first-event 概率），差异仅在于 κ' 更新方式。

### 7. 代码结构

#### 7.1 新增数据结构

```python
@dataclass
class ActionCandidate:
    kind: str          # "stop" | "edit"
    log_prob: float    # log p(action)
    edit: Optional[EditCandidate] = None
```

#### 7.2 新增参数（`sample_greedy_single_edit`）

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `explicit_stop` | bool | False | 启用显式 STOP |
| `kappa_mode` | str | `"ratio"` | κ 更新方式：`"poisson"` / `"frozen_hazard"` / `"ratio"` |
| `p_stop_mode` | str | `"absolute"` | `"absolute"` = e^{-U}；`"normalized"` = e^{-U/U_init} |
| `fh_warmup_steps` | int | 0 | warmup 步数（使用 depth κ 的初始步数） |

#### 7.3 核心逻辑

Per-step 的 per-sample 循环中，当 `explicit_stop=True` 时：

1. 计算 `U_global`（所有合法编辑的 base-rate 总量）
2. 计算 `p_stop = e^{-U}`，`log_p_stop = -U`
3. 对每个候选编辑 e：`log_p_e = log(1-p_stop) + log_u_e - log(U)`
4. 构造 `ActionCandidate("stop", log_p_stop)` 和 `ActionCandidate("edit", log_p_e, e)`
5. 选择 `log_prob` 最大的 action
6. 若 STOP 胜出 → 标记 finished；否则应用选中的 edit
7. 若 `kappa_mode` 为 `"poisson"` 或 `"frozen_hazard"`，调用对应公式更新 per-sample κ

#### 7.4 改动文件

| 文件 | 改动 |
|------|------|
| `edit_flows/sampling/beam.py` | 新增 `ActionCandidate`、`_update_fh_kappa`、`_update_poisson_kappa`；greedy 循环新增 explicit_stop 分支 |
| `scripts/sample_retro.py` | 新增 `--explicit_stop`、`--kappa_mode`、`--p_stop_mode`、`--fh_warmup_steps` |
| `experiments/exp8_explicit_stop/run_exp8.py` | A/B 对比实验脚本 |

---

## Part 3: Phase 1 — 实验结果

### 8. 实验设置

| 项目 | 内容 |
|------|------|
| Checkpoint | `2026-06-08_17-20-39/checkpoint_step1680000.pt` |
| 数据 | `test_dedup_seed42_1000` 前 200 条 |
| 采样 | `greedy_edit`, `max_edits=20` |
| 候选 | `k_ins_token=4, k_sub_token=4, k_edit_expand=16` |

脚本：`experiments/exp8_explicit_stop/run_exp8.py`

### 9. 总体结果

| Config | Top-1 | Invalid | Correct | Δ vs baseline | 方案 |
|--------|:-----:|:-------:|:-------:|:-------------:|------|
| `ratio_stop0.05` | 44.5% | 7.0% | 89 | — | 基线（RatioTimePolicy + 外部阈值） |
| **`poisson_abs`** | **45.5%** | **4.0%** | **91** | **+1.0pp** | **方案 A**（Poisson κ，无 warmup） |
| **`fh_abs`** | **46.0%** | **4.0%** | **92** | **+1.5pp** | **方案 B**（FH κ，无 warmup） |
| `poisson_abs_warmup2` | 41.0% | 6.0% | 82 | −3.5pp | 方案 A + 2 步 warmup |
| `fh_abs_warmup2` | 41.0% | 6.5% | 82 | −3.5pp | 方案 B + 2 步 warmup |
| `estop_ratio_abs` | 41.0% | 9.0% | 82 | −3.5pp | 混合（RatioTimePolicy κ + 显式 STOP） |
| `poisson_norm` | 5.0% | 1.0% | 10 | −39.5pp | 方案 A + e^{-U/U_init}（**废弃**） |
| `fh_norm` | 2.0% | 2.5% | 4 | −42.5pp | 方案 B + e^{-U/U_init}（**废弃**） |

> 注：`_norm` 配置因 p_stop = e^{-1} ≈ 0.37 在 step 0 对所有样本恒成立，导致几乎立即触发 STOP。该方向已废弃，不在后续讨论范围内。

### 10. 逐样本交叉分析

#### 10.1 方案 A vs 方案 B

| 分类 | 数量 | 说明 |
|------|:----:|------|
| A ✓ B ✓ | 90 | 两者都对 |
| A ✓ B ✗ | 1 | 仅 Poisson 对 |
| A ✗ B ✓ | 2 | 仅 FH 对 |
| A ✗ B ✗ | 107 | 两者都错 |

**方案 A 和 B 几乎等价。** 在 200 条样本上仅 3 条有分歧，两种 κ' 公式在实际数据上的行为差异极小。

#### 10.2 方案 B（最佳）vs 基线

| 分类 | 数量 | 说明 |
|------|:----:|------|
| 基线 ✓ FH ✓ | 78 | 两者都对 |
| 基线 ✓ FH ✗ | 11 | FH 破坏（基线对、FH 错） |
| 基线 ✗ FH ✓ | 14 | FH 修复（基线错、FH 对） |
| 基线 ✗ FH ✗ | 97 | 两者都错 |

**净收益：+3 样本（14 − 11）。** FH 修复的样本多为 Type A（过度编辑后更快停止），破坏的样本多为 FH 提前停止导致漏 token。

#### 10.3 方案 A vs 基线

| 分类 | 数量 |
|------|:----:|
| 基线 ✓ Poisson ✓ | 76 |
| 基线 ✓ Poisson ✗ | 13 |
| 基线 ✗ Poisson ✓ | 15 |
| 基线 ✗ Poisson ✗ | 96 |

**净收益：+2 样本（15 − 13），模式与方案 B 一致。**

#### 10.4 预测长度分析

| Config | avg_len | target avg_len | 说明 |
|--------|:-------:|:--------------:|------|
| `ratio_stop0.05` | 54.0 | 53.4 | 基线略长（过度编辑倾向） |
| `poisson_abs` | 52.6 | 53.4 | 稍短（停止更早） |
| `fh_abs` | 52.6 | 53.4 | 稍短（停止更早） |
| `poisson/fh_warmup2` | 53.5 | 53.4 | 接近基线 |

方案 A/B 无 warmup 的平均长度比基线短约 1.4 tokens，与 Invalid 降低（7.0%→4.0%）一致——显式 STOP 在破坏序列之前更早退出。warmup 配置的长度更接近基线，但正确率显著更低（82 vs 91/92），说明"更长"不代表"更正确"，而是在错误轨迹上浪费了更多步数。

### 11. 关键发现

#### 11.1 方案 A 未被否定——之前的 Kappa 失败在停止逻辑

原 `KappaTimePolicy`（exp7）使用与方案 A 完全相同的 κ' 公式，但 Top-1 仅 28.0%。其失败根因是 `u_tot < 1` 的硬停止条件——在 multi-edit 样本上过早触发（详见 exp4.md Part 10）。

当停止机制改为显式 STOP（p_stop 与 p_e 在同一概率空间竞争）后，方案 A 从 28.0% 跃升至 45.5%，反超基线 +1pp。这说明：**Poisson κ' 公式本身是可行的，之前的问题出在停止逻辑。**

#### 11.2 方案 A 和 B 在实际数据上等价

两种公式的差异仅在 FH 多了一个修正项 `-e^{-U}/(1-e^{-U})`：

- U 大时（U > 3）：该项 ≈ 0，两种公式几乎一致
- U 小时（U < 1）：Poisson 直接跳到 κ' ≈ 1，FH 渐进逼近 1

在实际轨迹中，大多数编辑步的 U 在 1-5 之间，两种公式的 κ' 差异不超过 0.05。200 条样本中仅 3 条有最终结果分歧。两者可以被视为同一思路的两个数值变体。

#### 11.3 Warmup 对自适应 κ 策略有害

两种方案在 warmup=0 时均优于 warmup=2（+3-4 样本）。warmup 期间使用非自适应的 depth κ，快速收敛样本在 warmup 中收到错误的时间信号，导致破坏性编辑。这与 exp4.md Part 9 对 RatioTimePolicy warmup 的轨迹分析一致——warmup 对 edit_dist=1 样本是双刃剑。

#### 11.4 显式 STOP 降低 Invalid 率

所有显式 STOP 配置的 Invalid 率（4.0-9.0%）均低于或接近基线（7.0%），即使正确率更低的配置也不例外。这说明显式 STOP 在"避免产生非法 SMILES"方面有天然优势——当序列被破坏后，u_tot 倾向于飙升，p_stop → 0，但一旦编辑回到合法区域，p_stop 重新上升，自然的停止行为减少了最极端的编辑混乱。

#### 11.5 归一化 p_stop（e^{-U/U_init}）方向不可行

该公式在 step 0 对所有样本恒有 p_stop = e^{-1} ≈ 0.37，log(p_stop) ≈ −1.0 远超任何 edit 的 log(p_e)（约 −5 ∼ −10）。几乎所有样本在第一步就选择 STOP。修正方向可能是使用训练集统计的全局常数替代 U_init（如 p_stop = e^{-U / median(U_init)}），或改用 sigmoid 形式。

#### 11.6 显式 STOP 不能完全替代外部阈值

`estop_ratio_abs`（RatioTimePolicy κ + 显式 STOP）Top-1 仅 41.0%，而基线（RatioTimePolicy κ + 外部阈值 0.05）为 44.5%。外部阈值 0.05 是经过实验调优的经验值，纯理论的 `e^{-U}` 公式在某些场景下不如经验阈值精确。但方案 A/B 通过不同的 κ 轨迹间接改善了 STOP 时机，弥补了这一差距。

---

## Part 4: 结论与后续方向

### 12. 总体结论

1. **显式 STOP 框架成立。** 方案 A/B 均优于基线（+1-2pp），且 Invalid 率从 7.0% 降至 4.0%。Phase 0 的诊断预测（Type A 过度编辑被修复）得到验证。

2. **方案 A/B 几乎等价，推荐方案 B 作为默认。** FH 公式在数值上略优（+0.5pp），且在 U→0 时行为更平滑（Poisson 在 U≤1 时直接跳到 κ≈1）。实际差异极小，选择 FH 主要是工程偏好。

3. **核心瓶颈仍然是模型的 u_tot 语义未校准。** 显式 STOP 的改善来自"相对比较"（p_stop vs p_e），而非 p_stop 的绝对校准。Phase 0 已显示 p_stop 在正确序列上中位数仅 0.073。要突破 50%+，需要模型本身在正确序列上输出更低的 u_tot。

4. **归一化 p_stop 需要重新设计。** e^{-U/U_init} 在 step 0 恒为 e^{-1}，不可用。后续可尝试 `e^{-U / c}`（c 来自训练集统计）或 `sigmoid(-α(U-β))`。

### 13. 推荐的后续方向

按优先级排列：

| 优先级 | 方向 | 说明 |
|:------:|------|------|
| **P0** | 混合 STOP（explicit_stop + 小阈值兜底） | 在 `fh_abs` 上叠加 `stop_u_tot_base ∈ {0.001, 0.005, 0.01}`，防止极端误停。成本最低，可能再推高 1-2pp |
| **P1** | 修复归一化 p_stop | 用全局常数 c 替代 U_init，或尝试 sigmoid 形式 |
| **P2** | Beam + 显式 STOP | 若 P0 将 greedy 推至 48%+，接入 beam 评估收益是否放大 |
| **P3** | 训练侧 STOP 语义 | 让模型在训练时就学会"序列正确时 u_tot → 0"，而不依赖外部公式 |

### 14. 相关文件

| 文件 | 说明 |
|------|------|
| `experiments/exp8_stop_diag/run_stop_diag.py` | Phase 0 诊断脚本 |
| `experiments/exp8_stop_diag/outputs/diag.jsonl` | 每步诊断记录（1678 条） |
| `experiments/exp8_stop_diag/outputs/summary.json` | 每样本汇总 |
| `experiments/exp8_stop_diag/summary.txt` | Phase 0 诊断汇总 |
| `experiments/exp8_explicit_stop/run_exp8.py` | Phase 1 A/B 对比实验脚本 |
| `experiments/exp8_explicit_stop/summary.txt` | Phase 1 完整结果汇总 |
| `experiments/exp8_explicit_stop/summary.json` | Phase 1 结果 JSON |
| `experiments/exp8_explicit_stop/outputs/*/predictions.txt` | 各配置 200 条预测 |
| `edit_flows/sampling/beam.py` | 显式 STOP 实现 |
| `docs/beam-search/todo5.md` | 显式 STOP 方案设计 |
| `docs/beam-search/exp4.md` | 时间 mismatch 实验（Ratio/Kappa 等） |
