# Beam Search 实验 #4：模型轨迹定性分析与时间 mismatch 验证

## 1. 背景

在 `docs/beam-search/todo4.md` 提出时间 mismatch 假说之后，实验 #4 分两步推进：

1. **定性分析**（50 条 greedy 轨迹逐条审查）：直接观察模型采样轨迹的失败模式
2. **定量验证（P1）**：在 oracle 轨迹上测量错误 t 信号对模型 edit ranking 的退化幅度

核心目的是判断时间 mismatch 是否是 78% → 35.5% gap 的主因，以及后续精力应投向采样侧还是训练侧。

---

## Part 1: 轨迹定性分析

### 2. 实验设置

| 项目 | 内容 |
|------|------|
| Checkpoint | `2026-06-08_17-20-39/checkpoint_step1680000.pt` |
| 数据 | `test_dedup_seed42_1000` 前 50 条 |
| 采样 | `greedy_edit`, `time_mode=depth`, `stop_u_tot_base=0.1` |
| 诊断方式 | 每步记录模型的 chosen edit、oracle best edit(s)、t_val、u_tot_base、oracle rank |

脚本：`experiments/trajectory_diag/run_trajectory_diag.py`

### 3. 失败模式分类

| 类型 | 机制 | 估计占比 | 时间 mismatch 角色 |
|---|---|---|---|
| A 过度编辑 | 序列已对但 u_tot 不降，继续编辑破坏 | 10-16% | **根因** |
| B 错误引爆 | 一步错 → u_tot 飙升 → 越修越坏 | 30-40% | **放大器** |
| C Token 失误 | 位置/类型对，token 差一点 | 10-16% | 基本无关 |
| D 预算不足 | edit_dist ≥ 23, max_edits=20 不够 | 10-16% | 无关 |

**类型 A 典型**（Sample 42, edit_dist=1）：
```
Step 0: t=0.048 → ins,19,C (正确)  序列已 = target
Step 1: t=0.095 → ins,9,)  (破坏)  u_tot 反升至 2.37
        Oracle 最佳候选 score=-8.84（本质无编辑需求）
```

**类型 B 典型**（Sample 39, edit_dist=2）：
```
Step 0: ins,18,C (正确)  u_tot=1.29
Step 1: ins,3,c  (错误)  u_tot 从 1.29 飙到 16.4
...跑满 20 步, u_tot 持续 >25
```

### 3.1 `u_tot_base` 动态

| 轨迹状态 | u_tot_base 行为 |
|----------|---------------|
| 成功轨迹 | 单调递减，完成编辑后降到 stop 阈值以下 |
| 类型 A 失败 | 序列已正确但 u_tot 不降（1.5-3.0），反映时间嵌入干扰 |
| 类型 B 失败 | 错误后急剧飙升（2→16, 3→28），模型检测到"序列坏了" |
| 类型 C 失败 | token 选错后快速下降，触发提前停止 |

---

## Part 2: P1 — 时间 mismatch 定量验证

### 4. 实验设计

**思路**：在 oracle 轨迹的每个 state 上，故意传入与训练分布不匹配的 t 值，测量模型 edit ranking 的退化幅度。

所有 mode 沿**同一 oracle 轨迹**推进（oracle edit 的选择和状态序列完全相同），唯一变量是传给模型的 t 值。

| Mode | t 值 | 模拟场景 |
|------|------|----------|
| `correct` | oracle 轨迹 t = (step+1)/(max_edits+1) | 诊断基线（不是严格意义上的训练分布内） |
| `small_t` | 恒定 t=0.1 | 模型总认为"还在早期" |
| `large_t` | 恒定 t=0.9 | 模型总认为"快结束了" |
| `step0_t` | 恒定 t=1/(max_edits+1) ≈ 0.048 | 模拟快速收敛：t 不随编辑进度增长 |

| 参数 | 值 |
|------|-----|
| Checkpoint / 数据 | 同上（200 条） |
| `max_edits` | 20 |
| `k_ins_token / k_sub_token / k_edit_expand` | 4 / 4 / 16 |
| GPU | NVIDIA A100-SXM4-40GB (GPU 2) |

脚本：`experiments/p1_time_mismatch/run_p1.py`

### 5. 结果

#### 5.1 总体指标

| 指标 | correct | small_t (0.1) | large_t (0.9) | step0_t (~0.048) |
|------|:--:|:--:|:--:|:--:|
| Overall Top-1 | **78.1%** | 73.0% (-5.0pp) | 76.6% (-1.4pp) | 72.3% (-5.8pp) |
| Overall Top-5 (cum) | 88.4% | 86.2% | 85.3% | 85.8% |
| Overall Top-16 (cum) | **95.8%** | 94.6% (-1.3pp) | 94.4% (-1.4pp) | 94.4% (-1.4pp) |
| Not in top-16 | 4.2% | 5.4% | 5.6% | 5.6% |
| Mean score gap | 0.20 | 0.26 | 0.36 | 0.27 |

**总体退化在 5-6pp，处于 todo4.md 决策阈值的"5-20pp = 重要因素"区间。**

#### 5.2 Per-step-bin 分解（关键）

| Step bin | correct | small_t (0.1) | large_t (0.9) | step0_t |
|:--|:--:|:--:|:--:|:--:|
| **step=0** (n=200) | 70.0% | 70.5% (**-0.0**) | **53.5% (-16.5)** | 70.0% (**-0.0**) |
| step=1-3 (n=446) | 77.6% | 77.6% (-0.0) | 75.6% (-2.0) | 77.8% (+0.2) |
| step=4-7 (n=249) | 67.9% | 64.7% (-3.2) | **72.3% (+4.4)** | 63.5% (-4.4) |
| **step>=8** (n=355) | 90.4% | **74.6% (-15.8)** | 94.1% (+3.7) | **73.0% (-17.4)** |

#### 5.3 Top-16 的 Per-step-bin 分解

| Step bin | correct | small_t | large_t | step0_t |
|:--|:--:|:--:|:--:|:--:|
| step=0 | 92.5% | 92.0% | **84.0% (-8.5)** | 92.5% |
| step=1-3 | 96.9% | 96.6% | 98.2% | 96.6% |
| step=4-7 | 92.8% | 92.4% | 93.6% | 93.2% |
| step>=8 | 98.6% | **94.9% (-3.7)** | 96.1% | **93.5% (-5.1)** |

### 6. 关键发现

**发现 1：退化是高度不对称的——取决于 (state, t) 是否匹配训练分布**

| (state, t) 组合 | 训练分布匹配？ | 退化幅度 |
|:--|:--|:--|
| 早期 state (x_0) + 小 t (0.048-0.1) | 匹配（训练中 t≈0 时 state 即 product） | ~0pp |
| 早期 state (x_0) + 大 t (0.9) | **严重 OOD**（训练中 t=0.9 时 state 约 86% reactant） | **-16.5pp** |
| 晚期 state (近 x_1) + 大 t (0.9) | 匹配（大 t 配近收敛 state） | +3.7pp（改善！） |
| 晚期 state (近 x_1) + 小 t (0.048) | **严重 OOD** | **-17.4pp** |

**发现 2：depth 时间映射的方向是对的（t 随步数增长），但幅度与编辑进度脱钩**

depth 映射：step=0→t=0.048, step>=8→t≥0.43。这个"方向"确保了整体退化仅 ~5pp。但在真实搜索中：
- **快速收敛样本**：3 步完成，t 仅到 0.19 → 晚期 state 配小 t → 直接对应 Type A 失败
- **慢速样本**：15+ 步仍未完成，t 到 0.76 → 如果进度落后于 t，则过早收到"晚期"信号

**发现 3：总体 5pp 退化掩盖了子场景 15-17pp 的严重退化**

整体平均是正负效应在不同 step bin 之间抵消的结果。在最相关的子场景（晚期 state + 小 t，对应快速收敛后过度编辑），退化高达 15-17pp——这是 "主要瓶颈" 级别。

**发现 4：Top-16 的退化远小于 Top-1**

所有 mode 的 Top-16 退化仅 1.3-1.4pp。正确编辑即使排不到第一，也几乎总在 top-16 内。这意味着 beam search 在时间修正后有望捕获这些编辑——前提是 state 本身不是 OOD。

---

## Part 3: 综合评估与计划更新

### 7. 时间 mismatch 假说的定位

| 论断 | P1 证据 |
|------|---------|
| 时间 mismatch 真实存在 | 证实。错误 t 确实导致 ranking 退化 |
| 是 78% → 35.5% 的主因 | **需要限定**。整体退化 ~5pp，特定子场景 ~16pp |
| 修正后 Beam 收益可能增大 | 合理推断。Top-16 几乎不变（-1.4pp），但 Top-1 在 OOD 场景下大幅退化，修正 t 后 Beam 候选应更有效分化 |

**更新后的工作假设**：

1. 时间 mismatch 是 Type A 失败（10-16% 样本）的**强候选根因**，但该判断仍需在模型自身轨迹 state 上复核
2. 时间 mismatch 是 Type B 失败（30-40% 样本）的**放大器**：错误发生后小 t 加剧了修复阶段的错误率，修正 t 可减弱但无法消除此类
3. Type C/D 与时间 mismatch 基本无关
4. 即使时间完美修正，per-step Top-1 上限仍是 ~78%（oracle 轨迹），多步累积后上限约 0.78^5.5 ≈ 25-30%，加上停止机制的幸运退出可达 ~35%。**要突破 50%+ 需要同时改善模型自身的轨迹稳定性。**

### 8. 更新后的执行计划

P1 结论：时间 mismatch 是重要因素（整体 ~5pp），在关键子场景下是主要瓶颈（~16pp）。采样侧改进有实质空间但非万能。

**立即执行**：

| 优先级 | 实验 | 说明 |
|:--:|------|------|
| P0 | **P3.5: `t = max(depth_t, 0.3)`** | 一行改动，直接测试"避免极小 t"能否减少 Type A 失败。成本最低，信号最直接 |
| P1 | **P3: fixed time sweep + u_tot 轨迹日志** | 验证固定 t 是否比 depth 更好。P1 数据预示总体可能持平，但需轨迹日志看失败模式是否转移 |
| P2 | **方向 A 原型: `time_mode="utot_progress"`** | 如果 P3.5 有效，进一步实现自适应 t。核心逻辑：`t = 1 - u_tot_base / u_tot_initial` |

**短期**：

- 若 P3.5/方向 A 将 Top-1 从 36% 推至 42-48%：采样侧改进有效，继续调优
- 若改善 <5pp：时间修正的采样侧天花板已到，Type B（错误引爆）是主要矛盾，需考虑训练侧改进

**中长期**：

- 训练侧改进（方向 C/D）始终在路线图上。P1 表明即使 t 完美，per-step Top-1=78%，多步累积后独立准确率约 25-30%。要突破 50%+ 需要模型本身更好——无论是在正确 state 上的排序精度（78% → 85%+）还是在错误 state 上的恢复能力。

---

## Part 4: P3 — FixedTimePolicy 扫参验证

### 10. 实验设计

在 200 条 test 子集上，greedy_edit + FixedTimePolicy，扫 `time_const` × `stop_u_tot_base`。Depth 基线复用 exp D 配置作为对照。

| 参数 | 值 |
|------|-----|
| Checkpoint / 数据 | 同上（200 条） |
| `time_policy` | fixed |
| `time_const` | 0.3, 0.5, 0.7, 0.9 |
| `stop_u_tot_base` | -1, 0.01, 0.05, 0.1, 0.5 |

脚本：`experiments/exp5_fixed_time/run_exp5.py`

### 11. 结果

| Config | Top-1 | Invalid |
|--------|:-----:|:-------:|
| depth_stop0.5 (基线) | 35.5% | 7.5% |
| fixed_tc0.3_stop0.5 | 33.5% | 12.5% |
| fixed_tc0.5_stop0.5 | 37.5% | 6.5% |
| **fixed_tc0.7_stop0.5** | **38.0%** | **5.0%** |
| fixed_tc0.9_stop0.5 | 32.5% | 2.0% |

完整 sweep 见 `experiments/exp5_fixed_time/summary.txt`。

### 12. 关键发现

1. **Fixed time 优于 depth time**。最佳 `tc=0.7, stop=0.5` 达 Top-1=38.0%（+2.5pp over depth），Correct 从 71→76。

2. **最优 time_const 在 0.5-0.7**。tc=0.3（太小）→ 过度编辑，Invalid 12.5%；tc=0.9（太大）→ 编辑不足，Top-1 仅 32.5%。与 P1 的 (state, t) OOD 退化模式一致。

3. **Stop 阈值在 fixed time 下依然关键**：tc=0.7 从 stop=-1 (6.0%) 到 stop=0.5 (38.0%)，差距 32pp。

4. **Fixed time 的改善幅度有限（+2.5pp）**，说明消除时间 mismatch 不是万能药。这与 P1 结论一致——时间 mismatch 是重要因素但不是唯一瓶颈。Ratio/Kappa 等自适应策略可能在 0.5-0.7 基线之上进一步改善。

### 13. 相关文件

| 文件 | 说明 |
|------|------|
| `experiments/trajectory_diag/run_trajectory_diag.py` | Part 1 轨迹定性分析脚本 |
| `experiments/trajectory_diag/output/` | 50 条逐步轨迹 + 样本汇总 |
| `experiments/p1_time_mismatch/run_p1.py` | Part 2 P1 实验脚本 |
| `experiments/p1_time_mismatch/outputs/` | P1 四 mode 对比报告 |
| `experiments/exp5_fixed_time/run_exp5.py` | Part 4 P3 sweep 脚本 |
| `experiments/exp5_fixed_time/outputs/` | 各配置预测 + eval |
| `experiments/exp5_fixed_time/summary.txt` | P3 完整结果汇总 |
| `experiments/exp6_ratio/run_exp6.py` | Part 5 P4 Ratio sweep 脚本 |
| `experiments/exp6_ratio/summary.txt` | P4 完整结果汇总 |
| `experiments/exp7_kappa/run_exp7.py` | Part 6 P5 Kappa 评估脚本 |
| `experiments/exp7_kappa/summary.txt` | P5 完整结果汇总 |
| `docs/beam-search/todo4.md` | 时间 mismatch 假说原始方案 |
| `docs/beam-search/exp3.md` | 上一轮定量实验 |

---

## Part 5: P4 — RatioTimePolicy 扫参验证

### 14. 实验设计

在 200 条 test 子集上，greedy_edit + RatioTimePolicy，扫 `stop_u_tot_base`。

Ratio 策略用模型速率总量的下降比例估计编辑进度。前两步使用 depth κ（编辑尚未体现在 u_tot 下降中），step ≥ 2 时 κ = clamp(1 - u_prev/u_init, ε, 1)。

| 参数 | 值 |
|------|-----|
| Checkpoint / 数据 | 同上（200 条） |
| `time_policy` | ratio |
| `stop_u_tot_base` | -1, 0.01, 0.05, 0.1, 0.5, 2.0 |

脚本：`experiments/exp6_ratio/run_exp6.py`

### 15. 结果

| Config | Top-1 | Invalid | Correct |
|--------|:-----:|:-------:|:-------:|
| ratio_stop-1.0 | 6.0% | 43.5% | 12 |
| ratio_stop0.01 | 39.0% | 7.0% | 78 |
| ratio_stop0.05 | **44.5%** | 7.0% | 89 |
| ratio_stop0.1 | 44.0% | 7.0% | 88 |
| ratio_stop0.5 | **44.5%** | 7.5% | 89 |
| ratio_stop2.0 | 1.5% | 34.0% | 3 |

基线对照：
| Policy | 最佳 Top-1 | Δ vs depth |
|--------|:---------:|:----------:|
| depth (exp5) | 35.5% | — |
| fixed tc=0.7 (exp5) | 38.0% | +2.5pp |
| **ratio (exp6)** | **44.5%** | **+9.0pp** |

完整 sweep 见 `experiments/exp6_ratio/summary.txt`。

### 16. 关键发现

1. **Ratio 自适应时间大幅优于 depth 和 fixed**。最佳 Top-1 = 44.5%（+9pp over depth，+6.5pp over fixed tc=0.7），Correct 从 71→89（+25%）。

2. **Stop 阈值 sweet spot 非常宽**。stop ∈ [0.05, 0.5] 全部在 44-44.5%，策略本身很鲁棒，不需要精调 stop 阈值。

3. **自适应 t 不能替代 stop 机制**。stop=-1 时仍然只有 6.0%（Invalid 43.5%）。即使 Ratio 自动推进 t，模型在序列已完成时仍然会有残留编辑速率，需要外部 stop 兜底。

4. **Fixed time 的天花板被证实**。Fixed 对所有样本用同一个 t，无法同时服务快速收敛样本（需要大 t）和慢速样本（需要小 t）。Ratio 按每样本的实际 u_tot 下降比例自适应地推进时间，直接解决了这个矛盾。

---

## Part 6: P5 — KappaTimePolicy 评估

### 17. 实验设计

Kappa 策略使用基于 flow 守恒的迭代公式：κ' = 1 - (1-κ)·(u-1)/u，假设 (1-κ) 和 u_tot 以相同速率衰减。内置停止条件：u_tot < 1（剩余编辑需求不足一次）。

| 参数 | 值 |
|------|-----|
| Checkpoint / 数据 | 同上（200 条） |
| `time_policy` | kappa |
| `stop_u_tot_base` | -1, 0.01, 0.05, 0.1, 0.5, 2.0 |

脚本：`experiments/exp7_kappa/run_exp7.py`

### 18. 结果

| Config | Top-1 | Invalid | Correct |
|--------|:-----:|:-------:|:-------:|
| kappa_stop-1.0 ~ stop0.5 | **28.0%** | 11.5% | 56 |
| kappa_stop2.0 | 1.0% | 30.0% | 2 |

所有 stop ∈ [-1, 0.5] 产生**完全相同**的结果（28.0%/11.5%/56）。

完整数据见 `experiments/exp7_kappa/summary.txt`。

### 19. 关键发现

1. **Kappa 不升反降**。Top-1 = 28.0%，比 depth 基线（35.5%）还低 7.5pp，比 Ratio 低 16.5pp。

2. **内置停止过早触发**。所有 stop ∈ [-1, 0.5] 结果完全一致，说明内置 `u_tot<1` 在外部阈值生效前就已停止所有样本。Correct 仅 56（vs Ratio 的 89）。

3. **根因：迭代公式对 u_tot 量级假设不成立**。公式中 `(u_tot-1)/u_tot` 隐含假设"一次编辑消耗 1 单位 u_tot"，但模型的实际 u_tot 与编辑计数之间没有这个精确对应关系。例如 u_tot 从 3 降到 2 不代表恰好完成了一次编辑。这导致 κ 推进过快，模型过早收到"晚期"信号，编辑不足。

4. **可能修复方向**（如需后续探索）：在公式中引入可调 scale factor `α`：κ' = 1 - (1-κ)·(u-α)/u；或放弃硬编码的 `-1`，改用 u_init 的比例作为停止条件。

---

## Part 7: 综合评估 — 四种策略排序

### 20. 最终对比

| 策略 | 最佳 Top-1 | Invalid | Correct | Δ vs depth |
|------|:---------:|:-------:|:-------:|:----------:|
| depth | 35.5% | 7.5% | 71 | — |
| fixed tc=0.7 | 38.0% | 5.0% | 76 | +2.5pp |
| kappa | 28.0% | 11.5% | 56 | −7.5pp |
| **ratio** | **44.5%** | **7.0%** | **89** | **+9.0pp** |

### 21. 结论

1. **RatioTimePolicy 是当前最佳时间调度策略**。以 u_tot/u_init 比值驱动 κ 的方案在理论上简单、在实践中有效。+9pp（相对提升 25%）证实了时间 mismatch 是 78%→35.5% gap 的重要成因，且自适应时间能大幅缓解。

2. **Kappa 的迭代方案需要重新设计**。当前公式的"一次编辑 = 1 单位 u_tot"假设不成立，导致 κ 推进过快。方向可以是取消硬编码常数或改用比例-based 停止。

3. **采样侧改进有实质空间**。45% 仍远低于 oracle 的 93%，但相比 depth 的 35.5% 是一次本质性跳跃。下一阶段应测试 beam + ratio（验证"时间修正后 beam 收益增大"的 exp4 假说）、以及在新 checkpoint 上复现。

---

## Part 8: 逐样本交叉对比 — Ratio vs Kappa vs Depth

### 22. 实验设计

**目的**：找出 Ratio 和 Kappa 各自成功/失败的样本，为后续轨迹分析筛选最有诊断价值的案例。

在 200 条 test 子集上，取各 policy 的最佳配置（ratio_stop0.05, kappa_stop-1.0, depth_stop0.5），做逐样本正确性标签的交叉制表。

### 23. 结果

| 类别 | Ratio | Kappa | 数量 | ed 范围 | 特征 |
|:----:|:-----:|:-----:|:----:|:-------:|------|
| **A** | ✓ | ✗ | 50 | 3-22 (avg 5.7) | **Kappa 独败**——主要诊断目标 |
| B | ✗ | ✓ | 17 | 1-16 (avg 3.4) | Kappa 独胜（多为 ed=1） |
| C | ✗ | ✗ | 94 | 1-35 (avg 10.5) | 两者共败 |
| D | ✓ | ✓ | 39 | 2-18 (avg 6.5) | 两者都对 |
| E | ✗ | ✗, ed≤5 | 36 | 1-5 (avg 3.8) | 简单但共败（C 的子集） |

### 24. Category A 的惊人规律

逐条检查 Category A 的 Kappa 预测输出后，发现**极其一致的失败模式**——几乎所有错误都是 "漏掉一个 token"：

```
Target:   O . N ( C )    →  Kappa:  . N ( C )      （漏 O）
Target:   Cl . N C C O   →  Kappa:  . N C C O      （漏 Cl）
Target:   Cl . [nH]      →  Kappa:  . [nH]         （漏 Cl）
Target:   Br . C = C     →  Kappa:  / C = C        （漏 Br 和 .）
Target:   Cl . O C C     →  Kappa:  . O C C        （漏 Cl）
```

这 50 个样本中绝大多数都是：Kappa 在最后一个 token-level 编辑之前就停了，`final_ed=1`。这强烈暗示 Kappa 的 `u_tot < 1` 停止条件在主要编辑完成后、但最后 token 级修正之前就触发了。

### 25. Category B：Kappa 为何独胜

17 个 Category B 样本中，大部分是 edit_dist=1（仅需改一个 token）。在这些极简单样本上，Ratio 倾向于"多做一步"破坏已完成序列，而 Kappa 的快速停止恰好正确。详见 Part 9 的轨迹分析。

### 26. 样本筛选

从各类别中共选出 35 条样本用于后续轨迹诊断（详见 Part 9），覆盖 A（12 条）、B（5 条）、C（8 条）、D（5 条）、E（5 条）。样本列表保存在 `experiments/trajectory_diag/selected_samples.json`。

---

## Part 9: 轨迹诊断 — Ratio / Kappa / Depth 三方对比

### 27. 实验设计

在选定的 35 条样本上，分别用 Ratio（stop=0.05）、Kappa（stop=-1）、Depth（stop=0.5）三种 TimePolicy 运行轨迹诊断。每步记录：t、κ、u_tot_base、u_tot_score、chosen edit、policy 内部状态（Ratio: u_init/u_prev/ratio；Kappa: κ_cur）、停止原因。

脚本：`experiments/trajectory_diag/run_trajectory_diag_v2.py`

| 参数 | 值 |
|------|-----|
| Checkpoint / 数据 | 同上（35 条选定样本） |
| `max_edits` | 20 |
| `k_ins_token / k_sub_token / k_edit_expand` | 4 / 4 / 16 |
| Oracle 对比 | 关闭（加速） |
| GPU | NVIDIA A100-SXM4-40GB (GPU 3/4/5 并行) |

### 28. 总体结果

| Policy | Correct | Accuracy |
|--------|:-------:|:--------:|
| **Ratio** | 17 | 48.6% |
| Depth | 16 | 45.7% |
| Kappa | 10 | 28.6% |

交叉矩阵：5 条三方都对，12 条 Ratio 独对，5 条 Kappa 独对，13 条三方都错。

### 29. 关键发现 1：Kappa 对 multi-edit 样本的一致性过早停止

Kappa 失败的 25 个样本中，**23 个因 policy_stop 触发**（`u_tot<1`），不是跑满 budget。典型轨迹：

```
Sample 17 (edit_dist=2): 需要 "N(C)C" → "O.N(C)C" (插入 '.' 再插入 'O')
  Ratio: 2 步完成 ✓
    Step 0: t=0.048, κ=0.0001 → ins,35,.   u_tot: 2.24
    Step 1: t=0.095, κ=0.0009 → ins,35,O   u_tot: 1.00
    Step 2: κ = 1-1.00/2.24 = 0.55, t=0.82 → u_tot=0.0006 → STOP (外部) ✓

  Kappa: 1 步停止 ✗
    Step 0: t=0.048, κ=0.0001 → ins,35,.   u_tot: 2.24
            update: κ' = 1-(1-0.0001)*(2.24-1)/2.24 = 0.447   (认为 55% 完成!)
    Step 1: t=0.765, κ=0.447 → u_tot=0.984
            → u_tot<1 → STOP!  序列还差 'O' 没插入, final_ed=1
```

**根因**：公式 `κ' = 1-(1-κ)·(u-1)/u` 中 `(u-1)/u` 对 multi-edit 样本过于激进。u_tot 从 2.24 降到 0.98 后，公式认为剩余进度只剩 `(0.98-1)/0.98 < 0`，说明假设"每个编辑消耗 ~1 单位 u_tot"本身就是错误的——一次编辑后 u_tot 的下降量可能从 0.1 到 10+ 不等。

在 Category A 的 12 个样本中：
- Kappa 步数：几乎全是 1（vs Ratio 的 2-7 步）
- `final_ed`：几乎全是 1
- 模式完全一致——做完第一个（大的）编辑后 u_tot 跌破 1，立即停止

### 30. 关键发现 2：Ratio 对 edit_dist=1 样本的过度编辑

```
Sample 42 (edit_dist=1): O → O C (仅需插入一个 'C')
  Ratio: 20 步跑满, 最终错误 ✗
    Step 0: t=0.048 (warmup) → ins,19,C ✓  序列已正确! u_tot=1.19
    Step 1: t=0.095 (warmup) → u_tot 反升至 2.37
            → 模型在 warmup t=0.095 认为"还需编辑" → ins,9,) → 破坏!
    Step 2: κ = 1-2.37/1.19 → 0, t≈0 → 模型持续破坏, 跑满 20 步

  Kappa: 1 步正确 ✓
    Step 0: t=0.048 → ins,19,C ✓  u_tot: 1.19→0.064
            κ' = 1-(1-0.0001)*(1.19-1)/1.19 = 0.84
    Step 1: t=0.943, u_tot=0.064<1 → STOP ✓
```

**根因**：Ratio 的前两步 warmup（强制 depth κ）在极简单样本上是双刃剑——edit_dist=1 时一步就完成了所有编辑，但 warmup 机制让模型继续使用早期 t (0.095)，时间嵌入推高编辑速率，模型对"已完成"序列再做破坏性编辑。随后 u_tot 反升导致 `κ→0`，t≈0 的 "最早期"信号驱动更多破坏，无法自愈。

Category B 的 5 个样本（Kappa 独胜）中，4 个是 edit_dist=1。Kappa 在这些样本上恰好正确的原因是：它没有 warmup，一步编辑后 κ 直接从 0.0001 跳到 0.8+，模型在 t≈0.9 下看到"已完成"序列，u_tot 快速跌破 1 触发停止。

### 31. 关键发现 3：Type B 错误引爆的 κ 动态

```
Sample 39 (edit_dist=2, 双方都败): /C=C/ → C=C 需要两个编辑
  Step 0: 模型选了 ins,3,c (位置不好)  实际需要的是别的编辑
  Ratio: u_tot 1.29→16.4→27.4 → κ→0, t≈0
         → 在 t≈0 下持续插入更多 c → 跑满 20 步 (final_ed=19)
  Kappa: κ 0.0001→0.775 → u_tot=0.133<1 → STOP at step 1
         → 序列已坏但停得快 (final_ed=2)
```

两种策略对 Type B 错误的处理截然不同：
- Ratio：u_tot 飙升 → κ→0 → t≈0 → **恶化循环**。模型在"最早期"信号下被最大化推着编辑，跑满步数。
- Kappa：快速停止。不修复但也不进一步破坏。

这解释了中 Kappa 的 Invalid 率更高（11.5% vs Ratio 的 7.0%）的原因——Kappa 在 Type B 场景下快速停止，停在错误的序列上；Ratio 虽然会在部分 Type B 上跑满步数，但更多正确样本的收益远超这个代价。

### 32. κ 动态对比总结

| 场景 | Ratio κ 行为 | Kappa κ 行为 |
|------|-------------|-------------|
| 简单样本 (ed=1) | Warmup → u_tot 反升 → κ→0 → 崩溃 | 一步到位 κ→0.8+ → 正确停止 |
| 普通样本 (ed=2-5) | Warmup → 逐步 κ→0.5-0.8 → 外部 stop ✓ | 一步后 κ→0.4-0.8 → u_tot<1 → **过早停止** ✗ |
| 错误引爆 (Type B) | u_tot↑ → κ→0 → t≈0 → **恶化循环** ✗ | 快速停止（少犯错但也未修复） |

---

## Part 10: Kappa 失败根因的数学解释

### 33. 为什么 `(u_tot-1)/u_tot` 假设不成立

Kappa 的迭代公式：

```
κ' = 1 - (1-κ) × (u_tot - 1) / u_tot
```

隐含假设：**一次编辑恰好消耗 1 单位 u_tot**。当 u_tot 从 U 降到 U-1 时，κ 推进的幅度恰好匹配一次编辑的进度。

但实际数据中，模型 u_tot 与所需编辑次数之间没有这个精确关系：

| Sample | Step 0 u_tot | 剩余编辑 | 每编辑对应 Δu_tot |
|--------|:-----------:|:------:|:-----------------:|
| 17 (ed=2) | 2.24 | 2 | ~1.12 |
| 42 (ed=1) | 1.19 | 1 | ~1.19 |
| 39 (ed=2) | 1.29 | 2 | ~0.64 |
| 76 (ed=2) | 19.15 | 2 | ~9.6 |

u_tot 的绝对值在不同样本间可以差一个数量级。公式 `(u_tot-1)/u_tot` 在 u_tot=2 时认为 "50% 完成"，在 u_tot=20 时认为 "95% 完成"——但实际上两者可能都还需要一次编辑。

### 34. Kappa 对 "u_tot<1" 停止条件的敏感性

停止条件 `u_tot < 1` 等价于 "剩余编辑需求不足一次"。但 u_tot 是模型的**预测值**，不是 ground truth。当 u_tot 在 0.5~1.5 之间时，模型的不确定性最高——此时恰好是决定"能否停下来"的关键窗口。Kappa 将这个脆弱的判断直接交给模型，没有任何缓冲。

相比之下，Ratio 使用外部的 `stop_u_tot_base` 阈值（0.05），提供了充分的缓冲空间——即使模型认为 u_tot 还有 ~0.5，序列可能已经是对的。

---

## Part 11: 更新结论与改进方向

### 35. 假设验证结果

| # | 假说 | 验证结果 |
|---|------|---------|
| H1 | Kappa κ 推进太慢 → Type A 过度编辑 | **推翻**。Kappa κ 推进**太快**（一步 κ 就跳 0.4+），导致过早停止，不是过度编辑 |
| H2 | Kappa 停止条件 `u_tot<1` 触发不当 | **证实**。u_tot<1 在 multi-edit 样本的最后 1-2 步前就触发，是最主要的失败原因 |
| H3 | Ratio 的 warmup 是关键优势 | **部分证实**。Warmup 对 multi-edit 样本是优势（防止 κ 过早跳变），但对 ed=1 样本是劣势（阻止 κ 快速前进） |
| H4 | `(u_tot-1)/u_tot` 假设不成立 | **证实**。u_tot 的绝对值与编辑次数无精确对应，每编辑 Δu_tot 可差 10 倍 |
| H5 | Type B 下 Kappa 能自动"倒车" | **推翻**。Kappa 在 Type B 下快速停止，不修复。Ratio 在 Type B 下反而更差（κ→0 恶化） |

### 36. 更新后的工作假设

1. **Ratio 是当前最佳策略**（+9pp），其核心优势来自 `u_tot/u_init` 比例的相对度量——不依赖 u_tot 绝对值，对不同样本自适应
2. **Kappa 的根本问题是 `(u_tot-1)` 中的硬编码常数 `-1`**，在 u_tot 绝对值跨度大的情况下失效
3. **两类错误有本质上的不对称**：Kappa 的过早停止（漏掉最后 1-2 token）远比 Ratio 的过度编辑（ed=1 样本）常见——Category A: 50 条 vs Category B: 17 条
4. **Type B（错误引爆）对所有策略都是难题**，但 Ratio 的 κ→0 退化使其在此场景下更差

### 37. 改进方向

基于轨迹分析，按优先级排列：

1. **Kappa + 安全边际（成本最低，信号最直接）**：将停止条件从 `u_tot < 1` 改为 `u_tot < 0.1` 或 `u_tot < 0.5`。这直接解决过早停止（Category A 的主要模式），同时保留对简单样本的快速停止优势。改动一行代码。

2. **Ratio + 最大 t 限制**：当 κ→0 时（Type B 恶化），限制 t 的最小值（如 t ≥ 0.3），防止模型在"最早期"信号下疯狂破坏。这可能同时改善 Category B 和 Type B 场景。

3. **Kappa + warmup**：给 Kappa 也加前两步 depth κ，缓解 step 0 后 κ 跳变过大的问题。结合改进 #1 的停止条件。

4. **Kappa 的 scale factor**：将公式改为 `κ' = 1 - (1-κ) * (u_tot - α) / u_tot`，其中 α 从数据统计中得到（或设为 u_init 的比例，如 α = u_init * 0.1）。

### 38. 相关文件

| 文件 | 说明 |
|------|------|
| `experiments/trajectory_diag/run_trajectory_diag.py` | Part 1 原始轨迹诊断（仅 depth） |
| `experiments/trajectory_diag/run_trajectory_diag_v2.py` | Part 9 新版轨迹诊断（支持 TimePolicy） |
| `experiments/trajectory_diag/output/trajectory_trace_{ratio,kappa,depth}.txt` | 三策略 35 条逐步轨迹 |
| `experiments/trajectory_diag/output/sample_summary_{ratio,kappa,depth}.txt` | 三策略样本汇总 |
| `experiments/trajectory_diag/selected_samples.json` | 35 条选定样本索引 |
| `edit_flows/sampling/time_policy.py` | TimePolicy 实现（四策略） |
