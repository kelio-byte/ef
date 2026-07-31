# Beam Search 实验路线图 v4：时间 mismatch 假说与下一步计划

## 1. 从初始实现到现在的完整脉络

### 1.1 初始想法

Edit Flows 的原始采样方式是 **Euler 多编辑并行采样**：在每个时间步，所有位置独立地以概率触发编辑。这种方式的问题是：多个编辑同时发生，可能互相干扰；且需要较多采样步数（n_steps=100+）。

一个自然的替代方案是 **单编辑搜索**：每步只执行一个最优编辑，通过贪心或束搜索逐步逼近 target。这类似于自回归解码中的 greedy/beam search，但操作对象是编辑而非 token。

### 1.2 实现与 Bug 修复历程

| 阶段 | 文档 | 关键内容 |
|------|------|----------|
| 方案设计 | `beam-search/todo1.md` | 单编辑搜索的初始设计 |
| 实现 | `beam-search/impl.md` | `beam.py` 初版实现 |
| Bug 修复 | `beam-search/todo2.md` | 发现并修复 5 个 correctness bug（beam stop 语义、dead-end 保活、BOS 保护、no-op 过滤、reverse-op 规则） |
| 实验 #1 | `beam-search/exp1.md` | 模型 Greedy/Beam Top-1=0%，初步归因"单编辑搜索不可行" |
| 实验 #2 | `beam-search/exp2.md` | Oracle Greedy=96%，**推翻 exp1 结论**；模型 ranking 诊断为"62% 正确编辑不在 top-16"→归因"模型排序极差" |
| 审查 | `beam-search/todo3.md` | 发现诊断脚本存在 t=0 边界 + tie 口径两个 bug；制定实验 A-E 计划 |
| 实验 #3 | `beam-search/exp3.md` | 修正口径后：模型 ranking 实际很强（Top-16=96.4%）；模型 Greedy/Beam 最高 Top-1=35.5%；Beam 收益仅 ~2pp |

### 1.3 当前代码状态

`edit_flows/sampling/beam.py`：
- Greedy 和 Beam 两种单编辑采样均已实现
- `time_mode="depth"`：`t = (step + 1) / (max_edits + 1)`（interior mapping，避免 t=0 边界）
- `time_mode="fixed"`：`t = time_const`
- `stop_u_tot_base`：基于模型基础速率总量的早停机制
- 评分使用 base rate（`use_rate_reparam=true` 时模型原始输出），`k(t)` 在分子分母中抵消
- 30 个单元测试通过

`scripts/edit_ranking_diag_v2.py`：
- 沿 oracle 轨迹推进，每步测量 oracle 最优编辑在模型候选池中的排名
- Tie-aware oracle 匹配
- Score 阈值早停（-3.0）过滤收敛后噪声
- Per-step-bin 分桶统计

### 1.4 当前核心数据

| 指标 | Test | Train-subset |
|------|:----:|:------------:|
| Per-step edit ranking Top-1（oracle 轨迹） | 77.9% | 83.7% |
| Per-step edit ranking Top-16（oracle 轨迹） | 96.4% | 99.4% |
| Greedy Top-1（最终生成） | 35.5% | — |
| Beam-5 Top-1（最终生成） | 36.0% | — |
| Beam Δ over Greedy | ~2pp（统计不显著） | — |

核心矛盾：**per-step 排序 78% 正确，但多步累积后最终只有 35.5%。**

---

## 2. 误差累积假说及其局限

### 2.1 纯误差累积估算

若 5.55 步（exp3 实测平均步数）中每步独立，per-step Top-1=77.9%，则最终成功率约为 0.779^5.55 ≈ 25%。实际 35.5% 高于此值，说明 stop 阈值帮助模型提前退出了部分原本会出错的轨迹。

### 2.2 误差累积假说无法解释的部分

1. **为什么 Beam 收益这么小？** 如果 per-step Top-1=77.9%、Top-16=96.4%，理论上 beam 应能捕获 ~18.5pp 的 ranking 差距。实际仅 ~2pp（且统计不显著）
2. **为什么 Oracle 轨迹 ranking（78%）与 first-step 静态 prediction（72%）都在同一水平，但最终生成差这么多？**
3. **为什么 stop_u_tot_base 如此关键？** 不禁用早停时 Top-1 仅 4%——这说明模型在持续进行破坏性编辑，而不仅仅是"某一步选错了"

这些矛盾指向一个更深层的问题：**模型自身轨迹上的 (state, t) 对与训练分布不匹配**。

---

## 3. 时间 mismatch 假说

### 3.1 训练时的时间语义

训练时 `t ~ Uniform(0, 1)`，`z_t` 通过条件概率路径采样：

```
z_t = z_1  with prob κ(t)      # 取 target token
z_t = z_0  with prob 1-κ(t)    # 保留 source token
```

Cubic scheduler 下 κ(t) = t³，因此 (state, t) 的联合分布为：

| t | κ(t) | z_t 中 target token 占比 | 模型学到的"典型状态" |
|:--:|:----:|:----------------------:|---------------------|
| 0.05 | 0.0001 | ~0% | 几乎完全是 product |
| 0.19 | 0.007 | ~0.7% | 绝大部分是 product |
| 0.5 | 0.125 | ~12.5% | 主要是 product，少量 reactant |
| 0.8 | 0.512 | ~51% | 约各半 |
| 0.95 | 0.857 | ~86% | 主要是 reactant |

模型的时间嵌入学到的是：**"在这个 t 下，通常还需要多少编辑"** 的先验。这与 k(t) 不同——k(t) 是速率缩放系数，时间嵌入是模型内部表示的一部分。

### 3.2 Beam/Greedy 搜索时的时间语义

`time_mode="depth"` 下 `t_val = (step + 1) / (max_edits + 1)`，max_edits=20 时：

| step | t_val | κ(t_val) | "期望"的 target% |
|:----:|:-----:|:--------:|:---------------:|
| 0 | 0.048 | 0.0001 | ~0% |
| 3 | 0.19 | 0.007 | ~0.7% |
| 5 | 0.286 | 0.023 | ~2.3% |
| 10 | 0.524 | 0.144 | ~14% |
| 15 | 0.762 | 0.442 | ~44% |
| 19 | 0.952 | 0.863 | ~86% |

**t 完全由步数决定，与实际编辑进度无关。**

### 3.3 Mismatch 的具体场景

#### 场景 A：快速收敛（最常见）

- 某样本只需 3 次编辑即可从 product → reactant
- Step 3 时：序列已接近 x_1（~90% 正确），但 `t_val = 0.19`
- 训练中 t=0.19 的 state 仅有 0.7% target token，模型学到"t=0.19 = 还早，需要大量编辑"
- **后果**：时间嵌入的"早期"先验推高模型速率输出 → `u_tot_base` 不下降 → 早停不触发 → 继续编辑破坏已完成序列
- **这就解释了为什么 stop=-1 时 Invalid=43.5%、Top-1=4%**：模型在"已完成"的状态上被 t 信号驱使继续输出高编辑速率

#### 场景 B：困难样本编辑不足

- 某样本需要 15+ 次编辑
- Step 15 时：序列可能只完成 60%，但 `t_val = 0.76`
- 训练中 t=0.76 的 state 已有 44% target token，模型学到"t=0.76 = 接近完成"
- **后果**：时间嵌入的"晚期"先验压低模型速率输出 → 提前触发早停 → 编辑不充分

### 3.4 该假说对现有数据的统一解释

| 实验 | (state, t) 分布 | t 与 state 是否匹配训练 | 观察到的 ranking |
|------|:--------------:|:---------------------:|:---------------:|
| exp3 ranking 诊断 | oracle 轨迹 state + 对应 t | **部分匹配**（t 方向合理，但 oracle 单编辑轨迹不等于训练时的 Bernoulli 路径态） | Top-1=78%, Top-16=96% |
| first-step analysis | `x_t = x_0` + 各 t 值 | **匹配**（`x_0` 就是 t→0 的训练分布） | Full First-Edit=72% |
| exp D greedy/beam | 模型自身轨迹 state + 步数 t | **不匹配**（编辑进度 ≠ t） | 最终 Top-1=35.5% |

以及：

| 现象 | 时间 mismatch 解释 |
|------|-------------------|
| stop=-1 时 Top-1 仅 4%（远低于 0.78^5.5≈25%） | 快速收敛后，小 t 驱使模型继续输出高编辑速率，主动破坏已完成序列 |
| stop=0.01 就产生巨大改善（+25pp） | 即使时间嵌入推高速率，u_tot_base 仍有下降趋势，轻微阈值即能捕获 |
| Beam 收益仅 ~2pp | 时间 mismatch 对所有候选路径的同向偏差，beam 保留多条路径无法纠正 |
| ranking Top-16=96% 但最终仅 36% | ranking 是在 oracle 引导的诊断轨迹上测量，不代表模型自身轨迹上的真实排序能力 |

---

## 4. 可能的解决方向

### 4.1 方向 A：基于编辑进度的自适应时间（推荐优先探索）

不按 `step/max_edits` 给 t，而是按实际编辑完成度：

```
t_progress = 1 - u_tot_base(t) / u_tot_base(t=0)
```

- 初始状态：`u_tot_base` 高 → `t_progress` 小 → 模型收到"早期"信号，输出强编辑
- 接近完成：`u_tot_base` 低 → `t_progress` 接近 1 → 模型收到"晚期"信号，输出弱编辑
- 语义与 flow 训练一致：`u_tot → 0` 等价于 `t → 1`

**优点**：
- 不需要知道 target
- 与现有 `stop_u_tot_base` 机制统一（stop 阈值等价于 t 接近 1 时的行为）
- `u_tot_base(t=0)` 只需在 step 0 计算一次

**潜在问题**：
- `u_tot_base` 本身也是被时间 mismatch 污染的模型输出。初始估计可能不可靠
- 需要用 ranking 诊断验证：在进度自适应 t 下，模型 ranking 是否优于深度 t

### 4.2 方向 B：固定时间（`time_mode="fixed"`）

exp2 中 Oracle Greedy 用 `time_const=0.5` 达到 96%，说明固定时间本身不是致命问题。

**优点**：实现简单，完全消除时间 mismatch
**缺点**：放弃了时间信息——模型在 t=0.5 时可能对"该编辑多少"的校准不是最优

如果选择这个方向，建议对 `time_const` 做一次扫描（0.3, 0.5, 0.7）。

### 4.3 方向 C：训练侧数据增强

在训练时混入"非标准"(state, t) 对，让模型学会不依赖时间先验：

- 对接近 x_0 的 state 给大 t
- 对接近 x_1 的 state 给小 t

**优点**：从根本上解决分布 mismatch
**缺点**：需要重训，周期长；改动训练逻辑有风险

### 4.4 方向 D：解耦时间嵌入与速率预测

将时间嵌入只用于 `k(t)` 速率缩放（已有 `use_rate_reparam` 做了一部分），让模型的 base rate 预测尽量不依赖时间嵌入：

- 当前架构：`x = token_emb + time_emb + pos_emb`，time_emb 贯穿所有层
- 可以尝试：将 time_emb 仅加到 rate head 而非主干，或使用自适应归一化（AdaLN）替代加法

**优点**：结构性修复
**缺点**：需要改模型架构并重训，周期最长

---

## 5. 建议的预实验与执行顺序

### 5.1 预实验 P1：验证时间 mismatch 假说（成本极低，优先级最高）

**目的**：确认"错误 t 会导致模型 ranking 退化"

**做法**：在已有 oracle 轨迹数据上，对每个 oracle state 故意传入不匹配的 t，测量 ranking 下降幅度

具体来说，可以用 `edit_ranking_diag_v2.py` 在已收集的 oracle 轨迹 state 上，额外做一组对照：
- **对照组**：每个 state 使用正确 t（oracle 轨迹的 t）→ 已有数据，Top-1=77.9%
- **实验组**：所有 state 统一使用固定小 t（如 0.05）或固定大 t（如 0.9）
- **实验组 2**：对 step k 的 state，使用 step 0 的 t（模拟"快速收敛"场景）

如果实验组的 ranking 显著低于对照组（比如 Top-1 从 78% 降到 40-50%），则时间 mismatch 假说得到直接证实。

**改动量**：在现有 `edit_ranking_diag_v2.py` 的 `_model_candidates_for_state` 调用处，增加一个 `t_override` 参数。约 10 行改动。无需重新跑模型 forward（如果复用已有 candidate 数据），或只需对 100 条样本重跑（几分钟）。

**决策阈值**：
- 若 ranking 下降 >20pp → 时间 mismatch 是主要瓶颈，全力推进方向 A
- 若 ranking 下降 5-20pp → 时间 mismatch 是重要因素，方向 A 和 B 并行
- 若 ranking 下降 <5pp → 时间 mismatch 不是主要原因，回到误差累积+轨迹 OOD 的解释线

### 5.2 预实验 P2：测量实际轨迹上的时间 mismatch 程度（低成本）

**目的**：量化模型自身轨迹上 t 与进度的偏差

**做法**：在 exp D 的 200 条 greedy 轨迹上，记录每条样本每步的：
- `t_val`（深度 t）
- `u_tot_base` / `u_tot_base(t=0)` 比值（估计的"完成度"）
- 对应的 `t_progress = 1 - ratio`

观察 `t_val` 与 `t_progress` 的散点图。如果两者相关性很低（比如 t_val=0.2 时 progress 已达 80%），则 mismatch 确实严重。

**改动量**：在 `beam.py` 的 `sample_greedy_single_edit` 中增加几条日志输出。约 10 行改动。

### 5.3 预实验 P3：固定时间对照（低成本）

**目的**：快速判断 `time_mode="fixed"` 能否绕过 mismatch 问题

**做法**：在 exp D 的 200 条样本上，用 `time_mode="fixed"` 跑 greedy_edit，扫几个 `time_const`（0.3, 0.5, 0.7），对比 `time_mode="depth"` 的 35.5%。

如果 `fixed` 的 Top-1 显著高于 `depth`（比如 40%+），则进一步证实时间 mismatch 假说，同时给方向 B 提供直接支持。

**改动量**：只需改运行参数，`run_exp_d.py` 已支持。无代码改动。

### 5.4 后续实验优先级

基于 P1/P2/P3 的结果分叉：

#### 如果时间 mismatch 被证实为主要瓶颈（P1 ranking 下降 >15pp）

1. **方向 A 实现**：在 `beam.py` 中新增 `time_mode="utot_progress"`，t = 1 - u_tot_base / u_tot_initial。在 200 条样本上对比 `depth` vs `utot_progress`
2. **方向 B 对照**：`time_mode="fixed"` 在不同 `time_const` 下的 sweep
3. **实验 E（oracle vs 模型轨迹）暂缓**：如果时间 mismatch 是主因，oracle 轨迹 OOD 问题可以被时间修正部分解决
4. **实验 C（位置/类型/token 分解）仍暂缓**

#### 如果时间 mismatch 被证实存在但影响有限（P1 ranking 下降 5-15pp）

1. **方向 A + B 并行**：小成本改进优先
2. **实验 E 优先级提升**：时间 mismatch 不能完全解释 gap，需查 oracle vs 模型轨迹的 ranking 差异
3. **训练侧改进（方向 C/D）列入中期计划**

#### 如果时间 mismatch 被排除（P1 ranking 下降 <5pp）

1. **回到实验 E**：oracle vs 模型轨迹 ranking 对比是下一步关键
2. **重新审视误差累积假说**：0.78^5.5 ≈ 25% vs 实际 35.5%，差距可能来自 stop 阈值的"幸运提前退出"而非真正的编辑质量
3. **考虑 per-step 准确率的提升路径**：如果 78% 是 hard ceiling，最终精度很难超过 30-40%

---

## 6. 执行建议

### 6.1 立即执行（本周）

| 优先级 | 实验 | 预计耗时 | 说明 |
|:------:|------|:--------:|------|
| **P0** | P1：时间 mismatch 验证 | 1-2h | 改动 ~10 行，对 100 条样本重跑 |
| **P1** | P3：固定时间对照 | 1h | 改参数重跑 exp D sweep |
| **P2** | P2：轨迹进度测量 | 1h | 改动 ~10 行，与 P1/P3 并行 |

三者可以并行：P1 改 `edit_ranking_diag_v2.py`，P2 改 `beam.py`，P3 只改运行参数。

### 6.2 根据 P1-P3 结果决定（下周）

- 若时间 mismatch 确认 → 实现方向 A（`time_mode="utot_progress"`），预计半天
- 若时间 mismatch 排除 → 启动实验 E（oracle vs 模型轨迹 ranking），预计 1-2 天

### 6.3 中长期

- 方向 C/D（训练侧改进）在当前阶段的优先级低于采样侧改进，因为：
  - 采样侧改进可以快速验证假说
  - 训练侧改动周期长，应在方向明确后再投入
  - 当前 checkpoint 已训练 1.68M steps，重训成本高

---

## 7. 工作假设更新

本轮分析后，对之前 `todo3.md` 的工作假设做如下更新：

1. **单编辑搜索框架可行** — 不变，Oracle Greedy=96% 已证实
2. **`beam.py` 主体实现无明显 bug** — 不变
3. **模型 edit ranking 实际很强（Top-16=96.4%）** — 更新，exp2 的"排序极差"结论已被推翻
4. **不存在轨迹退化** — 更新，之前"退化"是噪声污染假象
5. **新增：时间 mismatch 可能是 per-step 78% → final 35% 差距的主因** — 待 P1-P3 验证
6. **新增：Beam 收益有限（~2pp）可能是因为时间 mismatch 对所有候选路径产生同向偏差** — 与假说 #5 一致

---

## 8. 一句话总结

per-step 排序 78% 与最终生成 35.5% 之间的差距，最可能的原因是 `time_mode="depth"` 下模型收到的 t 信号与实际编辑进度不匹配——训练时模型学到的是 (state, t) 联合分布，而搜索时 t 仅由步数决定。三个低成本预实验（P1/P2/P3）可在 1-2 天内验证此假说，之后根据结果决定是推进自适应时间、固定时间，还是回到轨迹 OOD 解释线。
