# Beam Search 当前初步审查与下一步计划

## 1. 背景

当前 beam search 相关工作已经经历了三轮材料：

- 方案设计：`docs/beam-search/todo1.md`
- 实现与修复：`docs/beam-search/impl.md`、`docs/beam-search/todo2.md`
- 初步实验与诊断：`docs/beam-search/exp1.md`、`docs/beam-search/exp2.md`

其中 `docs/beam-search/exp2.md` 的核心结论是：

- Oracle Greedy 结果很好，说明单编辑搜索框架本身可行
- 真实模型的 edit ranking 很差，因此当前瓶颈主要在模型排序能力

但这一结论与 `first-step-analysis` 分支中的结果并不完全一致。后者表明：

- 模型第一步的静态能力并不差
- 尤其 token 预测能力较强
- 第一阶段更像是“位置/类型的泛化问题”而非“完全不会排正确 token”

为便于查找，这里明确给出另一分支中的相关材料位置：

- 分支：`first-step-analysis`
- 文档 1：`docs/first-step-analysis-finish.md`
- 文档 2：`docs/first-step-analysis2-finish.md`
- 辅助说明：`docs/first-step-analysis-impl.md`
- 初始计划：`docs/first-step-analysis-todo.md`

如果当前工作树不在该分支，可用下面命令直接查看：

```bash
git show first-step-analysis:docs/first-step-analysis-finish.md
git show first-step-analysis:docs/first-step-analysis2-finish.md
git show first-step-analysis:docs/first-step-analysis-impl.md
git show first-step-analysis:docs/first-step-analysis-todo.md
```

因此本轮目标不是重新否定 beam search，而是先判断：

1. 当前 `beam.py` 实现是否还存在明显错误
2. `exp2` 中“模型排序很差”的诊断证据是否足够干净
3. 下一步实验应优先确认什么

---

## 2. 初步审查结论

## 2.1 `beam.py` 主体实现目前没有再看到明显的致命语义错误

对照 `docs/beam-search/todo2.md` 中列出的主要问题，当前 `edit_flows/sampling/beam.py` 已实现：

- beam stop 语义前移：parent 满足 stop 时直接保留，不再额外展开 child
- dead-end state 保活：无候选 / 候选全被过滤时，parent 会作为 finished 保留
- finished state carry-over：后续轮次不会被直接丢失
- BOS 保护：禁止 `sub(pos=0, *)` / `del(pos=0)`
- no-op substitution 过滤：`sub(pos, current_token)` 不进入候选
- reverse-op 规则精确化：不再把所有同位置连续 substitute 都误判成回退

对应测试文件 `tests/sampling/test_beam.py` 当前通过：

- `pytest -q tests/sampling/test_beam.py`
- 结果：`30 passed`

因此，**截至本轮审查，beam/greedy 主体搜索语义本身看起来是基本正确的**。接下来更值得怀疑的是：

- 时间推进口径
- 诊断脚本口径
- 实验结论是否被边界点或统计口径污染

## 2.2 目前最值得怀疑的不是 beam 框架本身，而是诊断口径

`docs/beam-search/exp2.md` 用来支持“模型排序很差”的主要证据，是 `scripts/edit_ranking_diag.py`：

- 在 oracle 轨迹上逐步推进
- 每一步比较“oracle 最优编辑”在模型候选池中的排名

这个思路本身是合理的，但脚本原始实现里有两个会系统性压低结果的问题。

---

## 3. 本轮已做的小修改

## 3.1 修正 `depth` 时间推进：避免 step 0 恰好落在 `t=0`

### 问题

`beam.py` 原来在 `time_mode="depth"` 下使用：

```python
t = step / max_edits
```

这会导致第一步恰好在 `t=0`。

对于 cubic scheduler：

- `k(0) = 0`
- oracle 诊断里这一点已经被证明是病态边界点
- 即便模型使用 `use_rate_reparam=true`，评分时 `k(t)` 可抵消，`t=0` 也仍可能让模型时间嵌入落在分布边缘

因此，继续让 greedy/beam 默认从 `t=0` 开始并不稳妥。

### 修改

在 `edit_flows/sampling/beam.py` 中新增：

```python
def _depth_time_value(step, max_edits):
    return (step + 1) / (max_edits + 1)
```

并在 greedy / beam 两个入口里统一使用该映射。

### 影响

这相当于把 `depth` 模式的离散时间从闭区间边界点，改成开区间内部采样：

- step 0 不再是 `t=0`
- 最后一步也不会精确落在 `t=1`

这能减少 scheduler 边界点带来的数值和分布偏差，使 `depth` 模式更接近“把有限编辑步均匀铺在 (0,1) 内”的语义。

## 3.2 修正 `edit_ranking_diag.py`：时间口径与 oracle 真值口径

### 问题 1：时间口径仍从 `t=0` 起

`scripts/edit_ranking_diag.py` 原来同样使用：

```python
t_val = step / args.max_edits
```

这和上面的 `beam.py` 问题一致，会让 ranking diagnostic 的第一个决策点落在病态边界。

### 问题 2：oracle 真值口径过严

脚本原来只取：

- oracle 候选中的单个 top-1 edit

然后检查这个唯一 edit 在模型候选中的排名。

但这和 `first-step-analysis2` 里已经暴露出的问题高度一致：

- oracle 侧可能存在多个并列最优 edit
- 若只取一个 argmax，就可能把“命中了另一个同分正确 edit”的模型预测误判为错误

这本质上是 tie / multi-valid 问题在 edit-level 诊断上的再次出现。

### 修改

在 `scripts/edit_ranking_diag.py` 中做了两点修正：

1. 时间映射改为与 `beam.py` 一致的 interior mapping：

```python
t_val = (step + 1) / (max_edits + 1)
```

2. oracle 匹配改为 tie-aware：

- 先取 oracle top score
- 收集所有与 top score 近似相等的 oracle edits
- 只要模型命中其中任意一个，就算命中

### 影响

这意味着：

- `docs/beam-search/exp2.md` 中基于旧 ranking 脚本得到的“62% 不在 top-16”不能再直接视为最终可信结论
- 该数字很可能混入了边界点偏差和 tie 统计偏差

---

## 4. 目前对 `exp2` 结论的判断

## 4.1 Oracle Greedy 可行性结论依然可信

`docs/beam-search/exp2.md` 中关于 Oracle Greedy 的主结论仍然成立：

- 修复实现 bug 后，Oracle Greedy 可以达到很高 Top-1
- 单编辑离散化本身不是死路
- beam / greedy 框架理论上可行

这部分不需要推翻。

## 4.2 “模型排序很差”这个结论暂时不能直接定案

当前更稳妥的说法应是：

- 旧 ranking diagnostic 表明模型在 oracle 轨迹上的 edit ranking 可能存在明显问题
- 但该诊断脚本原先存在两个会系统性压低结果的口径问题
- 因此原报告中的具体数值，不宜直接拿来与 first-step analysis 对立比较，更不宜直接作为“模型非常不健康”的最终证据

换句话说，**方向性怀疑仍然可能对，但证据强度还不够**。

## 4.3 与 first-step analysis 的关系

当前两组结论并不必然互相否定。

一种很可能的情况是：

1. 模型在 `x_t = x_0` 的第一步静态能力确实较强
2. 但一旦沿轨迹推进，尤其进入后续编辑状态后，排序能力明显退化

如果是这样，那么真正的问题不是：

- “模型第一步就完全不会排”

而是：

- “模型的多步轨迹稳定性 / 后续步骤健康度不足”

这正是下一步实验最需要拆开的部分。

---

## 5. 下一步实验计划

下面的实验顺序已经按优先级调整，目标是先澄清诊断口径，再决定是否继续把精力投入 beam search 正式对比实验。

## 5.1 实验 A：重跑修正后的 edit ranking diagnostic

### 目的

重新回答最直接的问题：

- 在修正时间口径和 tie 口径后，模型在 oracle 轨迹上的 edit ranking 到底有多差？

### 做法

使用与 `docs/beam-search/exp2.md` 尽量一致的设置，重跑：

- 同一 checkpoint
- 同一数据子集
- 同样的 `k_ins_token / k_sub_token / k_edit_expand`

输出至少包括：

- oracle-best edit 命中模型 Top-1 / Top-5 / Top-16 的比例
- 不在候选池中的比例
- mean score gap

### 想确认什么

想确认旧结果中的“Top-1 11.8%、不在 Top-16 61.9%”有多少是模型真实问题，有多少是脚本口径造成的低估。

### 对后续正式实验的帮助

这一步会决定之后该把重点放在：

- 搜索策略调优

还是放在：

- 模型轨迹健康度分析 / 训练目标改进

如果修正后 ranking 显著回升，那么之前“模型排序极差”的结论需要降级处理。

---

## 5.2 实验 B：把 ranking diagnostic 按 step 分解，而不是只报总体平均

### 目的

区分问题发生在：

- step 0
- 早期几步
- 还是中后期轨迹

### 做法

在实验 A 的基础上，额外统计：

- `step=0` 单独指标
- `step=1~3`
- `step=4~7`
- `step>=8`

或者直接逐 step 报表。

### 想确认什么

想回答：

- 模型到底是“第一步就不行”
- 还是“第一步可以，但后续一走就坏”

这和 `first-step-analysis` 的关系非常关键。

### 对后续正式实验的帮助

如果问题主要集中在后续步骤，那么之后正式实验就不应只盯着：

- beam size
- stop threshold
- time_mode

而应增加对“轨迹中后期退化”的专门诊断或干预。

---

## 5.3 实验 C：将 ranking diagnostic 与 first-step analysis 指标体系对齐

### 目的

把 edit-level 排名诊断拆成 first-step analysis 更熟悉的三层：

- 位置
- 类型
- token

### 做法

对于 oracle 轨迹上的每个状态，不只看完整 edit 是否命中，还统计：

- oracle 正位置是否在模型高分位置中
- 在 oracle 位置上，类型是否正确
- 在 oracle 的 insert/substitute 位置上，token 是否命中 tie-aware valid set

必要时分别统计：

- “完整 edit 命中率”
- “位置命中率”
- “位置+类型命中率”
- “位置+类型+token 命中率”

### 想确认什么

这一步想回答：

- 如果 edit ranking 不理想，究竟主要错在位置、类型，还是 token

这可以直接对照 `first-step-analysis2` 的结论：

- token 很强
- 位置/类型更弱

### 对后续正式实验的帮助

它能把“模型排序差”这个笼统结论拆成更可行动的子问题，避免后续所有工作都陷入“调 beam 参数但不知道模型到底错在哪”。

---

## 5.4 实验 D：重跑模型 greedy / beam 小规模复现实验

### 目的

在修正 `depth` 时间后，重新看真实采样结果是否有实质变化。

### 做法

先做小规模、低成本复现：

- 数据：先用 `test_dedup_seed42_1000` 的一个更小子集，或直接 100/200 条
- 比较：
  - `greedy_edit, time_mode=depth`
  - `greedy_edit, time_mode=fixed, time_const=0.5`
  - `beam_edit, beam_size=3/5`
- 对 `stop_u_tot_base` 扫几档：
  - `-1`
  - `0.01`
  - `0.05`
  - `0.1`
  - `0.5`

主要记录：

- Top-1
- Invalid rate
- Unique rate
- 平均编辑步数
- 提前停止比例

### 想确认什么

想确认两件事：

1. `depth` 模式此前是否被 `t=0` 明显拖坏
2. 合理 stop 阈值是否能显著减少过度编辑 / invalid

### 对后续正式实验的帮助

如果这里就能看到：

- `depth` 修复后明显回升
- 或 stop 阈值能显著降 invalid

那么后续正式 beam 实验就有必要继续做；否则就不应过早扩大 beam 相关 sweep。

---

## 5.5 实验 E：做“第一步 vs 后续步骤”的统一诊断

### 目的

把两条已有分析线索接起来：

- first-step analysis：第一步静态预测
- ranking diagnostic：oracle 轨迹上的多步排序

### 做法

构造一份统一报表：

- step 0：使用当前状态 `x_t=x_0`
- step 1 以后：沿 oracle 轨迹推进
- 对每一步都报位置 / 类型 / token / 完整 edit 的命中情况

必要时再加一个对照：

- 沿模型自己 greedy 轨迹推进，而不是 oracle 轨迹推进

### 想确认什么

想最终回答：

- 模型到底是“不会做第一步”
- “会做第一步但不会维持轨迹”
- 还是“oracle 轨迹本身对模型来说分布外”

### 对后续正式实验的帮助

这一步直接决定长期方向：

- 若是第一步就差：优先改初始状态建模
- 若是后续退化：优先做轨迹稳定性 / 多步训练信号
- 若是 oracle 轨迹分布外：应谨慎解读 oracle-guided ranking 结果

---

## 6. 建议的执行顺序

建议按下面顺序推进：

1. **先重跑修正后的 ranking diagnostic**
   - 先验证原“模型排序很差”结论是否仍然成立
2. **再做 step 分解**
   - 先区分是第一步问题还是后续问题
3. **再做 greedy/beam 小规模复现**
   - 确认 `depth` 修复和 stop 阈值对真实生成是否有帮助
4. **最后再决定是否开展大规模 beam sweep**
   - 避免在诊断结论尚不稳时过早投入大量实验算力

---

## 7. 当前阶段的工作假设

本轮审查后，一个相对稳妥的工作假设是：

1. **单编辑 greedy/beam 框架本身是可行的**
   - Oracle Greedy 已经证明这一点
2. **当前 `beam.py` 主体实现没有再看到明显致命 bug**
   - 之前 `todo2` 里的主要 correctness 问题已基本修复
3. **“模型排序很差”这一判断很可能方向没错，但旧数值不够可靠**
   - 至少受到时间边界点和 tie 统计口径污染
4. **模型更可能是“后续轨迹健康度不足”，而不一定是“第一步完全不会排”**
   - 这与 first-step analysis 的结论更一致

---

## 8. 一句话总结

当前不应再把 beam search 的负面结果简单归因为“单编辑搜索无效”或“模型完全不健康”。更合理的下一步，是先用修正后的诊断脚本重新确认模型在 oracle 轨迹上的真实排序能力，再判断问题究竟主要出在第一步、后续轨迹，还是 beam 搜索本身。
