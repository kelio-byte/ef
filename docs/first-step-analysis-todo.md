# First-Step Analysis TODO

## 1. 背景

目前已有一个 `use_rate_reparam: true` 的训练模型：

- `checkpoints/USPTO_50K_PtoR_aug20_#global#/2026-06-08_17-20-39`

现在希望围绕“第一步”做两类分析：

1. **第一步预测是否准确**
2. **第一步真实编辑在多大程度上决定最终结果**

这里的“第一步”需要区分两种定义：

- **实验 1：初始预测**
  - 固定输入 `x_t = x_0` 不变，不真的执行采样
  - 只看模型在给定 `t` 下的 forward 输出
  - 目标是分析模型是否能在一开始找到反应中心 / 首个关键编辑

- **实验 2：第一次真实编辑**
  - 运行 Euler 采样
  - 记录轨迹里第一个实际发生的 edit event
  - 目标是分析“开局一步走对/走错”对最终生成质量的影响

这两个实验应分开做，否则“模型认知”和“采样行为”会混在一起。

---

## 2. 现有代码可复用点

### 2.1 初始预测相关

- `scripts/sample_retro.py`
  - 已经能加载 checkpoint、vocab、构造 `x_0`
- `edit_flows/models/transformer.py`
  - `model(x_t, t_model, x_pad_mask)` 输出 `log_rates, log_ins_probs, log_sub_probs`
- `edit_flows/core/rate_scale.py`
  - `apply_rate_parameterization()`
  - 可得到真正参与采样的 effective rates
- `edit_flows/sampling/oracle.py`
  - `compute_oracle_model_output()`
  - 在给定 `x_t, x_1, t` 时，可计算 oracle 的位置/类型/token 标签

### 2.2 第一次真实编辑相关

- `edit_flows/sampling/euler.py`
  - `sample_euler()` 已完整实现正常 Euler 采样逻辑
  - 适合作为“记录第一次真实编辑”的主骨架
- `edit_flows/sampling/ops.py`
  - `apply_ins_del_operations()`
  - 可直接复用来构造受控干预版本
- `edit_flows/sampling/oracle.py`
  - 可用于定义 oracle 的“当前正确编辑”

### 2.3 注意

- **不要复用会写 `predictions.txt` 的现有生成脚本作为实验入口**
  - 尤其不要直接跑 `scripts/eval_retro.py`
  - 新实验应写到独立输出目录

---

## 3. 数据范围建议

需要先澄清一点：目前仓库中的 `train_subsets` 来自 **train set 子集**，不是 test set 子集。

这意味着：

- `train_subsets`
  - 更适合回答“模型在见过分布甚至见过样本附近时，第一步是否学会了”
  - 更适合作为诊断集
- 新建 `test` 随机子集
  - 更适合回答“模型在正式评测分布上，第一步是否真的有效”
  - 更适合作为主结果集

因此建议不要只依赖 `train_subsets`。更稳妥的方案是同时准备两类数据：

### 3.1 诊断集：沿用现有 `train_subsets`

保留已有 oracle 分析使用过的 1000 条 deduplicated `#global#` 子集：

- 数据集：`USPTO_50K_PtoR_aug20_#global#`
- 处理方式：`deduplicate=20`
- 子集规模：先做 `1000` 条

理由：

- 与现有 oracle 文档口径一致
- 便于和已有 train-subset 上的生成/ oracle 分析对照
- 适合作为开发期快速诊断

但这组数据**不应作为主结论来源**，因为它来自 train set。

### 3.2 主结果集：新建 test set 随机子集

建议另外新建一个 deduplicated 的 test 随机子集，作为第一步分析的主结果集。

建议方案：

- 来源：`USPTO_50K_PtoR_aug20_#global#/test`
- 先按 `deduplicate=20` 取 unique product
- 再从 dedup 后样本中固定随机种子抽样
- 子集规模：
  - 第一轮 `1000` 条
  - 如需要更稳，再扩到 `2000` 或全 test

建议固定输出到新目录，例如：

- `analysis_subsets/USPTO_50K_PtoR_aug20_#global#/test_dedup_seed42_1000/`

目录内保存：

- `src-test.txt`
- `tgt-test.txt`
- `meta.json`

其中 `meta.json` 建议记录：

- 原始来源路径
- `deduplicate=20`
- 随机种子
- 抽样前总量
- 抽样后样本数
- 抽样索引

### 3.3 为什么建议新建 test 随机子集

原因有三点：

1. 避免把“记住训练样本”误判为“学会了第一步”
2. 后续若观察到第一步与最终生成强相关，test 子集上的结论更有说服力
3. 与已有 `checkpoints/*/*/eval/predictions.txt` 的 test 评测口径更一致

### 3.4 推荐使用顺序

建议分两层跑：

1. `train_subsets`
  - 用于开发和快速调试脚本
  - 看信号是否明显

2. 新建 `test` 随机子集
  - 用于正式汇报实验 1 / 实验 2 的主结果

### 3.5 后续扩展

如第一轮结论清晰，再扩展到：

- 全 test 集
- standard 数据集
- 其他 checkpoint 对照模型

这里的“其他 checkpoint 对照模型”可以包括：

- 未来新训练的 `use_rate_reparam: false` 模型
- 不同 scheduler 的模型
- 不同训练阶段 checkpoint

---

## 4. 实验 1：初始预测是否准确

## 4.1 目标

回答：

- 固定 `x_t = x_0` 时，模型是否能在初始阶段找到应编辑位置
- 模型是否能预测正确的编辑类型
- 对 insert / substitute，模型是否能给出正确 token
- 这些结论对 `t` 是否敏感

## 4.2 为什么不只看 `t=0`

虽然此时 token 输入都相同，只有时间嵌入变化，但目前尚不确定模型是否对 `t` 敏感。只看 `t=0` 容易把结论锁死在边界点上。

尤其对 `#global#` 来说，平均编辑距离约为 5，粗略上首个关键编辑更可能出现在 `t≈1/k≈0.2` 附近，因此除了 near-zero，也应加入 `0.1, 0.2` 这类时间点。

## 4.3 建议的时间点

主时间网格：

- `t = 0`
- `t = 1e-3`
- `t = 1e-2`
- `t = 5e-2`
- `t = 0.1`
- `t = 0.2`
- `t = 0.3`

可选补充：

- `t = 0.5`

报告时建议至少同时给三类结果：

- `t=0` 单点
- near-zero 区间
- `0.1~0.3` 区间

## 4.4 标签定义

对每个样本，固定：

- `x_t = x_0`
- `x_1 = target`

然后调用 `compute_oracle_model_output(x_t, x_1, t, scheduler, vocab_size)` 得到 oracle 输出。

关键点：

- 对固定的 `x_t=x_0`，不同 `t` 下 oracle 的**应编辑位置 / 编辑类型 / 正确 token**本质上不变
- 随 `t` 变化的主要是整体 rate scale

因此实验 1 应主要比较**排序与分布**，不要过度比较绝对 rate 大小。

## 4.5 指标设计

### A. 位置级指标

先把模型每个位置的总编辑倾向定义为：

- `score_pos(j) = lambda_ins(j) + lambda_sub(j) + lambda_del(j)`

oracle 位置标签定义为：

- 该位置在 oracle 的 X-space 聚合后，是否存在非零编辑需求

建议统计：

- `Center Hit@1`
  - 模型 top-1 编辑位置是否命中 oracle 正位置
- `Center Hit@3`
- `Center Hit@5`
- `Center MRR`
  - 第一个 oracle 正位置在模型排序中的倒数排名
- `Position AP`
  - 把“该位置是否需要编辑”看作多标签检索

说明：

- 这里的“center”并不一定严格等于化学图上的反应中心原子
- 但它是当前 Edit Flows token 空间下最自然、与代码一致的“应编辑位置”定义

### B. 编辑类型指标

在 oracle 正位置上，比较模型最优 edit type 是否正确：

- `Type Acc@oracle-pos`
  - `argmax(lambda_ins, lambda_sub, lambda_del)` 是否与 oracle 类型一致

也可以补：

- `Type CE@oracle-pos`
  - oracle 类型对模型 type 分布的交叉熵

### C. token 级指标

只对 oracle 为 insert / substitute 的位置统计：

- `Ins Token Acc@1`
- `Ins Token Acc@5`
- `Sub Token Acc@1`
- `Sub Token Acc@5`

### D. 完整首编辑指标

定义“完整首编辑正确”：

- 位置正确
- 类型正确
- 若为 insert/substitute，则 token 也正确

建议统计：

- `Full First-Edit Acc`
- `Full First-Edit Acc@k-pos`
  - 当位置取 top-k 命中后，再判断类型和 token

## 4.6 两套模型分数都应保存

为避免把参数化和 scheduler 影响混在一起，实验 1 最好同时导出两类分数：

1. **base output**
  - `model.forward()` 原始输出
  - 即模型自己的偏好

2. **effective output**
  - 经 `apply_rate_parameterization()` 后真正进入采样的速率
  - 即实际生成行为对应的分数

建议在结果里同时报告：

- `base_*`
- `effective_*`

这样后面比较不同 checkpoint 或不同参数化设置时更清楚。

## 4.7 输出建议

建议新写一个脚本，例如：

- `scripts/first_step_forward_analysis.py`

输入：

- `--checkpoint`
- `--products_file`
- `--targets_file`
- `--vocab_file`
- `--output_dir`
- `--scheduler`
- `--time_grid`
- `--deduplicate`
- `--max_lines`
- `--device`

输出：

- `summary.json`
  - 各个 `t` 下的聚合指标
- `per_example.parquet` 或 `per_example.pt`
  - 每条样本的详细排名与分数
- `report.md`
  - 人读摘要

## 4.8 额外建议

可以加两个补充分析：

1. 按初始编辑距离 `k=d(x_0, x_1)` 分桶
  - `k=1~3`
  - `k=4~6`
  - `k>=7`

2. 按序列长度分桶
  - 检查长序列是否更难在初始阶段找对位置

---

## 5. 实验 2：第一次真实编辑对最终结果的决定程度

## 5.1 目标

回答：

- 第一次真实编辑走对时，最终是否显著更容易成功
- 第一次真实编辑走错时，后续是否还能纠正
- 只修正第一步，最终结果能提升多少

## 5.2 “第一次真实编辑”的定义

采用 Euler 采样中的**第一次实际发生的 event**，而不是 `t=0` 的 forward。

在现有 `sample_euler()` 逻辑中，每轮会采样：

- `ins_mask`
- `del_sub_mask`
- `del_mask`
- `sub_mask`

因此第一次真实编辑可定义为轨迹中最早出现的下列任一事件：

- 某位置 `sub_mask=True`
- 某位置 `del_mask=True`
- 某位置 `ins_mask=True`

如果同一个 step 内多个事件同时发生，需要额外定义一个统一口径。

## 5.3 同一步多编辑问题

现有 Euler 采样允许同一步多个位置同时编辑，因此“第一次真实编辑”需要两层定义：

1. **first event step**
  - 第一次出现任意编辑的时间步

2. **first event set**
  - 该时间步内发生的全部编辑集合

建议实验 2 的主口径用：

- **first event set 是否完全正确**

即比较该步发生的整组编辑，与 oracle 在同一状态下采样最应发生的编辑集合之间的关系。

但为了更容易解释，也可以再构造一个简化口径：

- **anchor edit**
  - 该时间步中 hazard 最大的那一个编辑
  - 仅用于辅助可视化与分桶

## 5.4 oracle 对照定义

对采样到的当前状态 `x_t`，在该 step 开始时调用：

- `compute_oracle_model_output(x_t, x_1, t, scheduler, vocab_size)`

由此获得当前 oracle 的位置/类型/token 正确答案。

由于真实 Euler 有随机性，实验 2 的核心不是比较“采样到的编辑是否等于所有 oracle 非零位置”，而是比较：

- 模型第一次事件是否落在 oracle 高分 / 正确编辑区域内
- 如果我们强制第一次事件改成 oracle 正确编辑，后续结果如何变化

## 5.5 先做相关性分析

建议先做不干预的观测实验。

每条样本、每次采样记录：

- `first_event_step_idx`
- `first_event_t`
- `n_first_events`
- 第一次事件是否命中 oracle 正位置
- 第一次事件类型是否正确
- 第一次事件 token 是否正确
- 第一次事件集合是否完全正确
- 最终结果是否 exact match
- 最终结果是否 valid
- 最终最终编辑距离 `d(final, target)`

汇总的核心条件概率：

- `P(final correct | first event set correct)`
- `P(final correct | first event set wrong)`
- `P(final correct | first center hit)`
- `P(final correct | first center miss)`
- `P(valid | first event correct)`
- `E[d_final | first event correct]`

这组结果可以直接回答“第一步有多决定最终结果”。

## 5.6 再做受控干预实验

这是实验 2 最关键的部分。

建议新增一个支持“首步干预”的采样函数，例如：

- `sample_euler_with_first_step_intervention()`

三种模式：

1. `normal`
  - 完全按模型采样

2. `force_correct_first`
  - 第一次事件步，把模型事件替换为 oracle 正确编辑
  - 后续步全部恢复模型采样

3. `force_wrong_first`
  - 第一次事件步，强制执行一个错误但高分的编辑
  - 后续步恢复模型采样

对比指标：

- Top-1
- Top-k
- valid rate
- final edit distance

解释逻辑：

- 若 `force_correct_first` 提升很大，说明第一步是主要瓶颈
- 若提升有限，说明后续多步速率建模也明显有问题
- 若 `force_wrong_first` 显著拉低结果，说明模型对开局错误缺乏纠错能力

## 5.7 “force_wrong_first” 怎么定义

建议不要随机造错，而是使用“高置信错误”：

- 位置：取模型排序最高但不在 oracle 正位置中的位置
- 类型：该位置模型 top-1 错误类型
- token：对应 top-1 错误 token

这样更贴近真实失败模式。

## 5.8 扩展：oracle 前 N 步替换曲线

在首步干预之后，可以进一步做：

- oracle 前 `1` 步 + model 后续
- oracle 前 `2` 步 + model 后续
- oracle 前 `5` 步 + model 后续

观察 `N -> final accuracy` 曲线。

这能区分：

- 问题主要集中在开头几步
- 还是整条轨迹都在持续犯错

第一轮不一定要做完，但接口设计时应预留。

## 5.9 输出建议

建议新写脚本，例如：

- `scripts/first_event_impact_analysis.py`

输入：

- `--checkpoint`
- `--products_file`
- `--targets_file`
- `--vocab_file`
- `--output_dir`
- `--scheduler`
- `--n_steps`
- `--n_samples`
- `--deduplicate`
- `--max_lines`
- `--device`
- `--mode {correlation, intervention}`

输出：

- `correlation_summary.json`
- `intervention_summary.json`
- `per_sample_events.pt`
- `report.md`

---

## 6. 推荐的实施顺序

建议按下面顺序推进，避免一上来改动过多。

### Phase 1：静态初始预测

先完成实验 1：

- 固定 `x_t=x_0`
- 跑多个 `t`
- 输出位置/类型/token 三层指标

这是最便宜的一步，也最容易先看出模型是否具备“初始反应中心感知”。

### Phase 2：不干预的第一次真实编辑相关性

在 `sample_euler()` 基础上只加记录逻辑，不改采样行为。

先回答：

- 第一次真实编辑通常发生在什么 `t`
- 第一次真实编辑走对/走错与最终结果相关性多强

### Phase 3：首步干预

加入：

- `force_correct_first`
- `force_wrong_first`

这是最能支持因果解释的一步。

### Phase 4：前 N 步 oracle 替换

若前三阶段已证明首步很关键，再扩展到前 `N` 步。

---

## 7. 与 `use_rate_reparam` 的关系

当前这个计划针对的已知 checkpoint 是：

- `checkpoints/USPTO_50K_PtoR_aug20_#global#/2026-06-08_17-20-39`
- `use_rate_reparam: true`

因此第一轮实验应先围绕这个模型完成。

后续若训练出对照模型，例如：

- `use_rate_reparam: false`

那么两组实验都应完全复跑，并重点比较：

1. 实验 1 中：
  - `t=0` 的 `Center Hit@k`
  - `t=0` 的 `Full First-Edit Acc`
  - 指标随 `t` 的曲线是否更平滑、更合理

2. 实验 2 中：
  - 第一次真实编辑是否更早发生
  - 第一次事件正确率是否更高
  - `force_correct_first` 的边际收益是否减小

如果 `use_rate_reparam: true` 相比对照模型有效，预期会看到：

- 初始预测更强
- 首步真实编辑更准确
- 最终结果对首步修正的依赖略下降

因为模型本身已经更会在开头做正确编辑。

---

## 8. 当前最值得先做的最小可行版本

如果只做最小闭环，建议优先完成以下三项：

1. `scripts/first_step_forward_analysis.py`
  - 先在 `#global#` 1000 dedup 子集上，跑 `t={0, 1e-3, 1e-2, 0.1, 0.2, 0.3}`
  - 输出 `Center Hit@1/3/5`, `Type Acc`, `Full First-Edit Acc`

2. `sample_euler()` 的只读版记录扩展
  - 记录第一次真实编辑时间与正确性
  - 不改任何生成逻辑

3. `force_correct_first`
  - 只先做这一种干预
  - 比较 `normal` vs `force_correct_first`

如果这三项已经显示：

- 模型初始反应中心命中率不高
- 第一次真实编辑正确与最终成功高度相关
- 强制修正第一步能明显提升结果

那么就已经可以较有力地支持：

- “第一步”确实是当前模型的重要瓶颈之一

若后续加入 `use_rate_reparam: false` 对照模型，还可以进一步验证：

- `use_rate_reparam` 是否主要改善了初始反应中心感知
- 还是主要改善了真实采样时首步编辑的触发行为

---

## 9. 实现注意事项

1. 所有新脚本都应写入独立目录。
  - 例如 `analysis_outputs/first_step/...`
  - 不要写到现有 checkpoint 的 `eval/`

2. 不要调用会覆盖预测文件的旧评测链路。
  - 特别是不要直接使用 `scripts/eval_retro.py`

3. 记录结果时，尽量同时保存：
  - 聚合统计
  - 每样本明细
  - 少量可人工查看的 case study

4. 对“位置”指标要统一是否包含 `BOS`。
  - 默认建议排除 `BOS` 位置，不把它作为候选编辑位置

5. 对一次 step 内多个编辑的情况，要在文档和代码里明确口径。
  - `first event set`
  - `anchor edit`

6. 若后面比较不同 scheduler，实验 1 与实验 2 应分开解释。
  - 实验 1 主要是模型认知
  - 实验 2 同时受模型与离散化机制影响

---

## 10. 结论

本轮分析建议将“第一步”拆成两个问题分别研究：

- **静态初始预测是否好**
- **第一次真实编辑是否决定成败**

其中：

- 实验 1 更适合回答“模型有没有学会一开始就找反应中心”
- 实验 2 更适合回答“开局一步错了之后还有没有补救空间”

从当前代码基础看，这两组实验都可以在不影响现有训练/生成主链路的前提下完成，并且大部分逻辑都能直接复用现有的 `sample_euler`、`compute_oracle_model_output` 和 `apply_ins_del_operations`。
