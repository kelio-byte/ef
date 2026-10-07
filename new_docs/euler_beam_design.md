# Euler-Beam 当前设计与实现

> 状态：已按 2026-08-03 的代码更新。本文描述当前实现，不再把早期的“单后继、CPU
> 逐分支采样、旧反向路径分数”原型当作正式算法。完整实验流水账见
> `euler_beam_optimization_plan.md`，下一阶段计划见
> `euler_beam_next_stage_plan.md`。

## 1. 方法目标

普通 Euler 从每个输入状态采样一条 CTMC 轨迹。Euler-Beam 保留 Euler 的随机、并行编辑
语义，但在每个时间步维护一个最多含 K 个状态的搜索池；每个父状态产生 M 个后继，随后
按 token 状态合并并剪枝。它的目标是用一次父分支模型前向支持 K×M 个候选竞争，在可控
计算量下扩大正确反应的覆盖。

符号约定：

- K / `n_branches`：每个 run 的父分支上限；
- M / `n_children`：每个父分支每步产生的 child 数；
- R / `n_runs`：相互隔离的搜索池数量；
- 每个 augmentation 最终固定输出 `R*K` 个槽位，而不是只输出每个 run 的 winner。

## 2. 单步算法

每个 run 独立执行，单个 run 的一步可以写成：

```text
K 个当前父状态
  │
  ├─ 将所有活跃父状态组成一个 GPU batch
  ├─ Transformer 前向一次，得到 λ_ins/λ_sub/λ_del 与 Q_ins/Q_sub
  ├─ 每个父状态复制索引 M 次，不重复 Transformer 前向
  ├─ 用 (parent seed, step, child index) 派生独立 child 随机流
  ├─ GPU 向量化采样 K×M 组动作，并行应用编辑
  ├─ 对完整动作集合计分；构造新的 token 状态
  ├─ 相同 token 状态合并概率质量
  └─ 排名后保留最多 K 个状态
```

跨一个 `sample_euler_beam()` 调用时，同一 batch 内不同 product/run 的所有父状态也共享
一次模型前向。模型前向规模是活跃父分支数，M 只扩展模型输出、随机动作和编辑应用张量，
不会把 Transformer 计算重复 M 次。

## 3. 每个 child 内如何执行编辑

当前默认 `event_prob_mode=poisson`。在每个非 PAD 位置：

- 插入事件概率为 `1-exp(-h*λ_ins)`；
- 删除或替换这一互斥事件的概率为
  `1-exp(-h*(λ_sub+λ_del))`；
- 若删除/替换事件发生，再按 `λ_del : λ_sub` 决定类型；
- 插入和替换的 token 分别从 `Q_ins`、`Q_sub` 采样；
- 不同位置独立决定，因此一步可以发生 0、1 或多次编辑；插入还能与同一位置的
  删除/替换同时出现；所有动作最后批量应用。

这与普通 Euler 的编辑语义一致，不是“一步最多编辑一次”的传统 beam search。

需要注意的已知接口问题：当前 Euler 与 Euler-Beam 都只硬屏蔽 PAD，没有硬屏蔽 BOS 的
删除/替换，也没有硬屏蔽 Q 中的特殊 token。训练会把这些动作概率压低，但并非严格为
零；这是后续应先做诊断、再决定是否修正的推理正确性问题。

## 4. child 的独立性与重复

child 随机流由父 seed、step 和 child index 稳定派生，随机数直接在 GPU 上按
`(seed, step, position, stream)` 无状态生成。因此：

- child 之间随机流独立于 batch 划分，不需要 CPU `manual_seed` 循环；
- 改 K 不会重新编号已有 branch seed；
- “独立抽样”不代表结果必然不同。两个 child 可能都没有事件，或独立抽到同一组动作，
  因而得到完全相同的后继；这是正常碰撞，随后会被状态合并。

`stochastic_noop` 是例外：它只允许 M=2。child 0 始终正常随机采样；在
`step=floor(0.9*n_steps)`（不超过最后一步）时，child 1 被显式改成 no-op，保护父状态
不受该一步随机编辑影响。它是搜索启发式，不是来自目标 CTMC 的无偏 proposal。

## 5. 状态、计分、合并与剪枝

每个 `_BranchState` 保存：

```text
x_t          当前序列
path_log_p   完整轨迹动作 log-prob
log_mass     当前状态的 Monte Carlo 聚合质量
weight       M=1 兼容路径使用的历史共识权重
t            当前连续时间
seed         确定性随机流标识
```

### 5.1 M>1 正式路径

正式配置使用 `score_mode=full_probability`。每个父状态的 `log_mass` 在 M 个 child 间先
等分，即每个 child 减去 `log(M)`；相同 token 状态的 child 用 log-sum-exp 合并质量。
排名分数为：

```text
log_mass + changed_state_bonus * I[state != original product]
```

相同分数用原始 `log_mass` 和 seed 确定性破平局。当前推荐 bonus 为 0.5。这个 bonus
抵消早期/no-event 状态因大量重复 child 合并得到的搜索优势，但它是二元搜索先验，不是
模型似然，也不是每次编辑都累加的奖励；一个状态只要不同于原始 product，就只加一次。
例如 bonus=0.5 在排序上等价于把 changed state 相对原始状态的质量乘以
`exp(0.5)≈1.65`。实验已证明不能简单随 M 线性增加，M=3/4 的回归也不能只靠提高
bonus 修复。

bonus 在一次采样和一个冻结配置内必须是定值，不能按样本、target 或运行中间结果调整。
它也不必被假设为所有 K/M/R 的通用常数：若为新配置选择不同值，应只在 validation 上
预先筛选，随后冻结再进入 test。随时间或编辑进度变化的 bonus 属于新的搜索方法，需要
单独设计和验证，不能作为当前定值 bonus 的隐式自动调节。

`path_log_p` 会按完整动作概率计算，包括发生事件和未发生事件的概率，以及事件 token
概率，但当前 M>1 正式排名不使用它。它保留作诊断和 M=1 兼容用途。

### 5.2 M=1 兼容路径

M=1 没有 K×M 候选竞争。状态按 `path_log_p` 排序；相同状态合并。若合并后少于 K 个
分支，会机械复制已有分支并派生新 seed，以维持后续探索。这条路径主要用于兼容和消融，
不是当前推荐方法。

### 5.3 最终槽位

M>1 合并后可能只剩少于 K 个唯一状态。采样器不会在中途机械复制；最终为了保持固定文件
布局，会用最高排名状态补齐到 K 个槽位，并把 shortfall 写入 metadata/统计。评分前还会
做 canonical SMILES 合法性检查和去重，所以补齐不等于增加有效候选。

## 6. R、K、M 分别改变什么

- 增大 M：同一父状态有更多局部 proposal，child 采样和编辑成本上升，Transformer
  前向不重复；过大 M 会强化高碰撞/no-op 状态并增加低质量、非法探索。
- 增大 K：扩大同一池的全局竞争宽度，也扩大每步 Transformer batch；模式集中时可能
  同时淘汰多种搜索方向。
- 增大 R：建立更多相互隔离的搜索岛。不同 run 不合并、不相互剪枝，通常比等总宽度的
  单一大池更慢，但更能保护多样性。

输出文件采用 branch-rank-major、run-minor 顺序：先写 R 个 run 的 rank-1，再写 R 个
rank-2，依次到 rank-K。这样增加分支尾部不会把其它 run 的 winner 挤到很后面。

validation-200 的同预算对比说明 R 不能简单用 K 替代：R1K9 比 R3K3 快 32.7%，但
Top-1/2/3 分别低 2.0/5.0/5.5 个百分点。当前把 R3K3 视为准确率模式，R1K9 视为低延迟
模式。

## 7. 与普通 Euler 的关系

当 `R=10, K=1, M=1` 时：

- 每个 run 只有一条父轨迹和一个随机后继；
- 没有跨状态合并、Top-K 竞争、changed-state bonus 或补分支的实际作用；
- 从算法结构和目标分布看，它退化为 10 条独立的 Euler 风格轨迹。

但它不会与 `--sampler euler --n_samples 10` 逐字节相同：Euler-Beam 使用无状态 branch
RNG 和 inverse-CDF categorical；Euler 使用 PyTorch 全局 RNG 和 `multinomial`。两者
seed 映射、batch-size 不变性和 bookkeeping 也不同。因此应称为“分布语义上的退化”，
不能称为相同实现或相同随机序列。

## 8. 当前推荐配置与证据边界

准确率默认（`scripts/eval.py` 的 Euler-Beam 默认值）：

```text
K=3, M=2, R=3, n_steps=100, seed=42
score_mode=full_probability
changed_state_bonus=0.5
child_policy=stochastic_noop
matmul_precision=high        # RTX 3090 上启用 TF32
```

在 tiny 的 50 个原始反应上，固定其它变量：

| child policy | 时间 | Top-1/2/3 | rank 1/2/3 invalid |
|---|---:|---:|---:|
| stochastic | 123.8 s | 58/64/66 | 12.9/14.4/13.1% |
| stochastic_noop | 124.6 s | 60/64/70 | 12.5/14.5/13.4% |

因此 no-op 在这组已测配置上提高了 Top-1 2pp、Top-3 4pp，运行成本几乎不变；但样本
只有 50 个反应，而且 policy 是未校准启发式，不能保证换数据、K/R 或 seed 后仍提升。
R1K10 的 M2/M3 公平实验为了只改变 M，使用的是纯 `stochastic`，不能拿来证明
R1K10 下 no-op 的收益。

后续在未使用的validation reaction 200～399上完成了K10专属policy/bonus筛选。固定
R1K10M2时，`stochastic, bonus=0.5`的Top-1/3/10为47.0/67.5/78.0；迁移R3的
`stochastic_noop`得到46.5/67.5/78.5，属于Top-1和Top-10互换；bonus0.0得到
45.0/67.0/76.5而被支配，bonus1.0得到48.0/68.0/77.5但牺牲尾部且Oracle不变。
按预注册的多指标规则，R1K10继续推荐纯`stochastic`和bonus0.5，不能把R3的no-op默认
机械用于K10。

## 9. 效率特征

- 随机动作、token sampling、步分数和编辑应用均已 GPU 向量化；
- 正常分支的 tensor width 一致，状态 batch 使用一次 `torch.cat` 构造；不规则状态才回退
  到逐分支 padding。该 fast path 在 tiny 的固定 R1K10M2/batch64 实验中保持 10000 行预测
  完全一致，并把 sampling wall 从 99.16 秒降到 89.97 秒（-9.26%）；
- token-key 转 CPU、Python 字典合并和 Top-K 仍在 host 端，但 K/M 较小时不是主要
  Transformer 成本；
- `matmul_precision=high` 在 RTX 3090 使用 TF32，是日常实验的速度默认；最终严格
  FP32 复核可用 `highest`；
- `--euler_beam_profile` 会主动同步 CUDA，只应用于短 profiling，不能用它报告正常
  wall time。

不能只根据很短的 profile 增大 batch：R1K10M2 在 100 行上 batch128 看似较快，但完整
tiny 上 batch128 为 94.55 秒，慢于 batch64 的 89.97 秒，并改变 1392/10000 行预测。
当前继续保留 batch64。test-mini 的约半小时也不是旧脚本导致的 CPU 串行退化：它包含
20020 条需要独立前向的 augmentation 输入，是 tiny 的 20.02 倍；同机同输出数下，当前
Beam 反而比旧 Euler 入口更快。

准确率默认 R3K3M2 也在完整 tiny 上比较过 batch32/64/128：wall 为
109.78/110.12/117.13 秒，三者 Top-1～10 完全一致。32和64属于等速区间，128明确更慢；
为保持 validation/test-mini 历史基线可复现，正式默认仍为64，显存紧张时可使用32。

## 10. 当前限制

1. `log_mass` 是有限 child 频率与确定性 Top-K 剪枝得到的搜索量，不是严格 SMC 权重；
   `stochastic_noop` 又改变 proposal 而没有重要性校正。
2. 二元 changed-state bonus 只能区分“原状态/非原状态”，不能评价化学合理性或进度。
3. 大单池会发生模式坍缩；多 run 则牺牲 GPU 效率。
4. 特殊 token 和 BOS 的动作约束没有完全硬编码。
5. Euler-Beam API 的 trajectory/event recording 参数仍是预留项；当前完整 trajectory
   可视化使用普通 Euler 的多路径事件记录，不能误称为 Beam 搜索树可视化。
6. 当前指标主要来自 tiny、validation-200 和 test-mini-1001；在冻结方法前不应反复用
   test target 调参，完整 5007 反应测试应留作最终报告。

## 11. 后续设计方向

优先级从低风险到高风险：

1. 先诊断并测试 BOS/特殊 token 硬屏蔽，不改变已训练权重；
2. 在 validation 上做 Q temperature/top-p 的小规模预注册消融；
3. 将确定性 Top-K 改造成 ESS 触发的粒子重采样，形成独立 `euler_smc` 实验分支；
4. 有可用、与 test target 无关的化学 reward 后，再引入中间 reward twisting 和严格
   proposal/importance 校正；
5. reverse-rate、自校正和 localized edit flow 都需要新训练，不应直接塞入现 checkpoint。
