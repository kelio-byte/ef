# DGM 局部信用分配：一步后继监督方案

更新日期：2026-08-11  
状态：L0 已完成；L1（50 个未重叠训练反应的数据质量审计）待执行。

## 1. 要解决的具体问题

现有 action-level guidance 训练有三样离线数据：一个产物、某个 Euler 时间点的中间分子，以及从该中间分子继续采样直到终点后得到的反应物和其 forward-beam reward。

当前做法会把中间分子与**最终**反应物对齐：凡是“如果现在直接变到最终反应物”所需的插入、替换和删除，都会获得这条终点路径的 reward。这样实现简单，但有一个明确的信用分配错配：

```text
当前中间分子 x_t ──本步实际随机编辑──> x_(t+Δ) ──许多后续编辑──> 最终反应物 y

当前训练的目标：x_t ------------------------------> y 的全部编辑
真正需要引导的目标：x_t ----> x_(t+Δ) 的本步编辑
```

终点质量当然与整条路径有关，但它不能说明“从 `x_t` 出发、现在应该偏向哪个编辑”。终点和当前状态之间的差异会混入以后数十步的编辑，也可能混入以后对早期错误的修正。这为此前出现的现象提供了直接解释：guidance 可以学会终点 reward 的全局／组内排序，却在最终 Top-1 上没有稳定收益。

P1/P2 endpoint reward 校准已经关闭：P2 虽将同一共享状态内 reward AUC 提升到 0.7093，却仍使同候选池 Top-1/Top-3 回退。因此下一步不再改变终点 reward，而是改变 reward 被归因给**哪一个编辑集合**。

## 2. 方法本身：记录真实的一步后继，而不是事后对齐到终点

对每个共享中间状态 `x_t`，仍然生成 4 条随机 continuation，仍然用冻结的正向 Molecular Transformer 给各自最终反应物 `y_i` 计算相同的 raw forward-beam reward `r_i`。唯一新增的信息是：对每条 continuation，保存它从 `x_t` 开始的**第一步真实 Euler 转移后**的状态 `x_(t+Δ,i)`。

```text
同一 x_t
  ├─ 实际第一步编辑 a1 → x_(t+Δ,1) → … → y1 → r1
  ├─ 实际第一步编辑 a2 → x_(t+Δ,2) → … → y2 → r2
  ├─ 实际第一步编辑 a3 → x_(t+Δ,3) → … → y3 → r3
  └─ 实际第一步编辑 a4 → x_(t+Δ,4) → … → y4 → r4
```

训练时不再把 `x_t → y_i` 的所有编辑当作 label，而是只把 `x_t → x_(t+Δ,i)` 的实际本步编辑当作 label，并用该路径最终获得的 `r_i` 加权。于是模型学习的问题变成：

> 在完全相同的当前状态和时间下，哪些**已实际执行的一步编辑**更常通向高质量终点？

这仍是 Monte-Carlo return 的近似，而不是精确知道每一个编辑的长期因果价值；但它避免把未发生在当前步的未来编辑错误地标注为当前动作。

## 3. 与 DGM 的关系：更接近，但不冒充 exact DGM

离散 DGM 的理想引导来自“当前状态／下一状态在目标分布和基础分布之间的比值”。工程上无法枚举所有变长 SMILES 编辑及其精确未来价值，因此现有项目用 sampled trajectory 的终点质量作近似。

本方案比旧的终点对齐方式更贴近这个思想：每一条被加权的动作集确实是从当前状态走到一个**真实的一步后继**所采取的转移，而不是把未来整条路径压缩成当前一步。它还没有做到以下事情，必须诚实区分：

- 没有显式拟合严格的状态密度比或 Doob `h` 函数；
- 没有枚举所有可能编辑，只观察基础 Euler 实际采样到的 4 条一步后继；
- 终点 reward 仍然是 forward-beam proxy，仍会包含噪声；
- 推理仍沿用已有 action-weight adapter，不会在这一阶段新增昂贵的逐动作正向模型调用。

因此这是“更正确的局部 action-credit approximation”，不是论文意义上的 exact DGM 声明。

## 4. 适配到 Edit Flows 的最小实现

### 4.1 数据生成

共享 anchor 生成器增加一个默认关闭的选项。开启后，continuation 采样仍按原来的随机种子、模型调用和 100 步轨迹完成，但额外保存每条 child 第一次 Euler 更新后的 token 序列，临时字段命名为 `transition_tokens`。

必须避免为了这个字段把完整的 100 步 continuation 都常驻在 CPU 内存中。采样器只记录初始状态和第一个 post-edit state；默认的完整 trajectory 记录行为不能改变。

默认关闭时，旧文件的字节级随机行为、字段和训练结果都应保持兼容。

### 4.2 数据读取与训练

guidance 数据 collate 仅在所有记录都有 `transition_tokens` 时加入该张量；“一部分有、一部分没有”必须报错。训练脚本新增一个默认 `terminal` 的 target-source 选项：

- `terminal`：保持旧行为，构造 `x_t → y` 的 action mask；
- `transition`：构造 `x_t → x_(t+Δ)` 的 action mask。

同一设置同时作用于 Bregman、pairwise ranking 和 score-calibration loss，避免其中一项仍偷偷使用终点对齐。基础 Edit Flows checkpoint、正向 reward、guidance 网络结构、采样器和已有 terminal 模式都不改。

### 4.3 必需审计

新审计必须回答下列具体问题：

1. 每个 group 仍有 4 条记录，组内 `x_t` 和时间仍完全相同；
2. 每条 `transition_tokens` 确实来自该 child 的第一步 post-edit state，且与 `x_t` 的 action mask 数量合理；
3. 记录第一步状态没有改变同 seed 下的最终 terminal token、forward reward 或 RNG 轨迹；
4. 各时间点的“至少两条非空本步编辑且 reward 不同”的 group 比例；
5. 相比旧的 `x_t → y` mask，本步 mask 的平均动作数、零动作比例和各类编辑比例。

真实 target 只继续用于已有的离线 reward-quality 审计，不用于生成字段、计算 reward 或推理。

## 5. 执行顺序与停止门槛

### 阶段 L0：纯代码与小型真实 smoke

1. 为采样器增加“只保存第一步状态”的可选能力及单元测试；
2. 用 2 个反应、2 个 child、少量 Euler 步做 GPU smoke；关闭/开启记录时 terminal 必须完全一致；
3. 为数据 collate、terminal/transition target switch 和 pairwise path 增加单元测试；`terminal` 模式必须回归原 loss。

通过条件：测试全绿、first transition 存在、关闭开关不改变终点采样、没有大于必要的 trajectory 内存增长。

**状态：已完成（2026-08-11）。**

### 阶段 L1：小规模数据质量审计

用 50 个未重叠训练反应构造五个时间点、四后继的一步后继数据，并附加**现有 raw forward-beam reward**。这里只审计数据结构与局部 action 可用性，不训练、不看开发集。

通过条件：所有 group 完整；至少有足够数量（预期应超过 20%）的 group 同时具有两个以上非空一步 action set 且 reward 有差异；若大部分 late-time group 都是 no-op，先按时间点报告并重新决定是否保留它们，不能直接开始大规模训练。

### 阶段 L2：正式离线训练对照

仅在 L1 通过后，重新生成原始训练反应 0–999 和原始 validation 反应 0–199 的同配置数据（5 个时间点 × 4 children），并使用同一 frozen raw forward-beam reward。训练两套完全相同的 2,000-step guidance：

1. terminal target control；
2. transition target candidate。

除 target source 外，网络、batch、seed、学习率、Bregman guard、pairwise／calibration 权重和 checkpoint selection 都固定。比较 held-out Bregman、可用 pair 数、同组 pair accuracy 和 reward–action-set correlation；同时记录 wall/显存。

进入端到端的必要条件是：candidate 不突破 control 的 Bregman guard，且在**transition target 的 held-out 同组排序**上有可解释提升；仅因为 loss 更低不够。

### 阶段 L3：冻结后的 ordinary-Euler 开发集

只有 L2 通过，才用冻结 checkpoint、相同 `n_steps=100`、`n_samples=3`、batch 64、seed 42、β=0.10 和已有 1,000-reaction dev protocol 做一次 ordinary-Euler off/on 对照。Top-1 不下降且至少一个深层 Top-k／Oracle 有改善才允许确认集；否则记录负结果、停止这一 local-credit 支线。

## 6. 实验记录占位符

| 阶段 | 数据 | 方法 | 正确性／质量 | 效率 | 结论 |
|---|---|---|---|---|---|
| L0 | train 原始反应 1250--1251 | first-transition recording | 12 条旧字段逐条完全一致；6/6 group 同 state/time | 1.149s → 1.264s；峰值 GPU allocated/reserved 均为 228.9/243.3 MB | 通过，进入 L1 |
| L1 | 50 train reactions | data audit | 待运行 | 待运行 | 待运行 |
| L2 | train-1000 / val-200 | terminal vs transition target | 待运行 | 待运行 | 待运行 |
| L3 | dev-unique1000 | ordinary Euler off/on | 待运行 | 待运行 | 待运行 |

## 7. 当前结论

这是 P1/P2 endpoint reward 校准失败后唯一正在推进的下一项：它不声称“reward 更准确”，而检验“已有 reward 是否被赋给了正确时间尺度上的编辑”。L0 已确认记录机制不改变采样；在 L1 数据质量还未通过前，仍禁止重训 guidance、扫描 β、使用确认集或运行 `src-test`。

## 8. 阶段 L0 实现与验证记录（2026-08-11）

### 实现内容

- `sample_euler` 新增可选上限：记录轨迹时可只保留起点和指定数量的 post-edit
  state。`max_recorded_trajectory_steps=1` 只保留 `x_t` 与真实
  `x_(t+Δ)`，不增加模型前向，也不改变随机数调用。
- shared-anchor 数据生成器新增默认关闭的 `--record_first_transition`。开启后，每个
  child record 写入经过第一步实际 Euler 更新的 `transition_tokens`；旧默认文件不新增
  字段，也不改变原有采样路径。
- 数据读取只在一个 batch 的所有 record 都具有该字段时 collate 它；部分 record 缺失会
  立即报错，避免静默混合两类监督数据。
- guidance 训练和只读评估新增 `--action_target_source {terminal,transition}`，默认
  `terminal` 完全保留旧的 endpoint 对齐。选择 `transition` 时，Bregman、同 group
  pairwise ranking 和 score-calibration 三个 loss 都共同使用“当前状态到真实一步后继”的
  action mask，而不是只有其中一项切换。

### 自动化测试

- 本次相关模块定向测试：**69 passed**，覆盖采样 cap、旧终点路径、可选字段的 collate、
  transition target 的 Bregman/pairwise/calibration，以及原 `terminal` 默认 loss 回归。
- 全仓库测试结果为 **330 passed, 18 failed**。其中 17 个失败来自本次未改动的
  `tests/sampling/test_beam.py` 与当前 `EditCandidate` API/controlled-model 的既有不匹配；
  另一个是已有的、仅在整套测试顺序下出现的 `guidance_beta=0` 随机敏感测试，单独运行通过。
  `beam.py` 和其测试相对 L0 开始 commit 未变化，因此本阶段没有为让测试数字好看而改动
  无关 Euler-Beam 代码。

### 真实 CUDA smoke

使用冻结的 `new_checkpoints/checkpoint_step600000.pt`，从训练集原始反应区间
`[1250, 1252)` 生成 2 个反应、3 个 interior anchor、每个 anchor 2 个 child（共 12 条
record），`n_steps=4`、batch=2、seed=42。两个输出分别为：

- 旧模式：`/root/autodl-tmp/dgm_guidance_runs/local_credit_l0_base.pt`
  （SHA256 `579f319a88f1935c801a8a5f848ea10b1928f542af7087445e0a8f7b53b15417`）；
- 开启 first transition：
  `/root/autodl-tmp/dgm_guidance_runs/local_credit_l0_transition.pt`
  （SHA256 `ab576f15f5a7720f5fb097c0239a4845989bf115220fe8824d0f0538a7b0132f`）。

逐条比较显示 product/state/terminal token、time、reward、group/sample index 和两种 seed
全部完全相同；旧文件没有 `transition_tokens`，新文件 12/12 都有且均以 BOS 开头。6/6 个
shared-anchor group 的 state/time 严格相同。12 条中 6 条第一步为 no-op（一步后继与 anchor
相同），5 条后续没有再变化而使一步后继已等于终点；这是短 4-step smoke 的结构事实，不将其
解释为正式数据分布结论。开启记录的 wall 为 1.264s，旧模式为 1.149s，峰值 GPU
allocated/reserved 在两次均为 228.9/243.3 MB；这个极小任务的约 10% wall 差异主要包含启动和
CPU clone，L1 会在真实 100-step 设置下重新记录效率。

**L0 结论：通过。** 接下来只进行 L1 的数据结构与局部 action 可用性审计，不训练 guidance，
不读取开发/测试 target，也不做 Top-k。
