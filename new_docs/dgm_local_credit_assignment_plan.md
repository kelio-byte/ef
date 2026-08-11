# DGM 局部信用分配：一步后继监督方案

更新日期：2026-08-11  
状态：L0 已完成；L1 已完成但未通过预注册的数据可用性门槛；不得进入 L2。

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

这里的“两种 action set”在运行前进一步固定为**两种不同的 action mask**，而非仅仅两条
non-no-op child：同一第一步编辑可能由于后续随机性走向不同终点并获得不同 reward，但这不能给
当前 action 提供可区分的训练方向。审计同时报告较宽松的“两个非空 child”比例和更严格的
“两个不同 action mask 的平均 reward 有差异”比例；通过门槛使用后者。

**状态：已完成，未通过（2026-08-11）。** 单个 Euler 数值步的实际编辑概率远低于预期，
不训练 transition guidance，也不重建 train-1000/val-200。

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
| L1 | train 原始反应 1200--1249 | 5 time × 4 child 的 local-action audit | 250/250 group 结构正确，但仅 1/250（0.4%）有两种不同非空 action 且 reward 有差异 | trajectory 21.20s，forward reward 10.41s，峰值 544/1,030 MB | 未通过；禁止进入 L2 |
| L2 | train-1000 / val-200 | terminal vs transition target | 待运行 | 待运行 | 待运行 |
| L3 | dev-unique1000 | ordinary Euler off/on | 待运行 | 待运行 | 待运行 |

## 7. 当前结论

这是 P1/P2 endpoint reward 校准失败后唯一正在推进的下一项：它不声称“reward 更准确”，而检验“已有 reward 是否被赋给了正确时间尺度上的编辑”。L0 已确认记录机制不改变采样；L1 已表明“紧邻的一个数值 Euler step”几乎全为 no-op，故 transition target 不能直接用于 guidance 训练。除非先提出并通过新的局部时间尺度／event conditioning 数据诊断，仍禁止重训 guidance、扫描 β、使用确认集或运行 `src-test`。

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

## 9. 阶段 L1 数据质量审计记录（2026-08-11）

### 固定数据和 reward 构造

L1 使用训练 split 的原始完整反应块 `[1200, 1250)`。它与此前 guidance/reward 训练的
`[0,1000)`、reward calibration holdout 的 `[1000,1200)`、L0 smoke 的 `[1250,1252)` 都不重叠；
没有读取 validation 或 test target。配置在运行前固定为：100 个 Euler 数值步，anchor step
10/30/50/70/90，每个共享 anchor 4 条 child、batch 32、seed 42，并记录每个 child 的第一数值步
post-edit state。

先得到 1,000 条记录／250 个 group 的 validity 文件，再在**同一固定终点**上附加既有的
Molecular Transformer forward-beam=5 reciprocal-rank reward（canonical source、batch 16）。文件不进
Git，SHA-256 为：

```text
local_credit_l1_train50_start1200_validity.pt  a3c52c3d881e79b4c8698116ddb68ac19ef605b78449ce9c6e1b2a8987a162c1
local_credit_l1_train50_start1200_beam.pt      0a384760e6b9520cdb606cdeaa0bfd5c042263286b2403032fb35eba4b272a74
local_credit_l1_train50_start1200_action_audit.json
                                                78492aba89b9f59bc9729c429c6ffd40cea41b82b30f9ab61e5ace503158adef
```

轨迹生成 wall 为 **21.20 s**（47.18 records/s），峰值 CUDA allocated/reserved 为
**544.2/1029.7 MB**；forward reward wall 为 **10.41 s**，1,000 个终点的 forward-beam hit rate 为
67.9%，平均 reward 0.6157。此处 reward 只来自产物和已生成终点，未使用真实反应物。

### 结构正确性

审计确认 250/250 group 都恰有 4 条 record，250/250 group 的中间 state 和 time 分别完全一致；
所有 1,500 个同 group pair 的 state 都一致。因此失败不是共享 anchor、字段写入、padding 或
forward reward 附加错误造成的。

### 关键结果：单数值步的编辑过于稀疏

| anchor step | 真实一步 non-noop child | 有两种不同 action 且 reward 有差异的 group | 旧 terminal 对齐的 non-noop child |
|---:|---:|---:|---:|
| 10 | 0 / 200（0.0%） | 0 / 50（0.0%） | 193 / 200（96.5%） |
| 30 | 3 / 200（1.5%） | 0 / 50（0.0%） | 190 / 200（95.0%） |
| 50 | 7 / 200（3.5%） | 0 / 50（0.0%） | 188 / 200（94.0%） |
| 70 | 14 / 200（7.0%） | 0 / 50（0.0%） | 183 / 200（91.5%） |
| 90 | 27 / 200（13.5%） | 1 / 50（2.0%） | 103 / 200（51.5%） |
| 合计 | **51 / 1000（5.1%）** | **1 / 250（0.4%）** | **857 / 1000（85.7%）** |

一步 target 中的 53 个实际编辑有 49 个插入、4 个替换、0 个删除；其平均每条 record 的编辑数仅
0.053。相比之下旧 terminal 对齐平均每条有 2.877 个编辑，正是它把未来整条路径混入当前标签的
直接量化证据。

这不是采样 bug：Euler 的 100 个数值步把连续时间切得很细，某一步通常发生“什么也不编辑”。在
推理时这仍是正确的随机过程；但当前 action-mask Bregman/ranking 训练对 no-op 没有一个可加 reward
的具体编辑，因此绝大多数 record 只提供 background 信号。直接用这份数据跑 2,000-step training 会
把实验资源花在 94.9% 的无 action label 上，也无法公平比较 terminal/transition。

**L1 结论：不通过。** 预注册门槛为严格 local-discriminative group 比例超过 20%，实测仅 0.4%。
L2/L3 不启动。下一项只能是一个新的、先做小型数据诊断的方案：要么在当前状态显式按“发生编辑”
条件采样多个 action proposal，要么定义经证明足够短的多数值步局部窗口；不能把更多训练步、更多
β 扫描或更大训练集当作对这一数据缺陷的补救。

## 10. 后续候选 E：按事件条件抽样的原子编辑 proposal（已预注册，尚未实现）

### 10.1 为什么优先验证它

L1 排除了“只增加训练步数”这个解释：在 100-step Euler 中，一个数值步绝大多数时候没有编辑，
而当前 guidance 的三张权重表恰好描述的是**如果发生编辑，哪一个插入／替换／删除更值得选择**。
更重要的是，现有采样时的 per-position normalization 会保留每个位置的总编辑率；它只重新分配
这个编辑率到具体 token 和操作，并不试图学习 no-op 的概率。因此用 94.9% no-op record 训练具体
编辑权重既低效，也没有和推理时真正被改变的量对齐。

候选 E 保留当前共享状态、终点 rollout 和 forward reward，只改变“如何为当前 state 取得有信息量的
具体编辑”：从基础 Edit Flows 在当前时间给出的瞬时有效编辑率中，**条件于发生一个有效编辑**抽样
一个原子 action。对位置 `i` 和 token `v`，其未归一化质量是：

```text
insert(i, v)     = insert-rate(i)     × P_insert(v | i)
substitute(i, v) = substitute-rate(i) × P_substitute(v | i)
delete(i)        = delete-rate(i)
```

PAD、BOS、无效位置及“替换为当前同一 token”的 action 都会排除，再在剩余 action 上归一化抽样。
每个 child 因而必有且仅有一个真正改变 token 序列的编辑；随后从该编辑后的 state 在本次数值步结束
的真实时间继续普通 Euler 到终点，仍用冻结的 forward-beam reward 打分。

### 10.2 它解决什么、与 DGM 的距离是什么

它直接解决 L1 的样本有效率问题：训练不再依赖罕见的自然发生事件，而是从基础 rate field 的
**action identity distribution**主动取得 proposal。对 DGM/action guidance 而言，这更接近要学习的
“基础转移中哪些具体跳转应被放大”的量；同一 state/time 下的 child 仍可做公平的相对 reward 比较。

它不是完整 exact DGM，也不伪装成普通 Euler 单步的精确条件分布：普通 Euler 在一个数值步可同时
发生多个位置的编辑，而本方案强制一个原子编辑。它是小步长下连续时间 jump process 的
event-conditioned 近似。它也不学习总编辑强度或 no-op 概率；这与当前 per-position guidance
normalization 的能力边界一致。若未来改用会改变总 rate 的推理规则，必须重新定义这一训练数据，
不能复用 E 的结论。

### 10.3 最小适配方式

1. 新增一个独立的 GPU 向量化 helper，只从冻结 Edit Flows 的单次前向输出构造有效原子 action
   质量并为 batch 中每一行抽取一个 action；不得改动普通 `sample_euler` 的默认逻辑。
2. 新的 shared-anchor 生成器先按原流程生成 prefix；每个 anchor 对所有 child 做一次上述
   event-conditioned proposal，记录 proposal 的操作、位置、token、基础 log probability 和编辑后的
   `transition_tokens`；随后从本 Euler 数值步结束的时间 rollout 到 terminal。
3. 已有 collate/`action_target_source=transition`/loss 不需要为 E 重写：它们只读取
   `state → transition_tokens` 的 action mask。新 metadata 必须明确标记 proposal 条件化，避免和 L0/L1
   的自然 one-step 数据混用。
4. 使用同一 raw forward-beam reward、同一 canonicalization、同一 n_steps/anchor/child 和同一
   local-action audit。真实反应物仍不参与生成、reward 或训练。

### 10.4 执行顺序和门槛

| 阶段 | 工作 | 固定数据／配置 | 通过条件 | 不通过时的动作 |
|---|---|---|---|---|
| E0 | helper 单元测试 + 2 reaction CUDA smoke | 100 steps，5 anchor，4 child | 每 child 恰有一个有效 state-changing action；不改普通 Euler；rollout 起始时间正确（已通过） | 不适用；进入 E1 |
| E1 | 50-reaction 数据质量 pilot | 新的 train 原始块 `[1252,1302)`；同 L1 reward | group 完整共享；严格 local-discriminative group **>20%**；记录 action 类型、重复率、wall/显存 | 不进入正式数据／训练，记录负结果 |
| E2 | 正式离线对照 | train `[0,1000)` / val `[0,200)`，除 target source 外固定 | event proposal candidate 通过 held-out Bregman guard，且同 group 的局部 ranking/correlation 有可解释提升 | 不做 Top-k |
| E3 | frozen ordinary-Euler dev | 既有 1,000-reaction开发协议 | Top-1 不降且至少一项深层 Top-k/Oracle 改善 | 记录并关闭 E 支线 |

E1 只检查数据，不能因“每条都已有 action”便跳到 E2。由于 E 会额外增加每个 anchor 一次冻结模型
前向，smoke 和 E1 必须报告总 wall、records/s、峰值显存；在没有实测前，不宣称它更快或更慢。

### 10.5 E0 实现与 CUDA smoke（2026-08-11）

新增 `sample_event_conditioned_atomic_actions` 和
`sample_event_conditioned_euler_transition`：前者从当前 state 的 base action-rate mass 中为每行
抽出一项有效、会改变 token 序列的原子编辑；后者使用与普通 Euler 相同的时间映射、cross-scheduler
rate correction、rate parameterization 和 adaptive time increment，应用这一个编辑后返回该数值步的
真实结束时间。普通 `sample_euler` 没有调用这两个函数，默认采样路径不变。

新增独立脚本 `scripts/generate_event_conditioned_guidance.py`，以避免修改已有 natural shared-anchor
生成器。它记录 proposal 的类型、位置、token、基础 log-rate/log-probability 和 rollout 起始时间，
并在 metadata 中明确标注 `event_conditioned_atomic_rate`、被排除的结构 token 以及时间语义，防止
与 L0/L1 的自然数值步数据混用。

自动化验证：相关 Euler/guidance 定向测试 **67 passed**。其中覆盖了同 seed proposal 可复现、每行
恰有一个 action、BOS/PAD/结构 token 与 no-op substitute 均不会被抽到、无有效 action 会显式失败，
以及 transition 不原地修改输入且其 time advance 正确。

真实 smoke 使用冻结 checkpoint、训练原始反应 `[1302,1304)`、100 steps、anchor
10/30/50/70/90、4 child、batch=2、seed=42；只使用 cheap validity reward，不附加 forward beam。
生成 40 record／10 group，wall **4.391 s**（9.11 records/s），峰值 CUDA allocated/reserved
**234.4/247.5 MB**。审计确认：10/10 group 共享 state/time；40/40 transition 都与起始 state
不同、action mask 都恰为 1 个编辑、40/40 rollout start time 都等于相应 adaptive Euler step 的
真实终点。文件 SHA-256：

```text
event_proposal_e0_smoke_validity.pt     17527c4339efa332d080bdefbccd18274c6d466e8c996160626326c76be8880e
event_proposal_e0_smoke_action_audit.json
                                        c6e8d1e84ae7979107a9da4a41258f571000e38b22ed4bf15e58e591fa2d6440
```

该极小 smoke 的 40 个 proposal 都是插入，和 L1 自然事件中插入占多数的现象一致；样本太小且只用
validity reward，不能据此判断操作多样性或方法效果。其唯一结论是 E0 的数据和时间语义正确，允许
按预注册的 `[1252,1302)`、raw forward-beam E1 pilot 检查 action diversity 与 group reward 信号。
