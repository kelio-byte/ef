# Euler-Beam 下一阶段任务规划

更新日期：2026-08-03

## 1. 文档用途

本文档承接 [`euler_beam_optimization_plan.md`](euler_beam_optimization_plan.md) 中已经
完成的任务 0–8，负责记录下一阶段围绕 Top-2/Top-3 的诊断和改进。

每完成一项任务，都必须在对应位置补充：

- 实际修改；
- 测试命令与结果；
- 实验配置、运行时间和 Top-k；
- 是否保留实现；
- Git commit hash。

状态标记：

- `[ ]` 未开始
- `[-]` 进行中
- `[x]` 已完成并保留
- `[!]` 实验完成但方案被淘汰或受阻

## 2. 当前起点

### 2.1 推荐配置

```text
checkpoint: checkpoint_step600000.pt
dataset: USPTO_50K_PtoR_aug20_#global#/test/*-test-tiny.txt
K=3, M=2, R=3, n_steps=100, seed=42
score_mode=full_probability
changed_state_bonus=0.5
child_policy=stochastic_noop
matmul_precision=high（RTX 3090 TF32）
```

完整 tiny benchmark：

| 指标 | 当前值 |
|---|---:|
| 采样时间 | 约 122.6 秒 |
| Top-1 | 60% |
| Top-2 | 64% |
| Top-3 | 70% |
| Invalid rank 1/2/3 | 12.5% / 14.5% / 13.4% |

历史 58% 恢复版本为 58%/68%/76%，但其 seed、候选分布和评分语义不同，只能作为
研究线索，不能作为当前实现的严格对照。

### 2.2 当前核心问题

当前方法从 Top-1 到 Top-2 只增加 4 个百分点，到 Top-3 累计增加 10 个百分点；旧
恢复版本分别增加 10 和 18 个百分点。当前实现提高了最优候选质量和有效 SMILES 比例，
但三个独立 run 可能集中在相似模式。

必须区分两个原因：

1. **采样覆盖不足**：正确答案根本没有出现在全部 augmentation × run 候选中；
2. **聚合排序不足**：正确答案已经采到，但没有排进最终 Top-2/Top-3。

在完成这项归因前，不大范围修改 Euler-Beam，也不改变默认评分语义。

## 3. 固定实验协议

### 3.1 当前 Euler-Beam

```bash
python scripts/sample_retro.py \
    --checkpoint checkpoint_step600000.pt \
    --products_file "datasets/USPTO_50K_PtoR_aug20_#global#/test/src-test-tiny.txt" \
    --sampler euler_beam \
    --n_branches 3 \
    --n_children 2 \
    --n_runs 3 \
    --n_steps 100 \
    --batch_size 64 \
    --device cuda \
    --seed 42 \
    --euler_beam_score_mode full_probability \
    --euler_beam_changed_state_bonus 0.5 \
    --euler_beam_matmul_precision high \
    --euler_beam_child_policy stochastic_noop \
    --output_dir results/bench_beam/

python 'scripts/score_#global#.py' \
    --predictions results/bench_beam/predictions.txt \
    --targets "datasets/USPTO_50K_PtoR_aug20_#global#/test/tgt-test-tiny.txt" \
    --augmentation 20 \
    --beam_size 9 \
    --n_best 10 \
    --diagnostics
```

### 3.2 Euler 对照

```bash
python scripts/sample_retro.py \
    --checkpoint checkpoint_step600000.pt \
    --products_file "datasets/USPTO_50K_PtoR_aug20_#global#/test/src-test-tiny.txt" \
    --sampler euler \
    --n_samples 3 \
    --n_steps 100 \
    --batch_size 16 \
    --device cuda \
    --seed 42 \
    --output_dir results/bench_euler/

python 'scripts/score_#global#.py' \
    --predictions results/bench_euler/predictions.txt \
    --targets "datasets/USPTO_50K_PtoR_aug20_#global#/test/tgt-test-tiny.txt" \
    --augmentation 20 \
    --beam_size 3 \
    --n_best 5
```

任何评分前都检查：

```text
预测文件 = 本次采样输出
prediction lines = 原始反应数 × augmentation × beam_size
beam_size = n_runs × n_branches（Euler-Beam）或 n_samples（Euler）
```

## 4. 执行顺序

```text
任务 9：评分输入校验与覆盖率诊断
   |
   +-- 正确答案未被采到 --> 任务 10：异质 run / 受控探索
   |
   +-- 正确答案已采到但排名低 --> 任务 11：聚合方法消融
   |
   v
任务 12：输出元数据与可复现实验接口
   |
   v
任务 13：对胜出方法做完整正确性、准确率和性能验证
```

任务 10 和任务 11 不是预先二选一。任务 9 的结果决定先后顺序；如果两类损失都明显，
可以分别推进，但每次实验只改变一个因素。

## 5. 任务 9：评分校验与“覆盖—排序”归因

状态：`[x] 已完成；确认覆盖不足为主、聚合仍有次级损失`

### 9.1 严格输入校验

在不改变历史聚合结果的前提下，为 `score_#global#.py` 增加：

- `augmentation > 0`、`beam_size > 0`、`n_best > 0` 参数检查；
- prediction 行数必须严格整除 `augmentation * beam_size`；
- target 行数必须足以覆盖相同数量的原始反应；
- 默认拒绝静默截断；
- 当显式使用 `--length` 时，校验所需行数而不是无条件截断；
- 报告推断出的原始反应数和期望/实际行数。

验收标准：

- 正确输入的原 Top-k 数值逐项不变；
- 少一行、多一行、错误 beam size 都明确报错；
- 为上述情况增加快速单元测试，不调用模型和 GPU。

### 9.2 修正报告含义，不改变排名

新增名称明确的统计：

- 聚合前有效候选率；
- 每个 run 的 invalid rate；
- 每个 run 的 canonical duplicate rate；
- 每个反应全部 augmentation × run 的真实唯一候选数；
- run 两两之间的 canonical overlap。

保留旧 `Unique Rates` 输出以便复现历史日志，但标注为 legacy metric；新指标不能使用
`rank[:n_best]` 截断后的数量冒充原始多样性。

同时解除 Top-N 打印被 `beam_size` 限制的问题：聚合后确有 N 个候选时，可以报告到
`n_best`；缺失候选需要单独报告，不能计作 RDKit invalid。

### 9.3 Oracle/目标覆盖诊断

对每个原始反应记录：

- `oracle_any`：目标是否出现在任何 augmentation、任何 run；
- `target_aug_count`：包含目标的 augmentation 数量；
- `target_best_local_rank`：目标在单个 augmentation 内的最好 run 位置；
- `target_consensus_score`：原聚合规则下目标的共识分数；
- `target_final_rank`：目标聚合后的最终名次；
- `coverage_to_rank_loss`：已经覆盖但落在 Top-3 之外的数量。

关键输出：

| 指标 | 当前最佳 | stochastic | Euler | 58% 恢复版本 |
|---|---:|---:|---:|---:|
| Oracle-any | 80 | 78 | 90 | 92 |
| 最终 Top-1 | 60 | 58 | 54 | 58 |
| 最终 Top-2 | 64 | 64 | 66 | 68 |
| 最终 Top-3 | 70 | 66 | 74 | 76 |
| 已覆盖但未进 Top-3 | 10 | 12 | 16 | 16 |
| 平均真实唯一候选数 | 13.18 | 12.94 | 15.18 | 12.72 |
| run micro-Jaccard 范围 | 29.0–32.5 | 29.7–33.2 | 23.9–26.1 | 34.4–42.6 |

注意：不同 seed/语义的恢复版本只用于定性解释，不写成严格消融结论。

### 9.4 本任务完成记录

实际修改：

- 默认严格检查 prediction/target 行数和 `augmentation × beam_size` 布局；不再静默
  截断。`--length` 只允许读取具备完整行数的前缀。
- `canonicalize_smiles_clear_map()` 和 `compute_rank()` 不再依赖全局 `opt`，且
  `compute_rank()` 不再原地修改输入候选。
- 保留原 best-local-rank 聚合公式和旧 `Unique Rates` 数值。
- Top-N 报告解除 `beam_size` 限制；超过输入 rank 的位置将 invalid 显示为 N/A。
- 新增 `--diagnostics` 和 `--diagnostics_json`，记录 Oracle-any、target augmentation
  support、最好局部排名、最终排名、每 run 命中/invalid/重复率、真实唯一候选数、
  aggregated rank availability 和 run 两两 Jaccard overlap。
- 新增 `tests/test_score_global.py`，覆盖严格布局、显式 prefix、错误参数、旧聚合语义、
  非原地修改和诊断统计。

测试与实验：

- 新增评分测试：`9 passed`。
- 排除既有 `tests/sampling/test_beam.py` 后：`123 passed`。
- 全套测试：`141 passed, 17 failed`；17 项均位于本次未修改的 `test_beam.py`，主要是
  `EditCandidate(log_u_real=...)` 与当前 `beam.py` 接口不匹配，未越界修复。
- 2999 行预测配合 `augmentation=20, beam_size=3` 会在 canonicalization 前明确报错。
- 当前最佳预测在改动前后 Top-1/2/3、invalid 和旧 Unique 数值逐字一致：
  `60/64/70`、`12.5/14.5/13.4%`、`165.333%`。
- 当前最佳聚合候选 rank 1–3 可用率均为 100%，rank 4–5 为 98%。
- 四组诊断均复用已有 3000 行预测，没有重新采样或修改历史结果文件。

实验文件说明：当前最佳使用 `results/task8_noop_full/predictions.txt`，stochastic 使用
`results/task7_precision_full_tf32/predictions.txt`，Euler 使用
`results/bench_euler/predictions.txt`，恢复版本使用
`results/bench_beam_pre_full_score/predictions.txt`。当前 `bench_euler` 文件实测为
54/66/74，而早期文档曾记录 56/68/74；旧评分脚本对该文件同样得到 54/66/74，说明
这是实验文件来源差异，不是本次评分重构造成的变化，后续由任务 12 的元数据解决。

结论：

1. 当前最佳 Oracle-any 只有 80%，比 Euler 低 10pp、比恢复版本低 12pp；Top-2/3
   增长慢首先是高价值探索覆盖不足。
2. 当前仍有 5/50 个反应已经采到 target 但最终落在 Top-3 外，聚合排序存在次级空间。
3. 当前 Top-3/Oracle 转化率为 87.5%，高于 Euler 的 82.2% 和恢复版本的 82.6%；不能
   先修改默认聚合规则来追求表面 Top-3，优先进入任务 10 的离线异质 run。
4. 恢复版本的 run overlap 反而更高、真实唯一候选数更低但 Oracle 更高，说明“候选越
   多或 overlap 越低”并不自动带来目标覆盖；探索需要提高化学相关候选质量。

Commit：`b0c1695 Add strict global scoring diagnostics`

## 6. 任务 10：异质 run 与受控探索

状态：`[x] 简单 policy 混合淘汰；激进 exploration 候选池完成 5/20/50 验证`

目标是在尽量保持 Top-1 和有效率的同时，让三个最终输出承担不同角色，而不是仅依靠
不同 seed 运行同一种 policy。

### 10.1 先做离线候选混合

优先复用已经生成的完整预测，不重新调用模型。按每个增强输入的 run 位置，构造：

- 2 条 `stochastic_noop` + 1 条 `stochastic`；
- 1 条 `stochastic_noop` + 2 条 `stochastic`；
- 当前 3 条 `stochastic_noop` 对照；
- 当前 3 条 `stochastic` 对照。

离线混合脚本必须检查两个输入文件的行数和 `(reaction, augmentation, run)` 布局，并把
来源组合写入输出旁的元数据文件。不能简单拼接整份文件。

进入实现的证据要求：

- Oracle-any 或 Top-2/Top-3 相对当前最佳出现正向信号；
- Top-1 不出现无法解释的明显下降；
- 改善来自候选互补，而不是错误的行排列或评分截断。

执行结果（2026-08-01）：

- 新增 `scripts/mix_retro_runs.py`。每个输入用 `LABEL PATH` 注册，每个输出位置用
  `LABEL:RUN` 选择；严格检查两份文件的行数、augmentation、input beam 和 run 下标。
- 工具默认拒绝覆盖已有输出，同时写出 source 路径、SHA-256、行数、run 来源、布局和
  output SHA-256 到 `mixing_metadata.json`。
- no-op 与 stochastic 在 run 1/2/3 上分别有 59/79/76 行不同，即 5.9%/7.9%/7.6%。
- N 表示 `stochastic_noop`，S 表示 `stochastic`。固定相同 run 位置后，8 种完整组合：

| 组合 | Top-1/2/3 | Oracle-any | 已覆盖未进 Top-3 | 平均真实唯一候选 |
|---|---:|---:|---:|---:|
| NNN | 60/64/70 | 80 | 10 | 13.18 |
| NNS | 60/64/70 | 80 | 10 | 13.08 |
| NSN | 60/64/70 | 80 | 10 | 13.08 |
| NSS | 60/64/70 | 80 | 10 | 12.98 |
| SNN | 58/64/66 | 78 | 12 | 13.14 |
| SNS | 58/64/66 | 78 | 12 | 13.04 |
| SSN | 58/64/66 | 78 | 12 | 13.04 |
| SSS | 58/64/66 | 78 | 12 | 12.94 |

Oracle 集合进一步确认：NNN 命中 40 个、SSS 命中 39 个，39 个 SSS 命中全部包含在
NNN 中；SSS 没有提供任何独有 target，NNN 独有反应为 index 13。因此当前两种 policy
没有覆盖互补性，简单异质组合不能达到 10.2 的进入条件。

### 10.2 实现 per-run policy

只有离线混合有效时，才在 `sample_retro.py` 增加每个 run 独立的 policy 配置。设计要求：

- 单一 `--euler_beam_child_policy` 保持兼容；
- 新接口明确要求 policy 数量为 1 或 `n_runs`；
- product/run seed 保持稳定，不因 policy 列表或 batch size 改变；
- 不把异质 run 错误合并成一个内部 beam；
- 输出顺序仍为 product-major、run-minor。

决策：`[!] 不实施`。离线混合没有提高 Oracle-any 或任一 Top-k；现在增加 per-run
policy 只会扩大接口而没有方法收益。

### 10.3 若简单混合不足，再研究探索 proposal

研究顺序：

1. 只让一个 run 使用受控 exploration，另外两个保持当前最佳；
2. exploration 优先针对反应中心候选或模型不确定位置；
3. 明确记录目标概率 $p$、proposal $q$ 和是否做校正；
4. 先做 5/20 个反应短筛，再决定是否运行 50 个反应；
5. 不重复已经失败的 antithetic、多次 no-op、强制低概率编辑和内部低排名 branch 输出。

执行结果：

#### 普通 Euler 作为 exploration

用 E 表示普通 Euler，在相同 run 位置离线替换 N：

| 组合 | Top-1/2/3 | Oracle-any |
|---|---:|---:|
| NNN | 60/64/70 | 80 |
| NNE | 60/64/70 | 86 |
| NEN | 58/66/70 | 86 |
| NEE | 56/62/66 | 88 |
| ENN | 54/64/66 | 86 |
| ENE | 56/64/66 | 90 |
| EEN | 58/68/72 | 90 |
| EEE | 54/66/74 | 90 |

NNE 新增 3 个 target，但分别只出现在 1–2 个 augmentation，最终排名为 11/20/17；
即使去除 legacy best-rank 巨大惩罚，也没有足够共识进入 Top-3。保留全部 NNN 后追加
1/2/3 个 Euler run，Oracle 分别为 86/90/92，Top-2 最多由 64 提高到 66，Top-3 不变。

#### `legacy_triggered_reverse` 作为激进 exploration

旧 score mode 只累计触发事件并反向偏好低概率、多编辑路径。它不是目标 CTMC 的校准
概率，但已有候选表明它与当前 NNN 高度互补。历史 K=5,M=1 legacy 文件与 NNN 的
Oracle 并集为 96%，但历史采样约 479 秒，不具备当前效率。

因此用当前 K=3,M=2、TF32 批量实现重新采样：

```text
score_mode=legacy_triggered_reverse
child_policy=stochastic
K=3, M=2, R=3, n_steps=100, seed=42
```

- 5 反应：standalone 80/100/100，Oracle 100%；信号过小，仅用于进入 20 反应。
- 20 反应：standalone 55/60/60，Oracle 85%；与 NNN 六 run 并集 Oracle 95%。
- 50 反应：standalone 56/60/70，Oracle 88%，invalid rank 1/2/3 为
  19.6/19.2/21.5%；采样约 168 秒。
- 当前 NNN 与该 exploration 的完整并集覆盖 49/50，Oracle-any 为 98%，平均真实唯一
  候选为 25.2。

在 frequency-first 聚合下，追加当前 legacy prefix budget 的完整结果：

| 候选池 | 输出数 | Top-1/2/3 | Top-4/5 | Oracle-any | 采样成本说明 |
|---|---:|---:|---:|---:|---|
| NNN | 3 | 56/64/68 | 70/72 | 80 | 约 122.6 秒 |
| NNN + L1 | 4 | 56/64/72 | 74/76 | 96 | L1 单独耗时未测 |
| NNN + L1–2 | 5 | 62/66/78 | 80/82 | 98 | 预计低于完整 L1–3，待实测 |
| NNN + L1–3 | 6 | 62/68/76 | 80/84 | 98 | 约 122.6 + 168 秒 |

注意：上表使用同一个 frequency-first evaluator。历史默认聚合下，NNN 为 60/64/70，
NNN+L1–3 为 60/66/70，说明采样候选池本身在旧协议下只直接改善 Top-2；62/68/76
是“异质候选池 + 新聚合”的完整流程结果，不能全部归因于采样器。

### 10.4 本任务完成记录

当前阶段实际修改：新增经过布局校验、默认防覆盖并记录来源哈希的离线 run 混合工具；
复用 Euler-Beam 已有 opt-in legacy score mode，没有修改 `sample_retro.py`、
Euler-Beam、checkpoint 或历史结果。

测试与实验：混合工具与评分器相关测试共 `17 passed`；六份新混合输出均为 3000 行，
评分器严格识别为 `50 × 20 × 3`。NNN/SSS 直接复用原文件，其余六种输出写入新的
`results/task10_mix_*` 目录，不覆盖历史结果。

阶段结论：现有 no-op 与 stochastic 没有 target 互补性；普通 Euler 有互补 target，但
单 run 支持太弱；当前 K3M2 legacy-triggered exploration 与 NNN 形成强互补，Oracle
达到 98%。它以更高 invalid 和约 2.4× 顺序采样成本换取覆盖，需要在任务 12/13 中把
候选来源、运行配置和实际两-run预算耗时固定下来，再决定是否形成正式一体化接口。

Commit：`874825c Add auditable retrosynthesis run mixer`

## 7. 任务 11：评分聚合方法消融

状态：`[x] 四种无权重聚合已实现并完成离线消融；默认历史模式不变`

仅当 Oracle 诊断证明存在明显“已覆盖但排名靠后”时开展。

### 11.1 保留历史默认模式

当前规则以候选在任意 augmentation 中的最好局部位置为第一排序级，再比较 reciprocal
rank 累积分数。它必须保留为 `legacy_best_rank` 或等价模式，确保历史结果可复现。

### 11.2 候选聚合方案

以 opt-in 参数做单变量消融：

- `legacy_best_rank`：当前规则；
- `rrf`：只使用跨 augmentation reciprocal-rank 累积；
- `frequency_first`：出现的 augmentation 数优先，局部排名作次级排序；
- `hybrid`：RRF 加有界的最佳局部排名 bonus。

任何聚合方法都只使用预测本身能够提供的信息，不能利用 target 选参数或排序。

### 11.3 防止指标虚高

- 原历史模式始终同时报告；
- 先在预先定义的小范围方案上比较，不扫描大量权重；
- 若要调权重，划分独立验证集，不能在 tiny test target 上挑最佳值；
- 报告 Oracle 上限、最终 Top-k 和覆盖到排名的损失；
- 评分器变化不宣称为采样模型准确率提升。

### 11.4 本任务完成记录

实际修改：`score_#global#.py` 新增显式 `--aggregation_mode`：
`legacy_best_rank`、`rrf`、`frequency_first`、`hybrid`。默认仍为历史模式；没有暴露
连续权重，也没有修改 `score_alpha` 默认值。frequency-first 以出现 augmentation 数
为第一排序级、RRF 为 tie-break；hybrid 使用 `RRF + 1/(best_rank+1)`。

测试与实验：新增模式优先级测试后，评分器与混合工具共 `18 passed`；排除仓库既有
`test_beam.py` 失败后 `132 passed`。默认模式对当前 NNN 的 60/64/70、invalid 和
165.333% legacy Unique 逐字回归一致。

代表性消融：

| 候选池 | legacy | RRF | frequency-first | hybrid | Oracle |
|---|---:|---:|---:|---:|---:|
| NNN | 60/64/70 | 60/64/70 | 56/64/68 | 60/64/70 | 80 |
| NNE | 60/64/70 | 60/64/70 | 60/64/70 | 60/64/70 | 86 |
| EEE | 54/66/74 | 54/66/76 | 56/66/76 | 54/66/74 | 90 |
| NNN + current L1–2 | 60/64/70 | 60/64/72 | 62/66/78 | 60/64/72 | 98 |
| NNN + current L1–3 | 60/66/70 | 60/66/70 | 62/68/76 | 60/66/70 | 98 |

结论：

1. 单个 Euler exploration 新 target 支持过低，RRF/hybrid 不能把 NNE 的新增覆盖转化为
   Top-3，证明早先的排序损失不只来自 `-1e8`。
2. frequency-first 会让同质 NNN 的 Top-1/3 下降 4/2pp，不能无条件替换历史默认评分。
3. 在高度互补的 aggressive exploration 候选池上，frequency-first 才产生完整 Top-k
   收益；必须与 legacy 默认同时报告，并在独立更大评估集复核，不能称作纯采样提升。
4. tiny benchmark 每个反应为 2pp。L1–2 与 L1–3 的 Top-2/3 取舍不足以证明两-run预算
   普遍更好，任务 13 需要预先固定候选方案再扩大验证。

Commit：`344ae59 Add opt-in augmentation aggregation modes`

## 8. 任务 12：采样输出元数据和接口健壮性

状态：`[x] 已完成；采样轨迹与 predictions.txt 格式不变`

目标是避免再发生预测文件、beam size 或实验配置混淆。

计划：

- 修复 Euler-Beam 启动提示仍显示 `n_samples` 的问题；
- 在输出目录保存机器可读元数据，例如 `sampling_metadata.json`；
- 至少记录 checkpoint、输入文件、输入行数、sampler、K/M/R、n_steps、seed、precision、
  score mode、bonus、child policy、输出行数和 Git commit；
- 可选记录每条输出的 product index、augmentation index、run index 和 policy；
- 评分器若读到元数据，自动核对 `beam_size`、augmentation 和预测行数；
- 旧的纯文本 `predictions.txt` 格式保持不变。

如果要输出 path score、log mass 或分支来源，先确认这些量的语义能够跨 run 比较；不能
为了记录方便把启发式 mass 当成校准概率。

### 12.1 本任务完成记录

实际修改：

- 修正 Euler-Beam 启动提示：每产物输出数使用 `n_runs`，不再错误显示 `n_samples`。
- `sample_retro.py` 在成功完成采样后写入 `sampling_metadata.json`；记录 checkpoint
  路径/大小/mtime，输入文件路径/行数/SHA-256，sampler、K/M/R、有效 n_steps、scheduler、
  seed 及其作用范围、precision、score mode、bonus、child policy、origin mask、输出行数/
  SHA-256、wall time 和 Git commit/dirty 状态。
- augmentation 只从实际 `--products_file` 的明确 `augN` 路径推断；单条 `--product`
  记录为 `null`，不借用 checkpoint 训练目录进行不可靠推断。
- 输出完成时验证实际写入行数等于 `product_count × n_runs/n_samples`；原
  `predictions.txt` 内容和排列不变。
- `score_#global#.py` 自动发现并校验同目录的 `sampling_metadata.json` 或
  `mixing_metadata.json`，在 canonicalization 前核对 beam size、已知 augmentation、
  行数、SHA-256 及 product/reaction 布局；两份 manifest 同时存在时拒绝猜测。
- 没有 manifest 的历史纯文本结果继续兼容，并明确打印 legacy 输入提示。
- 新增 `tests/test_sample_retro_metadata.py`，扩展 `tests/test_score_global.py`，覆盖实际
  输出数、augmentation 推断、元数据字段、哈希、错误参数和歧义 manifest。

测试与实验：

- 元数据、评分和 run 混合的 CPU 快速测试：`27 passed in 0.63s`。
- 项目回归（排除已知且未修改的 `tests/sampling/test_beam.py`）：
  `141 passed, 7 warnings in 4.92s`。
- RTX 3090 冒烟测试：1 个 product，K=1、M=2、R=2、n_steps=1，采样主体约
  `0.738s`；启动提示显示 2 outputs，写出 2 行并由评分器成功验证 metadata。
- 相同配置/seed 重复运行的两份 `predictions.txt` SHA-256 均为
  `6f46744013a260dbe98238e05b7c45d6d3455967d6039405a365c74eee0adf6b`。
- 已有 `task10_mix_nns/mixing_metadata.json` 真实文件校验通过；无 metadata 的
  `task8_noop_full` 仍能按 legacy 路径评分。
- 首次冒烟测试暴露单条 product 被训练目录误推断为 aug20；收紧推断来源后重测通过。

结论：本任务只增加输出审计和评分前校验，没有改动模型 forward、候选生成、分支筛选、
seed 构造或聚合排名。相同 seed 的实际输出逐字节一致。今后的完整采样会自动留下可追溯
配置，能直接阻止 `n_runs/beam_size`、augmentation、旧文件或预测内容混用。

Commit：`f03bb61 Add auditable sampling metadata`

## 9. 任务 13：胜出方案完整验证与性能收口

状态：`[x] 已完成；保留效率默认与高覆盖实验模式两档配置`

只有任务 10 或 11 出现明确正向结果后执行。

### 13.0 预注册验证方案（运行前固定）

为避免在已经用于开发和筛选的 tiny 前 50 个反应上继续选择方案，本阶段使用
`src/tgt-test-mini.txt` 中紧随其后的 reaction 50–249：输入 product 行区间为
`[1000, 5000)`，共 200 个原始反应、4000 条 aug20 输入。mini 文件已核对为完整 test
文件前 20028 行的逐字节前缀；本实验区间与 tiny 不重叠。

运行前固定以下配置，不根据 holdout target 再调 K/M/bonus 或聚合权重：

| 名称 | K/M/R | score mode | child policy | bonus | 用途 |
|---|---:|---|---|---:|---|
| NNN | 3/2/3 | full_probability | stochastic_noop | 0.5 | 当前主方法 |
| LL | 3/2/2 | legacy_triggered_reverse | stochastic | 0.5 | 激进探索 |
| NNN+LL | 5 个输出 | 两来源拼接 | 两来源拼接 | — | 完整实验流程 |

固定评分：

- NNN 单独使用历史默认 `legacy_best_rank` 报告主方法 Top-k 和 Oracle；
- LL 单独报告历史默认指标，用于解释探索质量和 invalid 成本；
- NNN+LL 同时报告 `legacy_best_rank` 与预先选定的 `frequency_first`；不得只报告较好者；
- `n_best=5`，同时记录 Oracle、覆盖未进 Top-3、每 run invalid/duplicate/overlap；
- tiny 的结果仅保留为开发集记录，holdout 200 的结果作为是否保留组合方案的依据。

成本控制：按 tiny 实测线性外推，NNN 与 LL 顺序完整采样约 20 分钟；不直接运行约
1001 个反应的整个 mini（预计约 1.5 小时），除非 200 反应先给出稳定正向证据。通过
新增只读区间参数选择原文件行，不复制、不改写数据集；区间起点和长度必须按 augmentation
完整对齐并写入 metadata，评分 target offset 必须与之交叉校验。

接口准备已完成：

- `sample_retro.py` 新增 `--start_product/--max_products`，默认仍读取全部输入；显式区间
  越界或不满足 augN 完整块时拒绝运行，seed product index 加上源文件 offset。
- sampling metadata 记录源文件总行数和 selection 的起止 product 行。
- `score_#global#.py` 新增 `--target_offset`，targets 和 detailed sources 使用同一反应
  偏移；若 sampling metadata 的 selection 与 offset 不符则在 RDKit 前拒绝。
- `mix_retro_runs.py` 新增重复参数 `--source_beam_size LABEL:SIZE`，可严格组合 R=3 与
  R=2 等不同来源，不再错误要求原始 prediction 行数完全相同。
- CPU 接口测试 `30 passed`；排除已知旧 beam 测试后项目回归
  `144 passed, 7 warnings`。20 条 aug20 输入的 GPU 选择/评分冒烟通过，R3+R2 混合为
  beam=5 的真实 CLI 冒烟也通过。

Commit：`475be15 Add aligned holdout sampling support`

正式 holdout 运行前补充 CUDA `max_memory_allocated/max_memory_reserved` 到 sampling
metadata；只在一次完整采样的开始/结束同步，不在 Euler 步内插入同步点，因此不会污染
组件耗时或明显拖慢正式运行。Commit：`ca3d90f Record peak CUDA sampling memory`。

### 13.1 正确性

- 全部现有测试通过；
- 新增评分布局、seed、per-run policy 和元数据测试；
- 相同 seed/配置逐行可复现；
- batch size 改变不影响对应 product/run 的采样随机流；
- 未启用新功能时，历史默认输出逐字节一致或给出可解释差异。

### 13.2 准确率

固定报告：

- Top-1/2/3；
- Oracle-any；
- 覆盖但未进 Top-3；
- 各 rank invalid；
- 每 run 命中率、重复率和两两 overlap；
- 当前推荐配置、Euler 和历史恢复版本。

tiny benchmark 中一个反应对应 2 个百分点，单次 2pp 变化只能视为初步信号。最终
结论应在更大且预先固定的评估集上复核。

预注册 holdout 200 结果（reaction 50–249，未参与前序调参）：

| 候选池/聚合 | Top-1/2/3 | Top-4/5 | Oracle | 覆盖未进 Top-3 |
|---|---:|---:|---:|---:|
| NNN / legacy | 52.0/69.0/74.0 | 77.0/79.0 | 84.0 | 10.0 |
| LL / legacy | 51.0/66.5/73.5 | 77.5/78.5 | 86.0 | 12.5 |
| NNN+LL / legacy | 52.5/70.0/75.5 | 79.0/80.5 | 91.0 | 15.5 |
| NNN+LL / frequency-first | 54.0/71.5/79.5 | 83.0/84.5 | 91.0 | 11.5 |

NNN 的三个 run invalid 为 5.150/5.125/4.775%，真实唯一候选均值 9.825；LL 的两个
run invalid 均为 12.675%，真实唯一候选均值 11.840。组合后真实唯一候选均值 18.340，
NNN 与 LL run 的 micro Jaccard 仅 19.4–20.4%，确实形成互补而不是复制。

Oracle 逐反应分解为：NNN 独有 10、LL 独有 14、共同覆盖 158、两者都未覆盖 18，联合
覆盖 182/200。相对 NNN，frequency-first Top-1/2/3 分别净增 4/5/11 个反应：

| rank | 新增命中 | 丢失命中 | 净增 | exact McNemar p |
|---|---:|---:|---:|---:|
| Top-1 | 5 | 1 | 4 | 0.218750 |
| Top-2 | 10 | 5 | 5 | 0.301758 |
| Top-3 | 13 | 2 | 11 | 0.007385 |

结论：独立 holdout 支持组合流程提高 Top-3 和 Oracle；Top-1/2 虽为正向点估计，但当前
样本量下不能称为确定提升。NNN 自身在 holdout 为 52/69/74，Top-2/3 相对 Top-1 的
增量恢复为 +17/+22pp，因此 tiny 上“Top-2/3 上升慢”不是稳定的普遍现象。

### 13.3 性能

- 在 RTX 3090、TF32 `high` 下记录端到端 wall time；
- 分开记录模型 forward、候选生成/编辑、合并/排序和评分耗时；
- 记录峰值显存；
- 相同输入至少进行一次 warm-up 后再比较短任务；
- 不使用首次编译时间远大于完整采样收益的 `torch.compile` 路线；
- 不修改共享训练模型以迁就 BF16，除非未来单独扩大任务范围。

holdout 200 正式采样实测：

| 方法 | 输出行数 | wall time | peak CUDA allocated | peak CUDA reserved |
|---|---:|---:|---:|---:|
| NNN（R=3） | 12000 | 474.916s | 1,951,848,448 B | 24,750,587,904 B |
| LL（R=2） | 8000 | 462.018s | 1,478,682,112 B | 24,893,194,240 B |

完整 NNN+LL 顺序采样成本为 936.934 秒，约为 NNN 单独的 1.973×。NNN R3 与 LL R2
的正式 wall time 接近，但二者同时改变了方法语义，不能错误归因为“R 不缩放”。

新增 opt-in `--euler_beam_profile`：仅用于短任务，在阶段边界同步 CUDA 并累计分项耗时；
默认关闭，正式采样不增加逐步同步。5 个反应、batch=64 的完整 100 步结果：

| 方法 | R | wall | forward+rate | 父分支评估 | 子候选评估 |
|---|---:|---:|---:|---:|---:|
| NNN | 2 | 8.971s | 6.319s | 37,534 | 75,068 |
| NNN | 3 | 12.690s | 8.960s | 56,687 | 113,374 |
| LL | 2 | 13.181s | 9.238s | 56,091 | 112,182 |
| LL | 3 | 18.272s | 12.931s | 84,024 | 168,048 |

同一方法内 R2→R3 的父/子评估接近 1.5×，run 数确实缩放。LL R2 之所以与 NNN R3
同样慢，是因为激进状态保留更多活跃分支、产生更长序列；其父分支评估数已经接近 NNN
R3。NNN R3 分项占比为 forward+rate 70.8%、分支准备 10.9%、编辑 7.0%、proposal
5.3%、merge/prune 5.1%、step score 0.9%。因此 `_step_log_p` 已不是值得优先优化的瓶颈，
下一轮性能研究应针对模型 forward/动态长度 padding，而不是继续微调 CPU step score。

batch size 32/64/128 的 5 反应输出逐字节一致。NNN wall 为 12.80/12.69/13.54s，LL
为 13.43/13.18/13.87s；64 略优，增大 batch 的统一 padding 抵消批次数减少，故保留 64，
不继续扫描。

### 13.4 本任务完成记录

实际修改：

- 增加 augmentation 对齐且保持全局 seed index 的只读 holdout 区间接口；评分 target
  offset 与 metadata selection 交叉校验。
- run 混合工具支持不同来源 beam size，严格生成 NNN R3 + LL R2 五输出候选池。
- sampling metadata 增加 CUDA peak allocated/reserved。
- Euler-Beam 增加默认关闭的阶段 profiling；只在显式短 profile 中同步，不改变正常路径。

测试与实验：

- 排除仓库既有 `tests/sampling/test_beam.py` 后：`145 passed, 7 warnings`；该旧文件单独
  仍为完全相同的 `17 failed, 18 passed`，失败来自非 Euler-Beam 的旧
  `EditCandidate.log_u_real` 接口和 controlled-model 长度假设，本任务未修改对应代码。
- profile 开/关输出完全一致；batch 32/64/128 的相同 product/run 输出逐字节一致。
- 正式 holdout 两份输出分别为 12000/8000 行，metadata、selection、SHA-256 和评分布局
  均通过自动校验；混合文件为 20000 行、beam=5。
- NNN SHA-256：`18b2fa7ace6b33dc81649f757852706e5efccd131959fe1f606920c254a0e8b3`；
  LL SHA-256：`502e1406d764373afe14113af05375b89a1c25c7939cd0e481ad44aaa35a8ef4`。

结论：当前推荐分为两档。K3/M2/R3 full-probability/no-op、bonus 0.5、TF32 high 继续作为
效率默认；它在独立 holdout 为 52/69/74，证明 Top-2/3 增长慢不是稳定问题。需要更高
Top-3/coverage 且接受约 1.97× 采样成本时，使用 NNN+LL 五输出和 frequency-first；其
holdout 为 54/71.5/79.5、Oracle 91，Top-3 配对净增 11/200 且 p=0.0074。由于 LL 是
未校准的激进启发式且 Top-1/2 证据仍弱，不将组合流程静默设为默认，也不宣称模型本身
准确率提高。

Commits：`475be15`（holdout 接口）、`ca3d90f`（CUDA memory）、`e91e5ea`（stage
profiling）、`ce2bad8`（holdout 结果；本文档收口见后续 Git log）。

## 10. 任务 14：Validation 基准与 NNN Forward 效率

状态：`[x] 已完成；建立 validation-200 基线并完成三条 forward 优化短筛`

### 14.0 固定范围与目的

任务 14 只沿效率默认 NNN 主线推进：

```text
K=3, M=2, R=3, n_steps=100, seed=42
score_mode=full_probability
changed_state_bonus=0.5
child_policy=stochastic_noop
matmul_precision=high
batch_size=64
```

激进 LL 与 NNN+LL 高覆盖模式保持冻结，只作为后续离线参考，不参与本任务参数选择。
本任务不修改训练代码、checkpoint、数据集或 test 结果。

数据协议：

- 快速代码回归仍可使用 tiny，但不能据其 target 选择方案；
- 方法筛选改用 validation 的 reaction 0–199，即 `src-val.txt` product 行 `[0, 4000)`；
- 如果局部优化在 validation-200 上同时保持逐行输出和明显加速，再进入不重叠的
  validation 确认区间；
- 完整 5007 reaction test 只在方法和参数全部冻结后运行一次。

### 14.1 首个实验：不改代码的 NNN 基线

固定命令：

```bash
python scripts/sample_retro.py \
    --checkpoint checkpoint_step600000.pt \
    --products_file "datasets/USPTO_50K_PtoR_aug20_#global#/val/src-val.txt" \
    --start_product 0 --max_products 4000 \
    --sampler euler_beam \
    --n_branches 3 --n_children 2 --n_runs 3 \
    --n_steps 100 --batch_size 64 --device cuda --seed 42 \
    --euler_beam_score_mode full_probability \
    --euler_beam_changed_state_bonus 0.5 \
    --euler_beam_matmul_precision high \
    --euler_beam_child_policy stochastic_noop \
    --output_dir results/task14_val200_nnn_baseline/

python 'scripts/score_#global#.py' \
    --predictions results/task14_val200_nnn_baseline/predictions.txt \
    --targets "datasets/USPTO_50K_PtoR_aug20_#global#/val/tgt-val.txt" \
    --augmentation 20 --beam_size 3 --n_best 5 \
    --length 200 --target_offset 0 --process_number 16 \
    --aggregation_mode legacy_best_rank --diagnostics \
    --diagnostics_json results/task14_val200_nnn_baseline/diagnostics.json
```

基线完成结果：

| 指标 | validation reaction 0–199 |
|---|---:|
| Top-1/2/3 | 62.5/79.0/86.0% |
| Top-4/5 | 87.0/88.0% |
| Oracle-any | 93.0%（186/200） |
| 覆盖但未进 Top-3 | 7.0%（14/200） |
| invalid rank 1/2/3 | 5.850/5.675/5.250% |
| 平均真实唯一候选 | 9.820 |
| wall time | 487.303s |
| peak CUDA allocated | 1,600,589,824 B |
| peak CUDA reserved | 24,559,747,072 B |

输出严格为 12000 行，SHA-256 为
`795b150d7f1f7be59b1b939fb213cbb78142b412e06d7cdd85f6e240881af591`；sampling
metadata 记录 Git commit `ab185d0` 且 `dirty=false`。这组 validation 指标作为任务 14
唯一的正式优化基线，后续不使用 test target 选择方案。

### 14.2 性能诊断和修改门槛

基于任务 13 的 profile，优先检查模型 forward 和动态长度 padding，不继续优化只占不到
1% 的 `_step_log_p`。在修改采样代码前，先用 opt-in profile 记录：

- active parent/child 数；
- 实际 non-PAD token 数与 padded token 数；
- 每步最大长度和 padding 比例；
- forward、分支准备、proposal、编辑与 merge/prune 时间。

任何优化必须满足：

1. 默认不开启时不改变现有路径；
2. 相同 seed、product/run、不同 batch 划分时预测逐字节一致；
3. CPU 单元测试和 Euler-Beam 回归通过；
4. 短 profile 至少有明确的 padding/forward 证据；
5. validation-200 Top-k、Oracle 和逐行预测不变；
6. validation-200 wall time 至少降低 10%，否则不保留复杂实现。

首轮 padding profile 使用 validation 前 5 个反应、100 条 augmentation 输入、batch=64、
完整 100 步，共覆盖 69,909 次 active parent evaluation：

| 诊断 | 数值 |
|---|---:|
| 实际 / padded token | 3,517,796 / 4,918,873 |
| token padding 浪费 | 28.48% |
| 实际 / padded attention 长度平方代理 | 197,074,542 / 361,907,321 |
| attention padding 浪费 | 45.55% |
| 最大 active length | 103 |
| forward+rate 占分项时间 | 68.64% |

按长度直方图离线估计，固定 bucket width 8/16 可分别将当前 attention 长度平方代理降低
37.45%/25.43%；但它们会增加每步模型 forward 次数，真实收益必须通过 opt-in 短运行验证，
不能把理论 FLOP 降低直接当作 wall-time 加速。

长度分桶短筛结果（相同 validation 5 反应、完整 100 步）：

| 路径 | wall time | forward time | forward calls | peak allocated | 输出 |
|---|---:|---:|---:|---:|---|
| 单 forward 基线 | 14.117s | 9.614s | 200 | 1,087.6 MB | 基准 SHA |
| bucket width 8 | 16.397s | 11.538s | 929 | 582.2 MB | 逐字节一致 |
| bucket width 16 | 15.588s | 10.664s | 699 | 708.7 MB | 第 67 行起有差异 |

width 8 虽降低峰值 allocated 并保持输出，但 wall time 回归 16.2%；width 16 回归 10.4%
且不满足逐行正确性门槛。原因是每步 1 次 forward 增加到约 3.5–4.6 次，调用开销超过
padding FLOP 节省。按照预注册门槛，不运行 validation-200，不保留分桶实现；仅在
profile 中保留 `model_forward_calls` 计数作为诊断字段。

随后评估了“只按 product 初始长度重排外层 batch、输出时恢复原顺序”的低风险候选。
离线静态长度计算表明，它在 validation-200 上理论上可将初始 batch 的 padded attention
平方代理降低 43.22%，且不会增加每步 forward 次数。为避免把静态估计误当成实际收益，
先以默认关闭的隔离开关完成 validation 前 5 个反应短筛，再决定是否保留。

短筛仍使用相同的 100 条 augmented 输入和完整 100 步：动态 padded token 仅降低
4.05%，动态 padded attention 代理仅降低 6.58%；forward 从 9.614s 降到 9.530s
（仅 0.86%），总 wall 从 14.117s 增至 14.265s（回归 1.05%）。300 行中有 1 行发生
变化，首个差异在第 169 行；这说明 batch shape 引起的浮点差异已传播到随机离散决策。
该候选同时未通过逐行正确性与 10% 加速门槛，因此不运行 validation-200，代码和测试
开关已完整回退，仅保留本实验记录。

第三条候选检查 PyTorch attention forward。只读微基准使用首个 validation batch 的
代表性 `(576, 53)` 输入：复用 Q/K/V 的同一个 LayerNorm 输出可使单次 forward 从
约 55.4–57.7ms 降到 53.2ms，三个模型输出逐元素一致，但幅度不足以支持 10% 端到端
目标；进一步设置 `need_weights=False` 可降到 47.2ms，但模型输出最大绝对差约
0.006–0.010。

仍将后一方案做成默认关闭、训练默认路径不变的临时推理开关，并完成相同 val5 全流程
短筛。结果 forward 从 9.614s 降到 9.304s（3.22%），总 wall 从 14.117s 增至
14.718s（回归 4.26%）；300 行中同样只有第 169 行发生变化。微基准收益没有转化为
动态采样收益，且未满足逐行门槛，因此实现和测试已完整回退，不运行 validation-200。

### 14.3 本任务完成记录

实际修改：保留 opt-in profile 中的动态 token/attention padding 统计和
`model_forward_calls`；默认 profile 关闭，正式采样路径不增加同步。内层逐步长度分桶、
外层初始长度排序、efficient attention 三个候选均只做隔离短筛，失败后完整删除实现，
没有修改训练默认路径、checkpoint、数据集、历史结果或 NNN 方法语义。

测试与实验：validation reaction 0–199 的固定 NNN 基线为 Top-1/2/3
62.5/79.0/86.0%，Oracle 93.0%，wall 487.303s；padding profile 证明 attention padding
代理浪费 45.55%。三条候选分别为：width-8/16 内层分桶 wall 回归 16.2%/10.4%；
外层长度排序 wall 回归 1.05% 且 1/300 行变化；efficient attention wall 回归 4.26%
且 1/300 行变化。保留诊断后的项目回归为 `145 passed, 7 warnings`（排除仓库既有且
未修改的 `tests/sampling/test_beam.py`）。

结论：当前没有候选同时满足逐行正确性和 validation-200 预期加速门槛，因此不制造
“优化成功”的结论，也不重复运行没有胜出候选的 validation-200。继续保留 RTX 3090
上的 TF32 `high`、batch=64、K3/M2/R3 NNN 作为效率默认。进一步明显提速需要单独决定
是否允许极少量 seeded 输出变化，或扩大范围到模型 attention/训练实现；二者都超出本
任务“默认路径不变且逐行一致”的边界。

Commits：`1dfe8ae`（padding 诊断）、`06d8bbd`（拒绝内层分桶）、`c75b805`
（拒绝外层长度排序；本文档收口见后续 Git log）。

## 15. 任务 15：独立搜索岛与全局大分支池

状态：`[x] 已完成；seed42/43 均确认全局池更快但 Top-k/Oracle 较弱`

> 历史说明：本节记录当时只返回 Top-N 分支的实验接口。任务 18 已移除 `n_return`，
> 当前实现固定输出每个 run 的全部 K 个槽位；本节参数只用于解释历史结果，不能作为
> 当前命令模板。

### 15.0 研究问题与固定对照

本任务检验当前收益究竟来自 9 个总分支，还是来自 3 个互不剪枝的独立 run。只比较：

| 方案 | 搜索池 | 每步最大父/子候选 | 每 augmentation 输出 |
|---|---|---:|---:|
| 当前 NNN | `R=3 × K=3`，三个独立池各返回 Top-1 | 9 / 18 | 3 |
| 全局池 | `R=1 × K=9`，统一池返回 Top-3 | 9 / 18 | 3 |

两者固定 `M=2`、100 步、full-probability、stochastic-noop、bonus 0.5、TF32 high、
batch=64、seed=42。全局池复用当前 R3K3 的九条 `(virtual run, branch)` 初始随机流，
避免把 seed 更换混入“是否跨 run 竞争”的比较。除搜索池边界和最终返回 Top-3 外不改变
评分、模型、checkpoint 或数据。

### 15.1 实现与正确性门槛

- 新增默认 `1` 的 Euler-Beam Top-N 返回参数；默认输出和历史 metadata 不变；
- 新增显式初始 seed group 参数，仅用于把 R3K3 九条流放入 R1K9 全局池；
- 返回顺序保持 product-major、run-major、branch-rank-minor，固定输出 3 行；
- 全局池最终返回按现有状态质量和 changed-state bonus 排名的前三个状态；不足三个时
  必须采用确定性固定长度行为并记录，不得静默改变文件布局；
- 单元测试覆盖默认兼容、Top-N 顺序/数量、seed 等价、非法参数和 metadata；
- validation 前 5 个反应只检查运行、输出布局、显存和时间，不据其 target 调参。

实现后短测结果：默认 R3K3 的 300 行 SHA-256 与任务 14 基线完全相同
（`55c59b5b...`），证明 `n_return=1` 默认兼容。R1K9 同样输出 300 行，100 个输入均有
至少 3 个最终状态，return shortfall 为 0，且每组三条原始字符串均不同。R1K9 父/子
评估为 55,879/111,758，较 R3K3 的 69,909/139,818 降低 20.1%；profile wall 从
14.380s 降到 11.586s（19.4%），forward 从 9.839s 降到 8.099s。原因是跨原 run 的
相同状态可全局合并，后续无需继续评估三份重复分支。

仅作冒烟解释的 5 反应评分中，R3K3 Top-1/3/Oracle 为 60/80/100，R1K9 为
20/60/80；样本过小且该结果不用于调参或取消预注册 val200。

### 15.2 固定 validation 实验

实现通过短测后，只运行一个预注册配置：validation reaction 0–199、product 行
`[0, 4000)` 的 R1K9 Top-3。直接与任务 14 已存在的同区间 R3K3 基线比较，不重复运行
基线。评分固定 `augmentation=20`、`beam_size=3`、`n_best=10`、
`legacy_best_rank`，同时输出 diagnostics。

报告 Top-1 至 Top-10、Oracle、覆盖未进 Top-3、invalid、真实唯一候选、最终返回重复/
shortfall、父子候选评估数、wall time 和显存。该实验是方法消融，输出变化属于预期；
不得沿用任务 14 的逐行一致门槛。只有全局池在准确率/覆盖与时间上形成清晰 Pareto 收益
时才考虑替换默认，否则保留 R3K3 默认并把全局池标为 opt-in 消融。

### 15.3 本任务完成记录

实际修改：

- `sample_euler_beam()` 新增默认 `n_return=1`，可按现有最终搜索排名返回每个输入的
  Top-N 分支；默认返回形状和结果不变。
- 新增可选 `initial_branch_seeds`，CLI 通过
  `--euler_beam_initial_seed_groups 3` 把当前三个 virtual run、每组 K3 的九条初始随机流
  原样映射到单个 K9 池；没有用一组新的连续 seed 混淆对照。
- `sample_retro.py` 新增 `--euler_beam_n_return` 和上述 seed-group 参数；输出数、写出
  顺序、metadata beam size 均使用 `n_runs × n_return`，因此 R1K9 Top-3 仍为每条
  augmentation 三行，可继续用 `beam_size=3` 评分。
- 最终不同状态不足 `n_return` 时确定性重复最佳状态，保持固定文件布局；metadata 的
  `euler_beam_stats` 记录 shortfall、父/子评估和步数。默认 R3K3 不启用全局池，历史
  方法、checkpoint、训练和评分规则均未改变。

测试与实验：相关测试 `34 passed`，排除仓库既有 `test_beam.py` 后完整回归
`149 passed, 7 warnings`。默认 R3K3 val5 重跑与旧文件 300 行逐字节一致；全局池短测
无 shortfall 并显示约 19.4% wall 收益。正式 validation 0–199 均为 12000 行，固定
legacy 聚合的 Top-1～10 如下：

| 方法 | Top-1 | Top-2 | Top-3 | Top-4 | Top-5 | Top-6 | Top-7 | Top-8 | Top-9 | Top-10 | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R3K3 独立池 | 62.5 | 79.0 | 86.0 | 87.0 | 88.0 | 89.0 | 89.5 | 90.0 | 90.5 | 91.0 | 93.0 |
| R1K9 全局池 | 63.0 | 75.0 | 79.0 | 81.5 | 84.0 | 86.0 | 86.5 | 86.5 | 88.0 | 88.0 | 91.0 |

逐反应配对：Top-1 全局池新增 9、丢失 8、净增 1（exact McNemar `p=1.0`）；Top-3
新增 1、丢失 15、净损失 14（`p=0.000519`）；Oracle 新增 1、丢失 5、净损失 4。
覆盖但未进 Top-3 从 7% 增至 12%。

返回质量揭示了机制差异：R3K3 三个独立 run 的 target-hit 为 89.0/87.5/89.0%，invalid
为 5.850/5.675/5.250%；全局池三个最终 rank 的 target-hit 为 85.5/75.5/67.5%，
invalid 为 2.325/18.875/29.625%。全局 Top-1 更干净，但第二、第三分支明显较弱。
全局池平均真实唯一候选反而由 9.820 增到 13.490，但平均有效候选由 56.645 降到
49.835，说明问题不是字符串多样性不足，而是尾部分支的化学有效性和目标相关性较差。

4000 条输入中 223 条出现返回 shortfall（5.575%），共补齐 262/12000 行；39 组三个
输出全相同。全局池输出 SHA-256 为
`45cbe029a428859afd5c961411491e283c0c44661ba7fd8f9e0baa2848127a68`。

| 方法 | wall | peak allocated | peak reserved |
|---|---:|---:|---:|
| R3K3 | 487.303s | 1,600,589,824 B | 24,559,747,072 B |
| R1K9 | 329.360s | 1,479,646,208 B | 24,194,842,624 B |

全局池 wall 降低 32.41%（1.480× speedup），allocated 降低 7.56%。它在正式运行中
评估 1,568,857 个父分支和 3,137,714 个 child；跨 virtual run 合并使实际活跃分支低于
最大 K9，这既是加速来源，也是尾部覆盖损失来源。

结论：用户提出的尝试值得且已得到明确答案。统一 K9 能显著加速并维持相近 Top-1，
但会显著损害 Top-2/3 和 Oracle；因此不替换当前 R3K3 默认。若任务只重视 latency 与
Top-1，可把它作为显式快速消融；当前项目同时关注 Top-k 覆盖时，三个独立搜索岛仍是
更成熟的主方法。下一步若研究全局池，应针对“合并后不补充分支”和 rank2/3 invalid，
而不是继续盲目增大 K。

Commits：`02a387c`（预注册）、`0d7263a`（全局池实现；本实验结果收口见后续 Git log）。

### 15.4 Seed 稳健性复验（运行前固定）

相同 seed42 完整重跑只会重复同一无状态随机流，不能估计随机波动。为检验 seed42 的
结论是否偶然，固定使用 `seed=43`，在同一 validation reaction 0–199 上分别重跑
R3K3 与 R1K9；其余参数、输出数、评分和目录布局全部保持不变。必须成对重跑，不能拿
seed43 全局池与 seed42 基线比较。

本复验不扫描更多 seed，也不调 K/M/bonus。固定报告 Top-1～10、Oracle、invalid、
shortfall 和 wall；主要检查 seed42 的两个方向是否复现：R1K9 wall 明显下降，以及
R1K9 Top-2/3 低于 R3K3。

seed43 结果：

| 方法 | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Oracle | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| R3K3 | 61.5 | 78.0 | 82.5 | 88.0 | 90.0 | 92.5 | 484.468s |
| R1K9 | 58.0 | 74.0 | 82.0 | 84.0 | 87.0 | 89.5 | 324.007s |

全局池 wall 再次降低 33.12%（1.495×）；父分支评估由 2,430,702 降到 1,565,518。
Top-1/2/3 分别净损失 7/8/1 个反应，Oracle 净损失 6 个。R1K9 rank1/2/3 invalid 为
2.650/19.775/29.800%，与 seed42 的“第一名更有效、第二三名明显变差”一致；230/4000
条输入出现 shortfall，共补齐 276 行。

seed42 与 seed43 的损失幅度不同：Top-3 分别下降 7.0pp 与 0.5pp，说明单 seed 的具体
百分点确实有随机波动；但两个 seed 中 R1K9 的 Top-2、Oracle 和 Top-5～10 都更低，
约 32–33% 加速以及 rank2/3 高 invalid 均稳定复现。两 seed 简单平均的 R3K3 对 R1K9
为 Top-1 `62.0/60.5`、Top-2 `78.5/74.5`、Top-3 `84.25/80.5`、Oracle
`92.75/90.25`。因此 seed43 加强而非推翻原结论，不继续扫描 seed。

## 16. 后续实验固定报告格式

状态：`[x] 已按用户要求固定为后续所有正式采样实验的报告规范`

从任务 15 之后，每个正式实验的评分命令统一使用 `--n_best 10 --diagnostics`，并通过
`--diagnostics_json` 保存机器可读结果。报告至少包含：

- Top-1～Top-10 accuracy；
- 每个原始输出 rank 的 invalid rate 和 sorted invalid；当 `beam_size < 10` 时，
  Top-4～10 没有对应的原始 rank，invalid 必须显示 N/A，不能伪造；
- Oracle-any、覆盖但未进入 Top-3、target augmentation support；
- 每个输出通道的 target-hit、invalid、duplicate，以及通道间 Jaccard overlap；
- 平均 valid candidate、真实 unique candidate、Top-1～10 rank availability；
- sampling wall time、峰值 CUDA allocated/reserved、父/子评估数、final branch shortfall；
- checkpoint/input/output SHA-256、seed、K/M/R、score mode、bonus、policy、
  precision、batch size 和 Git commit。

固定评分模板：

```bash
python 'scripts/score_#global#.py' \
    --predictions RESULTS_DIR/predictions.txt \
    --targets TARGET_FILE \
    --augmentation 20 --beam_size OUTPUTS_PER_AUGMENTATION \
    --n_best 10 --length REACTION_COUNT --target_offset REACTION_OFFSET \
    --process_number 16 --aggregation_mode legacy_best_rank \
    --diagnostics --diagnostics_json RESULTS_DIR/diagnostics.json
```

`beam_size` 始终等于每条 augmentation 的实际输出数（Euler-Beam 当前为
`n_runs × n_branches`），而 `n_best=10` 表示跨 augmentation 聚合后报告到 Top-10；二者
不能混淆。若实验研究的是其他 aggregation mode，必须同时报告固定 legacy 默认，不能
只呈现较好的聚合结果。

## 17. 任务 16：统一评估入口与完整轨迹可视化

状态：`[x] 已完成`

本任务不改训练、数据、checkpoint、历史实验结果或 Euler-Beam 的搜索排序。目标是消除
评估命令中重复且容易不一致的参数，并修复诊断可视化，使方法行为更容易审计。

### 17.1 `eval.py` 的实现边界

- 新增一个命令入口，先调用现有 `sample_retro.py` 生成预测，再调用
  `score_#global#.py` 评分；采样和评分的核心实现仍各自只有一份，不复制算法。
- checkpoint、products、targets、output directory、augmentation 和选择区间只在统一
  命令中声明一次。评分所需的 `beam_size`、`length` 和 `target_offset` 优先从采样
  metadata 与实际选择区间推导，避免手工填错。
- 默认报告 Top-1～10、diagnostics 并保存 JSON；默认不得覆盖已有 predictions，只有
  显式指定 overwrite 时才允许覆盖。
- 子进程隔离不是主要性能瓶颈：采样完成后 GPU 模型无需在 RDKit CPU 评分阶段常驻，且
  继续复用两份已验证 CLI 比大范围重构更安全。提供 dry-run/只评分能力用于检查命令和
  复用已有预测。

### 17.2 可视化修复范围

- `visualize_trajectory.py` 不再只选择一个 exact/best sample；对每个反应显示所有请求的
  Euler 路径，包括没有发生编辑的路径。
- 每条路径按 Product、每次实际编辑后的完整 token 序列、Target 的顺序展示；每个中间
  序列末尾显示具体原子操作（插入、删除、替换及位置），多个同时编辑不得被隐藏。
- 事件记录增加编辑后的 `x_next`，只在诊断 recording 开启时保存，不改变普通采样结果、
  seed 或性能路径。
- 检查并修复 `visualize_first_step.py` 的重复导航、checkpoint/vocab/origin-mask 兼容问题，
  同时明确区分“真正随机采样”与“逐位置 argmax 诊断”，避免把后者错误标为 actual。
- 当前 `visualize_trajectory.py --n_branches` 调用的是尚未实现事件记录的 Euler-Beam 接口，
  需要先明确报错或补齐记录，不能继续保留会在运行时错误解包的伪支持。

### 17.3 验证与完成标准

1. 单元测试覆盖统一参数推导、覆盖保护、Top-10 diagnostics 命令，以及多路径/多操作的
   HTML 输出；
2. Euler 普通 sampling 在 recording 关闭时结果不变，并验证 event 的 pre/post state；
3. 用 checkpoint 做一个很短的 CUDA/HTML 冒烟，不运行无意义的完整数据实验；
4. 更新本节实际修改、测试结果和结论，创建范围明确的 commit 并推送当前任务分支。

### 17.4 完成记录

统一评估入口：新增 `scripts/eval.py`，以参数列表（非 shell 拼接）依次复用现有采样和
评分 CLI。checkpoint、products、targets、output directory、augmentation 和选择区间
只声明一次；评分的 `beam_size`、`length`、`target_offset` 从采样 metadata 推导。
默认 `n_best=10` 并保存 diagnostics JSON，默认拒绝覆盖，另支持 `--dry_run`、
`--score_only` 和显式 `--overwrite`。没有复制或改写采样/评分算法，子进程隔离还能在
RDKit CPU 评分前释放模型进程的 CUDA 上下文。实现 commit：`04c4895`。

轨迹诊断：Euler event 在仅开启 recording 时增加准确的 post-edit `x_next`；相同 seed
下开启/关闭 recording 的最终采样结果测试一致。trajectory HTML 不再按 target 选中
“最佳一条”，而是保留全部 `n_samples` 路径，包括零编辑路径。每条路径显示 Product、
每个 edit event 后的完整 token 序列、`-->` 后的所有具体插入/删除/替换动作和 Target；
同一步多个位置、同位置多个动作不再被 `elif` 隐藏。实际 CUDA 冒烟的两条路径分别显示
2/3 个 edit event，HTML 中确认出现 `+C`、`+)`、`n→[nH]` 等操作。

旧 `visualize_trajectory --n_branches` 实际调用了未实现 recording 的 Euler-Beam 并错误
解包 Tensor，本来会在运行时崩溃。现在会在加载 checkpoint 前明确报错，避免把“最终
最佳 branch”伪装成完整分支树；若以后需要完整剪枝树，必须在 Euler-Beam 中记录 node
ID、parent ID、候选是否合并/保留等信息。`visualize_first_step.py` 同时修复 vocab size
fallback、DDP `module.` 前缀和 origin-embedding 兼容，并把并未随机采样的逐位置最高
rate 行从 `ACTUAL` 更名为 `MODEL ARGMAX (NOT SAMPLED)`。实现 commit：`e57bf19`。

统一入口端到端冒烟使用 1 个反应的 20 条 augmentation、2 个采样步：成功生成 60 行
预测，自动验证为 `1 × 20 × beam_size 3`，并输出 Top-1～10、per-run invalid、Oracle/
coverage 和 diagnostics JSON。该短步结果只验证接口，不作为准确率实验。排除仓库已知的
`tests/sampling/test_beam.py` 后完整回归为 `161 passed, 8 warnings`。

### 17.5 Euler-Beam child 独立性审计

同一 parent 的 M 个 child 使用不同、可复现的无状态随机流：child 0 延续 parent seed，
child 1 及以后由 `(parent_seed, step, child_index)` 混合 seed；每个动作随机数再由
`(seed, step, position, stream)` 生成，插入触发、删除/替换触发、类型选择、插入 token、
替换 token 使用五个不同 stream。因此它们是给定 parent 状态和模型分布下的独立式
伪随机 proposal，不是共享同一随机数，也不受向量化 batch 顺序影响；但它们是确定性
hash 生成，不应称为数学或密码学意义的严格独立随机变量。

不同 child 完全可能得到相同状态：都采到 no-op、碰巧采到相同编辑，或模型分布高度
集中时都会发生；`stochastic_noop` 的指定时间步还会把 child 1 强制为 no-op，而随机
child 0 也可能恰好 no-op。不同 parent 也可能编辑后汇聚到相同 token 序列。当前
`full_probability, M>1` 会按最终 token key 合并这些候选，以 log-sum-exp 合并质量，再按
`log_mass + changed_state_bonus`、原始 `log_mass` 和确定性 seed tie-break 排序保留 Top-K。
所以“随机流不同”不等于“结果一定不同”；相同 child 是正常的离散采样碰撞，也是分支
数有时缩减的直接原因。

## 18. 任务 17：轨迹全局总览与发散—合流检测

状态：`[x] 已完成`

在不改变采样器结果的前提下，扩展 trajectory HTML：

1. 在任何逐 event 详情之前，先按 example 列出全部 `n_samples` 路径的 Product、每次
   edit 后状态、具体动作、Final 和 Target，作为不经过 target 选择的全局总览；
2. 对同一 example 内的每对路径，在相同 Euler step 上重建 post-step token state。只有
   路径先出现不同状态、随后恢复成完全相同 token 序列，才标记为 reconvergence，并记录
   divergence step、reconvergence step 和汇合状态；
3. 单独统计不同 example 在同一步出现完全相同 token state 的 cross-example collision，
   不把它与同一输入的随机路径合流混为一谈；
4. 检测采用精确 token 序列，不利用 target，也不把不同 SMILES 写法的化学等价性误报为
   序列相同。后续如需 canonical-SMILES 合流，应作为独立分析维度；
5. 用构造轨迹单测覆盖“持续分离”“先分离后合流”“再次分离后再次合流”和跨 example
   碰撞，再做短 CUDA HTML 冒烟，更新本节结果并提交、推送。

完成记录：trajectory HTML 现在先输出 `All Examples — Complete Path Overview`，逐个
example 展示全部路径的状态阶梯、具体编辑、Final/Target 和详情跳转；所有总览结束后才
进入 `Per-path Event Analysis` 的 oracle/model 表。导航栏也先跳到 overview，不再直接
跳过全局比较。

检测器从每条路径的初始状态和 event `x_next` 重建每个 0-based Euler step 结束后的精确
token state。对同一 example 的每对路径维护 divergence 区间，只有“不相等后重新相等”
才生成 episode，因此初始共同 Product 或一直相同不会误报；同一对路径可以记录多次
发散—合流。不同 example 的同一步相同状态单列为 cross-example collision，持续不变的
同一碰撞只报告首次，避免日志膨胀。

构造测试验证 Path 1/2 在 step `0→2` 和 `3→4` 的两次发散—合流均被识别，未合流的第三
条路径不误报；跨 example 在 step 1 的相同状态只进入 collision 报告。CUDA 冒烟使用
2 个 example × 3 paths × 6 steps，HTML 中 6 条 overview 路径全部位于 6 条详细路径前；
该真实短样本检测到 0 次合流，作为正常零结果保留，不据此推断完整数据分布。排除已知
`test_beam.py` 后回归为 `163 passed, 8 warnings`。实现 commit：`9e99bc0`。

## 19. 任务 18：全部分支输出、简洁轨迹与 R1/R3 公平比较

状态：`[x] 接口、测试和 validation 对比完成；test-mini 等待冻结速度/准确率目标`

### 19.1 接口和布局修改

- `visualize_trajectory.py` 新增 `--table`，默认 True；False 时仍生成完整的 all-example
  path overview、发散—合流和 cross-example collision，但不追加逐 event 的大表格。
- Euler-Beam 用户接口移除 `n_return`：每个 run 固定输出 K 个最终槽位，单条
  augmentation 的 `output_beam_size=R×K`。最终唯一状态不足 K 时仍用最高排名状态补齐
  固定布局，并在 metadata 中报告 final branch shortfall；canonical 去重会移除补齐重复。
- 多 run 输出采用 branch-rank-major、run-minor：先输出所有 run 的 rank1，再输出所有
  run 的 rank2，以此类推。这样 R3K3 新增 rank2/3 时，历史三个 run winner 仍占局部
  rank1～3，不因 run-major 布局被挤到 rank4/7。
- `eval.py` 自动推导 beam size，不再暴露 `euler_beam_n_return`。评分器的聚合算法不改，
  只接收新的固定 beam size 和 metadata layout。

### 19.2 预注册验证顺序

先用短测试验证形状、顺序、metadata、shortfall 和默认 Top-10 diagnostics；通过后在同一
validation reaction 0–199 上公平比较：

| 配置 | 搜索宽度 | 输出/augmentation | 目的 |
|---|---:|---:|---|
| R3K3 | 3 个隔离池 × K3 | 9 | 独立搜索岛基线 |
| R1K9 | 1 个全局池 × K9 | 9 | 相同初始随机流总量的全局竞争池 |

两者固定 M2、100 steps、seed42、相同 grouped initial seeds、bonus0.5、noop、TF32 high；
评分固定 beam9、Top-10、legacy 和 diagnostics。比较 Top-1～10、Oracle、有效/唯一候选、
各输出通道 invalid/duplicate、shortfall、wall 和父/子评估数。不能根据 test-mini target
选择 R/K。

现有 `src-test-mini.txt` 为 20028 行，不是完整 augmentation block（`20028 % 20 = 8`）。
若进入 test-mini，必须从完整 `src-test.txt` 选择前 20020 行，即 1001 个完整反应，target
使用完整 test 文件并由 metadata 推导 `length=1001,target_offset=0`。test-mini 只用于
冻结方案的规模化报告，不用于继续调参；完整 5007 test 留作最终一次评估。

### 19.3 完成记录

实现：

- `visualize_trajectory.py` 新增布尔参数 `--table`（默认 True）。`--table False` 仅省略
  `Per-path Event Analysis` 及逐 event oracle/model 表，完整路径总览、同 example
  发散—合流和跨 example collision 均保留。
- `sample_euler_beam()` 移除 `n_return`，始终返回每个 run 的 K 个最终槽位；状态不足 K
  时确定性复制最高排名状态并记录 shortfall，保证固定文件布局。
- `sample_retro.py` 和 `eval.py` 移除 `--euler_beam_n_return`，Euler-Beam 输出数和评分
  beam size 统一为 `R*K`。多 run 文件采用 branch-rank-major、run-minor，历史各 run
  winner 仍处于最前面的 R 个局部 rank。
- 评分聚合公式未改；`eval.py` 从 metadata 自动读取 beam size、反应数和 target offset。
- 评分诊断将原来容易误解的 `Run N` 控制台标签改为 `Input rank N`；兼容性所需的 JSON
  字段保留原名，指标和聚合数值均不改变。

验证：

- 新增/更新形状、顺序、shortfall、metadata、CLI 和 `--table` 测试；排除与本任务无关
  的既知 `tests/sampling/test_beam.py` 后为 `172 passed, 8 warnings`。
- CUDA 轨迹冒烟确认 `--table False` 的 HTML 含完整 overview，且不含详细分析标题或
  event table。
- 20 个 product 行、R3K3、2 steps 的 `eval.py` 冒烟写出 `20*9=180` 行，metadata
  自动驱动 beam9/Top-10 评分；每个 9 行块的前三行与旧 winner-only R3K3 完全一致。

validation reaction 0–199（4000 条 aug20 输入）的公平对比：

| 指标 | R3K3：三个隔离池 | R1K9：一个全局池 | 差值（R1-R3） |
|---|---:|---:|---:|
| 采样时间 | 482.18 s | 324.39 s | -32.7% |
| Top-1 | 64.5% | 62.5% | -2.0 pp |
| Top-2 | 80.5% | 75.5% | -5.0 pp |
| Top-3 | 85.0% | 79.5% | -5.5 pp |
| Top-5 | 88.0% | 83.5% | -4.5 pp |
| Top-10 | 90.5% | 89.0% | -1.5 pp |
| Oracle-any | 94.5% | 92.5% | -2.0 pp |
| 平均真实唯一候选 | 22.57 | 26.28 | +3.71 |
| 最终槽位不足占比 | 15.15% | 28.17% | +13.02 pp |

结论：R1K9 的单个大矩阵和跨分支合并使其快约三分之一，并产生更多不同候选，但所有
分支在同一 beam 中全局竞争，高分模式会共同淘汰其它搜索方向；R3K3 的三个 run 是隔离
搜索岛，因而并非冗余随机重复。当前证据形成明确速度—准确率取舍：R3K3 是准确率模式，
R1K9 是低延迟模式。不能使用 test-mini target 决定二者；应先冻结目标，再在从完整 test
文件截取的 1001 个完整反应上只做确认性报告。

实验目录：`results/task18_val200_r3k3_all/`、`results/task18_val200_r1k9_all/`。
实现 commit：`bdbd75a`；实验结论与文档收口 commit 见本次之后的 Git log。

## 20. 任务 19：test-mini 上的 R1K10 children 对照

状态：`[x] 新 mini 已构建；M2/M3 完整采样、评分和配对分析已完成`

用户指定在新建的完整 test-mini 上比较两个 R1K10 配置：

| 配置 | R | K | M | 每 augmentation 输出 |
|---|---:|---:|---:|---:|
| R1K10M2 | 1 | 10 | 2 | 10 |
| R1K10M3 | 1 | 10 | 3 | 10 |

为保证只改变 M，两组统一使用 `child_policy=stochastic`。现有 `stochastic_noop` 仅支持
M2，不能用于本对照。K10 不能被此前的 3 个 virtual seed group 整除，因此两组均不传
`euler_beam_initial_seed_groups`，使用相同 `seed=42` 和相同默认 K10 初始 branch seed
布局；本实验可以严格比较 M2/M3，但不能把它写成相对先前 grouped-seed K9 的单因素
结论。

数据固定从完整 `test/src-test.txt` 和 `test/tgt-test.txt` 各截取前 20020 行，写入新的
`src-test-mini-1001.txt`、`tgt-test-mini-1001.txt`，不覆盖既有 20028 行 mini。必须核验：

- source/target 均为 20020 行；
- 均可整除 augmentation 20；
- 新文件分别与完整文件的前 20020 行逐字节一致；
- 评分 metadata 推导为 1001 reactions、beam10、target offset 0。

两组固定 100 steps、batch64、CUDA、TF32 high、full probability、bonus0.5、legacy
aggregation、Top-1～10 diagnostics。分别写入独立结果目录，不覆盖历史结果。报告 wall、
显存、父/子评估、shortfall、Top-1～10、各 input-rank invalid、Oracle 和真实唯一候选。
这是用户指定的并列 test 报告；完成后不根据 test target 继续选择 bonus、policy 或其它
超参数。

### 20.1 数据构建结果

新文件均为 20020 行，即 1001 个完整 aug20 反应，且通过与完整 test 文件前 20020 行的
逐字节 `cmp`：

| 文件 | SHA-256 |
|---|---|
| `src-test-mini-1001.txt` | `6650a97484fbd64e9fa0992050c4a7a6c1221f8007417318e2c3b0395397d27c` |
| `tgt-test-mini-1001.txt` | `54c79c45fa6c44a0d2ab033bd8450a908d0062a4c22ef68741b50e7a760096e9` |

原 20028 行 mini 未改。数据目录由 `.gitignore` 排除，因此 Git 只记录文件名、构建方法和
哈希，不上传数据内容。

### 20.2 Top-k 与覆盖结果

两组 metadata 均验证为 `1001 reactions × 20 augmentations × beam10 = 200200` 行：
checkpoint SHA-256 为 `dad34b36c95f87674049a7907f11149a034fb841f271a8aa7d60e0f4d45d906b`；
M2/M3 prediction SHA-256 分别为 `1a60ea7d73a582e90a0faed29275127f9fb034180121b553177edefa798dba2f`
和 `81a49a7312858d00ca408e6007892cf879854a4c403172db76957baa08f1d337`。

| 配置 | Top-1 | Top-2 | Top-3 | Top-4 | Top-5 | Top-6 | Top-7 | Top-8 | Top-9 | Top-10 | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M2 | 52.547 | 67.033 | 71.628 | 75.225 | 76.523 | 78.022 | 79.221 | 80.420 | 80.819 | 81.618 | 87.213 |
| M3 | 51.249 | 63.636 | 68.831 | 72.228 | 74.326 | 75.524 | 77.123 | 78.222 | 78.721 | 79.421 | 83.916 |
| M3-M2 (pp) | -1.298 | -3.397 | -2.797 | -2.997 | -2.197 | -2.498 | -2.098 | -2.198 | -2.098 | -2.197 | -3.297 |

逐反应配对的 M2-only / M3-only 命中数：Top-1 为 `51/38`（双侧 exact McNemar
`p=0.203`），Top-2 为 `63/29`（`p=0.000509`），Top-3 为 `50/22`
（`p=0.00129`），Top-10 为 `44/22`（`p=0.00921`），Oracle 为 `46/13`
（`p=1.92e-5`）。Top-1 单项差异不显著，但 Top-2～10 和 Oracle 一致支持 M2。

### 20.3 效率、有效性与多样性

| 指标 | M2 | M3 | M3 相对 M2 |
|---|---:|---:|---:|
| sampling wall | 1687.14 s | 2065.51 s | +22.43% |
| parent evaluations | 8,234,855 | 9,634,193 | +17.0% |
| child evaluations | 16,469,710 | 28,902,579 | +75.5% |
| final-slot shortfall | 29.009% | 14.763% | -14.247 pp |
| mean valid candidates/reaction | 149.132 | 139.648 | -9.484 |
| mean true unique/reaction | 30.301 | 31.887 | +1.586 |
| peak CUDA allocated | 1,970,680,320 B | 1,929,543,168 B | -2.1% |
| peak CUDA reserved | 24,488,443,904 B | 24,652,021,760 B | +0.7% |

M2 input-rank 1～10 invalid 为 `3.207/17.363/25.844/32.008/35.659/36.693/
34.695/29.745/23.312/15.814%`；M3 为 `2.488/15.405/24.361/30.400/36.653/
39.970/41.873/42.582/38.666/29.361%`。M3 的前四个槽位 invalid 略低，但后半槽位明显
更高，最终有效候选反而更少。

结论：M3 确实减少槽位补齐并增加约 1.59 个真实唯一候选，但新增探索主要落在低价值或
非法区域，Oracle 下降 3.30pp，Top-2～10 全面下降，同时慢 22.43%。在本次固定 K10、
纯 stochastic 对照中，不保留 M3 为推荐配置，也不根据 test target 继续调整其
bonus/policy。就准确率与 wall-time 主目标而言 M2 支配 M3；M3 仅在 shortfall
和真实唯一数两个探索诊断上较好。结果目录为 `results/task19_testmini_r1k10_m2/` 和
`results/task19_testmini_r1k10_m3/`。

## 21. 任务 20：轨迹口径、训练审计与前沿方法研究

状态：`[x] 已完成代码口径修正、设计文档更新、训练审计和论文路线分析；未修改训练`

本任务回应六项研究问题，不运行新的长采样，也不根据 test-mini target 调参：

1. `stochastic_noop` 在 tiny 的固定 K3/M2/R3 消融中由 58/64/66 提升为 60/64/70，
   wall 基本不变；该结论只适用于已测配置，policy 仍是无 proposal 校正的启发式。
2. trajectory HTML 仍展示所有路径，但同 example 合流、跨 example state collision 和
   控制台计数只使用最终 canonical SMILES 命中 Target 的路径。0 条命中明确跳过，1 条
   命中因不足两条路径也跳过；原始 path 编号保持不变。
3. `euler_beam_design.md` 已从早期单后继/CPU RNG/旧排序原型改写为当前 K×M、无状态
   GPU RNG、full-probability state mass、R/K/M 和全部分支输出设计，并记录限制。
4. 对本地 Edit Flows 论文逐项核对训练代码。主 Bregman loss 与 Eq.23 一致；确认 Noam
   首次 update 顺序、fallback alignment、zip 静默截断、resume、无 validation/seed 等
   问题。现 checkpoint 权重有限，预对齐数据结构完整，潜在 fallback/data 缺陷没有证据
   影响该 checkpoint。全训练集目标编辑为 insert/sub/delete =
   91.201/7.875/0.924%，并识别出缺少不可编辑 product conditioning 的建模限制。
5. R10/K1/M1 在算法结构和分布语义上退化为十条独立 Euler 风格轨迹，但 RNG、seed
   映射、categorical 实现和 bookkeeping 不同，不会与 `euler --n_samples 10` 逐字节相同。
6. 论文路线形成 `frontier_methods_research.md`：近期优先 Q sharpening 与特殊 token
   诊断；独立 `euler_smc` 是最值得的新 sampler 方向；显式 product conditioning 是下一
   训练主方向；guidance/reverse/localized edits 和 RetroAgent 分别列明前提与成本。

验证：`tests/test_visualization_scripts.py` 为 `14 passed`。新增测试覆盖 mismatch path
过滤、原始 path index 映射、0-match 跳过和 cross-example 仅匹配路径语义。训练和 PDF
仅做只读审计，`PDF/` 保持用户未跟踪文件，不纳入 commit。

产出文档：

- `new_docs/euler_beam_design.md`
- `new_docs/training_code_audit.md`
- `new_docs/frontier_methods_research.md`

## 22. 任务 21：旧 Euler 性能复核与无损分支准备优化

状态：`[x] 已完成同机公平复核、无损局部优化和反例实验；保留 batch64`

本任务只分析采样性能，不修改训练、checkpoint、数据和历史结果。用户提供的
`old_sample_retro.py` 实际仍导入当前 `edit_flows.sampling.euler`；提供的 `old_euler.py`
与当前普通 Euler 主采样路径基本一致，当前文件新增的 event 记录以及 scheduler correction
下限不影响本实验使用的 cubic/cubic 正常路径。旧评分器与当前评分器都需要 RDKit
canonicalization；当前评分器额外做布局、哈希和覆盖诊断，但 test-mini 的 28.12 分钟已由
sampling metadata 单独计时，因此评分器不能解释这段采样耗时。

### 22.1 同机、同输入、同输出数复核

固定 RTX 3090、tiny 的 1000 条 augmentation 输入、100 steps、seed42，并都输出每条输入
10 个结果：

| 实现 | 配置 | sampling wall | 输出行数 |
|---|---|---:|---:|
| 旧 Euler 入口 | `n_samples=10,batch16` | 189.59 s | 10,000 |
| 当前 `sample_retro.py` + 当前 `euler.py` | `n_samples=10,batch16` | 185.77 s | 10,000 |
| 修改前 Euler-Beam | `R1K10M2,batch64,high` | 99.16 s | 10,000 |
| 本任务优化后 Euler-Beam | 同上 | 89.97 s | 10,000 |

当前入口直接调用当前 `edit_flows/sampling/euler.py` 的复核与旧入口相差仅 2.0%，证明上一轮
没有因误调用 `old_euler.py` 而夸大时间。因此当前 Euler-Beam 在公平小样本上并没有比旧
Euler 慢：修改前已快 47.7%，优化后相对当前 Euler 快 51.6%（约 2.06 倍）。原因是 M 只
扩展轻量 child proposal，不重复 Transformer；状态合并后平均活跃父分支也低于 10，而
普通 Euler 始终维持 10 条完整轨迹。

test-mini 有 20020 条 augmentation 输入，是 tiny 的 20.02 倍；虽然概念上是 1001 个
原始反应，模型仍必须分别处理每个反应的 20 个增强表示。M2 的 1687.14 秒采样产生
200200 行预测，因此“mini 约半小时”首先是数据规模结果。按本次无损提速估算，同配置
mini 约 25.5 分钟，完整 100140 行 test 约 2.1～2.5 小时；这只是外推，不替代完整实测。

当前机器和当前 `euler.py` 无法复现“旧 Euler 在完整 100140 行上 30 多分钟”：tiny
实测线性外推约 5.17 小时。旧脚本又不记录输入哈希、实际选择范围、采样 wall 和有效 seed，
所以现有证据不足以唯一还原历史条件；可能差异包括实际输入范围、steps/samples/batch、
计时范围、运行环境，或当时尚未修正的样本/seed 语义。不能据此认定当前指标被刻意压低。

### 22.2 保留的无损优化

原实现每个 Euler step 先在 GPU 分配 PAD tensor，再用 Python 循环逐分支执行小 slice
赋值，同时逐个写入时间。正常 Beam 分支来自同一个批量编辑结果，tensor width 一致；
现在走 uniform-width fast path，用一次 `torch.cat` 构造状态 batch，并一次构造时间 tensor。
仍保留原 padding fallback，以兼容不规则或外部构造的状态。

100 行 profiling 中，branch preparation 从 1.456 秒降到 0.106 秒（-92.7%），总 wall 从
11.263 秒降到 10.144 秒（-9.93%）。完整 tiny 中总 wall 从 99.157 秒降到 89.972 秒
（-9.26%）；修改前后 10000 行预测逐行完全一致，parent/child evaluation、shortfall 统计
也完全一致。profile 新增 `uniform_width_fast_path_steps`，短测试确认所有正常 step 均命中。

### 22.3 已否决的性能尝试

- 裁剪保留分支末尾 PAD：100 行 attention padding proxy 只下降 1.28%，wall 反而从
  11.263 增至 12.785 秒；TF32 因矩阵 shape 改变还造成 9/1000 行预测变化，已完整回退。
- `batch_size=128`：100 行短测看似比 64 快 5.9%，但完整 tiny 为 94.553 秒，反而比
  batch64 慢 5.1%；峰值 allocated 从 1.50 增至 2.80 GiB，并有 1392/10000 行预测变化。
  因此保留已验证的 batch64，不根据过短 profile 修改运行默认。

### 22.4 当前 Euler 的 batch 与 TF32 复核

普通 Euler 的脚本默认 batch32；用户此前对比命令明确传入 batch16。由于 `_make_batch()`
会把每个 product 复制 `n_samples=10` 次，模型实际 batch 分别约为 160/320/640/1280。
当前 `euler.py` 在完整 tiny、batch16 上实测 185.77 秒；同一入口在完整 test 文件的前
200 行进行 batch 扫描：

| matmul | product batch | sampling wall | peak allocated | peak reserved |
|---|---:|---:|---:|---:|
| FP32 `highest` | 16 | 36.32 s | 0.46 GiB | 4.06 GiB |
| FP32 `highest` | 32 | 38.32 s | 0.73 GiB | 8.85 GiB |
| FP32 `highest` | 64 | 36.97 s | 1.27 GiB | 18.31 GiB |
| FP32 `highest` | 128 | 36.43 s | 2.26 GiB | 22.98 GiB |
| TF32 override | 16 | 27.00 s | 0.46 GiB | 4.32 GiB |
| TF32 override | 64 | 25.66 s | 1.21 GiB | 17.90 GiB |
| TF32 override | 128 | 25.08 s | 2.26 GiB | 22.82 GiB |

严格 FP32 下扩大 batch 没有实际吞吐收益，只增加显存。TF32 可把短测加速约 25～31%，
但 batch128 相对 batch64 只再快 2.3%，显存余量很小；即便按最快短测外推，完整 100140
行仍约 3.5 小时，不能解释历史 30～40 分钟。

当前代码只为 Euler-Beam 设置 `matmul_precision=high`，普通 Euler 保持 PyTorch 2.7 默认
`highest`。更重要的是，普通 Euler metadata 显示 `seed_applied_to_sampler=False`：命令行
`--seed` 当前没有应用到普通 Euler。以上独立进程的预测不能用于严格逐行或准确率配对。
在决定给普通 Euler 启用 TF32 前，应先单独修复其 seed、再在同 seed 上做 FP32/TF32
准确率复核；当前不因短吞吐实验修改默认精度或 batch。Euler-Beam 的推荐 batch64 不变。

### 22.5 当前 R3K3 Euler-Beam 的完整 tiny batch 复核

此前任务 13 已在 100 行 profiling 上比较当前准确率配置 R3K3M2、`stochastic_noop`、
TF32 high 的 batch32/64/128：时间分别为 12.806/12.690/13.542 秒，三组输出 SHA 完全
一致，因此选择 batch64。但 R1K10 的实验表明过短样本可能误判 batch，故本任务又使用
当前代码在完整 tiny 1000 行上重跑三档：

| product batch | sampling wall | peak allocated | peak reserved | 相对 batch64 不同行 |
|---:|---:|---:|---:|---:|
| 32 | 109.783 s | 0.80 GiB | 12.44 GiB | 23/9000 |
| 64 | 110.119 s | 1.39 GiB | 23.15 GiB | 0 |
| 128 | 117.128 s | 2.56 GiB | 22.77 GiB | 13/9000 |

batch32 与64只差 0.336 秒（0.3%），小于正常 wall 波动，不能宣称32更快；batch128 则
稳定慢约 6.4%。三组评分的 Top-1～10 均为 `60/64/70/72/72/76/76/80/82/82%`，
Oracle 均为 90%。少量逐行差异来自 TF32 在不同矩阵 shape 下的数值路径，没有改变该
tiny 的 Top-k。

结论：Euler-Beam 继续以 batch64 作为可复现的正式默认，因为 validation/test-mini
历史基线均使用64，且短 profile 中64也略快；batch32 是速度相当、显存 allocated 更低的
安全备选；batch128 被否决。没有证据支持继续用 test target 对48/80/96等相邻 batch
做细粒度寻优。

## 23. 任务 22：Euler-Beam 与采样入口效率再审计

状态：`[x] 已完成当前代码 profiling 和性价比分级；本任务不保留低收益代码改动`

使用当前准确率默认 R3K3M2、`stochastic_noop`、TF32 high、batch64，在 tiny 前100行、
完整100步上重新开启分阶段 profile。profile 会在阶段边界同步CUDA，只用于占比诊断，
不能与正式 wall 直接混用：

| 阶段 | 时间 | 占 sampling wall |
|---|---:|---:|
| model forward + rate | 8.963 s | 74.40% |
| apply edits + token keys | 1.032 s | 8.57% |
| merge + prune | 0.944 s | 7.84% |
| child proposal | 0.710 s | 5.89% |
| step scoring | 0.129 s | 1.07% |
| branch batch preparation | 0.126 s | 1.05% |
| finalize output | 0.010 s | 0.08% |
| 入口/其它未归因 | 0.133 s | 1.10% |

本次 fast branch preparation 在200/200个step均命中；此前已占10.9%的准备阶段现在只占
1.05%，说明上一项优化已把该热点基本消除。动态 token padding浪费21.20%，attention
长度平方代理浪费36.95%；但任务14已证明内层长度分桶、外层初始长度排序和
`need_weights=False` attention短筛均没有转化成wall收益，不能重复同一路线。

### 23.1 低风险候选及收益上限

`apply_ins_del_operations()` 中除分配长度所需的 `.item()` 外，还有三处GPU tensor
`.any()`被Python判断，形成同步。使用约700×70的代表张量做不改仓库的等价微基准，删除
空mask分支后输出完全一致，单次从1.411降至1.311 ms（编辑helper快7.1%）；由于编辑阶段
只占8.57%，端到端预期仅约0.6%。该候选只适合以后与其它改动捆绑，不单独修改公共
`ops.py`。

其它低风险外围候选同样受Amdahl上限约束：

- M>1正式排名不使用`path_log_p`，条件跳过`_step_log_p_batch()`的绝对上限只有1.07%，
  还会损失现有轨迹诊断语义；
- 用GPU `arange/repeat_interleave`替换Python parent index列表、批量生成输出字符串、
  `writelines`、pinned-memory传输等，只覆盖proposal或入口的一小部分，预计各自低于1%；
- `sample_retro.py`的batch构建、CPU解码、写文件和其它入口开销合计仅约1.1%，不是当前
  mini/full采样时间的来源；现有batch64也已经过32/64/128完整tiny复核。

因此不为了微小benchmark数字堆积复杂路径，本任务不保留上述代码修改。

### 23.2 后续候选的性价比分级

| 候选 | 现实端到端收益预期 | 改动/风险 | 结论 |
|---|---:|---|---|
| 去掉编辑中的`.any()`同步 | 约0.5～1% | 低 | 只与其它改动捆绑 |
| 简化Python对象、连续slice合并 | 约2～4% | 中 | 可做下一轮低风险原型 |
| GPU状态key/组内去重，只回传保留索引 | 约4～8% | 高；需保证无hash碰撞、tie顺序 | Beam文件内最值得的隔离研究 |
| child维广播，减少K×M的rates/probs复制 | 约2～4% | 中高；显存和数值路径会变化 | 排在GPU去重之后 |
| inference-only SDPA/attention kernel | 约10～25%潜力 | 高；修改共享模型且可能改变预测 | 只有允许配对准确率复核时研究 |
| 将100 steps降为75/50 | 约25/50% | 方法与准确率改变，不是代码优化 | 单列validation研究，不能静默修改 |

纯粹消灭全部非forward阶段的理论加速上限也只有约1.34倍；要得到稳定的两位数提升，必须
处理占74.4%的模型forward。当前模型为10层、hidden256、FFN2048，且过去的full-model
`torch.compile`、BF16、padding bucket和attention开关均已有失败证据。若保持“逐行预测
不变、只改`euler_beam.py/sample_retro.py`”的严格边界，当前已经接近低风险收益上限；
下一项建议是先做默认关闭的GPU状态合并原型，而不是重写采样入口。

### 23.3 R1K10M2 的独立效率审计

当前低延迟实验配置为R1K10M2、纯`stochastic`、bonus0.5、TF32 high、batch64。使用当前
fast branch preparation在tiny前100行、完整100步的profile如下，并与同数据当前R3K3M2
profile并列：

| 阶段 | R1K10M2 | R3K3M2 |
|---|---:|---:|
| sampling wall（profile） | 10.144 s | 12.047 s |
| model forward + rate | 74.02% | 74.40% |
| apply edits + token keys | 8.41% | 8.57% |
| merge + prune | 7.41% | 7.84% |
| child proposal | 6.59% | 5.89% |
| step score | 1.26% | 1.07% |
| branch preparation | 1.04% | 1.05% |
| 入口/其它未归因 | 1.16% | 1.10% |
| parent / child evaluations | 58,365 / 116,730 | 69,191 / 138,382 |
| token / attention padding浪费 | 19.70 / 34.32% | 21.20 / 36.95% |

完整tiny的正常wall为R1K10 89.972秒、R3K3 110.119秒，R1快18.30%。R1的单一全局池会
合并原本属于不同run的相同状态，所以实际父/子评估更少、序列略短；它的热点结构并未
改变，forward仍占74%。因此上一节的性价比排序同样适用于R1K10：`sample_retro.py`不是
瓶颈，GPU key/合并是Beam文件内最有潜力但风险较高的候选，模型forward才有两位数空间。

这组速度不能直接证明R1K10方法优于R3K3：R1每augmentation输出10行，R3输出9行；R1
使用纯stochastic，R3使用stochastic_noop；K10也没有复用R3的三个virtual seed group。
它是两个实际运行配置的吞吐对照，不是单因素方法实验。

### 23.4 参数成熟度与完整test准入门槛

历史上真正公平的搜索池实验是R1K9与R3K3，而不是R1K10与R3K3。两者固定总宽度9、输出9、
M2、100 steps、bonus0.5、stochastic_noop、TF32 high、batch64，并把R3的三个virtual
seed group原样放进R1全局池。数据为validation reaction 0～199，即`src-val.txt`前4000
条aug20输入：全部分支接口的seed42结果为R3/R1 Top-1 `64.5/62.5`、Top-2
`80.5/75.5`、Top-3 `85.0/79.5`、Top-10 `90.5/89.0`，wall `482.18/324.39`秒。
早期winner-only口径又用seed42/43成对复验，均确认R1快约32～33%，但R1的Top-2、Oracle
和尾部有效性更弱。现有证据支持二者形成Pareto取舍，而不是一个配置全面支配另一个。

参数不能称为全局最优，只能分级描述：

- R3K3M2：M2、TF32、batch64、noop和多run保护均有实验支持，是当前最成熟的准确率配置；
  bonus0.5和100 steps仍是局部支持，不是穷举最优。
- R1K9M2：同预算、两seed validation对照充分支持其“更快但Top-k较弱”的定位，是最公平
  的低延迟对照；也不能称为全局最优。
- R1K10M2：M2在test-mini-1001上优于M3，但为保证M2/M3单因素比较使用纯stochastic，
  bonus0.5沿用而非为K10重新选择，只有seed42，且test-mini target已经被查看。因此它是
  可用工程配置，不是已经在validation上选出的最优配置，不能继续用test-mini调参。

完整`src-test.txt`为100140行、5007个反应。技术上现在即可运行，但科学上应等配置冻结
后只做一次最终报告。建议准入流程：

1. 若还要做GPU状态合并等代码优化，只允许保留逐行输出一致的实现；否则先完成或放弃，
   避免全量后又改变推理代码。
2. 在未使用过的validation reaction 200～1200（product行`[4000,24020)`，1001个完整
   aug20反应）做一次确认。主比较预注册为公平的R3K3与R1K9；若必须报告R1K10，也必须
   事先固定policy，不能看结果后切换。
3. confirmation只确认方向，不继续调bonus/K/M；随后锁定commit、checkpoint、seed42、
   scoring和metadata。
4. 在完整test上顺序运行冻结的两个配置，报告Top-1～10、Oracle、invalid、unique、wall
   和配对反应差异。完整test决定泛化表现，不再反向修改参数。

按当前tiny吞吐粗略外推，R3K3完整test约3.06小时，R1K10约2.50小时；公平R1K9预计约
2.0～2.3小时。两个主配置加评分适合一次约5～6小时的隔离长跑。validation-1001确认约
需1小时。因此不需要等待新训练；完成这一次未见validation确认并冻结代码后，下一轮
即可安排完整src-test，最好作为过夜实验。

## 24. 任务 23：R1K10M2 专属validation参数选择

状态：`[x] 分阶段validation筛选完成；没有新候选晋级，保留现R1K10基线`

R3K3的bonus/no-op证据不能直接外推到单一K10全局池。本任务只用validation选择R1K10
参数，不再使用已经看过target的test-mini；不修改训练、checkpoint、模型或评分算法。

固定项：R1K10M2、100 steps、cubic、full probability、TF32 high、batch64、seed42、
legacy aggregation、Top-1～10 diagnostics。M2已在规模化M2/M3实验中同时取得更高Top-k、
Oracle和更低wall，故本任务不重复M3；K、R、steps、precision和batch也不加入网格，避免
把一次policy/bonus选择扩大为组合搜索。

### 24.1 分阶段筛选

筛选数据固定为validation reaction 200～399，即`src-val.txt/tgt-val.txt` product行
`[4000,8000)`的200个完整aug20反应，与历史reaction 0～199不重叠。

第一阶段只比较：

| 候选 | child policy | bonus |
|---|---|---:|
| 当前K10基线 | stochastic | 0.5 |
| R3启发式迁移 | stochastic_noop | 0.5 |

第二阶段只对第一阶段胜出policy补跑bonus0.0和1.0，与已有bonus0.5组成三点筛选；不扫描
0.1间隔，也不改变no-op时刻。若policy没有清晰胜者，则保留纯stochastic作为保守基线，
第二阶段仍只在该policy上检查bonus，不能用bonus补偿另一个policy后反复交叉搜索。

选择规则预先固定：候选必须Top-1不下降，并使Top-3、Top-10、Oracle至少两项改善且其余
不出现超过1pp的回归，才视为清晰晋级；多候选同时满足时依次比较Top-1、Top-3、Top-10、
Oracle、invalid和wall。互有小幅胜负时保留`stochastic, bonus0.5`，不从噪声中挑最高点。

### 24.2 独立确认与test门槛

筛选胜者必须与当前K10基线在不重叠的validation reaction 400～1400，即product行
`[8000,28020)`的1001个完整反应上成对确认。确认阶段不再改变参数；报告逐反应paired
hit、Top-1～10、Oracle、invalid、unique、shortfall、父/子评估和wall。只有方向保持且
达到上述无明显回归规则，才更新R1K10推荐配置；否则维持当前基线。

所有采样写入新的`results/task23_r1k10_*`目录，不覆盖历史结果。R1K10确认完成后，才能
与冻结的R3K3/R1K9一起决定完整test报告名单；完整test只评估，不再调参。

### 24.3 筛选结果

四组均验证为validation reaction 200～399、4000条aug20输入、40000行预测，target
offset为200；其它参数严格固定。Top-1～10、Oracle和效率如下：

| policy / bonus | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Oracle | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| stochastic / 0.0 | 45.0 | 63.5 | 67.0 | 70.5 | 76.5 | 84.0 | 293.03 s |
| stochastic / 0.5 | 47.0 | 63.5 | 67.5 | 72.5 | 78.0 | 84.0 | 286.20 s |
| stochastic / 1.0 | 48.0 | 64.5 | 68.0 | 72.0 | 77.5 | 84.0 | 292.14 s |
| stochastic_noop / 0.5 | 46.5 | 63.5 | 67.5 | 72.0 | 78.5 | 84.0 | 288.09 s |

完整Top-1～10分别为：

- stochastic/0.0：`45.0/63.5/67.0/69.0/70.5/72.0/75.5/76.5/76.5/76.5`；
- stochastic/0.5：`47.0/63.5/67.5/70.5/72.5/73.0/77.0/77.5/77.5/78.0`；
- stochastic/1.0：`48.0/64.5/68.0/70.5/72.0/73.5/76.0/77.0/77.5/77.5`；
- noop/0.5：`46.5/63.5/67.5/70.5/72.0/72.5/76.5/77.0/77.5/78.5`。

policy阶段中，noop相对纯stochastic在bonus0.5下Top-1 `-0.5pp`、Top-3/Oracle不变、
Top-10 `+0.5pp`，属于互换而非晋级。逐反应Top-1 noop-only/baseline-only为`1/2`，
Top-3为`1/1`，Top-10为`2/1`，Oracle为`1/1`，均没有稳定方向。因此R3的no-op启发式
没有证据迁移到K10。

bonus0.0被bonus0.5支配：Top-1/3/10分别低`2.0/0.5/1.5pp`，三项逐反应都没有任何
bonus0-only命中。bonus1.0相对0.5虽增加Top-1/2/3 `1.0/1.0/0.5pp`，但Top-5/7/8/10
分别回退`0.5/1.0/0.5/0.5pp`，Oracle不变；Top-1仅`2/0`个反应交换、Top-10为`0/1`，
不满足“Top-3/Top-10/Oracle至少两项改善”的预注册规则。

四组mean valid/true unique分别为`153.165/31.340`、`152.420/31.650`、
`151.430/31.730`、`151.915/31.500`；final-slot shortfall均约28.5～28.8%。更高bonus只
略增unique并降低valid，不能证明化学覆盖更好。wall差异不作为bonus收益，父/子评估均
约1.62M/3.24M。

结论：在本次预注册的R1K10M2局部搜索空间内，保留`child_policy=stochastic`、
`changed_state_bonus=0.5`。没有新候选晋级，故按提前停止规则不浪费约一小时在独立
1001反应区间重复“新参数确认”；原基线已经保留，不需要用confirmation重新批准自己。
该结论是validation支持的局部推荐，不声称K/M/steps的全局最优。结果目录为：

- `results/task23_r1k10_val200_stochastic_b00/`
- `results/task23_r1k10_val200_stochastic_b05/`
- `results/task23_r1k10_val200_stochastic_b10/`
- `results/task23_r1k10_val200_noop_b05/`

验证：Euler-Beam 与采样 metadata 定向测试为 `36 passed`；排除仓库中既知且与本任务
无关的 `tests/sampling/test_beam.py` 后，完整回归为 `176 passed, 8 warnings`。实现 commit：
`f921ee8`。用户的 `old_*.py`、`PDF/` 和既有 `visualize_trajectory.py` 本地修改均保持
未跟踪/未提交，不纳入本任务。

### 24.4 样本规模、低 Top-1 与同切片对照

用户质疑筛选样本过少以及47% Top-1与历史60%+不一致。这里首先统一计数口径：test-tiny
有1000条aug20输入，即50个独立反应；任务23使用4000条aug20输入，即200个独立反应，
独立样本数是tiny的4倍。因此它适合局部validation筛选，但200反应对1～3pp的小差异仍然
有限，不能替代最终完整test。

历史60%+不是同一个实验：当前可复核的主要来源包括test-tiny上的R3K3 Top-1 60.0%，
以及validation reaction 0～199上公平宽度实验的R3K3/R1K9 Top-1 64.5/62.5%。它们不能
直接与reaction 200～399上的R1K10 Top-1 47.0%横比，因为数据切片、K/R以及child policy
不同。作为同切片参考，补跑以下冻结配置，不扫描参数：R3K3M2、3 runs、stochastic_noop、
bonus0.5、100 steps、batch64、TF32 high、seed42、legacy aggregation。输入和target offset
与任务23完全相同，采样411.64秒，结果为：

| 同一 validation reaction 200～399 | Top-1 | Top-2 | Top-3 | Top-10 | Oracle |
|---|---:|---:|---:|---:|---:|
| R1K10M2 stochastic / bonus0.5 | 47.0 | 63.5 | 67.5 | 78.0 | 84.0 |
| R3K3M2 stochastic_noop / bonus0.5 | 50.5 | 63.0 | 68.5 | 80.5 | 88.0 |

逐反应的R3-only/R1-only命中数分别为Top-1 `11/4`、Top-2 `5/6`、Top-3 `6/4`、
Top-10 `7/2`、Oracle `11/3`。这说明该切片上R3覆盖整体更好，但200反应下3.5pp的Top-1
差异仍只应视为方向性证据，不升级为最终方法结论。

更明显的是数据切片差异：同为R3K3，validation reaction 0～199为
Top-1/2/3/10/Oracle `64.5/80.5/85.0/90.5/94.5`，reaction 200～399则为
`50.5/63.0/68.5/80.5/88.0`。两段不是相同反应，不能做逐反应配对，但足以证明不能把
前一切片的60%+当作后一切片的预期基线。47%主要来自更困难的切片组成，另有R1单全局池
相对R3独立池的覆盖损失；没有证据表明评分器刻意压低指标，或bonus0.5本身导致这次断崖。

本补充对照不改变任务23的选择：stochastic/bonus0.5仍是R1K10局部推荐，因为其它K10
候选没有满足预注册晋级条件。后续完整test比较必须冻结配置，并优先报告相同输入、相同
总宽度和相同输出数的R1K9与R3K3；R1K10可作为用户关心的快速配置另行报告。参考结果位于
`results/task23_r3k3_val200_reference/`，预测与metadata已通过评分脚本的布局、输入哈希和
target offset校验。

## 25. 任务 24：test-mini-1001 最终工程选型

状态：`[x] R3相对R1的选型完成；其默认地位随后被任务25的R9结果取代`

目标是在严格的test-mini-1001（20020条aug20输入、1001个完整反应）上，用两个方法各自
已经由validation支持的最佳参数做一次工程选型。该子集来自test且已有结果被查看，因此
只能用于后续工程路线定型，不能再表述为完全未见测试集；选定后不再根据mini target调整
K/M/R、policy、bonus、steps或聚合方式，完整test只作冻结评估。

固定候选：

| 候选 | K/M/R | child policy | bonus | 每个aug输出 |
|---|---|---|---:|---:|
| R3K3 accuracy | 3/2/3 | stochastic_noop | 0.5 | 9 |
| R1K10 speed | 10/2/1 | stochastic | 0.5 | 10 |

两者都使用checkpoint step600000、100 steps、cubic、full probability、batch64、CUDA、
seed42、TF32 high、legacy aggregation与Top-1～10 diagnostics。两者输出宽度9/10不同，
因此这是“各自最佳参数”的实际方法比较，不声称是控制总宽度的单因素消融；公平机制证据
仍以历史R3K3/R1K9实验为准。

选择规则在补跑R3前冻结：Top-1为主指标并检查逐反应配对命中；若Top-1没有清晰方向，
再依次看Top-3、Top-10和Oracle。若准确率互有胜负、没有稳定覆盖优势，则选择wall更低的
R1K10；若R3在主指标和覆盖指标上形成一致优势，则选择R3K3。invalid、unique、shortfall
和显存作为正确性/效率诊断，不用单一辅助指标推翻整体Top-k方向。

现有R1结果`results/task19_testmini_r1k10_m2/`的metadata已确认输入为严格
`src-test-mini-1001.txt`、20020行、SHA-256
`6650a97484fbd64e9fa0992050c4a7a6c1221f8007417318e2c3b0395397d27c`，配置与上表一致，
采样耗时1687.14秒，无需重复运行。R3写入新目录
`results/task24_testmini1001_r3k3_best/`，不会覆盖历史结果。

### 25.1 准确率与配对结果

R3 metadata验证为`1001 reactions × 20 augmentations × beam9 = 180180`行，输入哈希与
R1完全相同；prediction SHA-256为
`68f8ae53c62f91302bfc14f735e4c566b729d3c0a8d39efd66eddcf197e18a89`。两组Top-k为：

| 配置 | Top-1 | Top-2 | Top-3 | Top-4 | Top-5 | Top-6 | Top-7 | Top-8 | Top-9 | Top-10 | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1K10 | 52.547 | 67.033 | 71.628 | 75.225 | 76.523 | 78.022 | 79.221 | 80.420 | 80.819 | 81.618 | 87.213 |
| R3K3 | 55.145 | 68.332 | 74.026 | 77.223 | 78.821 | 80.619 | 81.618 | 82.817 | 83.516 | 84.515 | 89.311 |
| R3-R1 (pp) | +2.598 | +1.299 | +2.398 | +1.998 | +2.298 | +2.597 | +2.397 | +2.397 | +2.697 | +2.897 | +2.098 |

逐反应R3-only/R1-only为Top-1 `47/21`（双侧exact McNemar `p=0.00219`）、Top-2
`34/21`（`p=0.105`）、Top-3 `43/19`（`p=0.00316`）、Top-10 `53/24`
（`p=0.00126`）、Oracle `38/17`（`p=0.00646`）。除Top-2外，主报告节点均形成一致且
显著的R3优势；不是tiny上少数反应交换造成的偶然排序。

### 25.2 效率与最终选择

| 指标 | R1K10 | R3K3 | R3相对R1 |
|---|---:|---:|---:|
| sampling wall | 1687.14 s | 2071.54 s | +22.78% |
| input throughput | 11.87/s | 9.66/s | -18.55% |
| peak CUDA allocated | 1,970,680,320 B | 1,892,043,776 B | -3.99% |
| final-slot shortfall | 29.009% | 14.524% | -14.485 pp |
| mean valid / available slots | 149.132/200 (74.566%) | 143.043/180 (79.468%) | +4.902 pp |
| mean true unique/reaction | 30.301 | 24.031 | -6.270 |

R1因每个augmentation输出10条而非9条，绝对unique更多，但R3用更少输出取得更高Top-k和
Oracle，且有效槽位比例更高。R3慢384.40秒、约22.78%，显存占用没有增加。按照预先冻结
的规则，准确率主指标与覆盖指标的一致收益足以支付该时间开销，因此后续默认方法确定为：

```text
R3K3M2, n_runs=3, n_branches=3, n_children=2,
stochastic_noop, changed_state_bonus=0.5,
n_steps=100, batch_size=64, full_probability, TF32 high, seed=42
```

R1K10M2不再与R3并列纠结，只保留为明确的速度优先备选：当约23%的wall节省比2～3pp
Top-k更重要时使用。这里的R3默认结论是R1/R3二者比较阶段的决策；随后用户要求的最后
一个R9结构challenger在任务25满足预注册替代规则，因此当前默认以任务25为准。

## 26. 任务 25：R9K1 独立轨迹结构消融

状态：`[x] 完整mini-1001采样、评分和配对完成；准确率默认冻结为R9K1`

用户要求在默认方法冻结前补充R9K1。该实验固定总输出宽度9，并与R3K3共用M2、
stochastic_noop、bonus0.5、100 steps、full probability、batch64、TF32 high、seed42和
legacy aggregation，只把`R=3,K=3`改为`R=9,K=1`。因此它回答独立轨迹和run内分支
竞争/合并的结构差异；自然run seed布局也随R改变，这是实际R9方法的一部分。

R9K1M2不是普通Euler。每个run每步仍生成两个child并剪到一个，但不同父分支之间不发生
合并或竞争；除强制no-op步外，等质量的不同child主要通过确定性seed tie-break保留。
它更接近9条独立的Euler-Beam轨迹，而不是9粒子的共享beam。

完整实验前冻结选择规则：R9只有在Top-1提高，并且Top-3、Top-10、Oracle至少两项同步
提高且其它主指标无超过1pp回退时，才替代R3；否则保持R3。wall、invalid、unique和
shortfall用于解释取舍，不从test-mini继续调R9的bonus/policy/M。该实验是最后一个
test-mini结构challenger，完成后不追加其它R/K组合。

前100条aug20输入（5反应）smoke成功输出900行，metadata验证为`5 × 20 × beam9`，
K1/M2/noop路径无shape、seed或评分布局错误。smoke准确率因样本极少不参与方法选择；
结果目录为`results/task25_r9k1_smoke5/`。完整结果写入独立目录
`results/task25_testmini1001_r9k1/`。

### 26.1 完整结果

R9 metadata验证为`1001 reactions × 20 augmentations × beam9 = 180180`行，prediction
SHA-256为`8ab60adba5fc3fbe6e2a39dfa90e20668f45678ada09467a772b2fee392656bc`。
三个已冻结配置的统一结果为：

| 配置 | Top-1 | Top-2 | Top-3 | Top-4 | Top-5 | Top-6 | Top-7 | Top-8 | Top-9 | Top-10 | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1K10 | 52.547 | 67.033 | 71.628 | 75.225 | 76.523 | 78.022 | 79.221 | 80.420 | 80.819 | 81.618 | 87.213 |
| R3K3 | 55.145 | 68.332 | 74.026 | 77.223 | 78.821 | 80.619 | 81.618 | 82.817 | 83.516 | 84.515 | 89.311 |
| R9K1 | 57.043 | 71.528 | 78.122 | 81.319 | 82.617 | 83.816 | 84.815 | 85.115 | 85.814 | 86.114 | 91.808 |
| R9-R3 (pp) | +1.898 | +3.196 | +4.096 | +4.096 | +3.796 | +3.197 | +3.197 | +2.298 | +2.298 | +1.599 | +2.497 |

R9-only/R3-only逐反应命中为Top-1 `60/41`（双侧exact McNemar `p=0.0728`）、Top-2
`65/33`（`p=0.00160`）、Top-3 `69/28`（`p=3.80e-5`）、Top-5 `66/28`
（`p=0.000111`）、Top-10 `45/29`（`p=0.0805`）、Oracle `40/15`
（`p=0.00102`）。Top-1/10单点没有越过0.05，但没有回退；Top-2/3/5和Oracle均形成
显著一致优势，满足实验前冻结的替代规则。

### 26.2 效率与机制结论

| 指标 | R1K10 | R3K3 | R9K1 |
|---|---:|---:|---:|
| sampling wall | 1687.14 s | 2071.54 s | 3059.83 s |
| input throughput | 11.87/s | 9.66/s | 6.54/s |
| peak CUDA allocated | 1,970,680,320 B | 1,892,043,776 B | 2,110,356,992 B |
| final-slot shortfall | 29.009% | 14.524% | 0.000% |
| valid candidates / slots | 74.566% | 79.468% | 86.192% |
| mean true unique/reaction | 30.301 | 24.031 | 25.807 |

R9比R3慢47.71%，原因不是Transformer前向宽度增加：两者活跃轨迹总宽度都是9；而是R9
把每个product展开为9个独立sample容器，逐sample的Python状态合并/剪枝循环是R3的3倍。
另一方面，R9没有跨父分支的mass剪枝和最终槽位补齐，valid比例、真实unique、Top-k和
Oracle都优于R3。这是明确的机制证据：当前启发式mass/bonus竞争会过早丢掉一部分有用的
独立轨迹，增加run带来的独立性比run内beam合并更有价值。

按照预注册规则，后续最高准确率默认冻结为：

```text
R9K1M2, n_runs=9, n_branches=1, n_children=2,
stochastic_noop, changed_state_bonus=0.5,
n_steps=100, batch_size=64, full_probability, TF32 high, seed=42
```

R3K3保留为平衡速度配置（wall少32.3%，但Top-1/3/10低1.90/4.10/1.60pp），R1K10保留
为最快配置。后续正式准确率实验统一使用R9；仅在研究mass合并/beam剪枝本身时使用R3
作结构参照。本任务至此关闭test-mini上的R/K搜索，不再追加组合。

## 27. 下一阶段：推理低风险改进与训练基础设施

当前checkpoint不因训练审计而失效：主Bregman loss与论文Eq.23一致，权重有限，现有
预对齐数据完整且没有证据走到损坏的fallback。但下次重训前存在必须修复的问题：首次
Adam update错误使用默认`lr=1e-3`后才进入Noam；fallback alignment会把PAD当真实token；
src/tgt用`zip()`会静默截断；resume会重复一步；没有统一seed/RNG state；没有validation
loop、最佳checkpoint选择或早停。

因此不建议现在直接开新600k训练，也不建议立即大规模实现需要外部reward的Euler-SMC。
按成本和依赖关系，后续顺序冻结为：

1. 在R9默认上做BOS/特殊token非法事件诊断；只有证据充分才加入采样硬约束。
2. validation-only完成Q temperature最小消融，首轮只比较`1.0/0.9/0.8`，不同时扫描
   top-p/top-k；这是可复用当前checkpoint且额外成本近似为零的前沿推理改进。
3. 独立修复训练基础设施P0/P1并加scheduler、数据、resume和reproducibility测试；只跑
   synthetic及10k～30k pilot，不直接替换当前checkpoint。
4. 基础设施通过后，最高价值的新训练单变量是显式product conditioning；完整product
   作为不可编辑条件保留，再研究CFG。它比继续堆R/K更可能带来模型级提升。
5. Euler-SMC先只做synthetic mechanics与proposal/weight正确性；在没有独立forward
   consistency或feasibility reward前，不把启发式bonus包装成理论SMC。

也就是说，立即下一步仍在推理层做低风险、低成本的特殊token诊断和Q sharpening；同时
训练代码的基础设施修复是任何新checkpoint之前的硬门槛。完成这两层后，再进入显式条件
模型，而不是在当前Euler-Beam上继续增加分支组合。

## 28. 任务 26：R10K1 的 M1/M2 最终补充实验

状态：`[x] smoke、完整mini-1001采样、评分与逐反应配对均已完成；R9保持默认`

用户在R9K1定型后要求补充R10K1的M1/M2。两组使用同一严格test-mini-1001、R10/K1、
100 steps、batch64、CUDA、TF32 high、full probability、seed42、legacy aggregation，
每个augmentation都输出10条并统一报告Top-1～10与Oracle：

| 配置 | M | child policy | bonus | 语义 |
|---|---:|---|---:|---|
| R10K1M1 | 1 | stochastic | 0.0 | 每run每步一个后继，无child竞争 |
| R10K1M2 | 2 | stochastic_noop | 0.5 | 每run每步两个后继剪到一个 |

M1不能使用只支持M2的`stochastic_noop`；K1/M1没有候选剪枝竞争，changed-state bonus没有
实际作用，故显式设0而不记录无效的0.5。M1接近10条独立Euler轨迹，但随机实现、seed流
和Euler sampler并不完全相同，不能直接宣称与`euler --n_samples 10`严格等价。

选择规则在结果前冻结：先用逐反应Top-1/3/10/Oracle和wall比较M1/M2；R10胜者只有在
覆盖收益足以解释相对R9多出的第十条输出和wall时才替代R9，否则R9保持准确率默认。
本任务完成后关闭run数量搜索，不继续测试R11/R12或从test-mini调整bonus/policy。
结果分别写入`results/task26_testmini1001_r10k1_m1/`和
`results/task26_testmini1001_r10k1_m2/`，不覆盖历史实验。

两组5反应smoke均通过：各自输出`5 × 20 × 10 = 1000`行，metadata中的R/K/M、policy、
bonus、输出布局和评分反应数均正确。完整实验均输出`1001 × 20 × 10 = 200200`行，
无final-slot shortfall。M1 prediction SHA-256为
`20f6dad8a7899e4392215ca8eaa517fb11830cc0fc2b3aa9b0efcdce0d5e2eed`，M2为
`bd9ee6b35dd4e6d93963db969e95b2bc2ca1092e6862c0d96088708a89e882e1`。

### 28.1 准确率与配对结果

| 配置 | Top-1 | Top-2 | Top-3 | Top-4 | Top-5 | Top-6 | Top-7 | Top-8 | Top-9 | Top-10 | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R9K1M2 | 57.043 | 71.528 | 78.122 | 81.319 | 82.617 | 83.816 | 84.815 | 85.115 | 85.814 | 86.114 | 91.808 |
| R10K1M1 | 56.344 | 70.529 | 76.224 | 77.922 | 80.120 | 81.319 | 82.917 | 84.116 | 84.515 | 84.915 | 92.308 |
| R10K1M2 | 56.943 | 71.828 | 78.422 | 81.419 | 82.418 | 83.816 | 85.015 | 85.215 | 85.814 | 86.014 | 92.408 |
| R10M2-R9M2 (pp) | -0.100 | +0.300 | +0.300 | +0.100 | -0.199 | 0.000 | +0.200 | +0.100 | 0.000 | -0.100 | +0.600 |

M2相对M1的逐反应M2-only/M1-only为Top-1 `50/44`（双侧exact McNemar
`p=0.606`）、Top-3 `42/20`（`p=0.00715`）、Top-5 `41/18`（`p=0.00379`）、
Top-10 `41/30`（`p=0.235`）、Oracle `21/20`（`p=1.0`）。因此M2的中段Top-k优势有
配对证据，M1虽有更低invalid和相近Oracle，但不能作为准确率配置替代M2。

R10M2相对R9M2的逐反应R10-only/R9-only为Top-1 `1/2`、Top-2 `3/0`、Top-3
`3/0`、Top-5 `0/2`、Top-10 `0/1`、Oracle `6/0`。第10个run确实新增6个Oracle覆盖
（exact `p=0.03125`），但聚合重排后Top-1和Top-10各净少1个命中；除Oracle外，所有
Top-k变化都只有0～3个反应，属于当前1001反应benchmark上的微小波动，而不是稳定收益。

### 28.2 效率与最终决定

| 指标 | R9K1M2 | R10K1M1 | R10K1M2 |
|---|---:|---:|---:|
| sampling wall | 3059.83 s | 3022.81 s | 3425.59 s |
| input throughput | 6.543/s | 6.623/s | 5.844/s |
| peak CUDA allocated | 2,110,356,992 B | 2,213,384,192 B | 2,320,353,792 B |
| valid candidates / slots | 86.192% | 87.229% | 86.170% |
| mean true unique/reaction | 25.807 | 26.458 | 27.641 |
| final-slot shortfall | 0 | 0 | 0 |

R10M2比R9M2多11.11%的最终输出，wall增加365.76秒（11.95%），unique增加1.834，
但Top-1/10分别回退0.10pp，只有Oracle增加0.60pp。新增覆盖主要停留在最终聚合Top-10
之外，尚不能支付计算成本。R10M1因child评估数减半，尽管多一个run仍与R9M2耗时相近，
但Top-3/5分别低1.898/2.497pp；它也不满足替代条件。

因此任务26结论是：**R9K1M2继续作为最高准确率默认，R3K3继续作为速度平衡配置；不把
R10K1M1或R10K1M2加入推荐配置，也不继续扩大R。** R10实验揭示的6个新增Oracle候选应
在下一阶段通过聚合排序或Q sharpening转化，而不是继续线性堆独立run。

## 29. 任务 27：Euler、局部 child 选择与全局 beam 的机制复盘

状态：`[x] 两项补充实验、逐反应配对和机制复盘完成`

本任务不继续扫描参数，而是回答当前创新究竟由哪一部分产生：普通Euler、每步M个child
的局部择优、K个长期分支的合并剪枝，以及R个隔离搜索池分别贡献什么。已有实验先作为
证据盘点；只补两个无法由历史结果严格回答的对照：

1. `R9K1M2`对`R1K9M2`：完整test-mini-1001、宽度和输出均为9、M2、noop、bonus0.5、
   100 steps、batch64、TF32 high、seed42。R1使用`initial_seed_groups=9`，把R9的9个
   run初始流原样放进同一个K9池，唯一主要变量是隔离边界/跨流竞争。R9复用已存在结果，
   R1写入新目录，不覆盖历史实验。
2. 普通Euler对当前R9：使用同一test-mini前200个完整反应、每augmentation输出9条、
   100 steps和TF32 high。当前Euler入口的`--seed`尚未接入采样器，故本诊断通过同进程
   `torch.manual_seed(42)`固定全局CUDA RNG，并固定batch16；metadata仍会如实标记CLI
   seed未接入。它足以提供算法级方向证据，但在正式修复per-product稳定seed前不宣称为
   最终可复现Euler基准。R9直接从完整mini diagnostics/predictions取相同前200反应。

两项都统一使用legacy augmentation aggregation、Top-1～10、Oracle、invalid、unique、
wall和逐反应配对。结果解释规则提前冻结：

- 若M2/R9只改善Top-1却损害Top-3/10或Oracle，则创新只能表述为排序集中化，不称全面
  优于Euler；若主Top-k和覆盖一致改善，才称多child局部选择有效。
- 若R1K9更快但Top-k/Oracle下降，则全局K分支剪枝定位为速度优化而非准确率创新；若同
  seed流下同时提高准确率和wall，才替代R9。
- 本任务完成后不依据test-mini继续扫描R/K/M/bonus；下一步只针对实验暴露的具体失败
  环节立项。

### 29.1 纯Euler同9输出预算

使用test-mini前200反应。普通Euler N9通过外部同进程seed42固定当前全局RNG，TF32 high、
batch16，采样523.35秒；当前CLI seed尚未实际传入Euler，故只作为方向实验。R9复用完整
mini结果的同一前缀：

| 方法 | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| Euler N9 | 54.0 | 69.5 | 77.5 | 82.5 | 88.0 | 94.5 |
| R9K1M2 | 58.0 | 74.0 | 81.0 | 86.0 | 88.0 | 96.0 |

R9-only/Euler-only为Top-1 `13/5`（exact `p=0.0963`）、Top-2 `15/6`
（`p=0.0784`）、Top-3 `15/8`（`p=0.210`）、Top-10 `6/6`、Oracle `5/2`。
当前创新有把正确候选前移的正向证据，但Top-10不增、Oracle仅+1.5pp，且200反应配对尚
未显著；不能声称已经严格、全面超越纯Euler。Euler输出SHA-256为
`e219cd08cef023341af0c217ea32748a16e706f050918e180068aa445752f4f3`。

### 29.2 R9隔离池与R1全局池

R1使用9个initial seed groups，严格复用R9九条初始流。完整mini结果：

| 方法 | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Oracle | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1K9M2 | 54.146 | 67.333 | 72.927 | 77.123 | 83.516 | 87.013 | 1396.58s |
| R9K1M2 | 57.043 | 71.528 | 78.122 | 82.617 | 86.114 | 91.808 | 3059.83s |

R9-only/R1-only为Top-1 `71/42`（`p=0.00815`）、Top-2 `74/32`
（`p=5.5e-5`）、Top-3 `78/26`（`p=3.28e-7`）、Top-10 `55/29`
（`p=0.00604`）、Oracle `62/14`（`p=2.32e-8`）。R1通过跨lineage合并和全局淘汰把
父评估从18,018,000降到7,726,148，wall降低54.4%；但valid/slots从86.192%降至
74.614%，final-slot shortfall为27.421%，正确低质量模式发生显著竞争性灭绝。
R1输出SHA-256为`90f037f5df66932a798e24092823a509b6da79ff1a387ed8c4bce2727062ac84`。

### 29.3 阶段结论与下一步

当前证据只支持“多child局部选择方案”改善前段/中段Top-k，不支持“宽K全局beam”提高
准确率。R9K1M2继续作为准确率配置，R3K3M2为平衡配置，R1K9M2为速度配置。详细机制、
变量表和下一步优先级见`new_docs/euler_beam_current_situation.md`。

后续用户明确普通Euler暂不进入当前主线。R9重复父状态profile和forward共享已经完成；
下一阶段直接研究可复用当前checkpoint的前沿推理改进，再逐步进入独立新sampler。停止
继续增加R/K/M或从test-mini扫描bonus。

## 30. 任务 28：R9受保护分支的效率优化

状态：`[x] forward共享、K1M2快速选择和batch复核完成`

用户决定优先优化当前最高准确率`R9K1M2`。概念上可把R9描述为9条受保护的独立分支，
但不能直接改写成现有K9：R分支互不淘汰，K分支共享状态质量并全局竞争。接口命名可以在
方法稳定后简化，本任务先保持历史CLI和metadata可复现。

代码复核确认`sample_retro.py`已经把`B_product × R`行一次传入`sample_euler_beam()`，
每个step也把全部K1状态展平成单次模型batch；R9不是9次串行模型调用。相对R1K9的主要
wall差异来自R9始终保留9条lineage，而R1会合并并删除状态，使parent evaluations下降
57%。因此本任务不能通过“把R放进batch”获得虚假加速。

分阶段执行：

1. 默认关闭的profile记录同一product内受保护lineage的逻辑parent行数、唯一parent状态
   数和理论可共享forward行数；同时记录K1/M2的child相同、bonus决策和seed tie-break
   次数。profile不得改变任何预测。
2. 在5反应短集记录R9各阶段wall与共享上限。只有重复parent比例足够，才实现“唯一state
   forward一次、结果映射回全部lineage”；搜索状态、seed和输出仍保持9条。
3. 若forward共享上限不足，再评估K1专用dense/GPU选择路径；不为消除少量Python开销大
   范围重写通用K分支实现。
4. 所有优化先要求单元测试和真实checkpoint短集prediction SHA逐行一致，再运行完整tiny
   性能。TF32因batch形状可能改变末位数值；若无法逐行一致，必须作为opt-in数值变更另行
   决策，不能静默替换R9默认。

本任务不改变R/K/M、bonus、policy、checkpoint、数据、评分或历史结果；新实验写入独立
目录。首个实现commit只允许增加诊断，不宣称加速。

### 30.1 五反应profile结果

使用test-mini前5个完整反应（100条augmentation输入），固定R9K1M2、noop、bonus0.5、
100 steps、batch64、TF32 high、seed42。无profile基线wall为14.189秒；开启同步计时和
状态统计后为14.981秒，两个`predictions.txt`逐字节一致。短集准确率仅用于完整性检查，
不用于选择算法参数。

- 逻辑parent行90,000，product内精确唯一parent行61,263，可共享28,737行（31.93%）。
- 前两步每批576个逻辑parent只有64个唯一状态，即理论可共享88.89%；随着seed导致路径
  发散，共享率逐步下降。
- 模型前向与rate计算11.020秒，占profile总wall的73.56%；merge/prune为1.018秒，编辑
  应用1.068秒，child proposal为0.744秒。因此第一优化目标是重复模型前向，不是先重写
  K1剪枝。
- 90,000个M2 child pair中73,775对（81.97%）落到同一状态；15,334次不同状态选择由
  seed平局规则决定，891次由changed-state bonus决定，另有900个90%时间点的noop锚点。
  这些是后续校准选择规则的证据，本任务暂不改变它们。

结论：实现一个默认关闭的“相同product、相同`t`、相同token状态只forward一次，再映射
回各自seed lineage”原型。它只共享确定性模型计算，不合并lineage、不共享随机动作、不
改变局部child选择。真实checkpoint短集若不能逐字节复现，或wall没有实际下降，则不启用
为默认路径。

### 30.2 相同状态forward共享原型

已增加默认关闭的`--euler_beam_share_identical_forwards`。实现为每步按
`(product group, t, token state)`去重模型输入，模型输出再映射回原来的逻辑parent行；
child seed、随机动作、局部M2选择、lineage数量和最终输出槽位均不合并。metadata记录开关，
runtime同时记录逻辑parent、实际model-forward parent和共享行数。

单元与入口测试共47项通过。真实checkpoint的5反应短集在TF32 high下预测逐字节一致，
wall由14.189秒降到12.107秒（14.67%）。随后使用完整tiny（1000条augmentation输入、
50个反应）成对验证：

| tiny指标 | 原R9K1M2 | forward共享 | 变化 |
|---|---:|---:|---:|
| sampling wall | 154.025s | 114.058s | -25.95% |
| 逻辑parent行 | 900,000 | 900,000 | 0 |
| 实际model parent行 | 900,000 | 558,907 | -37.90% |
| peak CUDA allocated | 1,681,829,888 B | 1,553,148,416 B | -7.65% |
| 不同预测行 | 0/9,000 | 2/9,000 | 0.022% |

两版tiny的Top-1～10（60/70/80/84/84/84/84/84/84/84）、Oracle 98%、invalid、
valid/unique和全部coverage diagnostics完全相同。基线输出SHA-256为
`542f2a8582547b30752c3c6df28ebf454f4d4673b6c79e78f2b79397d12c2d48`，共享版为
`9748b25877b1595f39670fa09aebc133db513dad36eaabb97d1a42e3129daeb6`。

进一步对包含两处差异的input区间`[200,300)`使用`highest` FP32复测，两版输出SHA均为
`653ddd740c64f37f244bfce60d2c4e3d2d7180aaec95885195367416193bb5c1`，逐字节一致；wall
由17.662秒降至13.436秒（23.93%）。由此确认tiny的两行差异来自TF32在不同GEMM形状下
的数值漂移，而非跨product共享或seed lineage错误。

阶段决定：该功能以opt-in效率模式提交。3090/TF32下它有稳定的大幅收益且tiny指标不变，
但因未满足TF32逐字节门槛，暂不静默改为默认；`highest`下可视为逐字节等价优化。下一步
继续实现K1M2的等价快速选择，之后再决定是否组合进入推荐命令。

### 30.3 K1M2等价快速选择

profile中merge/prune约占短集wall的6.8%。当前K1M2仍为每个sample创建dict、执行通用状态
合并并对最多两个元素排序；但K=1时只需在两个child中选一个。下一小步只为
`n_branches=1, n_children=2, full_probability`增加专用选择器，严格复现以下规则：相同
状态做logaddexp并保留较小seed代表；不同状态仍按changed-state bonus、log mass、seed
依次比较。其他K/M/score mode继续走原逻辑。

门槛：helper与通用实现属性等价、现有测试全通过、真实checkpoint预测SHA一致；性能先用
5反应组合共享模式成对测量，收益若低于正常计时噪声则保留清晰实现但不宣称额外提速，
不为该路径扩大重写范围。

实现和测试已完成。新增专用选择器仅处理K1M2/full-probability，三类helper逐属性对照加上
全套相关测试共50项通过。真实checkpoint组合共享模式的pre/post预测SHA均为
`62eaf2e0cf4fca7f2d5bb15ab02e1b08f73aa5677c0b9459303205abccef6d95`。一次非同步短测
wall从11.875秒到11.396秒，但该4.04%差异可能包含GPU波动；同步profile中merge/prune
从1.018秒降到0.945秒（该阶段-7.17%，总wall绝对约-0.073秒），其他阶段存在更大的计时
波动。因此结论只记为小幅等价优化，不宣称稳定4%端到端提速，也不继续扩大K1剪枝重写。

### 30.4 forward共享后的batch size复核

共享版5反应profile中模型阶段由11.150秒降至8.426秒，仍占总wall 65.7%；apply edits、
merge/prune和child proposal分别只有9.7%、7.8%、6.6%。完全相同state的共享已穷尽，
继续重写后三级路径的预期收益较低。由于共享把物理model rows显著压缩，非共享R9采用的
product batch64不一定仍能充分利用RTX 3090。

固定R9K1M2、noop、bonus0.5、TF32 high、seed42、100 steps和共享开关，在tiny前400条
输入上比较batch64/128/256。只以sampling wall、input throughput、物理model rows、峰值
显存和预测SHA为判据；不从20个反应的accuracy选择batch。若更大batch没有稳定收益或显存
增长不成比例，继续保留64，不再扩大扫描。

结果如下：

| product batch | wall | input throughput | physical model rows | peak CUDA allocated |
|---:|---:|---:|---:|---:|
| 64 | 44.920s | 8.905/s | 237,439 | 1,553,148,416 B |
| 128 | 47.587s | 8.406/s | 237,438 | 2,800,537,600 B |
| 256 | 48.906s | 8.179/s | 237,438 | 4,082,305,024 B |

batch128/256相对64分别慢5.94%/8.87%，峰值allocated增长80.3%/162.8%，没有减少实质
搜索工作。TF32不同batch形状相对batch64分别改变7/3600和9/3600个输出行，也没有数值
稳定性优势。因此RTX 3090共享模式继续推荐batch64，不再扩大batch扫描。

任务28阶段结论：保留R与K的历史接口语义；把当前R9K1称为9条受保护分支是准确的，但
不能把竞争式K>1也改称同一种分支。共享开关在tiny上将R9 sampling wall降低25.95%，
指标完全不变，是本任务的主要成果；因TF32下2/9000输出行漂移，继续保持opt-in。相关
commit为profile `98cbcfb`、forward共享`ccf273a`、K1M2选择`14f9523`。

## 31. 后续方法与改进的强制记录格式

从本节开始，无论是新增sampler、论文方法适配，还是现有Euler-Beam的局部改进，都必须
先在本文建立对应条目，再修改代码。每个条目必须依次包含以下内容，不能只写参数和结果：

1. **方法/改进介绍**：说明方法本身做什么、作用在哪一层、是否改变目标分布、proposal、
   搜索、排序或仅改变计算实现，并区分论文原方法与本项目实际采用的版本。
2. **为什么要做**：使用已有代码诊断或实验数据说明动机，禁止只因“论文更新”就接入。
3. **对应当前什么问题及预期好处**：明确它试图改善Top-1、Top-k、Oracle、invalid、
   diversity、wall、显存还是方法正确性；同时记录可能损害的指标和适用边界。
4. **如何适配本任务**：写清作用位置、公式/状态、接口、metadata、seed、输出布局、与现有
   R/K/M/child policy的关系，以及哪些逻辑保持不变。需要新checkpoint、reward或训练时
   必须显式注明，不能伪装成纯推理开关。
5. **实验预注册与结果占位符**：在执行前冻结数据、baseline、变量、指标、正确性门槛、
   停止条件和输出目录；预留代码修改、测试、结果表、分析、结论、commit字段。实验后回到
   同一位置填写，不能只在聊天或终端输出中保留结论。

统一模板如下：

```text
### 方法名称
状态：[待研究/待实现/实验中/完成/停止]

#### A. 方法/改进介绍
#### B. 为什么要做
#### C. 对应当前问题、预期好处与风险
#### D. 适配到本项目的具体方案
#### E. 实验预注册
#### F. 实现与正确性测试（占位）
#### G. 实验结果（占位）
#### H. 结果分析与结论（占位）
#### I. Git记录（占位）
```

## 32. 任务 29：Q sharpening推理改进

状态：`[x] 接口实现、T=1兼容、validation-A/B/C及tiny补充对照完成；默认仍为T=1.0`

### 32.A 方法/改进介绍

Q sharpening作用于模型给出的insert/substitute token条件分布`Q_ins`、`Q_sub`。温度版本
把token log-probability变换为：

```text
log Q_T(token) = log_softmax(log Q(token) / T)
```

`T=1`为当前实现；`T<1`提高高概率token的相对权重。它不改变insert/substitute/delete的
event rate，也不改变某个位置是否发生编辑，只改变编辑触发后抽到哪个insert/substitute
token。该方向来自Edit Flows原论文的推理策略，属于复用当前checkpoint的proposal改进，
不是新训练模型，也不是最终候选reranker。

### 32.B 为什么要做

当前R9K1M2 profile显示，90,000个child pair中81.97%产生相同状态；真正不同的pair里多数
由seed平局而非具有预测力的value决定。已有trajectory分析还观察到错误insert token、
反复修正和invalid SMILES。继续增加R或M只会扩大同一proposal的采样预算，未直接改善
token proposal质量。Q temperature几乎不增加计算量，适合先验证模型分布是否过平。

### 32.C 对应当前问题、预期好处与风险

试图应对的问题：错误token造成的无效编辑、低质量分支和Top-k噪声。可能收益是降低
invalid、让正确高概率token更早出现、提高Top-1/3；额外wall近似为零。主要风险是分布
过尖，使9条受保护分支趋同，true unique、Top-10或Oracle下降。因此不能只用Top-1选择
温度，也不能直接在test-mini扫描。

### 32.D 适配到本项目的具体方案

1. 在`euler_beam.py`的模型输出之后、child token采样和step log-prob计分之前，对
   `log_ins_probs`和`log_sub_probs`统一应用温度；rate head保持原值。
2. 新增显式参数`q_temperature`及CLI
   `--euler_beam_q_temperature`，要求大于0，默认1.0。
3. `T=1.0`必须不增加额外变换或保证prediction SHA逐字节兼容；非1温度必须使用变换后
   的log-prob进行token采样及proposal/path概率记录，不能“按新分布采样、按旧分布计分”。
4. 参数贯通`sample_retro.py`、`eval.py`、sampling metadata和命令打印；不修改checkpoint、
   R/K/M、rate、seed、输出行数、augmentation聚合或评分器。
5. 第一阶段只实现temperature，不同时加入top-k/top-p，避免无法归因。

### 32.E 实验预注册

- 数据：validation `src-val/tgt-val` 的前1000行（50个完整augmentation反应）作为A，
  紧接的1000行（50个不重叠反应）作为B；不使用test选择温度。若A/B方向一致，再扩大
  到validation-200，而不是直接运行完整validation。
- 固定baseline：R9K1M2、100 steps、batch64、TF32 high、seed42、bonus0.5、
  stochastic_noop、forward共享。
- 单变量：`T ∈ {1.0, 0.9, 0.8}`；第一轮不扫描top-k/top-p。
- 指标：Top-1～10、Oracle、invalid/slots、true unique、rank availability、wall、峰值显存；
  对Top-1/3/10/Oracle做逐反应配对。
- 正确性门槛：T=1与现baseline输出SHA一致；metadata和eval命令测试通过；相同seed可复现。
- 继续门槛：Top-1不出现明确回退，且Top-3/10、invalid或候选可用性至少一项在A/B方向
  一致改善。若只提高Top-1却降低Top-10/Oracle，记录为集中化权衡，不替换默认配置。
- 输出目录：`results/task29_qtemp_<split>_t<temperature>/`。

### 32.F 实现与正确性测试

- 修改文件：`edit_flows/sampling/euler_beam.py`、`scripts/sample_retro.py`、
  `scripts/eval.py`及对应测试。
- 单元/入口测试：52项通过。
- T=1 prediction SHA：`62eaf2e0cf4fca7f2d5bb15ab02e1b08f73aa5677c0b9459303205abccef6d95`，
  与既有隐式T=1共享版smoke逐字节一致。
- 正确性异常及处理：T=1显式/隐式输出对照、温度log-prob归一化和`q_temperature<=0`
  参数校验已覆盖；A/B/C真实checkpoint实验均完成，metadata记录了q_temperature。

### 32.G 实验结果

| split | T | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | invalid/slots | true unique | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation-A (50) | 1.0 | 74 | 92 | 96 | 96 | 98 | 18.8% | 27.40 | 105.338s |
| validation-A (50) | 0.9 | 72 | 94 | 96 | 96 | 98 | 19.3% | 26.94 | 105.858s |
| validation-A (50) | 0.8 | 74 | 94 | 96 | 96 | 98 | 18.1% | 26.36 | 105.460s |
| validation-B (50) | 1.0 | 62 | 84 | 88 | 96 | 98 | 9.4% | 18.84 | 100.408s |
| validation-B (50) | 0.9 | 62 | 86 | 90 | 96 | 98 | 8.8% | 18.58 | 99.469s |
| validation-B (50) | 0.8 | 60 | 86 | 88 | 98 | 98 | 8.5% | 17.96 | 99.890s |
| validation-C (200) | 1.0 | 50.0 | 74.5 | 81.5 | 86.0 | 91.5 | 13.675% | 26.315 | 395.965s |
| validation-C (200) | 0.9 | 51.0 | 74.0 | 81.5 | 86.0 | 91.0 | 13.425% | 25.775 | 386.366s |

Top-2、完整Top-1～10和逐反应配对见各目录的`diagnostics.json`；A/B/C目录分别为
`results/task29_qtemp_valA_t{1,09,08}/`、`results/task29_qtemp_valB_t{1,09,08}/`和
`results/task29_qtemp_valC200_t{1,09}/`。

### 32.G.1 tiny补充对照（post-hoc，不用于选温度）

用户要求把任务29放回历史tiny开发集，与既有方法在同一评分器下做描述性比较。tiny为
`src-test-tiny.txt`前1000行，即50个完整aug20反应；由于该test子集此前已被反复查看，
本补充不改变validation结论，也不以tiny target重新选择温度。任务29三组固定
R9K1M2、100 steps、batch64、seed42、TF32 high、shared identical forwards、
`stochastic_noop`和`bonus=0.5`，每个product输出9条。

| 方法/温度 | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Oracle | rank-1 invalid | true unique | sampling wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 普通Euler（历史N=3） | 54 | 66 | 74 | 76 | 84 | — | 21.3% | — | 81.14s* |
| 旧R3K3（历史beam=3） | 60 | 64 | 70 | 72 | 80 | 80 | 12.5% | 13.18 | 122.09s |
| 旧R9K1M2 shared | 60 | 70 | 80 | 84 | 84 | 98 | 22.3% | 31.00 | 114.06s |
| Q temperature T=1.0 | 60 | 70 | 80 | 84 | 84 | 98 | 22.3% | 31.00 | 113.09s |
| Q temperature T=0.9 | 60 | 74 | 78 | 80 | 82 | 96 | 21.1% | 30.08 | 110.73s |
| Q temperature T=0.8 | 58 | 76 | 78 | 82 | 86 | 90 | 21.0% | 29.14 | 113.46s |

`T=1.0`预测SHA为`9748b25877b1595f39670fa09aebc133db513dad36eaabb97d1a42e3129daeb6`，
与任务28 shared tiny结果逐字节一致。普通Euler的旧文本没有sampling metadata，时间来自历史
固定基准，且Euler/R3的每条输入输出数少于R9，因此只作背景参考；三组Q温度之间才是严格
单变量比较。

### 32.H 结果分析与结论

- proposal质量是否改善：A/B中T=0.9的Top-3分别比T=1高2个百分点，Top-5在B高2个
  百分点；invalid略降，但true unique也下降。C200中Top-3反而低0.5个百分点，Oracle
  低0.5个百分点，说明proposal集中化收益不稳定。
- 多样性/覆盖是否受损：T=0.8的true unique在A/B降至26.36/17.96，尽管B的Top-10高
  2个百分点；T=0.9在三段均略降true unique，不能把invalid下降解释为全面质量提升。
- 收益是否能在不重叠validation复现：A/B的Top-3方向一致，但更大不重叠C200没有复现；
  T=1 vs T=0.9在C的Top-1/3/10逐反应only差为2/4、2/1、2/2，未形成稳定优势。
- 是否替换默认T=1、停止或继续top-k/top-p：按预注册门槛停止本轮temperature改进，
  默认保持T=1.0；不在test上继续调温度，也不立即叠加top-k/top-p。若未来有更大validation
  或新的proposal证据，可重新建立独立任务。
- tiny补充是否改变结论：不改变。T=0.9在tiny只提高Top-2，Top-3、Top-10和Oracle
  下降；T=0.8的Top-1和Oracle进一步下降，true unique也随温度变尖而减少。tiny与validation
  的局部排序不同，进一步说明不能用该test子集挑温度。

### 32.I Git记录（占位）

- 预注册commit：`e56021e`
- 实现commit：`67ae17d`
- 实验结论commit：`7adf233`
- tiny补充对照commit：`7206878`

## 33. 任务 30：Euler-SMC独立新方法

状态：`[~] 独立synthetic mechanics已实现，尚未接入checkpoint或采样入口`

前沿论文汇报材料：`new_docs/frontier_inference_scaling_report.md`。该文档区分了论文
SMC方法、当前Euler-Beam的启发式分支机制、已完成的mechanics/bootstrap验证，以及下一步
独立terminal reward适配，不把尚未完成的accuracy实验记为已验证收益。

### 33.A 方法/改进介绍

Sequential Monte Carlo用一组带权粒子近似逐时间目标分布。每一步从proposal生成child，
使用目标/提议概率比更新log-weight，根据ESS判断是否重采样，并保留ancestor lineage。
它与当前Euler-Beam的“多分支、多child、剪枝”外观相似，但当前方法采用确定性Top-K、
启发式changed-state bonus和碰撞mass，不具备完整importance correction或ESS语义。新方法
必须作为独立sampler实现，不能只把Euler-Beam改名为SMC。

### 33.B 为什么要做

R1K9说明跨lineage确定性竞争会过早删除正确低质量模式；R9K1保留独立lineage提高准确率，
但固定保留所有粒子成本高。与此同时，当前K1M2在不同child间经常由seed或bonus决定，
没有估计“这个状态未来能否形成正确reactant”的分数。SMC提供明确的proposal、weight、
ESS和重采样框架，可研究何时共享预算、何时必须保护多样性，而不是继续扫描R/K/M。

### 33.C 对应当前问题、预期好处与风险

它试图解决启发式child选择、固定粒子预算和竞争性路径灭绝问题。潜在好处是相同计算预算
下更合理地分配粒子、在高ESS时避免无效重采样、在低ESS时恢复有效探索，并提供可诊断的
权重/祖先轨迹。主要风险是：若没有独立的化学reward，bootstrap target与proposal相同，
理论上权重应接近一致，不能凭空带来准确率；不正确的importance ratio反而会造成更严重
的粒子坍缩。接入forward reward还会显著增加推理成本并引入额外模型依赖。

### 33.D 适配到本项目的具体方案

1. 新建独立`euler_smc.py`及sampler入口，保留`euler_beam.py`和历史结果不变。
2. 粒子状态至少记录tokens、t、seed、log proposal、log target/weight、ancestor id和
   resampling次数；相同state的模型forward仍可共享，但粒子身份不能合并。
3. 第一阶段proposal完全复用Euler transition，target也设为同一base transition；此时
   importance increment应接近0，用于验证mechanics，而不是追求准确率。
4. 实现normalized log-weight、ESS和systematic resampling；seed必须按product/particle/
   step稳定，batch切分不得改变逻辑随机流。
5. terminal/twisted reward只能来自train/validation构建的独立forward consistency、
   feasibility或明确化学约束；严禁使用测试Target。changed-state bonus不能直接当成理论
   reward。
6. 输出布局、metadata和Top-1～10评分保持兼容；新增ESS、祖先多样性和额外forward wall。

### 33.E 实验预注册

阶段1仅验证mechanics：

- synthetic离散CTMC：已知目标分布和proposal，检查importance estimate、ESS、systematic
  resampling频率和ancestor传播。
- bootstrap invariant：target=proposal时权重增量应接近0；固定seed下开关无必要
  resampling不应改变边际结果。
- 小型真实checkpoint smoke：只检查shape、seed、metadata、invalid数值和wall，不宣称
  Top-k提升。

阶段2只有在阶段1通过且独立reward定义完成后，才在validation比较固定总child budget的
R9K1M2与Euler-SMC。指标包括Top-1～10、Oracle、invalid、true unique、ESS曲线、祖先数、
resampling次数、forward次数和wall。输出目录：`results/task30_euler_smc_<stage>_<run>/`。

### 33.F 实现与正确性测试

- 目标/proposal数学定义：`advance_particles()`接收每个child的
  `log_target_increment`和`log_proposal_increment`，将其差值加到选定父粒子的归一化
  log-weight；按ESS阈值触发systematic resampling，并传播`ancestor_ids`。
- 修改/新增文件：新增`edit_flows/sampling/euler_smc.py`和
  `tests/sampling/test_euler_smc.py`；没有修改`euler_beam.py`、`sample_retro.py`、
  checkpoint或训练代码。
- synthetic测试结果：11项测试通过，覆盖log-sum-exp归一化、ESS、确定性systematic
  resampling、批次布局独立性、importance ratio、Euler proposal闭合和非法输入校验。
- transition adapter：新增`euler_transition_step()`，一次只执行一个Euler proposal，
  复用Euler-Beam的无状态seed动作、Poisson事件概率、编辑应用和完整step log-prob；
  `target=proposal`时可直接送入`advance_particles()`做bootstrap smoke。为避免把linear
  事件概率用Poisson scorer误计，当前明确拒绝`event_prob_mode='linear'`。
- 回归测试：Task30相关测试及Euler-Beam/入口测试共`58 passed`；完整
  `tests/sampling`另有17个既有`beam.py` API不匹配失败（测试构造仍传入已移除的
  `log_u_real`，且一个受控模型序列长度不一致），本次新增文件未触及这些代码，故不把它们
  误记为SMC通过或失败。
- bootstrap invariant结果：当target=proposal时增量为0、`ESS=N`、evidence增量为0，
  在阈值`ESS<N`时不发生resampling，权重保持均匀。
- seed/batch invariance：每个product/step使用稳定派生seed；同一行在单行和多行batch
  中的resampling结果一致，且不受无关product行的随机消耗影响。

### 33.G 实验结果

| stage/config | particle budget | Top-1 | Top-3 | Top-10 | Oracle | mean ESS | ancestors | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic mechanics | 3 | — | — | — | — | invariant pass | genealogy pass | <1s CPU |
| bootstrap invariant | 4 | — | — | — | — | 4.0 | no resampling | <1s CPU |
| checkpoint transition smoke (2 rows, 1 step) | 2 | — | — | — | — | not scored | not scored | shape/finite pass |
| checkpoint bootstrap smoke (1 product, N=3, 4 steps) | 3 | — | — | — | — | 3.0 each step | no resampling | ~0.99s GPU* |
| validation baseline R9 | not run | — | — | — | — | — | — | — |
| validation Euler-SMC | not run | — | — | — | — | — | — | — |

真实checkpoint已完成单步transition smoke和4步、N=3的target=proposal bootstrap smoke；
`~0.99s GPU`含4次模型forward，仅作接口量级记录，不是与Euler-Beam的正式效率比较。
尚未运行带独立target/reward的多步粒子采样，因此本阶段没有Top-k、invalid或准确率改进结论。

### 33.H 结果分析与结论

- mechanics是否满足理论不变量：是；log-weight归一化、ESS、target/proposal比值和
  systematic resampling的单元测试均通过。
- Euler transition接口是否闭合：是；合成模型的proposal可稳定复现，并在真实
  `checkpoint_step600000.pt`上完成一次2行GPU forward、编辑应用和有限log-prob输出。
- 多步bootstrap是否闭合：是；真实checkpoint的3粒子、4步rollout保持`ESS=3`、无
  resampling且累计evidence为0，说明target=proposal时没有凭空制造权重偏置。
- 是否出现粒子坍缩/谱系灭绝：尚未在真实transition/reward上评估；合成测试只验证了
  resampling后谱系传播，不把一次抽样误写成坍缩结论。
- reward是否提供独立于base模型的排序信息：尚未定义或测量。bootstrap设置中
  target=proposal，不能产生额外排序信息。
- 固定预算下准确率与wall权衡：尚无真实checkpoint结果；当前实现只建立CPU mechanics，
  不代表已接入后的推理效率。
- 继续reward twisting、停止或转向训练方法：先完成Euler transition adapter和小规模
  smoke，证明proposal/target接口及metadata正确后，再决定是否引入独立reward；在此之前
  不替换默认Euler-Beam。

### 33.I Git记录（占位）

- 预注册commit：`e56021e`
- mechanics实现commit：`fd149f1`
- transition adapter commit：`3553b1b`
- bootstrap rollout commit：`c251843`
- validation结论commit：`[待填写]`

## 34. 任务 31：独立target/reward选择与Terminal Twisting

状态：`[ ] 已完成候选方案说明和实验占位；等待确定研究目标，尚未接入reward`

### 34.A 方法/改进介绍

Euler-SMC的bootstrap阶段令`target=proposal`，因此权重恒等、ESS不提供额外排序信息。
下一步可在终点状态`x_1`加入一个不依赖测试Target的独立reward`R(x_1; product)`，定义

```text
pi(x_1 | product) ∝ q_euler(x_1 | product) * exp(beta * R(x_1; product))
log_target_increment = log_proposal_increment + beta * ΔR
```

第一版只做terminal twisting：中间Euler step仍按base proposal推进，最后一步把终点reward
作为target/proposal比值输入`advance_particles()`。之后若reward有稳定信号，才考虑随时间
平滑打开的intermediate twisting；不同时引入learned proposal或CFG。

### 34.B 为什么要做

bootstrap已证明SMC mechanics不会凭空改变分布，但也证明没有独立target就不可能凭空提升
Top-k。terminal reward提供一个可审计的“最终状态质量”来源，理论上可在固定粒子预算下
提高有效候选质量，ESS则可以诊断reward是否把粒子过早集中到单一模式。

### 34.C 对应当前问题、预期好处与风险

- 独立forward consistency分数：最符合逆合成目标，可能同时提升Top-1和Top-10，但需要
  一个不共享反向checkpoint的forward模型或可复现实验数据。
- feasibility/classifier分数：成本居中，可提供反应可行性信号，但必须先确认训练/验证
  来源和校准，避免把数据集偏差当成目标。
- RDKit validity/价态/片段约束：无需新模型、效率较高，适合作为弱reward诊断；但“有效
  SMILES”不等于正确反应，过去invalid下降伴随Top-k下降的结果说明不能直接把它设为默认。
- 反向模型自身log-prob或共识：只能作为proposal诊断，不能冒充独立reward，否则只是
  把现有排序重新包装为SMC。

主要风险是`beta`过大导致粒子坍缩、Oracle/Top-10下降和invalid指标虚假改善；reward计算
也可能成为新的wall瓶颈。因此必须固定总child budget，报告ESS曲线、祖先数、reward分布、
Top-1～10、Oracle、invalid、true unique和wall，不能只看Top-1。

### 34.D 适配到本项目的具体方案

1. 先确定唯一主reward来源及其训练/验证边界；任何实现都不得读取test target或用test调
   `beta`。
2. 定义纯函数`terminal_reward(states, product_context)`，只接受token状态和product，
   返回每粒子的有限标量；记录reward版本、参数和来源到metadata。
3. 在`run_euler_smc_bootstrap`之外新增独立terminal-twisting入口，保持proposal、seed、
   输出布局和现有Euler-Beam不变；仅在最终transition向`advance_particles()`传入
   `log_target_increment = log_proposal_increment + beta * ΔR`。
4. 先用synthetic已知target验证importance estimate，再做validation-小段的R9K1M2固定
   预算对照；若reward使ESS快速降到1附近或Oracle/Top-10下降，停止twisting并记录失败。
5. 只有terminal reward在不重叠validation区间稳定改善，才研究intermediate twisting；
   不在同一轮叠加Q temperature、changed-state bonus或新的child policy。

### 34.E 实验预注册（待reward选择后冻结）

- baseline：当前默认Euler-Beam R9K1M2，以及target=proposal的Euler-SMC bootstrap；
  proposal、n_steps=100、TF32/FP32、batch、seed和总child budget固定。
- reward候选：只选择34.C中的一个主reward；其余最多作为诊断，不混合成不可解释分数。
- beta：在validation-A上预注册小集合并在validation-B/C复核；不使用test调参。
- 指标：Top-1～10、Oracle、invalid、true unique、mean/p10 ESS、resampling次数、祖先
  多样性、reward分布、forward/reward调用次数、peak memory和wall。
- 成功门槛：Top-1不明显回退，且Top-3/10或Oracle在不重叠validation上稳定改善，同时
  ESS不出现系统性坍缩、invalid不以牺牲覆盖为代价下降。

### 34.F 实现与正确性测试（占位）

- 主reward及独立数据边界：`[待选择/填写]`
- terminal-twisting入口与metadata：`[待实现]`
- synthetic已知target importance estimate：`[待实现]`
- validation-A/B/C结果：`[待实验]`

### 34.G 结果表（占位）

| reward/beta | budget | Top-1 | Top-3 | Top-10 | Oracle | invalid | mean ESS | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bootstrap target=proposal | `[固定]` | — | — | — | — | — | — | — |
| terminal reward | `[固定]` | — | — | — | — | — | — | — |

### 34.H Git记录（占位）

- 预注册/候选方案commit：`[待填写]`
- reward实现commit：`[待填写]`
- validation结论commit：`[待填写]`

### 34.I 新训练 checkpoint tiny 回归（2026-08-07）

状态：`[x] 新旧 checkpoint 完成同配置 R9K1M2 配对复测`

在 A6000 完成新模型训练后，先在 tiny（50 个完整反应）上冻结 R9K1M2、100 steps、
seed42、bonus0.5、`stochastic_noop`、TF32 high 和相同状态 forward sharing。新 checkpoint
相对旧 checkpoint 的 Top-1/2/3/5/10 为 `58/74/78/84/90` 对 `60/70/80/84/84`，Oracle
为 `92` 对 `98`，采样时间约 `113.49s` 对 `114.72s`。结论是覆盖下降但尾部排序改善，
tiny 不足以决定替换模型；详细过程、SHA 和结果目录见
[`new_checkpoint_tiny_evaluation.md`](new_checkpoint_tiny_evaluation.md)。

本次还修复了 PyTorch 2.6+ 的 checkpoint 加载兼容问题，代码 commit 为 `88a0f2e`。

## 11. 决策门槛

继续推进无需等待确认，但以下情况必须停止并请用户决定：

1. 两种研究方案相互冲突，一种优化 Top-1、另一种优化 Top-3，且没有明确的主指标；
2. 需要更换 checkpoint、训练配置、数据集或 target 才能继续；
3. 继续实验可能覆盖历史结果或破坏无法隔离的用户文件；
4. 评分改动只能通过利用测试 target 调参获得提升；
5. 需要大范围修改与 Euler-Beam 无关的模型或训练代码。

以下情况不能单独作为停止理由：

- 一次短筛没有提升；
- 完整 GPU 实验较慢，但存在离线分析或更小规模筛选；
- 某一条启发式路线失败；
- 仍可通过只读诊断明确下一步。

## 12. Git 和文档记录规则

取得明确阶段性进展时：

1. 更新本文件中对应任务的状态、修改、测试、结果和结论；
2. 检查并排除与本任务无关的用户改动；
3. 创建范围清晰的 commit；
4. 推送到当前 GitHub 分支；
5. 在本文件记录 commit hash。

建议 commit 范围：

```text
Add strict retrosynthesis scoring validation
Add sampling coverage diagnostics
Evaluate heterogeneous Euler-Beam runs
Add per-run Euler-Beam policies
Compare augmentation aggregation strategies
Record final Top-k and performance validation
```
