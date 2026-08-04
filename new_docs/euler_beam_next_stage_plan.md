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

验证：Euler-Beam 与采样 metadata 定向测试为 `36 passed`；排除仓库中既知且与本任务
无关的 `tests/sampling/test_beam.py` 后，完整回归为 `176 passed, 8 warnings`。实现 commit：
`f921ee8`。用户的 `old_*.py`、`PDF/` 和既有 `visualize_trajectory.py` 本地修改均保持
未跟踪/未提交，不纳入本任务。

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
