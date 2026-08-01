# Euler-Beam 下一阶段任务规划

更新日期：2026-08-01

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
    --beam_size 3 \
    --n_best 5
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
beam_size = n_runs（Euler-Beam）或 n_samples（Euler）
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

状态：`[ ] 未开始`

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
| Oracle-any | 待测 | 待测 | 待测 | 待测 |
| 最终 Top-1 | 60 | 58 | 56 | 58 |
| 最终 Top-2 | 64 | 64 | 68 | 68 |
| 最终 Top-3 | 70 | 66 | 74 | 76 |
| 已覆盖但未进 Top-3 | 待测 | 待测 | 待测 | 待测 |
| run overlap | 待测 | 待测 | 待测 | 待测 |

注意：不同 seed/语义的恢复版本只用于定性解释，不写成严格消融结论。

### 9.4 本任务完成记录

实际修改：待填写。

测试与实验：待填写。

结论：待填写。

Commit：待填写。

## 6. 任务 10：异质 run 与受控探索

状态：`[ ] 等待任务 9 归因`

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

### 10.2 实现 per-run policy

只有离线混合有效时，才在 `sample_retro.py` 增加每个 run 独立的 policy 配置。设计要求：

- 单一 `--euler_beam_child_policy` 保持兼容；
- 新接口明确要求 policy 数量为 1 或 `n_runs`；
- product/run seed 保持稳定，不因 policy 列表或 batch size 改变；
- 不把异质 run 错误合并成一个内部 beam；
- 输出顺序仍为 product-major、run-minor。

### 10.3 若简单混合不足，再研究探索 proposal

研究顺序：

1. 只让一个 run 使用受控 exploration，另外两个保持当前最佳；
2. exploration 优先针对反应中心候选或模型不确定位置；
3. 明确记录目标概率 $p$、proposal $q$ 和是否做校正；
4. 先做 5/20 个反应短筛，再决定是否运行 50 个反应；
5. 不重复已经失败的 antithetic、多次 no-op、强制低概率编辑和内部低排名 branch 输出。

### 10.4 本任务完成记录

实际修改：待填写。

测试与实验：待填写。

结论：待填写。

Commit：待填写。

## 7. 任务 11：评分聚合方法消融

状态：`[ ] 等待任务 9 归因`

仅当 Oracle 诊断证明存在明显“已覆盖但排名靠后”时开展。

### 11.1 保留历史默认模式

当前规则以候选在任意 augmentation 中的最好局部位置为第一排序级，再比较 reciprocal
rank 累积分数。它必须保留为 `legacy_best_rank` 或等价模式，确保历史结果可复现。

### 11.2 候选聚合方案

以 opt-in 参数做单变量消融：

- `legacy_best_rank`：当前规则；
- `rrf`：只使用跨 augmentation reciprocal-rank 累积；
- `frequency_first`：出现的 augmentation 数优先，局部排名作次级排序；
- `hybrid`：标准化的频次、reciprocal rank 和最佳局部排名组合。

任何聚合方法都只使用预测本身能够提供的信息，不能利用 target 选参数或排序。

### 11.3 防止指标虚高

- 原历史模式始终同时报告；
- 先在预先定义的小范围方案上比较，不扫描大量权重；
- 若要调权重，划分独立验证集，不能在 tiny test target 上挑最佳值；
- 报告 Oracle 上限、最终 Top-k 和覆盖到排名的损失；
- 评分器变化不宣称为采样模型准确率提升。

### 11.4 本任务完成记录

实际修改：待填写。

测试与实验：待填写。

结论：待填写。

Commit：待填写。

## 8. 任务 12：采样输出元数据和接口健壮性

状态：`[ ] 未开始`

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

实际修改：待填写。

测试与实验：待填写。

结论：待填写。

Commit：待填写。

## 9. 任务 13：胜出方案完整验证与性能收口

状态：`[ ] 未开始`

只有任务 10 或 11 出现明确正向结果后执行。

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

### 13.3 性能

- 在 RTX 3090、TF32 `high` 下记录端到端 wall time；
- 分开记录模型 forward、候选生成/编辑、合并/排序和评分耗时；
- 记录峰值显存；
- 相同输入至少进行一次 warm-up 后再比较短任务；
- 不使用首次编译时间远大于完整采样收益的 `torch.compile` 路线；
- 不修改共享训练模型以迁就 BF16，除非未来单独扩大任务范围。

### 13.4 本任务完成记录

实际修改：待填写。

测试与实验：待填写。

结论：待填写。

Commit：待填写。

## 10. 决策门槛

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

## 11. Git 和文档记录规则

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
