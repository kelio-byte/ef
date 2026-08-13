# 首个编辑多样性策略：R9K1M2 A/B 结果

日期：2026-08-13

## 结论

关闭 `--euler_beam_first_edit_diversity`。它增加了少量候选多样性和 Oracle，但显著损害 Top-1；在当前 dev gate 上不进入 confirm、final 或完整 `src-test`。

## 对比配置

两组使用同一新 checkpoint、同一输入、同一 seed 和同一评分协议：

- checkpoint：`new_checkpoints/checkpoint_step600000.pt`
- 数据：`evaluation_v2/dev_unique1000_aug20`，1,000 个 reaction、20,000 个 augmented product 输入
- 采样：`n_steps=100`、`batch_size=64`、CUDA、seed 42、cubic scheduler
- Euler-Beam：`n_runs=9`、`n_branches=1`、`n_children=2`
- score：`full_probability`、`changed_state_bonus=0.5`、`q_temperature=1.0`、`stochastic_noop`
- 输出：每个 augmented input 9 条，合计 180,000 条
- 聚合：`legacy_best_rank`
- `share_identical_forwards=false`

| 组别 | 结果目录 | 唯一变化 |
| --- | --- | --- |
| A 原始 R9K1M2 | [`results/r9_vs_euler_dev1000_r9k1m2_seed42`](../results/r9_vs_euler_dev1000_r9k1m2_seed42) | 无新开关；已存在结果 |
| B 首个编辑多样性 | [`results/r9_vs_euler_dev1000_r9k1m2_first_edit_seed42`](../results/r9_vs_euler_dev1000_r9k1m2_first_edit_seed42) | 增加 `--euler_beam_first_edit_diversity` |

## Top-k 与覆盖

| 指标 | A | B | B−A |
| --- | ---: | ---: | ---: |
| Top-1 | 58.7% | 57.4% | −1.3 pp |
| Top-2 | 73.1% | 71.8% | −1.3 pp |
| Top-3 | 77.4% | 77.1% | −0.3 pp |
| Top-5 | 81.3% | 81.1% | −0.2 pp |
| Top-10 | 85.4% | 85.0% | −0.4 pp |
| Oracle-any | 90.8% | 91.0% | +0.2 pp |
| Mean true unique candidates/reaction | 23.237 | 24.354 | +1.117 |
| Mean valid candidates/reaction | 157.599 | 157.563 | −0.036 |
| Invalid rate（180 个 raw candidate 的估算） | 12.445% | 12.465% | +0.020 pp |

Reaction-level paired bootstrap（5,000 次）结果：

- Top-1：−1.3 pp，95% CI `[-2.4, -0.2]`
- Top-2：−1.3 pp，95% CI `[-2.3, -0.4]`
- Top-3：−0.3 pp，95% CI `[-1.2, +0.5]`
- Top-5：−0.2 pp，95% CI `[-0.9, +0.5]`
- Top-10：−0.4 pp，95% CI `[-1.1, +0.2]`
- Oracle：+0.2 pp，95% CI `[0.0, +0.5]`

配对统计以原始 reaction 为单位，20 条 augmentation 没有被当作独立样本。完整 JSON 见 [`comparison_to_baseline.json`](../results/r9_vs_euler_dev1000_r9k1m2_first_edit_seed42/comparison_to_baseline.json)。

## 效率与机制诊断

- A 采样耗时：3,004.7 s；B：3,128.1 s，增加 123.5 s（约 4.1%）。
- 两组模型前向父分支数均为 18,000,000，child candidate evaluations 均为 36,000,000；新策略没有增加模型前向或输出预算。
- B 的首次编辑 signature 统计：223,538 个候选被赋予 signature；累计保留 signature 槽位 18,000,000；最终没有 branch shortfall。
- 多样性确实增加：mean true unique candidates/reaction 从 23.237 到 24.354。
- 但新增候选没有转化为更好的排序，且 target 的平均最终 rank 从 2.811 变为 2.951；因此策略主要增加了低排序价值的候选。

## 判定

这个实验验证了机制假设的一半：首步分层可以增加候选差异，但当前“按首个 `(操作类型, 位置, token)` 强行分散 9 条 run”的选择标准不能判断哪些差异有用。它牺牲了 Top-1/Top-2，收益只停留在 +0.2 pp Oracle，故关闭该版本，不继续扫描 signature 定义、seed 或分层强度，也不消耗 confirm/final/test。

实现保留为默认关闭的 opt-in 功能，便于未来有更可靠的 future-value/ranking 信号时复用。A/B 的原始预测、metadata、diagnostics 和配对结果均保留在上述结果目录中。
