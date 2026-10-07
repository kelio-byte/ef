> **最终修正版请先阅读 [`motivation_report.md`](motivation_report.md)。**
> 本文件前面的 P1/P2 章节包含早期单条 Levenshtein 对齐和旧版干预的历史结果；
> 它们已被顺序无关重分析替代，不能作为最终机制结论。新版原始结果位于
> `results/intervention_order_invariant_v3/`。

# P0：轨迹纠错诊断 Smoke 报告

日期：2026-08-21

## 目的

验证新增 compact trajectory recorder 是否能记录首个编辑、后续 SUB/DEL 和最终 canonical 命中，同时确认它不会改变普通 Euler 的采样结果。

本次只是工具链 smoke，不用于判断 SPE 的纠错机制是否成立。

## 实现

- 在 `sample_euler` 中新增可选的 `record_compact_events=True`；默认关闭时不改变原有路径。
- compact event 只保存 `step/t`、编辑位置/类型/token、oracle 支持、编辑前后状态，不保存完整 logits。
- 新增 token-aware 判定：位置、INS/SUB/DEL 类型和 completion token 必须同时匹配 oracle；不能只比较编辑位置。
- 新增正式的 `join token → inverse_global_align → RDKit canonical SMILES` 终点判定。
- 新增独立入口 `scripts/trajectory_correction_analysis.py`。

## 验证结果

### 自动测试

`tests/test_trajectory_correction.py` 与 `tests/sampling/test_euler.py`：

```text
39 passed, 12 warnings
```

覆盖了 token 错误、位置错误、全事件一致性、canonical 等价性，以及 compact recording 前后输出一致性。

### 两个真实 checkpoint smoke

| 模型 | reaction | 轨迹数 | 首事件数 | 首事件 fully-oracle | 首事件 off-oracle | 最终 valid | 最终 hit | compact 前后输出 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Atom@600K | 10 | 20 | 18 | 44.44% | 55.56% | 95.00% | 25.00% | 3/3 batches identical |
| SPE-M500@490K | 10 | 20 | 20 | 50.00% | 50.00% | 85.00% | 30.00% | 3/3 batches identical |

Smoke 中的条件统计仅供检查数据流：

- Atom：首错轨迹的自然恢复率为 0/10；首错后平均发生 0.4 个 SUB/DEL action，平均 1.2 次后续 event 使 token edit distance 降低。
- SPE-M500：首错轨迹的自然恢复率为 0/10；首错后平均发生 0.6 个 SUB/DEL action，平均 0.3 次后续 event 使 token edit distance 降低。

样本量太小，不能据此下机制结论。

## P0 结论

P0 通过：诊断代码、token-aware oracle 判定、canonical 终点判定和 compact recorder 均可运行；记录功能没有改变两模型的采样输出。

P1/P2 已在后续章节完成；P0 通过后使用固定 1,000 reaction、N=9、seed 42/7/123 的协议继续验证，并按 reaction block 做 paired/cluster bootstrap。

详细命令和文件哈希见 [`commands.md`](commands.md)；原始 JSON/JSONL 位于：

- `revision/results/p0_smoke/atom_600k/`
- `revision/results/p0_smoke/m500_490k/`

---

# P1/P2：SPE 后续纠错机制验证结果

日期：2026-08-21

## 1. 实验目的与冻结协议

本轮不是重新比较所有 sampler，而是验证一个具体 Motivation：如果 SPE-M500 的前面编辑出错，是否更容易通过后续 SUB/DEL 把序列纠正回来。

冻结的对照为 Atom@600K 与改进后 global R-SMILES 上的 SPE-M500@490K。两者都使用自己的 tokenizer/data projection，但 reaction index 完全对应。每个模型在 `dev_unique1000_aug20` 的 1,000 个 reaction 上取第一条 augmentation，普通 Euler、cubic、100 steps、N=9，seed 为 42/7/123。所有置信区间均以 reaction 为 bootstrap 单位，而不是把同一 reaction 的 9 条轨迹当成独立样本。

## 2. P1：自然轨迹

| 指标 | Atom@600K | SPE-M500@490K | M500−Atom | 95% CI |
|---|---:|---:|---:|---:|
| First off-oracle rate | 43.28% | 39.89% | −3.39 pp | [−5.35, −1.43] pp |
| Natural recovery rate | 3.07% | 2.64% | −0.43 pp | [−1.34, +0.47] pp |
| Clean-path success | 49.11% | 46.44% | −2.67 pp | [−4.66, −0.56] pp |
| Final hit rate | 35.32% | 36.38% | +1.06 pp | [−0.47, +2.56] pp |
| Final valid rate | 88.47% | 87.16% | −1.31 pp | [−2.31, −0.25] pp |
| 首错后平均 SUB/DEL | 0.328 | 0.875 | +0.547 | [+0.487, +0.604] |
| 首错后平均距离下降事件 | 1.492 | 1.031 | −0.461 | [−0.622, −0.306] |

### P1 分析

M500 确实更少出现首个 off-oracle 事件，但自然 recovery 没有提高，三 seed 的总体结果也不支持“首错后更会修复”。M500 后续 SUB/DEL 更多，却没有带来更多距离下降，说明动作数量不能直接当作纠错成功的证据。P1 更支持 H1（初始局部编辑/token completion 更好），不支持 H2（后续纠错更强）。

## 3. P2：首 completion 的受控干预

在每条轨迹首个可用的 oracle INS/SUB 位置，保持位置和动作类型不变：

- `force-correct`：使用 oracle support 中的正确 token；
- `force-wrong`：使用同一位置/类型上模型认为合法、但排除 oracle support 后的最高概率 token。

| 模型 | 条件 | 干预覆盖率 | 最终命中 | 最终有效 | 后续 SUB/DEL | 后续距离下降 |
|---|---|---:|---:|---:|---:|---:|
| Atom | force-correct | 99.16% | 53.22% | 86.93% | 0.387 | 3.039 |
| Atom | force-wrong | 99.19% | 3.90% | 76.80% | 0.425 | 2.328 |
| SPE-M500 | force-correct | 100.00% | 51.17% | 87.59% | 1.177 | 2.121 |
| SPE-M500 | force-wrong | 100.00% | 2.47% | 79.43% | 1.187 | 1.809 |

正确首 completion 到错误首 completion 的最终命中下降为：Atom `49.32 pp`，M500 `48.70 pp`。M500−Atom 的下降差异为 `−0.63 pp`，95% CI `[−2.98, +1.73]` pp，跨过 0。

### P2 类型分层

| 模型 | 条件 | INS 数/命中 | SUB 数/命中 |
|---|---|---:|---:|
| Atom | force-correct | 23,982 / 54.67% | 1,619 / 43.85% |
| Atom | force-wrong | 24,003 / 4.15% | 1,609 / 2.11% |
| SPE-M500 | force-correct | 21,687 / 52.06% | 4,357 / 47.62% |
| SPE-M500 | force-wrong | 21,658 / 2.47% | 4,355 / 2.71% |

M500 的 fragment 表示确实覆盖到更多 SUB completion，但 force-wrong 后的命中率仍只有约 2.5%。因此“后续 SUB/DEL 数量更多”并没有转化为“错误首 token 更容易被纠正”。跨 tokenizer 的 atom/fragment 错误破坏程度并不完全相同，所以这里主要使用各模型自身的 correct→wrong drop；该 drop 没有显著差异。

## 4. 数据与实现审计

- P0 自动测试：`39 passed, 12 warnings`。
- compact recorder 与普通 Euler 的最终 token 输出在 smoke 的全部 batch 中完全一致。
- P1：2 个模型 × 3 seeds，每模型 27,000 条自然轨迹。
- P2：2 个模型 × 2 个干预条件 × 3 seeds，每个条件 27,000 条轨迹。
- 对每个模型/seed，correct 与 wrong 的首事件 anchor 在有干预的配对轨迹中位置、类型、oracle token 完全一致；无 anchor 的情况仅来自未检测到可干预首事件。
- 结果保存在 `revision/results/natural/` 与 `revision/results/intervention/`，汇总脚本分别为 `scripts/summarize_trajectory_correction.py` 和 `scripts/summarize_trajectory_intervention.py`。

## 5. 最终结论与停止决定

本轮没有证据支持“**SPE-M500 的优势主要来自首错后的 SUB/DEL 纠错能力**”。更准确的结论是：SPE-M500 的优势更接近于降低首个局部编辑/token completion 出错率；一旦首 completion 被强制改错，后续轨迹通常无法修回，无论 Atom 还是 M500 都会显著掉点，M500 也没有更强的恢复能力。

因此暂不进入“专门强化后续纠错训练”的分支，不追加纠错 loss、错误注入训练或 checkpoint/sampler 搜索。后续若继续做推理改进，仍以 `SPE-M500@490K`（改进后 global R-SMILES，普通 Euler N=9）作为 fragment-level baseline；改进目标应优先放在首编辑质量、候选覆盖和采样效率，而不是假设模型能可靠修复任意错误首 token。

详细执行计划及停止规则见 [`plan.md`](plan.md)，P1/P2 原始汇总分别见 [`results/natural/summary/summary.md`](results/natural/summary/summary.md) 和 [`results/intervention/summary/summary.md`](results/intervention/summary/summary.md)。

---

# 修正后的最终结论

请以 [`motivation_report.md`](motivation_report.md) 和
[`results/intervention_order_invariant_v3_summary/summary.md`](results/intervention_order_invariant_v3_summary/summary.md)
为准。修正后的 P1 不再把某条对齐路径当成唯一正确顺序；修正后的 P2 只接受实际首事件
距离增加 1 的 token 干预。结果仍不支持 SPE-M500 主要依靠后续 SUB/DEL 纠错，后续 baseline
仍为改进后 global R-SMILES 上的 SPE-M500@490K + Euler N=9。
