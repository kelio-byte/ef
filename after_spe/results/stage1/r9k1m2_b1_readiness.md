# R9K1M2 + B1：实现与全量 test 结果

日期：2026-08-24。本文记录 B1 接入 R9K1M2 的实现、验证和一次完整 test 的 oracle 结果。

## 为什么做这件事

先前 B1 只在普通 Euler N=9 上测试过：它在每条轨迹的第一个真实编辑发生前，利用真实反应中心提高中心附近位置被编辑的机会。用户希望再用当前更强的 R9K1M2 推理策略做一次完整 test 的 oracle 对照，判断“反应中心信息”在这个候选生成策略下是否仍有价值。

这仍是一个 **oracle 上界实验**。B1 的中心来自真实反应物/答案，实际部署时并不知道它，因此无论结果好坏，都不能当作可直接使用的推理方法。

## R9K1M2 中的 B1 到底怎么接入

对每个输入表示，R9K1M2 有 9 条相互独立的 run；每条 run 每一步从一个 parent 产生 2 个 child，再按原有 K=1、M=2 规则留下一条。

- 给这 9 条 run 分别分配原来 B1 使用的中心组件分数。
- 一条 run 在**还没有发生任何真实 INS / SUB / DEL**之前，对位置速率施加 B1 偏向：中心位置分数为 1、相邻位置分数为 0.5、其他位置为 0；最高相对倍率为 3。
- INS、SUB、DEL 三个 mode 各自的合法总速率保持不变；只是在该 mode 内把概率从远处位置移向中心附近。token 的完成分布 Q 不变。
- 两个 child 都从这个同一个“首编辑偏向后的”分布采样；child 的原有选择规则（状态变化 bonus、seed 平局规则）没有改动，也不在 9 条 run 之间竞争。
- 被选中的 lineage 一旦发生第一组真实编辑，就永久回到普通 R9K1M2。若选中的是 no-op child，则下一步仍可使用 B1。

实现只接受冻结的 R9K1M2 协议：`n_runs=9, n_branches=1, n_children=2, full_probability, stochastic_noop, changed_state_bonus=0.5, q_temperature=1.0`，并拒绝混入 first-edit-diversity、forward sharing 或其他 beam 设置，避免误把不同方法称为 R9K1M2。

## 已完成验证

| 检查 | 结果 |
|---|---|
| 全量自动测试 | `450 passed` |
| B0 中性检查 | 在 dev 的 1 个完整 aug20 block（20 views，180 条 R9 输出）上，普通 R9K1M2 与 B1 倍率=1 的预测逐字节一致 |
| B1 是否真的生效 | 倍率=3 时，180/180 条最终 lineage 记录到首事件，180/180 条的首事件位置速率确实发生重加权 |
| hazard 守恒 | 最大相对误差 `1.754e-6` |
| 输出预算 | 20 views × 9 runs = 180 条预测，符合 R9K1M2 协议 |
| 诊断开销 | B1 的 `full` 事件记录与默认 `summary` 记录产生完全相同的预测；后者只保留计数，不保存每条轨迹的完整事件 JSON |
| full-test sidecar 构建器 | 旧的 dev 小样本（1 reaction × 20 views）构建成功，20/20 状态正常；构建器已支持完整 test 的所有 augmentation block |
| 完整 test 映射 | 5,007/5,007 条反应一一匹配；100,140/100,140 个 augmentation view 的中心标签均为 `ok` |

小样本只用于检查代码和协议，不能据此推断准确率提升。

## 全量 test 的冻结协议

| 项目 | 值 |
|---|---|
| checkpoint | `new_checkpoints/spe_m500_checkpoints/checkpoint_step490000.pt` |
| 数据 | `datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/test` |
| 采样 | R9K1M2，`n_steps=100`，seed=42，cubic scheduler |
| B1 | true/oracle center，最高倍率 3，只用于每条 selected lineage 的首个真实编辑前 |
| test 规模 | 100,140 个 augmentation views = 5,007 reactions × 20；预期 901,260 条原始预测（每 view 9 条） |
| 事件诊断 | `summary` 模式：保留首事件数量、重加权数量、角色计数和 hazard 误差；避免保存约 90 万条逐轨迹 JSON |

## 完整 test 执行结果

已按上述冻结协议运行。CPU 预处理先将 5,007 条 raw reaction 与 `#global#` 的 test block 一一对齐，再为 M500 的 100,140 个 view 生成 oracle 中心 sidecar；无缺失、无 sidecar 错误。采样输出数为预期的 901,260（5,007 reactions × 20 views × 9 runs），采样和评分均以 exit code 0 完成。

| 指标 | 结果 |
|---|---:|
| Top-1 | 62.592% |
| Top-3 | 79.529% |
| Top-5 | 83.104% |
| Top-10 | 86.079% |
| Oracle-any | 91.232% (4,568 / 5,007) |
| Invalid@1 | 11.614% |
| Mean valid candidates / reaction | 158.831 |
| Mean true-unique candidates / reaction | 24.795 |
| Sampling + scoring wall time | 2 h 13 m 30 s |

中心偏向的运行诊断也正常：

- 901,260 条最终轨迹中，893,266 条（99.11%）记录到首个真实编辑；其余 7,994 条在 100 步内没有真实编辑。
- 9 条 run 都使用 B1，`first_event_position_only=true`；一旦选中的 lineage 发生首个真实编辑，后续立即回到普通 R9K1M2。
- 每个 mode 的总合法速率保持不变；最大相对误差为 `2.496e-6`，说明概率重分配没有意外改变编辑总强度。

结果文件（未纳入 Git，因为含 90 多万条预测）位于：

```text
results/after_spe_stage1/r9k1m2_b1/r9k1m2_b1_fulltest_20260825_full/
```

此前的“获准后如何执行”命令保留如下，供复现使用：

```bash
python scripts/build_reaction_center_labels.py \
  --processed_dir datasets/USPTO_50K_PtoR_aug20_#global# \
  --splits test --workers 8

python scripts/build_center_bias_sidecar.py --all_processed_blocks \
  --global_products datasets/USPTO_50K_PtoR_aug20_#global#/test/src-test.txt \
  --m500_products datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/test/src-test.txt \
  --raw_csv datasets/USPTO_50K/raw_test.csv \
  --labels results/after_spe_stage1/cache/reaction_centers_test.jsonl \
  --crosswalk results/after_spe_stage1/cache/raw_to_processed_test.jsonl \
  --output_dir results/after_spe_stage1/center_sidecars/test_all_aug20 \
  --workers 8
```

随后使用：

```bash
ALLOW_FULL_ORACLE_TEST=YES bash scripts/run_r9k1m2_b1_oracle.sh full
```

该脚本在没有 `ALLOW_FULL_ORACLE_TEST=YES` 时会拒绝全量运行，以防误启动。

## 当前结论

完整的 **oracle** R9K1M2+B1 test 已完成，且中心偏向在全量数据上确实被应用、没有破坏既有 R9K1M2 的模式总速率或输出预算。上表是“若已知真实反应中心，首编辑位置偏向可达到什么表现”的上界数据。

不过，这次运行本身没有附带相同 checkpoint、相同 test、相同 seed/protocol 的 B0 全量结果；本机也未找到该 B0 全量结果文件。因此不能仅从 Top-1=62.592% 推出 B1 带来了多少提升，更不能把 oracle B1 当作可部署方法。下一步若要判断 B1 是否值得继续，应把这次输出与匹配的 B0 full-test 指标做逐项差分；若 B1 明显更好，才值得进一步替换 oracle 中心为可从 product 得到的预测中心。
