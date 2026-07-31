# 下一步 TODO List

> 综合 `analysis_first_step_vs_trajectory.md`、`analysis_trajectory_20260728_173057.md`、`analysis_20aug_10run.md` 三份报告, 按优先级排列。

---

## P0 — 快速修复 (本日可完成)

- [ ] **Event correctness 增加 token 检查**
  - 当前只检查位置+类型 (INS/SUB/DEL), 不检查 token
  - 导致 #16780 那样 4/4 events "correct" 却 MISMATCH 的误导
  - 修改 `visualize_trajectory.py` 中 `_build_example_section` 的 event 判定逻辑
  - 来源: [analysis_first_step](analysis_first_step_vs_trajectory.md#32-event-correctness-的盲区), [analysis_traj](analysis_trajectory_20260728_173057.md#34-事件正确但结果错误)

- [ ] **First-step time_grid 扩展默认值**
  - 当前 `"0,0.1,0.3,0.5"` 漏掉 t>0.5 的编辑 (产物 4837 的编辑在 t=0.89~0.98)
  - 默认改为 `"0,0.1,0.3,0.5,0.7,0.9"` 或全自动生成
  - 来源: [analysis_first_step](analysis_first_step_vs_trajectory.md#210-产物-4837)

---

## P1 — 深度分析 (本周)

- [ ] **分析产物 1828 的"事件爆炸"**
  - 平均 16.2 次编辑, 是其他产物的 3~16 倍
  - 打开 trajectory HTML, 检查是否在同一位置反复做 INS/SUB/DEL
  - 如果确认是"试错-修正"模式 → 考虑优化 token 预测的确定性
  - 来源: [analysis_20aug](analysis_20aug_10run.md#6-产物-1828-的事件爆炸)

- [ ] **分析产物 1143 的失败根因**
  - 唯一在 200 次采样下仍只有 70% 命中的产物
  - per-sample 命中率仅 0.7%, 且大量 augmentation 产生 0 事件
  - 对比 first_step 和 trajectory, 确认模型是"不知道编辑哪里"还是"编辑了但都错"
  - 来源: [analysis_20aug](analysis_20aug_10run.md#3-scorepy-模拟)

- [ ] **深入分析"挣扎"行为**
  - 产物 204/712/2253: **失败时编辑数 > 成功时编辑数**
  - 说明模型在错误的编辑路径上越走越远
  - 对比 success/fail 的编辑序列差异, 看第一步错在哪里
  - 来源: [analysis_20aug](analysis_20aug_10run.md#5-编辑事件分析-成功-vs-失败)

---

## P2 — 实验验证 (下周)

- [ ] **验证非均匀时间步长**
  - 当前 100 步均匀分布, 46% 编辑挤在 t>0.8
  - 尝试: 前 80 步稀疏 (step to t=0.5), 后 20 步密集 (step to t=1.0)
  - 对比 match rate 和事件时间分布变化
  - 来源: [analysis_traj](analysis_trajectory_20260728_173057.md#31-编辑事件时间分布)

- [ ] **验证多 augmentation 投票策略**
  - 当前 per-sample 命中率低 (26.3%), 但 200 次下几乎必然命中
  - 测试: n_samples=5, 跨 20 augs 投票取 consensus
  - 对比: n_samples=100 单 aug 的命中率
  - 来源: [analysis_20aug](analysis_20aug_10run.md#3-scorepy-模拟)

- [ ] **修改 score.py 支持独立采样排序**
  - 当前 `1/(position+1)` 权重假设 beam search (位置 0 最优)
  - 对 Euler 独立采样, 位置是任意的, 应用均匀权重或纯频次排序
  - 对比新旧排序对 Top-1 ACC 的影响
  - 来源: [session-handoff 讨论]

---

## P3 — 长期改进

- [ ] **训练时增加 DELETE 操作权重**
  - 产物 712 的纯 DELETE 操作全败, 模型倾向 INSERT
  - 检查训练数据中 DELETE 的比例, 考虑 re-weighting
  - 来源: [analysis_first_step](analysis_first_step_vs_trajectory.md#33-编辑类型偏好)

- [ ] **降低 augmentation 敏感性**
  - 产物 4837 的 CV=161%, 不同 SMILES 排列差异 10 倍
  - 可能的方案: 训练时对同一产物使用更多 atom ordering
  - 来源: [analysis_20aug](analysis_20aug_10run.md#4-augmentation-敏感性)

- [ ] **添加"零事件"诊断**
  - 当 100 步产生 0 个编辑事件时, 自动输出警告
  - 帮助区分"模型静默"与"采样未触发"
  - 来源: [analysis_traj](analysis_trajectory_20260728_173057.md#35-零事件样本)
