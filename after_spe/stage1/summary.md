# Stage 1 总结：reaction-center-aware 推理路线

日期：2026-08-25
状态：**oracle 中心在 R9K1M2 下有效；尚未训练或评估可部署的 product-only 中心预测器。**

## 这份目录在研究什么

`after_spe/stage1/` 专门记录 **reaction-center-aware 推理/采样**：如果知道产物中真正会发生化学变化的位置，能否让模型在生成反应物时更好地完成第一次编辑。

它不是 SPE tokenizer 训练总结，也不是 product memory、DGM 或其他训练改进。目录中的“标签构建、crosswalk、token 映射”也不是训练步骤；它们只是为了把化学反应中心可靠地投影到模型实际编辑的 M500 token 位置。

当前模型和表示固定为：

```text
improved global R-SMILES + SPE-M500@490K
```

中心引导的共同规则为：只在每条轨迹的**首个真实编辑发生之前**，把 INS / SUB / DEL 各自的位置概率轻微移向中心及一跳邻域；每个操作的总编辑强度、token completion 分布和后续步骤均保持原样。它是软偏向，不是强制 mask。

## 为什么值得尝试

真实反应中心是产物中发生断键、成键、键级/电荷/手性变化，或新片段接入的位置。训练数据的静态回看显示，编辑入口高度集中在这里：

| 静态统计（global R-SMILES + M500，train 中固定 1,000 条 reaction × 20 views） | 数值 |
|---|---:|
| 中心及一跳邻域占全部 M500 token 的比例 | 28.737% |
| 真实已有-token 编辑位置落在该区域的比例 | 91.088% |
| 真实新增片段插入入口落在该区域的比例 | 96.343% |

这说明中心附近是很强的位置先验：只关注约 29% 的 token，已覆盖绝大多数真实编辑入口。但它只是“值得做端到端实验”的理由，不等于模型一定能得到更高准确率。

## 数据和映射是否可靠

在训练和验证数据上，raw atom-mapped reaction 与当前 `#global#` 表示均可可靠对应：

| split | processed reaction block | 成功对应 | 未匹配 |
|---|---:|---:|---:|
| train | 40,003 | 40,003 | 0 |
| val | 5,001 | 5,001 | 0 |

M500 provenance 映射在固定的 1,000 条 reaction、20,000 个 augmentation view 上也全部成功，且重放 SPE 合并后与保存的 M500 token 逐 token 一致。故后续实验确实是在“中心附近的位置”施加偏向，而不是因 token 对应错误产生表面收益。

详情见 [s1_center_label_report.md](s1_center_label_report.md)、[s2_mapping_report.md](s2_mapping_report.md) 和 [rc0_locality.md](rc0_locality.md)。

## 实验一：普通 Euler 下，真实中心是否改善第一次编辑

三个组的含义：

- **B0**：普通 Euler N=9，不使用中心信息；
- **B1**：9 条轨迹都使用真实反应中心的首编辑软偏向，属于 oracle；
- **B2**：把同样的偏向施加到同一产物中的错误局部区域，作为负对照，也属于 oracle。

所有组使用相同模型、9 条轨迹、100 steps、seed 42 和 20 augmentation 聚合。B1 的首次编辑更常在真实中心附近，也更常让当前 token 序列接近真实目标：

| split | B0：首步靠近真实中心 | B1：首步靠近真实中心 | B0：首步后更接近目标 | B1：首步后更接近目标 |
|---|---:|---:|---:|---:|
| dev-1000 | 84.72% | 88.17% | 65.29% | 67.10% |
| confirm-1000 | 84.53% | 88.06% | 66.14% | 67.96% |

因此，“正确的化学位置能改善首步”在两个独立 1,000-reaction split 上均复现。B1 也优于错误区域 B2 的 Top-1：dev 为 `+2.2 pp`，confirm 为 `+1.3 pp`；这说明收益不是任意提高一小块位置的概率就会出现。

### 端到端准确率：全 B1

| split | 方法 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid@1 | 真正不同候选 / reaction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dev-1000 | B0 | 60.1% | 76.6% | 80.5% | 83.7% | 90.0% | 12.850% | 24.719 |
| dev-1000 | B1 | **61.2%** | **77.0%** | 80.1% | 83.7% | 89.2% | 12.625% | 24.457 |
| confirm-1000 | B0 | 58.5% | 77.6% | 81.5% | 84.9% | 90.0% | 11.830% | 24.625 |
| confirm-1000 | B1 | **60.0%** | **78.2%** | **81.8%** | **85.0%** | **90.1%** | 12.330% | 24.276 |

正确解读不是“B1 失败”：它在两个 split 上的 Top-1 均提高，confirm 上各个 Top-K 的点估计也均为正。限制在于每个 split 只有 1,000 条 reaction，B1−B0 的准确率 bootstrap 区间仍包含 0；同时真正不同候选数持续下降（confirm 为 `−0.349`，95% CI `[-0.630, -0.063]`）。也就是说，B1 有明确的 Top-1 和首步机制信号，但会使九条轨迹更集中，深层 Top-K / Oracle 的收益不够稳定。

完整数据见 [stage1_report.md](stage1_report.md)、[rc1_confirm1000_report.md](rc1_confirm1000_report.md)。

## 实验二：能否保留部分普通轨迹来恢复多样性

为避免 9 条轨迹都集中到中心，固定尝试了 **RC1.5：3 条 B1 引导轨迹 + 6 条普通 Euler 轨迹**。它没有扫描更多倍率或比例。

| split | RC1.5 − B0 Top-1 | Top-3 | Top-5 | Top-10 | Oracle | 结果 |
|---|---:|---:|---:|---:|---:|---|
| dev-1000 | +1.1 pp | +0.6 pp | +0.6 pp | +0.5 pp | +0.2 pp | 点估计全面正向 |
| confirm-1000 | +0.7 pp | −0.7 pp | −1.0 pp | −0.5 pp | −0.7 pp | 覆盖指标反向，未通过确认 |

RC1.5 的意义是一个重要的负结果：把一部分轨迹恢复为普通 Euler 的确略微恢复了候选多样性，但没有稳定保住 Top-K / Oracle。**停止的是这种混合 Euler 设计和其后的比例扫参，不是“真实中心对首步无用”这一机制结论。**

详情见 [rc15_dev1000_report.md](rc15_dev1000_report.md) 和 [rc15_confirm1000_report.md](rc15_confirm1000_report.md)。

## 实验三：R9K1M2 下的完整 test oracle 验证

随后将同一 B1 机制接入更强的 R9K1M2 采样器：9 条独立 run，每条 run 在每一步产生两个 child，沿用原有 K=1/M=2 child 选择规则。B1 仍只作用到每条 selected lineage 的首个真实编辑之前，且不改变每个 mode 的总编辑强度。

使用同一 SPE-M500@490K checkpoint，在完整 test（5,007 条 reaction × 20 views）上与已有普通 R9K1M2 B0 直接比较：

| 方法 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid@1 |
|---|---:|---:|---:|---:|---:|---:|
| B0：普通 R9K1M2 | 61.434% | 78.670% | 82.225% | 85.480% | 90.913% | 11.921% |
| B1：oracle 首编辑中心偏向 | **62.592%** | **79.529%** | **83.104%** | **86.079%** | **91.232%** | **11.614%** |
| B1 − B0 | **+1.158 pp** | **+0.859 pp** | **+0.879 pp** | **+0.599 pp** | **+0.319 pp** | **−0.307 pp** |

这组结果是当前最强证据：中心偏向不仅改善 Top-1，也一致改善所有 Top-K、Oracle 和 Invalid。增益从 Top-1 向 Oracle 递减，说明其主要作用是让正确路径更早、更稳定地被优先生成，而不是大幅扩张模型原本完全无法覆盖的反应空间。

实现诊断也通过：5,007 条 test reaction 均成功映射；100,140 个 view 均有可用中心标签；901,260 条输出符合既定预算；99.11% 最终轨迹记录到首个真实编辑；总编辑强度的最大相对守恒误差为 `2.50e-6`。详情见 [r9k1m2_b1_readiness.md](r9k1m2_b1_readiness.md)。

## 总体结论

1. **反应中心是有效的编辑位置先验。** 静态数据、首次编辑诊断、B2 错误位置对照和 R9K1M2 全量 test 共同支持这一点。
2. **不能把普通 Euler 的 RC1.5 负结果误读成 B1 无效。** 全 B1 在两个小 split 上已有正向 Top-1 信号；失败的是“3 条引导 + 6 条普通”的混合策略未能稳定改善覆盖。
3. **R9K1M2 是更匹配中心偏向的候选生成器。** 在完整 test 上，B1 对 B0 的所有核心指标方向一致地变好，因而不应放弃 reaction-center-aware 路线。
4. **当前 B1 仍不可部署。** 它读取真实反应物才能得到中心，实际预测时没有这项信息；因此现有结果只能说明“如果中心能从 product 得到，它有潜在价值”。

## 下一步：Stage 2 要做什么

下一步不再扫描 oracle B1 的倍率、轨迹比例或 checkpoint，也不再尝试普通 Euler 的 RC1.5。要回答的新问题是：

> 只看 product，能否预测出足够好的反应中心，使 R9K1M2 获得 oracle B1 的一部分收益？

建议按以下顺序推进：

1. **训练 product-only 反应中心预测器。** 只使用 train split 的中心标签；输入是 product 图，输出每个原子/键属于反应中心的概率。dev、confirm 和 test 都不能参与训练。
2. **先做离线预测质量检查。** 在 dev 与 confirm 上报告中心/邻域覆盖、precision-recall，以及把预测中心投影到 M500 token 后的编辑入口覆盖；与本 Stage 1 的 oracle 覆盖作差，确认 predictor 没有退化到无信息位置。
3. **接入固定的 R9K1M2 B1。** 保持 checkpoint、100 steps、seed、倍率 3 和采样预算不变，仅将 true-center sidecar 换成 predicted-center sidecar；和普通 R9K1M2 B0 做配对比较。
4. **预先设定继续门槛。** predicted-center B1 必须在 dev 与 confirm 都保持正向的 Top-1，并且不能以 Top-5 / Oracle / true-unique 的明显下降换取该收益；达到后才冻结实现。

完整 test 已经用于 oracle 机制验证，因此后续 predictor 的结构、阈值和校准只能在 train/dev/confirm 上决定，不应再反复用 test 挑选设计。未来再次报告 test 时，应明确它是已观察过的参考集，而非新的盲测。

## 文档导航

| 文件 | 内容 |
|---|---|
| [s1_center_label_report.md](s1_center_label_report.md) | raw reaction 与 global 数据的对应、中心标签质量 |
| [s2_mapping_report.md](s2_mapping_report.md) | 化学中心如何映射到 M500 token / 插入入口 |
| [rc0_locality.md](rc0_locality.md) | 静态局部性与编辑入口覆盖 |
| [stage1_report.md](stage1_report.md) | 普通 Euler B0/B1/B2 与 RC1.5 的完整过程 |
| [rc1_confirm1000_report.md](rc1_confirm1000_report.md) | 全 B1 的独立 confirm 数据 |
| [rc15_confirm1000_report.md](rc15_confirm1000_report.md) | RC1.5 的独立 confirm 负结果 |
| [r9k1m2_b1_readiness.md](r9k1m2_b1_readiness.md) | R9K1M2 + B1 的实现、full test 与 B0 对比 |
