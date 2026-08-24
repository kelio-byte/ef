# Stage1 结论：反应中心引导首次编辑

日期：2026-08-24
状态：已完成；**不进入 RC2/RC3 的中心预测器训练。**

## 一句话结论

真实反应中心能让 SPE-M500 的第一次编辑更常朝目标靠近，也明显优于故意放错位置的伪中心；但即使把真实中心直接提供给采样器，端到端收益也只有不稳定的 Top-1 `+1.1 pp`，Top-10 不变、Oracle 反而下降。因此，当前“只重分配首次编辑位置”的机制没有足够的上界空间，训练一个信息更弱的 product-only 中心预测器不值得继续投入。

这不是“反应中心没有化学信息”的结论，而是：**在当前 M500 + Euler N=9 框架中，它不足以成为下一步有效的采样改进。**

## 已完成的前置验证

Stage1 先完成了从 raw atom-mapped reaction 到 global R-SMILES、再到 SPE-M500 token 的可审计映射；dev-1000 的 20,000 个 augmentation 均成功映射。RC0 说明中心本身具有很强的局部性：radius-1 只覆盖 `28.737%` 的 M500 token，却覆盖 `91.088%` 的已有-token 编辑和 `96.343%` 的插入入口。

这说明“中心应当有用”在静态数据上成立，但不能替代端到端验证。RC1 因此只改变每条轨迹第一次真实编辑前的**位置分配**，保持 INS/SUB/DEL 各自总 hazard 和 completion 分布不变；第一次非空编辑后立即恢复普通 Euler。

## RC1 正式实验

固定协议：改进后 global R-SMILES 的 `SPE-M500@490K`、Euler `N=9`、`100` steps、seed `42`、20 augmentation、`dev_unique1000_aug20` 的 1,000 条反应。每组生成 180,000 条候选。

| 条件 | 含义 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle-any | Invalid@1 | 真正 unique / reaction | 有效候选 / reaction | 采样时间 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 普通 Euler | 60.1% | 76.6% | 80.5% | 83.7% | 90.0% | 12.850% | 24.719 | 157.155 | 24.14 min |
| B1 | **真实中心（oracle，不可部署）**，倍率 3 | 61.2% | 77.0% | 80.1% | 83.7% | 89.2% | 12.625% | 24.457 | 157.815 | 26.24 min |
| B2 | 同 product 的远离真实中心 pseudo-center，倍率 3 | 59.0% | 76.0% | 79.5% | 83.8% | 89.0% | 12.605% | 24.880 | 157.198 | 26.12 min |

`B0-trace` 用真实中心 sidecar 但倍率为 1，仅记录诊断。它与 B0 的 180,000 条预测**逐字节一致**，证明记录逻辑和中性对照没有改变普通 Euler；其采样时间为 25.58 min，仅作诊断，不计入方法效率。

所有中心条件的每个 mode 总 hazard 最大相对误差不超过 `2.48e-6`，没有 NaN。B1 相比 B0 的额外采样时间约 `8.7%`，显存开销很小（peak reserved 5.56 → 5.80 GiB）。

## 统计判断

以下为以原始 reaction 为单位、10,000 次 paired bootstrap 的差异；区间为 95% CI，单位为百分点（unique/valid 除外）。20 个 augmentation 只用于候选聚合，没有被当成独立样本。

| B1 − B0 | 点估计 | 95% CI |
|---|---:|---:|
| Top-1 | +1.1 pp | [-0.5, +2.7] |
| Top-3 | +0.4 pp | [-0.9, +1.7] |
| Top-5 | -0.4 pp | [-1.8, +0.9] |
| Top-10 | 0.0 pp | [-1.4, +1.4] |
| Oracle-any | -0.8 pp | [-2.0, +0.4] |
| true unique candidates | -0.262 | [-0.550, +0.027] |

真实中心不是随便一个位置扰动：B1 相比 B2 的 Top-1 为 `+2.2 pp`，95% CI `[+0.5, +3.9]`。这确认中心方向本身有效；问题在于它相对已经很强的普通 Euler 并没有给出稳定、可覆盖 Top-k 的上界收益。

## 第一次编辑发生了什么

“更接近/不变/更远”指第一次完整非空编辑后，相对同一 augmentation target 的 token 编辑距离变化；它不把不同的合法编辑顺序强行判为对或错。

| 条件 | 有首次编辑的轨迹 | 首次编辑落在真实中心或一跳邻域 | 距离更近 | 距离不变 | 距离更远 | 平均距离改善 |
|---|---:|---:|---:|---:|---:|---:|
| B0-trace | 96.70% | 84.72% | 65.29% | 16.38% | 18.33% | +0.475 |
| B1 true-center | 96.69% | 88.17% | 67.10% | 16.92% | 15.97% | +0.516 |
| B2 pseudo-center | 96.65% | 16.85% | 64.60% | 16.15% | 19.26% | +0.458 |

B1 的局部作用是清楚的：相比 B0，首次编辑更接近目标 `+1.82 pp`，更远的比例 `-2.35 pp`。但 B0 本身已有 `84.72%` 的首次编辑落在真实中心或一跳邻域，剩余可改善空间不大；B1 同时略微减少 candidate diversity 和 Oracle 覆盖，抵消了局部收益。

## 决策与后续边界

RC1 的必要 sanity 条件大多通过：真实中心优于 pseudo-center，局部首编辑更好，invalid 未升高。然而它只是**oracle upper bound**，相对 B0 的主要 Top-k 指标没有稳定提升，且 Oracle 下降。product-only predictor 不可能比真实中心更准确，因此没有合理依据期待它保留、放大这点微弱收益。

因此本阶段在这里停止：

- 不训练 RC2 product-only reaction-center predictor；
- 不运行 RC3 predicted-center sampler；
- 不在 confirm/final/test 上继续试，也不扫描更多 bias 强度，避免在 dev 上挖掘偶然波动；
- 保留中心标签、SPE provenance、sidecar 和 sampler 实现，作为可复现的负结果；
- 后续改进应转向不只改变首编辑位置的方向，例如 product memory、completion/候选覆盖或训练目标，而不是把资源投入当前中心-first sampler。

当前可继续使用的生成 baseline 不变：`SPE-M500@490K + ordinary Euler N=9, 100 steps`。

## 结果与复现材料

- 结构兼容性审计：[rc0_locality.md](rc0_locality.md)
- GPU 前置检查与 sidecar 质量：[rc1_preflight.md](rc1_preflight.md)
- 完整数值、hash、bootstrap 和首编辑统计：[rc1_dev1000_summary.json](rc1_dev1000_summary.json)
- 可复现运行入口：[rc1_commands.md](rc1_commands.md)
- 运行输出（未提交的大文件）：`results/after_spe_stage1/rc1_runs/20260823T225828Z_dev1000/`
