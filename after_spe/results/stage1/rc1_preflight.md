# Stage1 RC1：GPU 前置检查

日期：2026-08-24

## 当前结论

RC1 的 CPU 准备已经完成，现在只差 GPU 上的 B0/B1/B2 推理。当前实现没有训练新模型，也没有修改 M500 checkpoint；它只在每条轨迹第一次真正发生编辑前，把同一种编辑操作的位置概率更多地分配到反应中心附近。首个非空编辑步结束后，该轨迹立即恢复普通 Euler。

## 已完成内容

- 从 atom-mapped raw reaction 提取 bond、atom 和 attachment center；
- 建立 raw reaction、global R-SMILES、SPE-M500 token 的可审计映射；
- 为 dev-1000 的 1,000 个 reaction、全部 20 个 augmentation 生成 true-center 与 same-product pseudo-center sidecar，共 20,000 行；
- 九条轨迹按 `trajectory_index % min(3, component_count)` 分配中心；
- 中心分数固定为：中心 `1.0`、一跳邻域 `0.5`、其余 `0.0`；
- 中心最大位置倍率固定为 `3.0`，INS/SUB/DEL 各自的全序列总 hazard 保持不变；
- 记录首次编辑的时间、模式、位置、completion token、中心分数和中心 component；
- `max_multiplier=1.0` 是逐 bit 中性的诊断对照，可记录 B0 的首事件而不改变采样分布。

## Sidecar 质量

| 项目 | 结果 |
|---|---:|
| reaction / augmentation 行 | 1,000 / 20,000 |
| 成功生成 | 20,000 / 20,000 |
| true/pseudo center components | 各 1,038 个 |
| pseudo center 在真实 radius-2 外 | 1,023 / 1,038 |
| pseudo center 在真实 radius-1 外 | 15 / 1,038 |
| 不得不落入真实中心 | 0 |
| scores 文件 SHA256 | `0f319ad2e9e0cc7851053ea5a428692fb188dbad8a80825ffdcfcf3bb6b41472` |

逐行 sidecar 位于 `results/after_spe_stage1/center_sidecars/dev_unique1000_aug20/`，属于可重建的运行缓存，不提交 Git；生成脚本、数据哈希和汇总报告提交 Git。

## 代码验证

- 本次中心、SPE provenance、Euler、CLI 和 metadata 相关测试：`82 passed`；
- 训练相关尾部测试分组运行：`26 passed`；
- CPU-DDP 集成测试：`1 passed, 1 skipped`；
- 全仓 433 项一次运行在约 97% 时因当前实例内存限制收到 exit 137；被杀前没有断言失败，尾部测试随后分组通过；
- `compileall` 和 `git diff --check` 通过。

真实 M500@490K checkpoint 的 CPU plumbing 使用 1 个 reaction × 20 augmentation × 9 trajectories、12 steps：

| 检查 | 结果 |
|---|---:|
| 输出行数 | 180 |
| 记录到首事件的轨迹 | 180 / 180 |
| 首事件动作数 | 197（163 条轨迹单动作，17 条双动作） |
| 中心分数 1.0 / 0.5 / 0.0 的动作 | 196 / 0 / 1 |
| 模式 | INS 145，SUB 52，DEL 0 |
| 每个 mode 总 hazard 最大相对误差 | `1.482e-6` |

这只是链路测试，`n_steps=12` 的单条反应准确率不能作为方法结果。

## GPU 后的执行门槛

先运行 10 reaction × 20 augmentation 的四组 smoke：

- B0：普通 Euler；
- B0-trace：倍率为 1 的中性诊断，预测必须与 B0 完全一致；
- B1：真实中心，倍率 3；
- B2：同 product 伪中心，倍率 3。

只有输出数量、B0 一致性、首事件覆盖、hazard 误差和无 NaN 都通过，才跑固定 100 reaction；100 reaction 没有明显退化后，再跑 dev-1000。不能跳过 smoke 直接消耗完整推理时间。
