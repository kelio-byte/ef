# 新训练 checkpoint：validation 参数消融与 mini-1001 前冻结

日期：2026-08-07

## 1. 目的

新 checkpoint 在 A6000 上重新训练完成后，不能直接沿用旧 checkpoint 上为
Euler-Beam 选出的参数。tiny 只有 50 个完整反应，适合做加载和布局回归，不适合选
参数。因此本轮先在 validation 的前 200 个完整反应（4000 条 augmentation 输入）上，
固定总输出预算为 9 条/产物，比较 R/K、后继数 M、child policy、Q temperature 和
changed-state bonus；随后冻结一套配置，再在 test mini-1001 上做较大规模确认。

所有 validation 采样均使用：100 Euler steps、batch=64、seed=42、
`full_probability`、TF32 matmul `high`、相同状态 forward sharing、augmentation=20、
Top-1～10、legacy best-rank aggregation。预测和 diagnostics 暂存于
`/tmp/ef_euler_beam_sweep_20260807/`，因为项目挂载点在本轮实验时没有可用空间；仓库中
没有覆盖或删除历史结果。

## 2. 重新训练前实际修改了什么

新 checkpoint 的模型结构、训练目标和主要超参数没有改变：hidden=256、10 层、8 heads、
FFN=2048、dropout/attention dropout=0.3、batch=128、600k updates、Noam warmup=8000、
cubic flow scheduler、`use_origin_mask=False`。权重中仍没有 `origin_embedding`。其嵌入
config 实际记录 `num_workers=4`（模板 `retro_v2.yaml` 默认 2），这只改变 DataLoader
吞吐/预取，不改变模型目标；validation 仍从 100k 开始、每 20k 一次。

会改变模型参数轨迹的改动有两类：

1. **Noam 更新顺序修复（实质性）**：旧脚本在第一次 `optimizer.step()` 后才调用
   `NoamScheduler.step()`，所以第一次更新使用 Adam 默认 `1e-3`，而不是 Noam 的
   step-1 学习率（约 `8.73e-8`）。新脚本先设置 update `n` 的学习率，再做该 update。
   这会改变整个优化轨迹，因此新 checkpoint 不是旧 checkpoint 的 bitwise 重训。
2. **可复现的数据顺序与 RNG（实质性但非目标函数改动）**：加入 seed=42、按
   `seed+epoch` 的可恢复 permutation、Python/NumPy/PyTorch CPU/CUDA/DataLoader RNG 保存与
   resume。它固定了新训练中的随机轨迹；旧 checkpoint 没有这些状态，无法追溯其原始
   batch 顺序。

以下改动不直接改变新模型的每次 optimizer 更新：

- validation、TensorBoard、最佳 checkpoint、时间戳日志和 checkpoint 字段只增加监控/恢复
  能力；validation 前后恢复 RNG。
- raw/aligned 文件的行数 fail-fast 和 fallback alignment 的 PAD 修复只在相应数据分支
  触发。本次训练使用已有预对齐 train 文件，故没有证据表明它改变了这次 600k 的训练
  样本。
- `scripts/sample_retro.py` 的 `torch.load(..., weights_only=False)` 是 PyTorch 2.6+
  采样加载兼容修复，不影响训练权重或采样数学逻辑。

因此新旧性能差异不能只归因于 Euler-Beam 参数，也不能只归因于“模型变差”；必须在同一
checkpoint、同一协议下选采样参数，再用更大数据确认。

## 3. 新旧 checkpoint 的同协议证据

仓库已有的旧 checkpoint validation-200 R3K3M2 结果，与本轮新 checkpoint 使用相同的
前 200 个 validation 反应、M=2、bonus=0.5、`stochastic_noop`、T=1.0 和 9 条输出预算。

| checkpoint | Top-1 | Top-3 | Top-10 | Oracle | rank-1 invalid | sampling wall |
|---|---:|---:|---:|---:|---:|---:|
| 旧 `checkpoint_step600000.pt` | 64.5% | 85.0% | 90.5% | 94.5% | 5.85% | 482.18 s |
| 新 `new_checkpoints/checkpoint_step600000.pt` | 62.0% | 83.5% | 89.0% | 94.0% | 5.675% | 335.57 s |

新模型在该小片段上的 Top-1/3/10 分别低 2.5/1.5/1.5 个百分点，Oracle 低 0.5 个百分点；
invalid 基本相同。wall 的下降来自当前采样代码/环境，不应解释成模型准确率提升。200 个
反应仍有抽样误差，不能据此废弃新 checkpoint。

## 4. 新 checkpoint validation-200 消融结果

| 配置（总输出均为 9） | Top-1 | Top-2 | Top-3 | Top-10 | Oracle | rank-1 invalid | valid / true-unique | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R9K1M2, no-op, T=1.0, bonus=.5（基准） | 70.0 | 84.5 | 90.5 | 93.5 | 95.5 | 12.10% | 158.50 / 21.255 | 411.64 s |
| R3K3M2, no-op, T=1.0, bonus=.5 | 62.0 | 77.5 | 83.5 | 89.0 | 94.0 | 5.675% | 143.335 / 21.315 | 335.57 s |
| R9K1M2, stochastic, T=1.0, bonus=.5 | 70.0 | 84.5 | 91.0 | 93.5 | 95.5 | 12.075% | 158.50 / 21.245 | 407.77 s |
| R9K1M2, no-op, T=0.9, bonus=.5 | 69.5 | 85.0 | 91.5 | 94.0 | 95.5 | 11.75% | 159.16 / 20.590 | 410.08 s |
| R9K1M1, stochastic, T=1.0, bonus=.5 | 69.5 | 84.5 | 88.5 | 92.5 | 96.0 | 11.325% | 159.725 / 20.940 | 319.30 s |
| R9K1M2, no-op, T=1.0, bonus=.8 | 70.0 | 84.5 | 90.5 | 93.5 | 95.5 | 12.10% | 158.50 / 21.255 | 415.76 s |
| R1K9M2, stochastic, T=1.0, bonus=.5 | 61.0 | 76.5 | 80.5 | 87.0 | 90.0 | 2.575% | 134.80 / 24.865 | 288.18 s |

`bonus=.5` 与 `.8` 的 prediction SHA-256 完全相同（分别为
`2be7a1a2847d4902dbfe518757867e2661b929b940ed94f0bc651f4b2434a6f0`），说明本轮 bonus
没有改变排序，而不是仅仅由于四舍五入得到相同指标。

## 5. 解释与冻结决策

- **R/K 组织方式影响最大**：在新 checkpoint 上，R9K1M2 比 R3K3M2 高 8.0/7.0/4.5 个
  Top-1/3/10 百分点，Oracle 也高 1.5 个百分点；R3K3 快约 76 秒。当前不把“九个分支”
  视为完全等价的实现。
- **M=2 有排序收益**：相对 M=1，Top-3 提升 2.5、Top-10 提升 1.0 个百分点，Top-1
  提升 0.5；Oracle 没有提升，wall 增加约 92 秒。多个后继主要改善候选排序，而非保证
  找到更多真实模式。
- **child policy 不是主要因素**：`stochastic` 与 `stochastic_noop` 的 Top-1、Top-10、
  Oracle 相同，Top-3 只差 0.5 个百分点。
- **T=0.9 只产生轻微尾部变化**：Top-1 下降 0.5，Top-3/10 上升 1.0/0.5，Oracle 不变；
  在 200 个反应上不足以证明应改变默认概率语义。
- **bonus 不能直接从旧 checkpoint 迁移出收益**：0.5 和 0.8 逐行一致，当前数据上没有
  可观测作用。

因此 mini-1001 冻结配置为：**R9K1M2、full_probability、stochastic_noop、bonus=0.5、
Q temperature=1.0、TF32 high、batch=64、seed=42、100 steps**。这不是根据 mini target
调参，而是 validation-200 上预先确定的基准配置；mini 只用于更大规模确认。

## 6. mini-1001 实验记录（进行中）

- 输入：`test/src-test-mini-1001.txt` / `tgt-test-mini-1001.txt`，1001 个完整反应、
  20020 条输入。
- 预测预算：`20020 × 9 = 180180` 行；评分 Top-1～10。
- 暂存目录：`/tmp/ef_euler_beam_sweep_20260807/new_ckpt_mini1001_r9k1m2/`。

采样和评分已完成，metadata 校验通过，最终 branch shortfall 为 0：

| 指标 | 结果 |
|---|---:|
| Top-1 / Top-2 / Top-3 | 58.242% / 72.028% / 77.922% |
| Top-4 / Top-5 / Top-6 | 81.019% / 83.017% / 84.016% |
| Top-7 / Top-8 / Top-9 / Top-10 | 84.515% / 85.315% / 86.014% / 86.414% |
| Oracle-any | 91.508% (916/1001) |
| rank-1 invalid | 12.767% |
| mean valid / true-unique candidates | 157.376 / 23.737 |
| mean target final rank when covered | 2.783 |
| sampling wall | 2029.18 s (33 min 49 s) |
| peak CUDA allocated / reserved | 1.73 / 2.06 GB |
| prediction SHA-256 | `3e2db73986cc476e85794d5a1e3704ef9faf449c6d9ce493a43d29345412d5ee` |

mini 是 test 的较大规模确认子集，不用于回调参数。它的 Top-1 低于 validation-200 的
70.0%，但样本规模从 200 增至 1001 后，指标不再被 tiny 的 50 个反应主导；在正式完整
src-test 前仍应把这组数字视为当前配置的探索性 test 结果，而不是训练模型的最终结论。

## 7. 当前结论

新 checkpoint 的性能变化确实与采样参数有关，但训练代码的 Noam 修复也改变了模型本身，
所以不能用旧 checkpoint 的 58～60% tiny 数字直接推断新模型。validation-200 和
mini-1001 共同支持当前冻结配置：R9K1M2、M=2、T=1.0、bonus=.5、`stochastic_noop`。
mini 结果没有提供足够理由在 test 上继续调参；下一步可在明确研究目标后运行完整
src-test，或进入独立 reward/SMC 研究，但不能把 mini 的 test target 用于二次参数搜索。
