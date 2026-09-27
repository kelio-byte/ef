# FastFlow 启发的 EFRetro 推理加速实验记录

执行规则见 [任务文档](../../FastFlow_EFRetro_Acceleration_Task_Plan.md)。本文件只写已运行的实验及其结论；原始预测和逐 batch 计时保存在忽略 Git 的 `outputs/fastflow_accel/` 下。

## 固定口径

| 项目 | 值 |
|---|---|
| 代码提交 | `bf8ab00`（阶段 0 计时和划分工具） |
| 权重 SHA-256 | `7fe982dcdbea59eee999fcee7fd491664c6615a9fc9498afca02e7abbff5facf` |
| dev1000 输入 SHA-256 | `54384b145933d85ce707f2eb5b7551e4ba3d3ab9197686ea5f1664e608fd8439` |
| pilot 输入 SHA-256 | `44eaa6d653c3e49a87e78ae449bb0950df1d5c724933745ed69291381f7161e5` |
| 划分 | seed `20260926`；200 个 pilot 反应，800 个未调参反应；索引及源文件在 `outputs/fastflow_accel/split/` |
| 采样设置 | seed 42；100 步；cubic；R=9；M=2；K=1；batch size 32 |
| 环境 | NVIDIA GeForce RTX 4090；PyTorch `2.13.0+cu130`；RDKit `2026.03.4`；`PYTHONPATH=/root/autodl-tmp/efretro` |
| 时间范围 | 模型加载和输入分词后开始；包含产品编码、采样、预测写出；CUDA 同步后结束；不含评分及文件哈希 |

**Hazard 参数口径：**当前状态下，`Λ` 是模型预测的插入、替换、删除速率，在各自合法位置掩码后对全序列求和；普通步长取 cubic 调度器上限（最多 `0.04`）、`0.15 / max(Λ, 1e-8)` 和剩余时间 `1−t` 的最小值。当距 `t=1` 不超过 `0.001` 时，代码会直接走完剩余尾段，因此这最后一步可能不满足 `Λh≤0.15`。所以运行名称中的 `Λh≤0.15` 指普通步的强度限制，不代表终点补步也受该限制；详细公式和示例见[组会汇报](GROUP_MEETING.md)。

## A. 运行台账

| run_id | 阶段/方法 | 划分 | 代码提交 | 输入 SHA-256 | 输出目录 | 状态 |
|---|---|---|---|---|---|---|
| `baseline_pilot` | Full-100 | pilot200 | `bf8ab00` | `44eaa6d6...` | `outputs/fastflow_accel/baseline_pilot/` | 完成；预测 SHA-256 `17f7c190...` |
| `baseline_dev1000` | Full-100 | dev1000 | `bf8ab00` | `54384b14...` | `outputs/fastflow_accel/baseline_dev1000/` | 完成；预测 SHA-256 `f3406c06...` |
| `baseline_heldout` | Full-100 的评分子集 | heldout800 | `bf8ab00` | `2bf01200...` | `outputs/fastflow_accel/baseline_heldout/` | 完成；从 `baseline_dev1000` 提取，无额外推理；预测 SHA-256 `aca2b8e8...` |
| `profile_pilot_batch` | Full-100 诊断 profiling | pilot 中位长度批次（batch 53） | `066cb59` | `44eaa6d6...` | `outputs/fastflow_accel/profile_baseline.json` | 完成；与基线对应 288 条预测完全一致；计时含 profiler 开销，不作速度比较 |
| `trajectory_pilot` | Full-100 轨迹诊断 | pilot200 | `066cb59` | `44eaa6d6...` | `outputs/fastflow_accel/trajectory_pilot/` | 完成；预测 SHA 与基线相同；评分与 pilot 基线一致 |
| `exact_gpu_select_pilot` | Full-100 + M2 GPU 选择 | pilot200 | `ac39a8c` | `44eaa6d6...` | `outputs/fastflow_accel/exact_gpu_select_pilot/` | 完成；预测逐字节与基线一致；预测 SHA-256 `17f7c190...` |
| `static50_pilot` | Static-50 uniform + M2 GPU 选择 | pilot200 | `a1aa63f` | `44eaa6d6...` | `outputs/fastflow_accel/static50_pilot/` | 完成；预测 SHA-256 `5f58ef36...` |
| `static25_pilot` | Static-25 uniform + M2 GPU 选择 | pilot200 | `a1aa63f` | `44eaa6d6...` | `outputs/fastflow_accel/static25_pilot/` | 完成；预测 SHA-256 `557a633f...` |
| `static75_pilot` | Static-75 uniform + M2 GPU 选择 | pilot200 | `a1aa63f` | `44eaa6d6...` | `outputs/fastflow_accel/static75_pilot/` | 完成；预测 SHA-256 `e3ce07ec...` |
| `hazard04_l015_pilot` | Hazard 自适应（h≤0.04，Λh≤0.15）+ M2 GPU 选择 | pilot200 | `d99cc2a` | `44eaa6d6...` | `outputs/fastflow_accel/hazard04_l015_pilot/` | 完成；所有轨迹到达 t=1；预测 SHA-256 `89ed64dc...` |
| `hazard04_l015_dev1000` | Hazard 自适应（h≤0.04，Λh≤0.15，max NFE 200）+ Exact-M2-GPU | dev1000 | `d99cc2a` | `54384b14...` | `outputs/fastflow_accel/hazard04_l015_dev1000/` | 完成；预测 SHA-256 `ae522bdd...`；全量速度与原始 Full-100 的对比包含 M2 GPU 选择优化 |
| `hazard04_l015_heldout` | 同次 Hazard + Exact-M2-GPU dev1000 输出的 heldout800 评分子集 | heldout800 | `d99cc2a` | `2bf01200...` | `outputs/fastflow_accel/hazard04_l015_heldout/` | 完成；parent prediction SHA-256 与 dev1000 相同；无额外推理 |
| `hazard04_l015_test_attempt1` | Hazard 自适应（h≤0.04，Λh≤0.15，max NFE 200）+ Exact-M2-GPU | test | `d99cc2a` | `1b3663f6...`（100,140 个输入；目标 SHA-256 `429bc620...`） | `outputs/fastflow_accel/hazard04_l015_test_partial_200cap/` | 失败作废：完成 1,400/3,130 batches（44,800 个输入），一条轨迹触发 max NFE；未评分；部分预测 SHA-256 `75542d57...` |
| `hazard04_l015_test` | 冻结 Hazard 自适应方法（h≤0.04，Λh≤0.15，max NFE 500）+ Exact-M2-GPU | test | `d99cc2a` | `1b3663f6...`（100,140 个输入；目标 SHA-256 `429bc620...`） | `outputs/fastflow_accel/hazard04_l015_test/` | 完成；预测 SHA-256 `9529e9eb...`；max NFE 200 的首次尝试因一条轨迹超限而作废，500 上限重跑完整 |

## B. 速度与质量

| 方法 | 划分 | 采样秒数 | 相对基线加速 | 平均/中位 NFE | Top-1 | Top-3 | Top-10 | Oracle-any | invalid-at-1 | batch p95 秒 | 显存峰值 MiB |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Full-100 | pilot200 | 217.68 | 1.00× | 100/100 | 58.5% | 80.5% | 88.0% | 91.5% | 8.65% | 2.005 | 556.2 |
| Full-100 | dev1000 | 1071.01 | 1.00× | 100/100 | 61.7% | 80.7% | 88.0% | 90.4% | 8.635% | 1.953 | 650.9 |
| Full-100 | heldout800（从 dev1000 输出评分） | — | — | 100/100 | 62.375% | 80.875% | 87.625% | 89.875% | 8.69375% | — | — |
| Exact-M2-GPU | pilot200 | 188.31 | 1.156× | 100/100 | 58.5% | 80.5% | 88.0% | 91.5% | 8.65% | 1.733 | 556.3 |
| Static-50 + Exact-M2-GPU | pilot200 | 93.82 | 2.320× | 50/50 | 59.5% | 79.5% | 89.0% | 92.0% | 10.425% | 0.858 | 577.2 |
| Static-25 + Exact-M2-GPU | pilot200 | 52.68 | 4.132× | 25/25 | 60.0% | 78.0% | 89.0% | 92.0% | 16.8% | 0.523 | 555.7 |
| Static-75 + Exact-M2-GPU | pilot200 | 136.87 | 1.590× | 75/75 | 58.0% | 78.0% | 88.5% | 91.0% | 9.375% | 1.280 | 566.4 |
| Hazard + Exact-M2-GPU，h≤0.04，Λh≤0.15 | pilot200 | 161.75 | 1.346× | 38.56/33 | 59.0% | 80.5% | 88.5% | 92.0% | 8.675% | 1.977 | 513.2 |
| Hazard + Exact-M2-GPU，h≤0.04，Λh≤0.15 | dev1000 | 822.83 | 1.302× | 39.24/34 | 62.4% | 81.4% | 87.3% | 90.5% | 8.015% | 1.981 | 577.4 |
| Hazard + Exact-M2-GPU，h≤0.04，Λh≤0.15 | heldout800（同次 dev1000 预测） | — | — | —（dev1000 全量 39.24/34） | 63.25% | 81.75% | 86.875% | 89.875% | 8.04375% | — | — |

完整基线共处理 20,000 个增强输入、180,000 条轨迹；总 NFE 为 18,000,000，625 个 batch 共调用模型 62,500 次。`heldout800` 只从同一次完整预测提取对应反应并重新评分，因此没有独立采样耗时或显存值。历史 dev1000 的 1088.07 秒使用另一权重及旧计时口径，不参与加速比计算。

**全量速度的归因说明：**`baseline_dev1000` 在 `bf8ab00` 上运行，当时 M=2 子分支选择在 Python/CPU 上完成；最终 Hazard 候选在 `d99cc2a` 上运行，包含 `ac39a8c` 加入的 Exact-M2-GPU 选择优化。该优化的 pilot 预测与旧选择逐字节一致，但没有单独做全量 dev1000。因此 1.302× 是“GPU 选择优化 + Hazard 自适应步长”整套候选相对原始 Full-100 的全量速度比，不是 Hazard 单项的全量 ablation。可直接拆分的 pilot 对照为：同样使用 Exact-M2-GPU 时，Full-100 `188.31` 秒、Hazard `161.75` 秒，后者快 `1.164×`、耗时少 `14.1%`；该数值仅是 200 反应 pilot 结果。

### 配对质量差值及不确定性

候选与基线按同一反应配对，以反应为单位有放回抽样 10,000 次，随机种子 `20260926`，报告百分比点（pp）的 2.5%～97.5% percentile 区间。预先约定的质量预算仍按点估计判断；区间用于说明不确定性，不把“包含 0”作为自动通过或失败。

| 划分 | 指标 | 候选−基线 (pp) | 配对 bootstrap 95% 区间 (pp) |
|---|---|---:|---:|
| heldout800 | Top-1 | +0.875 | [−0.500, +2.375] |
| heldout800 | Top-3 | +0.875 | [−0.375, +2.250] |
| heldout800 | Top-10 | −0.750 | [−2.250, +0.750] |
| heldout800 | oracle-any | 0.000 | [−1.250, +1.250] |
| heldout800 | invalid-at-1 | −0.650 | [−1.181, −0.125] |
| dev1000 | Top-1 | +0.700 | [−0.700, +2.100] |
| dev1000 | Top-3 | +0.700 | [−0.400, +1.800] |
| dev1000 | Top-10 | −0.700 | [−2.000, +0.600] |
| dev1000 | oracle-any | +0.100 | [−1.000, +1.200] |
| dev1000 | invalid-at-1 | −0.620 | [−1.085, −0.155] |

Top-k 与 oracle 的区间均覆盖 0；这批样本没有清晰证据说明候选在这些指标上优于或劣于基线。heldout800 点估计满足原先固定的损失上限，invalid-at-1 的区间则显示候选低于基线。

### 完整 test：历史参考与冻结方案

| 方法 | 划分 | 完整采样秒数 | 相对同划分基线加速 | 平均/中位/最大 NFE | Top-1 | Top-3 | Top-10 | Oracle-any | invalid-at-1 | batch p95 秒 | 显存峰值 MiB |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 历史 100 步 test（不同 checkpoint/子候选策略；仅作参考） | test（5,007 反应；100,140 个增强输入） | 5897.02 | 不适用（非同条件基线） | 未记录逐轨迹分布（配置 100 步） | 63.751% | 82.584% | 89.734% | 92.111% | 8.908% | 未记录 | 未记录 |
| Hazard + Exact-M2-GPU，h≤0.04，Λh≤0.15（max NFE 500） | test（5,007 反应；100,140 个增强输入） | 3967.39 | 不适用（无同权重、同策略 test 基线） | 39.04/34/222 | 64.170% | 82.564% | 89.415% | 92.251% | 8.300% | 1.942 | 677.9 |

历史参考值来自 [`training_run/train_09-25/test_step500000/sampling_metadata.json`](../../training_run/train_09-25/test_step500000/sampling_metadata.json) 和 [`metrics.json`](../../training_run/train_09-25/test_step500000/metrics.json)：原始秒数 `5897.024958723923`，Top-1/3/10 为 `63.75074895146795/82.58438186538845/89.73437187936888%`，oracle-any 为 `92.11104453764729%`，invalid-at-1 为 `8.90752945875774%`。两次评分的目标文件 SHA-256 相同，聚合模式均为 `legacy_best_rank`；历史 checkpoint SHA-256 以 `2dcf037c` 开头，本轮以 `7fe982dc` 开头，历史 `child_policy=stochastic_noop`，且未记录 GPU 型号。因此表中两行可以展示实测耗时和评分，不能用于计算同条件的 test 加速比或质量变化归因。

本次完整 test 采样输出 901,260 条候选，预测 SHA-256 为 `9529e9eb87c7a80b52dd037ac1b81ff022d8a8b42df9cf42bdceae1e08b1a523`；输入和目标各 100,140 行，SHA-256 分别为 `1b3663f61f7941229bb090ce463c3f68cf3eb4bc4038870c0fc14afac5a08f79` 和 `429bc62014b60ed52b0c0e74403dd32e68fa50b2f875e96db2b7a8d4160a8f99`。采样耗时为 3967.39 秒（约 66 分 7 秒），不含评分；环境为 RTX 4090、PyTorch `2.13.0+cu130`、CUDA `13.0`。本次 NFE histogram 覆盖 901,260 条轨迹，平均 39.04、中位数 34、最大 222；模型调用 291,039 次。首次以 max NFE 200 运行在 1,400/3,130 个 batch 处因一条轨迹达到保护上限而中断，部分结果未评分；冻结配置和 hazard 参数不变，将终止保护上限提高到 500 后完整完成。test 没有同权重、同策略基线，因此只报告泛化指标，不计算 test 加速比或套用 dev 的质量预算，也没有据此调参。

复现命令（使用该仓库源码而非环境中其他可编辑安装）：

```bash
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/evaluate.py --split dev1000 --output outputs/fastflow_accel/baseline_dev1000 --record-performance --workers 8
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/accel_protocol.py score-heldout --split-dir outputs/fastflow_accel/split --full-output outputs/fastflow_accel/baseline_dev1000 --output outputs/fastflow_accel/baseline_heldout --workers 8
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/profile_sampler_batch.py --products outputs/fastflow_accel/split/pilot/src.txt --reference-predictions outputs/fastflow_accel/baseline_pilot/predictions.txt --output outputs/fastflow_accel/profile_baseline.json
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/sample.py --products outputs/fastflow_accel/split/pilot/src.txt --output outputs/fastflow_accel/trajectory_pilot --record-trajectory-diagnostics
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/score.py --predictions outputs/fastflow_accel/trajectory_pilot/predictions.txt --targets outputs/fastflow_accel/split/pilot/tgt.txt --workers 8
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/sample.py --products outputs/fastflow_accel/split/pilot/src.txt --output outputs/fastflow_accel/exact_gpu_select_pilot --record-performance
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/sample.py --products outputs/fastflow_accel/split/pilot/src.txt --output outputs/fastflow_accel/static50_pilot --n-steps 50 --time-grid uniform --record-performance
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/sample.py --products outputs/fastflow_accel/split/pilot/src.txt --output outputs/fastflow_accel/static25_pilot --n-steps 25 --time-grid uniform --record-performance
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/sample.py --products outputs/fastflow_accel/split/pilot/src.txt --output outputs/fastflow_accel/static75_pilot --n-steps 75 --time-grid uniform --record-performance
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/sample.py --products outputs/fastflow_accel/split/pilot/src.txt --output outputs/fastflow_accel/hazard04_l015_pilot --n-steps 200 --time-grid hazard --hazard-max-step 0.04 --hazard-limit 0.15 --record-performance
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/evaluate.py --split dev1000 --output outputs/fastflow_accel/hazard04_l015_dev1000 --n-steps 200 --time-grid hazard --hazard-max-step 0.04 --hazard-limit 0.15 --record-performance --workers 8
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/accel_protocol.py score-heldout --split-dir outputs/fastflow_accel/split --full-output outputs/fastflow_accel/hazard04_l015_dev1000 --output outputs/fastflow_accel/hazard04_l015_heldout --workers 8
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/evaluate.py --split test --output outputs/fastflow_accel/hazard04_l015_test --n-steps 500 --time-grid hazard --hazard-max-step 0.04 --hazard-limit 0.15 --record-performance --workers 8
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/paired_bootstrap.py --baseline-metrics outputs/fastflow_accel/baseline_heldout/metrics.json --candidate-metrics outputs/fastflow_accel/hazard04_l015_heldout/metrics.json --output outputs/fastflow_accel/hazard04_l015_heldout/paired_bootstrap.json
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/paired_bootstrap.py --baseline-metrics outputs/fastflow_accel/baseline_dev1000/metrics.json --candidate-metrics outputs/fastflow_accel/hazard04_l015_dev1000/metrics.json --output outputs/fastflow_accel/hazard04_l015_dev1000/paired_bootstrap.json
```

## C. 性能构成与轨迹诊断

| 时间段 | Transformer 占标记 CUDA 时间 | 同步函数 CPU 占比¹ | 选中状态不变比例 | 双 child 无事件比例 | 合法 hazard p50/p90 | 状态不变时 1 格强度变化 p50/p90 | 理想可省 NFE 上限 | 结论 |
|---|---:|---:|---:|---:|---|---|---:|---|
| 0.00～0.25 | 82.76% | 15.17% | 99.575% | 99.562% | 0.132 / 0.557 | 0.180 / 0.554 | 896,175（99.575%） | 计算路径主要在 Transformer；状态高度稳定，强度 1 格变化仍较大 |
| 0.25～0.50 | 82.60% | 15.99% | 97.533% | 96.973% | 1.266 / 2.908 | 0.060 / 0.081 | 877,798（97.533%） | 状态稳定，强度变化较平缓 |
| 0.50～0.75 | 83.48% | 16.23% | 95.333% | 92.336% | 3.361 / 7.861 | 0.049 / 0.054 | 857,997（95.333%） | 状态稳定，短跨度强度变化最低 |
| 0.75～1.00 | 83.23% | 16.90% | 92.114% | 86.085% | 5.487 / 21.314 | 0.078 / 0.209 | 829,025（92.114%） | hazard 上升且分布尾部变宽，后段需谨慎跳步 |

¹“同步函数 CPU 占比”是 `state_keys`、步长转 CPU 列表、编辑应用三个含同步操作函数的 CPU 总时间，占 profiling 中已标记采样函数 CPU 总时间的比例；包括这些函数里的其他主机工作，不代表纯 GPU 等待时间。Transformer 百分比以四个时间段里已标记 CUDA 函数的设备时间之和为分母。单批 Python child selection 另计 58.45 ms/28,800 次调用。profiling batch 为 pilot 按输入 token 数排序后的中位批次；包含 profiler 时总 wall time 约 7.00 秒，不作为采样基准。

强度变化是同一状态、位置对应的合法 INS/SUB/DEL 每 token 强度的相对 L1 差，分母为前一时刻强度总量；诊断每个 batch 跟踪 16 条代表轨迹。强度变化跨度补充如下：

| 时间段 | 1 格 p50/p90 | 2 格 p50/p90 | 4 格 p50/p90 |
|---|---|---|---|
| 0.00～0.25 | 0.180 / 0.554 | 0.394 / 1.403 | 0.957 / 4.361 |
| 0.25～0.50 | 0.060 / 0.081 | 0.125 / 0.172 | 0.274 / 0.392 |
| 0.50～0.75 | 0.049 / 0.054 | 0.101 / 0.110 | 0.211 / 0.230 |
| 0.75～1.00 | 0.078 / 0.209 | 0.154 / 0.387 | 0.304 / 0.668 |

hazard 每四步轮转抽样 1/4 的轨迹（每时间段 225,000 个值）；无事件率和状态变化率使用全部轨迹（每段 900,000 个轨迹步）。理想 NFE 上限把每个“选中状态未变”的转移都算作下一格可复用一次精确 forward；假设知道未来结果且复用免费，因此只能用于筛选分支，不能当作预计加速比。原始统计、分位误差样本和 profiling 明细保存在对应输出目录。

## D. 阶段决定

| 阶段 | 假设 | 已有证据 | 决定 | 下一步 |
|---|---|---|---|---|
| 0：新基线 | 同一权重和代码可以得到可复现的速度、质量口径 | pilot200：217.68 秒；dev1000：1071.01 秒，Top-1 61.7%、Top-10 88.0%；heldout800：Top-1 62.375%、Top-10 87.625%；所有轨迹均为 100 NFE | 阶段 0 完成，固定后续质量和速度对照 | 阶段 1 |
| 1：性能与轨迹诊断 | 热点及状态稳定区足以支持后续优化筛选 | Transformer 占标记 CUDA 时间 82.6%～83.5%；状态不变率 92.1%～99.6%；理想可省 NFE 上限 92.1%～99.6%；诊断输出与基线预测哈希、评分一致 | 阶段 1 完成；足以解释热点；无事件跳步及强度复用都值得小规模验证，后段 hazard/误差更高 | 阶段 2 先做保持语义的 CPU 同步/编辑开销优化；之后按计划对照 Static-50/25 |
| 2a：M2 选择工程优化 | 避免把全部子候选状态传回 CPU 并逐条 Python 分组，保留原选择结果 | pilot：188.31 秒，对比 217.68 秒；提速 13.49%；预测 SHA、Top-k、oracle、invalid 全部一致；NFE 仍为 100 | pilot 达到 Exact 保留线（逐字节一致且快至少 5%）；仅有 pilot 级证据，尚未全量确认 | 阶段 3 固定步数对照；该优化合入候选 |
| 3a：Static-50 | 均匀 50 格到达 `t=1` 能减少一半 NFE 并保持有竞争力的质量 | pilot：93.82 秒、2.320×；Top-1/3/10 `59.5/79.5/89.0%`；invalid-at-1 `10.425%`（基线 `8.65%`） | 很快；首选无效率在 pilot 上上升 1.775 pp，未列为正式候选；pilot 只作排序，继续测 Static-25 完成曲线端点 | Static-25 |
| 3b：Static-25 | 更少 NFE 能否提供有用的质量/速度端点 | pilot：52.68 秒、4.132×；Top-1/3/10 `60.0/78.0/89.0%`；invalid-at-1 `16.8%`（基线 `8.65%`） | 速度快但首选无效率上升 8.15 pp，排除为候选；Static-50 也高于预算，依预案补测 Static-75 | Static-75 |
| 3c：Static-75 | 中间固定步数能否恢复质量并保留至少 1.20×速度 | pilot：136.87 秒、1.590×；Top-1/3/10 `58.0/78.0/88.5%`；invalid-at-1 `9.375%` | invalid 上升 0.725 pp 且 Top-3 下降 2.5 pp；三个固定步数点均未在 pilot 显示可接受质量，跳过该分支全量验证 | 阶段 4 Hazard 自适应步长；利用分时 hazard 区间筛保守阈值 |
| 4a：Hazard h≤0.04 | 用总编辑 hazard 限制单步累计强度，在允许时跨大步 | pilot：161.75 秒、对原始 Full-100 为 1.346×；相对已优化 Exact-M2-GPU Full-100 为 1.164×；平均/中位 NFE 38.56/33，最大 157/200；Top-1/3/10 `59.0/80.5/88.5%`；invalid-at-1 `8.675%` | 同选择实现的 pilot 显示 Hazard 调步额外减少耗时 14.1%；pilot 质量用于筛选。全量最终候选的加速还包含 GPU 选择优化 | 阶段 8：冻结参数，完整 dev1000 + 同次预测的 heldout800 评分 |
| 8：最终确认 | 最终候选整套实现相对原始 Full-100 达到目标，并在未调参的 800 个反应上保持质量预算 | dev1000：822.83 秒（1.302×，包含 Exact-M2-GPU 与 Hazard）；heldout800 相对基线 ΔTop-1 `+0.875 pp`、ΔTop-3 `+0.875 pp`、ΔTop-10 `−0.75 pp`、Δoracle `0 pp`、Δinvalid `−0.65 pp`；heldout Top-k 95% 配对区间均包含 0，invalid 区间 `[−1.181, −0.125] pp`；平均/中位/最大 NFE `39.24/34/185` | 按预先固定的点估计预算，整套候选在 dev1000 和 heldout800 均通过；目前没有 Hazard 单项的全量速度消融 | 冻结整套参数，进入完整 test 单次评估 |
| 9：完整 test 泛化评估 | 冻结方法能否在未参与选择的完整 test 上完成推理并给出泛化指标 | 5,007 个反应；3967.39 秒；平均/中位/最大 NFE `39.04/34/222`；Top-1/3/10 `64.170/82.564/89.415%`；oracle `92.251%`；invalid-at-1 `8.300%`；901,260 条候选输出，文件哈希已记录 | test 完整运行和评分通过；因没有同策略 test 基线，不计算 test 加速比或按 test 调参；NFE 200 上限导致的首次中断已由 500 保护上限完整重跑 | 本轮实验完成；保留冻结配置及全部结果记录，后续如需部署再单独评估运行成本与接口集成 |

## E. 同条件轨迹案例诊断

为解释步长如何改变模型调用分布，从 dev1000 中**未参与调参的 800 个反应**选取反应编号 `0`、`4`、`5`（均为从 0 起的编号），各取第 1 个增强视图和第 1 次独立采样。案例是看到输出后选出的示例，不用于估计总体质量或速度。`scripts/visualize_trajectory.py` 用正式 checkpoint、完整 32 输入 batch、每输入 `R=9`、每步 `M=2`、seed `42` 重跑 Full-100 与 Hazard；两边使用同一 GPU 子候选选择实现，只改变时间步规则。Hazard 运行使用 `h≤0.04`、普通步 `Λh≤0.15`；保护上限 500，大于这些轨迹的实际 NFE。六条最终预测均逐字节匹配已保存的 dev1000 预测文件。下表按调用开始时间 `t_start` 分成 `[0,0.25)`、`[0.25,0.50)`、`[0.50,0.75)`、`[0.75,1)` 四段；每格为“加速前 → 加速后”，四段调用数之和等于全程 NFE。

| 反应编号 | 第1段调用 | 第2段调用 | 第3段调用 | 第4段调用 | 全程 NFE | 单条轨迹最终结果 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | `26 → 7` | `25 → 6` | `25 → 8` | `24 → 29` | `100 → 50` | 命中目标 → 命中目标 |
| 4 | `26 → 7` | `25 → 7` | `25 → 20` | `24 → 23` | `100 → 57` | 未命中目标 → 命中目标 |
| 5 | `26 → 7` | `25 → 6` | `25 → 10` | `24 → 38` | `100 → 61` | 命中目标 → 无效分子 |

“单轨迹命中”以规范化产物与目标的精确匹配判断，不等于反应级 Top-k。旧表中的“保留状态变化”是每步两个子候选中选中的状态相对前一步是否变化；三个反应的计数分别为 `4→4`、`12→7`、`5→4`。它不是事件总数，也不能直接衡量速度或质量，故不列入汇报主表，但仍保存在原始轨迹中。反应 0 的最终分子保持正确而 NFE 减半；反应 4 的单轨迹结果改善；反应 5 的单轨迹结果退化并变成无效分子。第1至第3段调用都减少，第4段可因强度升高而更密集：反应 5 的 Hazard 轨迹在实际模型调用点观测到的 `Λ` 中位数，第1段为 `0.204`，第4段为 `22.269`；相应区间调用数为 `7` 和 `38`。逐步图与数据见 [trajectories.html](case_study/trajectories.html) 和 [trajectories.json.gz](case_study/trajectories.json.gz)。HTML 已将对照标为“加速前／加速后”；JSON 仍使用 `baseline`／`hazard` 键以保持内部数据口径。新增的 `rate_detail_step=18` 为六条轨迹保存第 18 次模型调用的逐位置、掩码后速率；[组会叙述文档](GROUP_MEETING_NARRATIVE.md)选用反应 4 的加速后轨迹作为真实计算示例，其 `t=0.56174071`、`Λ=9.51519966`、`h=0.01576425`。两文件的 SHA-256 分别为 `5db5b0b400ce8ee099e3e864b56fde9beb3cdbe9d8981dddbb854d91bf033177` 和 `69fd2038d52da02a7c2c0d9f87457ff9281edbaac05cfbf2b3859c1f7e4f18d2`。轨迹记录会增加同步开销，因此这些案例**只比较 NFE 和采样路径，不比较墙钟时间**。

复现命令：

```bash
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/visualize_trajectory.py \
  --indices 0,80,100 --run-index 0 --rate-detail-step 18 \
  --baseline-predictions outputs/fastflow_accel/baseline_dev1000/predictions.txt \
  --hazard-predictions outputs/fastflow_accel/hazard04_l015_dev1000/predictions.txt \
  --output experiments/fastflow_acceleration/case_study/trajectories.html --force
```

## F. 固定随机抽样的 10 个补充案例

从未参与调参的 800 个反应中排除上节的 `0`、`4`、`5`，对剩余反应编号排序后用 `random.Random(20260927).sample(..., 10)` 抽取，选中 `99`、`107`、`164`、`316`、`410`、`468`、`524`、`637`、`646`、`882`（编号从 0 起）。每例仍取第 1 个增强视图和第 1 次独立采样；重跑完整 32 输入 batch、每输入 9 次采样，两种方法使用同一 checkpoint 与 GPU 子候选选择实现。20 条最终预测都与已保存的 dev1000 预测逐条一致。逐例四段调用次数、结果和链接见[随机案例索引](case_study/random10/README.md)，独立页面均命名为 `反应编号-trajectory.html`。

| 范围 | 平均 NFE：加速前 → 后 | 第1段调用合计 | 第2段调用合计 | 第3段调用合计 | 第4段调用合计 | 最终 token 完全相同 | 单条轨迹命中：前 → 后 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 随机 10 例，仅作轨迹分析 | 100 → 38.4（后者范围 26～60） | 260 → 70 | 250 → 60 | 250 → 93 | 240 → 161 | 9/10 | 7/10 → 6/10 |

四段按模型调用的 `t_start` 分为 `[0,0.25)`、`[0.25,0.50)`、`[0.50,0.75)`、`[0.75,1)`。唯一最终输出不同的是反应 `882`：加速前命中，加速后生成有效但未命中的分子。10 例中的单轨迹命中数不代表反应级 Top-k，也不能用于估计总体质量；总体结论仍以完整 dev1000、未调参 800 反应和完整 test 为准。轨迹记录增加同步开销，不用这些页面的运行时间计算加速比。

[合并轨迹页面](case_study/random10/index.html) SHA-256 为 `2ff579a286df52b96ace874e7891b31523419bdbbec6d3586b20d15fc88f369a`；[压缩逐步数据](case_study/random10/index.json.gz) SHA-256 为 `403ea159060edac43685db87391f19385e8332a61f5deef61fa1d2360fd4a3b7`。复现命令见案例索引。
