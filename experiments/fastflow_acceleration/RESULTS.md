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

## A. 运行台账

| run_id | 阶段/方法 | 划分 | 代码提交 | 输入 SHA-256 | 输出目录 | 状态 |
|---|---|---|---|---|---|---|
| `baseline_pilot` | Full-100 | pilot200 | `bf8ab00` | `44eaa6d6...` | `outputs/fastflow_accel/baseline_pilot/` | 完成；预测 SHA-256 `17f7c190...` |
| `baseline_dev1000` | Full-100 | dev1000 | `bf8ab00` | `54384b14...` | `outputs/fastflow_accel/baseline_dev1000/` | 完成；预测 SHA-256 `f3406c06...` |
| `baseline_heldout` | Full-100 的评分子集 | heldout800 | `bf8ab00` | `2bf01200...` | `outputs/fastflow_accel/baseline_heldout/` | 完成；从 `baseline_dev1000` 提取，无额外推理；预测 SHA-256 `aca2b8e8...` |

## B. 速度与质量

| 方法 | 划分 | 采样秒数 | 相对基线加速 | 平均/中位 NFE | Top-1 | Top-3 | Top-10 | Oracle-any | invalid-at-1 | batch p95 秒 | 显存峰值 MiB |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Full-100 | pilot200 | 217.68 | 1.00× | 100/100 | 58.5% | 80.5% | 88.0% | 91.5% | 8.65% | 2.005 | 556.2 |
| Full-100 | dev1000 | 1071.01 | 1.00× | 100/100 | 61.7% | 80.7% | 88.0% | 90.4% | 8.635% | 1.953 | 650.9 |
| Full-100 | heldout800（从 dev1000 输出评分） | — | — | 100/100 | 62.375% | 80.875% | 87.625% | 89.875% | 8.69375% | — | — |

完整基线共处理 20,000 个增强输入、180,000 条轨迹；总 NFE 为 18,000,000，625 个 batch 共调用模型 62,500 次。`heldout800` 只从同一次完整预测提取对应反应并重新评分，因此没有独立采样耗时或显存值。历史 dev1000 的 1088.07 秒使用另一权重及旧计时口径，不参与加速比计算。

复现命令（使用该仓库源码而非环境中其他可编辑安装）：

```bash
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/evaluate.py --split dev1000 --output outputs/fastflow_accel/baseline_dev1000 --record-performance --workers 8
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/accel_protocol.py score-heldout --split-dir outputs/fastflow_accel/split --full-output outputs/fastflow_accel/baseline_dev1000 --output outputs/fastflow_accel/baseline_heldout --workers 8
```

## C. 性能构成与轨迹诊断

| 时间段 | Transformer 占比 | 同步占比 | 选中状态不变比例 | 双 child 无事件比例 | 合法 hazard p50/p90 | 强度变化 p50/p90 | 理想可省 NFE | 结论 |
|---|---:|---:|---:|---:|---|---|---:|---|
| 0.00～0.25 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 0.25～0.50 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 0.50～0.75 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 0.75～1.00 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

## D. 阶段决定

| 阶段 | 假设 | 已有证据 | 决定 | 下一步 |
|---|---|---|---|---|
| 0：新基线 | 同一权重和代码可以得到可复现的速度、质量口径 | pilot200：217.68 秒；dev1000：1071.01 秒，Top-1 61.7%、Top-10 88.0%；heldout800：Top-1 62.375%、Top-10 87.625%；所有轨迹均为 100 NFE | 阶段 0 完成，已固定后续质量和速度对照 | 应用户要求暂停；恢复后进入阶段 1 性能构成与轨迹诊断 |
