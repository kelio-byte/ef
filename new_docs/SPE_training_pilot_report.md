# SPE Edit Flows 训练 pilot 报告

执行日期：2026-08-14 UTC。环境：conda `ef`，NVIDIA GeForce RTX 3090。

## 结论

SPE 分支可以稳定训练，并且 Euler 采样约快 2.84 倍、每个 reaction 的 true-unique 候选更多；但在当前 50k pilot checkpoint 上，Top-K、Oracle 和 invalid rate 都明显劣于已经训练 600k steps 的原 tokenizer baseline。因此本轮不启动完整 600k 训练，也不把 SPE 设为默认 baseline。

这不是“证明 600k SPE 一定无效”：当前比较包含 50k vs 600k 的训练步数差异。但按本轮 pilot 的 go/no-go 标准，SPE 尚未显示足以抵消质量损失的收益；后续只有在明确优先优化推理速度/候选多样性时，才值得另行安排更长训练。

## 1. 实现和独立性

新增独立配置 [`configs/retro_spe_pilot.yaml`](../configs/retro_spe_pilot.yaml)，保持 `retro_v2.yaml` 的模型结构、Edit Flows objective、optimizer、batch size、scheduler 和 seed protocol；只改变 SPE data directory、SPE vocabulary 和 pilot horizon `total_steps=50000`。原始 `configs/retro.yaml`、`configs/retro_v2.yaml`、原始 dataset 和已有 checkpoint 均未覆盖。

训练脚本 [`scripts/train_retro.py`](../scripts/train_retro.py) 增加了可选的 pilot monitoring：每 1000 steps 写入 loss/u 值、梯度 norm、最大梯度、非有限值计数、吞吐和 CUDA memory，并在结束时保存 `training_summary.json`。该功能由配置开关控制，不改变未启用 monitoring 时的训练逻辑。

本次运行目录为：

```text
checkpoints/retro_spe_pilot/USPTO_50K_PtoR_aug20_#global#_SPE/2026-08-13_22-53-18/
```

其中包含 `checkpoint_best.pt`、`checkpoint_step50000.pt`、`train.log`、`training_monitor.jsonl`、TensorBoard event 和 `training_summary.json`。这些运行产物保留在本地，但不进入 Git。

## 2. 兼容性与 sanity check

- SPE train/val 数据分别为 800,060 / 100,020 对，loader 为 6,250 / 782 batches；使用已有 pre-aligned 数据路径。
- SPE vocabulary 为 3,035 个真实 token，加 `<PAD>/<BOS>/<GAP>/<UNK>` 后 model vocabulary 为 3,039；special IDs 仍为 0/1/2/3。
- 模型参数量为 15,820,993；原 tokenizer checkpoint 为 13,537,173。差异来自 vocabulary/output heads，不是 hidden/layer/attention 配置改变。
- 实际 GPU batch 的 source/target 与 sampled state shape 为 `[128, 22]`；一次 forward/backward 的 loss 为 `56.8043`，`u_tot=30.3088`，所有参数和梯度有限。
- 相关测试最终为 **44 passed, 4 warnings**；warnings 来自 PyTorch Transformer nested-tensor 提示，不是异常。

评估时确认 SPE 仍继承项目的 `#global#` 表示。新增 [`scripts/evaluate_spe_euler.py`](../scripts/evaluate_spe_euler.py) 先拼接 SPE token，再调用 `inverse_global_align`，最后用 RDKit canonicalize；空 canonical molecule 计为 invalid。项目既有 `scripts/score_#global#.py` 也用同一逆变换作为正式 Top-K scorer。

## 3. 训练结果

### 3.1 Validation milestones

| step | val loss | val u_tot | val INS | val DEL | val SUB |
|---:|---:|---:|---:|---:|---:|
| 5,000 | 14.9464 | 4.40 | 2.81 | 0.03 | 1.56 |
| 10,000 | 12.0569 | 4.50 | 2.91 | 0.01 | 1.57 |
| 20,000 | 9.0048 | 4.01 | 2.76 | 0.01 | 1.23 |
| 30,000 | 7.9359 | 3.57 | 2.38 | 0.01 | 1.18 |
| 40,000 | 7.4976 | 3.84 | 2.63 | 0.01 | 1.20 |
| 50,000 | **6.8347** | **3.48** | 2.33 | 0.01 | 1.14 |

### 3.2 训练资源和异常

- 完成 50,000 / 50,000 steps，未发生中断。
- wall time：3,580.74 s，约 59.68 min；平均 13.964 steps/s。
- monitor：50 条记录，`monitor_anomalies=0`。
- train monitor loss：22.3425（step 1k）降至 6.8799（step 50k），全程 finite。
- 最大 gradient norm：31.57；最大 absolute gradient：7.60；非有限 gradient/parameter 数均为 0。
- 峰值 CUDA allocated：3,437,280,256 bytes（约 3.44 GB）；reserved：5,360,320,512 bytes（约 5.36 GB）。

按当前吞吐线性外推，600k steps 约需 11.9 小时；本次没有启动 600k。

## 4. Euler sampling protocol

为避免 10 个 reaction 的偶然性，最终 pilot sampling 使用前 100 个原始 reaction：

- 20 条 augmentation × 每条 10 个普通 Euler samples；共 2,000 个 product rows、20,000 个候选；
- 100 Euler steps、cubic scheduler、batch 64、seed 42、CUDA；
- SPE 使用 `checkpoint_best.pt`（step 50k）；baseline 使用 `new_checkpoints/checkpoint_step600000.pt`；
- 两组使用项目正式 `#global#` scorer 的 `legacy_best_rank` 聚合；新增 evaluator 只用于 first-seen coverage/unique 诊断。

采样输出和 JSON 诊断分别保存在：

- SPE：[`results/spe_pilot_euler_100rxn_n10/`](../results/spe_pilot_euler_100rxn_n10/)
- baseline：[`results/original_baseline_euler_100rxn_n10/`](../results/original_baseline_euler_100rxn_n10/)

## 5. 对比结果

正式 Top-K 来自 `scripts/score_#global#.py`；unique/valid 来自同 protocol 的 sampling diagnostics。Invalid 同时报告 input rank 1 和 10 个 rank 的总体值。

| 指标（100 reaction） | SPE 50k | 原 tokenizer baseline 600k | SPE − baseline |
|---|---:|---:|---:|
| Top-1 | 23.0% | 53.0% | −30.0 pp |
| Top-3 | 43.0% | 72.0% | −29.0 pp |
| Top-5 | 51.0% | 79.0% | −28.0 pp |
| Top-10 | 56.0% | 88.0% | −32.0 pp |
| Oracle-any | 70.0% | 96.0% | −26.0 pp |
| Invalid，input rank 1 | 43.75% | 16.60% | +27.15 pp |
| Invalid，10 ranks 总体 | 44.58% | 15.56% | +29.02 pp |
| mean valid candidates / reaction | 110.84 | 168.88 | −58.04 |
| mean true-unique candidates / reaction | **43.04** | 27.92 | **+15.12** |
| true-unique / 200 candidate slots | 21.52% | 13.96% | +7.56 pp |
| sampling wall time | 143.45 s | 407.73 s | 2.84× faster for SPE |
| peak CUDA allocated | 3.92 GB | 2.15 GB | SPE higher |
| peak CUDA reserved | 24.86 GB | 24.93 GB | approximately equal |

每个 input rank 的 invalid rate（SPE / baseline）分别为：

```text
rank 1: 43.75% / 16.60%    rank 2: 45.05% / 15.50%
rank 3: 43.75% / 15.45%    rank 4: 47.00% / 16.35%
rank 5: 44.55% / 15.85%    rank 6: 44.90% / 15.65%
rank 7: 42.60% / 16.15%    rank 8: 43.95% / 14.00%
rank 9: 44.15% / 14.95%    rank 10: 46.10% / 15.10%
```

## 6. 判断

SPE pilot 证明了两件事：

1. 训练链路、vocabulary 适配和 checkpoint 保存是稳定的；
2. 更短的 SPE 序列确实带来更快采样和更多表面候选多样性。

但当前候选质量没有达到 baseline 的可接受范围：Oracle 下降 26 pp，Top-1 下降 30 pp，invalid 增加约 27–29 pp。增加的 unique candidates 主要伴随 invalid，并没有转化为有效 Top-K 覆盖。

因此本轮结论是：**暂不进入完整 600k，不替换 baseline，SPE 保留为独立 opt-in 分支。** 若未来要继续，必须先把“invalid 大幅升高”作为首要问题，完成一个新的中期质量门槛后再决定是否支付约 12 小时的完整训练成本；本报告不把 50k vs 600k 的差异解释成 SPE 架构本身的最终上限。
