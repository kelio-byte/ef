# DGM 后续执行：P0 冻结协议与基线记录

状态：P0 已完成。本文档只记录执行协议和 provenance；它不把历史 guidance 结果重新解释为新方法结论。

## 冻结的端到端协议

- 基础模型：`new_checkpoints/checkpoint_step600000.pt`
- 输入：`datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/dev_unique1000_aug20/src.txt`
- 目标：同目录 `tgt.txt`
- 统计单位：1000 个原始反应；每个反应 20 条 SMILES augmentation，不把 augmentation 当作独立反应
- 采样：普通 Euler，100 steps，cubic scheduler，每条 augmentation 3 个候选，seed 42
- 批大小与设备：batch 64，`cuda`（NVIDIA GeForce RTX 3090，conda 环境 `ef`）
- 评分：固定现有 `score_#global#.py` / diagnostics 聚合逻辑，`n_best=10`

正式复跑命令为：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ef
python scripts/eval.py \
  --checkpoint new_checkpoints/checkpoint_step600000.pt \
  --products_file datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/dev_unique1000_aug20/src.txt \
  --targets datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/dev_unique1000_aug20/tgt.txt \
  --output_dir results/dgm_execution/p0_euler_baseline_seed42 \
  --sampler euler --n_samples 3 --n_steps 100 --batch_size 64 \
  --device cuda --seed 42 --augmentation 20 \
  --start_product 0 --max_products 20000 --n_best 10
```

## 基线产物与校验

- 预测：`results/dgm_execution/p0_euler_baseline_seed42/predictions.txt`
  - SHA-256：`9e645acca9718a55ff8ef7c63fafead82f66e2f4e7f126ca2817ed33e59b5b38`
- metadata：`results/dgm_execution/p0_euler_baseline_seed42/sampling_metadata.json`
  - 输入 `src.txt` SHA-256：`c20e337496f52bbeacc7e5870e3eefbb27a653800fa3d3b0fd8dbca8cfd098f6`
  - 基础 checkpoint 大小：162,640,263 bytes
  - 运行耗时：1287.16 s；峰值 CUDA allocated/reserved：约 0.75/14.33 GB
- diagnostics：`results/dgm_execution/p0_euler_baseline_seed42/diagnostics.json`
  - Top-1：58.2%
  - Top-3：75.5%
  - Top-5：79.8%
  - Top-10：83.5%
  - Oracle-any：86.6%
  - invalid（输入 rank 1/2/3）：11.875% / 11.425% / 12.115%

新复跑与既有 E1 的 Top-k、Oracle 和 invalid 结果一致到报告精度，说明基线输入布局、seed 和聚合逻辑可复现。当前 metadata 的 `dirty=true` 仅因为仓库中尚未提交的研究文档存在；采样代码 revision 已固定为 `c2e1e83`，后续报告同时保留此 provenance。

## 历史 E7 对照及停止规则

当前 action-level guidance 的冻结历史产物是：
`results/dgm_evaluation_v2/dev_multitime_guidance_2000_beta010_seed42/`，使用相同基础 checkpoint、100 steps、3 samples、seed 42、`beta=0.10` 和 `per_position` normalization。其 guidance checkpoint 为：
`/root/autodl-tmp/dgm_guidance_runs/shared_anchor_multitime_2000_lam025_cal010_seed42/guidance_best.pt`。

将新 P0 diagnostics 与该历史 E7 diagnostics 做 reaction-level paired bootstrap（5000 次，seed 20260812）得到：

| 指标 | 普通 Euler | 历史 E7 | E7 − Euler |
| --- | ---: | ---: | ---: |
| Top-1 | 58.2% | 56.7% | −1.5 pp（95% CI −3.3～+0.3） |
| Top-3 | 75.5% | 75.9% | +0.4 pp（95% CI −0.9～+1.7） |
| Top-10 | 83.5% | 83.8% | +0.3 pp（95% CI −1.2～+1.7） |
| Oracle-any | 86.6% | 86.6% | 0.0 pp（95% CI −1.2～+1.2） |

这只是历史对照，不是新的可调参候选。E7 的训练时长、`beta`、checkpoint、seed 和当前 reward-calibration/local-credit 分支均按计划关闭；不得在 dev 上继续扫描来挽救该结果。confirm、final 和完整 test 在 P3/P4 gate 通过前保持未使用。

## P0 后的允许动作

1. 在已使用的 dev-1000 上完成改进前 Euler/E7 的成对 `scripts/visualize_trajectory.py` 诊断（P1）。
2. 在 train-1000 构造新的 correctness reward，并仅用 reward holdout-200 做一次冻结的 P2 AUC + P3 endpoint-rerank 报告。
3. 只有 P2 与 P3 同时通过，才允许用该 reward 重建 guidance、跑 dev，并复用同一个 P1 面板做改进后轨迹复核。

