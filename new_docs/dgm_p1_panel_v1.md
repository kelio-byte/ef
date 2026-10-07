# P1-panel-v1：改进前 Euler / E7 成对轨迹面板

状态：样本清单已冻结；P1 只做机制诊断，不用于调参、选择 checkpoint 或进入保留评估集。

## 固定协议

- 基线：P0 的普通 Euler（`new_checkpoints/checkpoint_step600000.pt`，100 steps，3-path 历史协议）
- 对照：冻结 E7 action-level guidance
  - checkpoint：`/root/autodl-tmp/dgm_guidance_runs/shared_anchor_multitime_2000_lam025_cal010_seed42/guidance_best.pt`
  - `beta=0.10`
  - `guidance_rate_normalization=per_position`
- 运行脚本：`scripts/visualize_trajectory.py` 的 paired 模式；Euler 与 guidance 在同一输入、同一 augmentation、同一 seed 下分别重置随机数
- 输入：`datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/dev_unique1000_aug20/{src,tgt}.txt`
- 每个面板运行：`n_steps=100`、`n_samples=1`、`device=cuda`、`html=true`、`table=true`
- 固定 seeds：42、43、44
- 统计单位仍为原始反应；本面板的 visualization 行只取每个反应的第一个 augmentation（`example_id = reaction_index * 20`），不把 20 个 augmentation 当作 20 个反应。

## 分层规则

在 P0 Euler diagnostics 与冻结 E7 diagnostics 中，`target_final_rank <= 1` 记为该方法 Top-1 正确；无 target 或 rank>1 记为错误。按同一 reaction index 对齐后得到：

| 层 | 全部反应数 | 固定抽样数 | reaction indices | visualization example ids |
| --- | ---: | ---: | --- | --- |
| Euler 正确 / E7 错误 | 50 | 8 | 103, 106, 548, 723, 841, 848, 856, 953 | 2060, 2120, 10960, 14460, 16820, 16960, 17120, 19060 |
| E7 正确 / Euler 错误 | 35 | 8 | 44, 291, 331, 407, 496, 629, 664, 853 | 880, 5820, 6620, 8140, 9920, 12580, 13280, 17060 |
| 两者正确 | 532 | 8 | 104, 315, 348, 351, 388, 500, 544, 680 | 2080, 6300, 6960, 7020, 7760, 10000, 10880, 13600 |
| 两者错误 | 383 | 8 | 209, 624, 625, 699, 728, 804, 945, 966 | 4180, 12480, 12500, 13980, 14560, 16080, 18900, 19320 |

抽样算法是对每层升序 reaction index 使用 `random.Random(20260812).sample(..., 8)`，再升序保存。该清单在 P4-D（若 P2/P3 通过且新 guidance 通过 dev gate）中原样复用，不得换例。

## 运行清单

每行代表一个输出目录；目录名和 seed 是 provenance 的一部分。

```text
visualizations/dgm_p1_pre/P1-panel-v1/euler_correct_e7_wrong/seed42
visualizations/dgm_p1_pre/P1-panel-v1/euler_correct_e7_wrong/seed43
visualizations/dgm_p1_pre/P1-panel-v1/euler_correct_e7_wrong/seed44
visualizations/dgm_p1_pre/P1-panel-v1/e7_correct_euler_wrong/seed42
visualizations/dgm_p1_pre/P1-panel-v1/e7_correct_euler_wrong/seed43
visualizations/dgm_p1_pre/P1-panel-v1/e7_correct_euler_wrong/seed44
visualizations/dgm_p1_pre/P1-panel-v1/both_correct/seed42
visualizations/dgm_p1_pre/P1-panel-v1/both_correct/seed43
visualizations/dgm_p1_pre/P1-panel-v1/both_correct/seed44
visualizations/dgm_p1_pre/P1-panel-v1/both_wrong/seed42
visualizations/dgm_p1_pre/P1-panel-v1/both_wrong/seed43
visualizations/dgm_p1_pre/P1-panel-v1/both_wrong/seed44
```

每个目录只生成脚本输出的 HTML 与 `.metadata.json`。真实 target 只用于 HTML 的 ORACLE/终点诊断，不进入 Euler 或 guidance 的采样输入。

