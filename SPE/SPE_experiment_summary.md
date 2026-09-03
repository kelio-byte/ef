# SPE 实验总览与证据索引

记录更新：2026-08-20 UTC。本文是 SPE（Subword/SMILES Pair Encoding）分支的**总入口**：集中说明做过什么、哪些结果可横向比较、证据在哪里，以及当前应如何使用 checkpoint。它不替代原始报告；每项都保留了来源链接和结果目录。

## 0. 先读这个结论

- 已完成的成熟实验绝大多数基于**改进后的 R-SMILES**，即数据目录带 `#global#`；下文没有特别标注“原始”的结果都属于这一分支。
- Full-SPE（全部 3,002 条 merge rule）明确带来约 `2.4×` 的 Euler 推理加速，但初版 B128 质量明显落后 Atom；B256 重训虽改善，仍稳定弱于同协议的 M500。
- SPE-M500 是目前最有价值的 fragment-level 分支。配对三 seed 下：
  - `M500@490K` 更适合首选排序：Top-1/3/5 为 `59.27 / 76.40 / 80.63%`，Oracle `89.90%`；
  - `M500@500K` 的 Top-10 与 Invalid 更好：`83.97%` / `12.47%`。
  二者差异接近 sampling-seed 波动，不能称任一 checkpoint 全面胜出。若当前目标是 Top-1/Top-5，临时 baseline 选 `M500@490K`；500K 保留作 Top-10/invalid 对照。
- 改进前 R-SMILES 的 M500 已有一份外机 checkpoint sweep：最强的 550K 为 Top-1 `45.5%`、Top-10 `66.7%`、Oracle `83.4%`，明显低于改进后 M500。该报告的原始 result 目录、manifest、diagnostics 和训练日志尚未迁回本机，因此它是**强信号，但不是已验证的配对因果比较**。
- 完整 `src-test` 尚未作为 checkpoint 选择集使用；目前的成熟 checkpoint 筛选都在冻结的 `dev_unique1000_aug20` 上完成。

## 1. 原始记录在哪里

| 文档 / 产物 | 记录内容 | 当前作用 |
|---|---|---|
| [SPE_preprocessing_plan.md](../new_docs/SPE_preprocessing_plan.md) | 初版 Full-SPE 的预处理设计 | 历史计划 |
| [SPE_preprocessing_report.md](../new_docs/SPE_preprocessing_report.md) | Full-SPE（K=3002）round-trip、长度、编辑、OOV 审计 | 初版数据正确性证据 |
| [SPE_training_pilot_report.md](../new_docs/SPE_training_pilot_report.md) | 50K pilot、训练链路与早期采样 | smoke；质量数字受训练预算和旧 sampler bug 混杂 |
| [SPE_600k_experiment_report.md](../new_docs/SPE_600k_experiment_report.md) | 初版 Full-SPE B128 600K、20% test、dev、吞吐 probe、早期后续计划 | 初版训练/效率证据；其中 P1 状态已被后续结果更新 |
| [SPE_prefix_rules_experiment.md](../new_docs/SPE_prefix_rules_experiment.md) | K=500/1000/2000/full 数据集构造与统计 | tokenizer 深度选择依据 |
| [SPE_dev1000_checkpoint_evaluation.md](../new_docs/SPE_dev1000_checkpoint_evaluation.md) | Full-B256 与 M500 checkpoint sweep、M500 多 seed、R9K1M2 | **当前主性能表** |
| [RSMILES_SPE_M500_four_way_dataset_audit.md](../new_docs/RSMILES_SPE_M500_four_way_dataset_audit.md) | 原始/改进 R-SMILES × Atom/M500 四组数据审计 | 解释原始 R-SMILES M500 低分的机制证据 |
| [ori_rsmiles_spe_m500_evaluation.md](ori_rsmiles_spe_m500_evaluation.md) | 原始 R-SMILES M500 的外机 checkpoint sweep | 已报告；原始 artifacts 待迁回核验 |
| [SPE_atom_fragment_required_data.txt](../new_docs/SPE_atom_fragment_required_data.txt) | Atom/Full-SPE 的长度、词表、硬件、初版超参清单 | 历史数据快照 |
| [extracted_metrics.txt](../extracted_metrics.txt) | Full-SPE 600K 的 `n_steps=60…200` 小规模 sweep 原始指标 | 探索性步数敏感性记录 |
| `results/**/sampling_metadata.json`、`diagnostics.json`、`logs/*.log` | 可复算的采样时间、有效/去重候选、Oracle、Top-K 日志 | 原始评估证据 |

支持性而非主实验记录：`new_docs/sampler_semantics_audit.md` 解释了早期 pilot 的 action-support 问题；`new_docs/editretro_2024_transfer_analysis.md` 是文献迁移分析，不是 SPE 性能实验。

## 2. 表示与数据集实验

### 2.1 数据集命名

| R-SMILES 表示 | Atom-level | Full-SPE | SPE-M500 | 状态 |
|---|---|---|---|---|
| 改进后 global R-SMILES | `USPTO_50K_PtoR_aug20_#global#` | `..._#global#_SPE` | `..._#global#_SPE_m500` | Full 与 M500 均已训练、评估 |
| 改进前 R-SMILES | `USPTO_50K_PtoR_aug20` | `..._SPE_full` | `..._SPE_m500` | 数据已构造；M500 外机 sweep 已报告，原始 artifacts 待迁回 |

`#global#` 表示 global-aligned R-SMILES，而不是一个额外模型技巧。它改变产物/反应物的书写和对齐结构；SPE 则在该字符串表示上进一步做 merge tokenization。

### 2.2 merge-depth 数据实验（改进后 R-SMILES）

所有 split 都通过 SPE round-trip、raw/aligned 投影和行数检查；训练集均为 800,060 条 augmentation pair。下表为训练集统计。

| 表示 | merge 规则 | 真实词表 | 平均 aligned 长度 | 平均编辑距离 | 平均归一化编辑 | INS / DEL / SUB |
|---|---:|---:|---:|---:|---:|---:|
| Atom | — | 69 | 50.809 | 5.741 | 11.756% | 91.201 / 0.924 / 7.875% |
| SPE-M500 | 500 | 568 | 16.047 | 4.127 | 27.017% | 67.341 / 0.614 / 32.045% |
| SPE-M1000 | 1,000 | 1,066 | 14.259 | 4.053 | 29.621% | 65.362 / 0.529 / 34.109% |
| SPE-M2000 | 2,000 | 2,056 | 12.722 | 3.869 | 31.865% | 62.677 / 0.562 / 36.761% |
| Full-SPE | 3,002 | 3,035 | 12.057 | 3.857 | 33.486% | 62.274 / 0.611 / 37.115% |

结论：merge 越深，序列越短，但词表、fragment SUB 比例和编辑密度越高。M1000/M2000 仅完成数据构造和审计，**没有本地训练/性能结果**；当前已完成训练的中等粒度点是 M500。

### 2.3 原始 vs 改进 R-SMILES 的 M500 数据审计

| 数据表示 | M500 对齐长度 | M500 编辑距离 | 编辑密度（均值 / P95） | KEEP 比例 | SUB 比例 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 原始 R-SMILES M500 | 15.24 | 6.84 | 44.24% / 86.67% | 55.13% | 47.1% | 更短，但几乎一半位置要改，fragment 替换很多 |
| 改进 R-SMILES M500 | 16.05 | 4.13 | 26.93% / 54.55% | 74.28% | 32.1% | 略长，但编辑更稀疏、对齐更稳定 |

这排除了 OOV、SPE round-trip、数据错位和 `max_seq_len=96` 截断（M500 最大对齐长度仅 63/60）作为原始 M500 低分的主因。更可信的机制是：原始 R-SMILES 将 Edit Flows 变成了高密度、fragment-SUB 主导的任务。详见四组审计文档。

## 3. 训练实验清单

模型骨架在 SPE 训练中保持为 Transformer hidden=256、10 layers、8 heads、FFN=2048；变化主要来自词表、batch、merge 深度和训练预算。

| ID | 表示 / tokenizer | 训练设置 | 状态 | 主要观察 |
|---|---|---|---|---|
| T0 | global Full-SPE | 50K、B128 | 完成 | 链路、词表、对齐、训练/采样均正常；仅作 smoke |
| T1 | global Full-SPE | 600K、B128、LR factor 1.0、warmup 8K、无 clip | 完成 | 11.36 h，训练稳定；best val loss 在 580K，但生成质量不由 val loss 单独决定 |
| T1-throughput | global Full-SPE | 约 1K-step probe：B128 / B256 / B512 | 完成 | pairs/s 为 1817 / 1928 / 1891–1922；大 batch 只带来约 4–6% 吞吐提升，显存明显增加 |
| T1-P1A | global Full-SPE | 从 T1 600K 续训至 800K，B128、factor 1.0、无 clip | 完成 | 有现成 best/final 预测；结果见第 5.2，未达到原预注册质量门槛 |
| T1-P1B | global Full-SPE | 计划 600K→800K，factor 1.3 | 未运行 | 仅有配置 `configs/retro_spe_p1b_800k.yaml`，无 checkpoint/result |
| T2 | global Full-SPE | B256 重训，多个 190–600K checkpoint | 完成 | 后期 invalid 下降，但没有稳定追平 M500 |
| T3 | global SPE-M500 | B256，多个 350–600K checkpoint | 完成 | 当前主线；M500 同步改善质量与约 2.4× 推理速度 |
| T4 | 原始 R-SMILES M500 | B256、600K；300/450/490/500/550/600K 已评估 | 外机完成、报告已迁回 | 最佳 550K 为 Top-1 45.5、Top-10 66.7、Oracle 83.4；原始 artifacts 未迁回，尚不能做 paired attribution |

关于 T2/T3：当前本机保存的是 checkpoint 和评估产物，完整的原始训练 run/config 并不都在本机，因此本表只把可确认的 B256、tokenizer、checkpoint step 当作事实；不要把当前目录中同名的“新训练配置”倒推为所有历史 run 的精确超参。

## 4. 已完成的正式评估

### 4.1 统一 validation-dev 协议

除特别标注外，当前 checkpoint 选择使用：改进后对应 tokenizer 的 `dev_unique1000_aug20`，1,000 reaction blocks / 20,000 augmentation 输入，普通 Euler `N=9`、100 steps、cubic、batch=32、augmentation=20、seed=42、Top-K 到 10。每个目录的 `sampling_metadata.json` 与 `diagnostics.json` 是可复核来源。

### 4.2 初版 Full-SPE B128：600K 与 Atom 的公平预评估

这组使用从完整 test 随机冻结的 1,001 blocks（20% shuffled test），不是完整 test。

| 模型 / checkpoint / sampler | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid@1 | 采样时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Atom @600K / Euler | 58.541 | 77.622 | 81.918 | 85.614 | 90.909 | 12.008 | 58.4 min |
| Full-SPE @580K best / Euler | 50.649 | 70.629 | 74.825 | 77.223 | 87.113 | 22.807 | — |
| Full-SPE @580K best / R9K1M2 | 53.447 | 71.329 | 75.225 | 78.422 | 88.212 | 23.801 | — |
| Full-SPE @600K final / Euler | 51.349 | 70.330 | 74.825 | 77.323 | 87.612 | 22.193 | 24.1 min |
| Full-SPE @600K final / R9K1M2 | 54.346 | 72.228 | 76.224 | 78.521 | 87.912 | 22.807 | — |

初版 Full-SPE 的明确收益是 2.42× 采样加速；代价是相对 Atom 的 Top-1/3/5/10 低约 7–8 pp，invalid 高约 10 pp。R9K1M2 是同一 checkpoint 的 sampler 消融，不能用它替代表示方法的公平对照。

### 4.3 Full-SPE B256 checkpoint sweep（global，seed 42）

| step | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid@1 | 采样时间 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 190K | 51.8 | 70.6 | 75.6 | 78.9 | 88.4 | 22.740 | 23.84 min |
| 200K (`checkpoint_best`) | 53.7 | 71.0 | 75.7 | 79.2 | **89.0** | 19.510 | 23.86 min |
| 210K | 52.0 | 73.3 | 76.6 | 80.1 | 87.9 | 21.905 | 23.66 min |
| 400K | 56.3 | 73.2 | 77.4 | 80.4 | 87.6 | 19.885 | 23.65 min |
| 470K | **56.4** | 73.3 | 78.0 | 81.0 | 88.0 | 16.490 | 23.85 min |
| 500K | 55.2 | **73.6** | **79.0** | **81.5** | 88.1 | 17.755 | 24.35 min |
| 550K | 56.3 | 73.5 | 77.3 | 80.2 | 87.7 | 17.680 | 24.41 min |
| 600K | 55.5 | 72.4 | 77.3 | 81.4 | 88.8 | **15.400** | 24.17 min |

Full-SPE 没有单个 checkpoint 同时支配质量、Oracle、invalid 和多样性：470K 偏 Top-1，500K 偏 Top-K，600K 偏 valid/Oracle。所有点都是单 seed；不建议继续把 Full-SPE 作为主线做更多 checkpoint sweep。

### 4.4 SPE-M500 checkpoint sweep（global，seed 42）

本机完整 diagnostics 的 checkpoint：

| step | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid@1 | valid / reaction | true-unique / reaction | 时间 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 350K | 57.7 | 75.6 | 79.7 | 83.9 | 89.2 | 14.475 | 154.089 | 25.961 | 24.28 min |
| 400K | 57.6 | 76.2 | 80.4 | 83.9 | 88.9 | 12.625 | 157.694 | 24.972 | 24.51 min |
| 450K | 59.1 | 76.5 | 80.0 | 83.9 | 89.9 | 14.075 | 154.787 | 24.877 | 24.31 min |
| 500K | **59.9** | 75.9 | 80.1 | **84.3** | 89.7 | 12.245 | 157.513 | 24.281 | 24.47 min |
| 550K | 58.7 | 76.3 | **80.5** | 83.9 | 89.1 | **11.665** | 158.566 | 23.283 | 24.23 min |
| 600K | 58.1 | **77.2** | 80.4 | 83.7 | 89.0 | 11.985 | 158.319 | 22.649 | 24.59 min |

外机补充的密集单-seed sweep（同协议，只提供核心指标）：

| step | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid@1 |
|---:|---:|---:|---:|---:|---:|---:|
| 140K best | 55.4 | 73.4 | 78.0 | 81.2 | 89.6 | 15.615 |
| 150K | 55.8 | 73.3 | 78.0 | 82.0 | 89.7 | 15.300 |
| 200K | 57.7 | 74.4 | 78.6 | 81.6 | 89.5 | 14.240 |
| 250K | 57.7 | 73.9 | 78.4 | 83.0 | 89.8 | 14.255 |
| 300K | 57.4 | 74.7 | 80.4 | 83.7 | 89.2 | 13.170 |
| 460K | 58.3 | 76.7 | 80.0 | 83.6 | 89.4 | 13.030 |
| 470K | 58.9 | 75.6 | 80.0 | 84.0 | 89.6 | 13.415 |
| 480K | 58.6 | 75.1 | 79.2 | 83.6 | 89.6 | 12.460 |
| 490K | **60.1** | 76.6 | **80.5** | 83.7 | **90.0** | 12.850 |
| 500K | 59.9 | 75.9 | 80.1 | 84.3 | 89.7 | 12.245 |
| 510K | 58.2 | **76.8** | 80.4 | 83.2 | 89.1 | 14.265 |
| 520K | 58.3 | 76.5 | 79.8 | **84.7** | 89.9 | 12.840 |
| 530K | 58.6 | 76.5 | 80.2 | **84.7** | 89.7 | 12.950 |
| 540K | 58.1 | 75.9 | 80.0 | 83.8 | 89.4 | 13.995 |
| 600K | 58.1 | 77.2 | 80.4 | 83.7 | 89.0 | **11.985** |

### 4.5 M500 的 paired 三 seed 选择实验

这是当前最可靠的 checkpoint 选择证据：两个 checkpoint 在相同 dev block、同一普通 Euler 配置下使用 seed `42/7/123` 配对重跑。

| checkpoint | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid@1 |
|---|---:|---:|---:|---:|---:|---:|
| M500@490K，3-seed mean ± sample SD | **59.27 ± 0.74** | **76.40 ± 0.20** | **80.63 ± 0.81** | 83.67 ± 0.65 | **89.90 ± 0.66** | 12.65 ± 0.18 |
| M500@500K，3-seed mean ± sample SD | 58.90 ± 0.87 | 76.13 ± 0.49 | 80.13 ± 0.15 | **83.97 ± 0.35** | 89.83 ± 0.15 | **12.47 ± 0.19** |

490K 相对 500K 的均值变化为 Top-1/3/5/Oracle `+0.37/+0.27/+0.50/+0.07 pp`；500K 则有 Top-10 `+0.30 pp` 和 invalid `-0.19 pp`。这些量级不足以证明严格支配关系。

### 4.6 SPE-M500@490K sampler 消融

| sampler | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid@1 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| Euler N=9 | 60.1 | 76.6 | 80.5 | **83.7** | 90.0 | 12.850 | checkpoint sweep 的普通采样基线 |
| R9K1M2 | 60.0 | **77.3** | **80.7** | 83.6 | **90.4** | **12.395** | 提升 Top-3/5、Oracle、validity，但没有提高 Top-1/10 |

R9K1M2 的本机采样时间为 25.55 min；它是采样器研究的备选，不应替代普通 Euler 作为 SPE 表示对照。

### 4.7 Full-SPE P1-A：600K→800K continuation 的补录

`checkpoints/retro_spe_p1a_800k/.../checkpoint_step800000.pt` 已完成，但旧计划文档仍标为“待执行”。本次汇总直接对已保存的 1,000-block prediction 以正式 `score_#global#.py` 重评分（未重新采样）。

| checkpoint | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid@1 | valid | true-unique | 采样时间 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1-A `checkpoint_best.pt` | 49.9 | 69.8 | 74.5 | 77.8 | 87.8 | 20.685 | 142.494 | 36.333 | 24.63 min |
| P1-A final @800K | 51.5 | 71.3 | 74.8 | 78.4 | 86.8 | 20.470 | 142.549 | 34.414 | 24.68 min |

相对初版 Full-SPE 600K dev（50.0 / 68.7 / 73.3 / 77.2 / 87.2 / 22.275），final 800K 的 Top-1/3/5/10 增加 `+1.5/+2.6/+1.5/+1.2 pp`，invalid 降 `1.805 pp`，但 Oracle 降 `0.4 pp`。它未达到原定的 `Top-1 +≥2 pp` 且 `Oracle +≥1 pp` 或 `invalid −≥3 pp` 门槛；因此“单纯多训 200K”不是 Full-SPE 的充分解法。

### 4.8 原始 R-SMILES SPE-M500 checkpoint sweep（外机报告）

报告标注的协议为：`USPTO_50K_PtoR_aug20_SPE_m500/evaluation_v2/dev_unique1000_aug20`、Euler N=9、100 steps、cubic、batch=32、seed=42、augmentation=20、`n_best=10`。这些设置与改进后 M500 的主协议相同；但对应 evaluation subset、checkpoint、result 目录、`sampling_metadata.json`、`diagnostics.json` 和训练日志均不在本机，尚不能证明两个 `dev_unique1000` 使用完全相同的 reaction indices。

| step | Top-1 | Top-3 | Top-5 | Top-10 | Oracle |
|---:|---:|---:|---:|---:|---:|
| 300K | 42.4 | 60.5 | 64.0 | 65.9 | 81.7 |
| 450K | 43.5 | **62.0** | **65.1** | 66.6 | 82.9 |
| 490K | 41.4 | 59.8 | 63.4 | 64.7 | 81.5 |
| 500K | 42.6 | 60.7 | 64.2 | 65.5 | 81.7 |
| 550K | **45.5** | 61.5 | 64.9 | **66.7** | **83.4** |
| 600K | 43.8 | 61.9 | 64.9 | 66.0 | 82.8 |

即使只作非配对的方向性比较，差距也远大于改进后 M500 的 seed 波动：原始最佳 550K 相对改进后 M500@490K 的 Top-1/3/5/10/Oracle 低 `14.6/15.1/15.6/17.0/6.6 pp`。其中 Oracle 的 `-6.6 pp` 说明问题不只是候选排序：目标根本没有被采到的 reaction 更多；而原始 550K 的 `Oracle − Top-10 = 16.7 pp`（改进后 490K 为 `6.3 pp`）又说明即使目标出现，跨 augmentation 的排序/聚合也更难把它送进 Top-10。

这与第 2.3 节的数据机制一致：原始 M500 虽然更短（aligned length 15.24 vs 16.05），却有更高编辑密度（44.24% vs 26.93%）和更多 SUB（47.1% vs 32.1%）。因此“序列短”并不能抵消 global alignment 缺失造成的编辑困难；DP Levenshtein 在两边都正确执行，但它只能对既定的书写顺序求最优对齐，不能把原始 R-SMILES 的跨位置/跨 fragment 变化变成稳定的保留和插入。

## 5. 探索性/非决定性实验

### 5.1 50K pilot

50K Full-SPE 与 600K Atom 的 100-reaction 比较显示更快（2.84×）但 Top-1 低 30 pp、invalid 高约 27–29 pp。它的主要价值是验证 pipeline；由于训练预算不匹配且当时存在 `INS(pos=0)` action-support bug，不能与后续结果混合做质量结论。

### 5.2 推理步数 sweep（初版 Full-SPE 600K）

来自 `extracted_metrics.txt`：每次 `max_products=2000`，即 100 个完整 reaction block × 20 augmentation；不是 dev1000。只有单 seed，且输出目录被 `--overwrite` 重用，因此只作为敏感性提示。

| Euler steps | Top-1 | Top-3 | Top-5 | Top-10 | Invalid@1 |
|---:|---:|---:|---:|---:|---:|
| 60 | 45 | 61 | 65 | 70 | 22.55 |
| 80 | 43 | 63 | 66 | 71 | 23.15 |
| 90 | 43 | **66** | **69** | 72 | 21.05 |
| 100 | 44 | 60 | 65 | 71 | 22.05 |
| 110 | 44 | 63 | 66 | 71 | 22.40 |
| 120 | **45** | 60 | 64 | 68 | 22.60 |
| 140 | 44 | 59 | 64 | 70 | 21.20 |
| 160 | 44 | 59 | 67 | **74** | 21.70 |
| 180 | 44 | 63 | **69** | 73 | 20.55 |
| 200 | 43 | 61 | 67 | 71 | 21.65 |

没有单调收益，也没有 invalid 的系统性改善；不支持把“100 步不够”当作当前 Full-SPE 质量问题的主要解释。

### 5.3 P1-A inference batch benchmark（100 blocks）

同一 P1-A best checkpoint、相同 100-step Euler N=9、2,000 augmentation 输入：batch 32/64/128 的 sampler time 分别为 `151.58 / 150.63 / 153.07 s`。batch 64 略快，但差异不足以改变正式评估默认 batch=32 的结论；大 batch 的显存 reserved 更高。

## 6. 当前边界与下一步

1. **主 baseline：**做 fragment-level 方法改进时，优先使用改进后 R-SMILES 的 `SPE-M500@490K` + 普通 Euler N=9；同时保留 `@500K` 作为 Top-10/invalid 备选。
2. **Full-SPE：**不再扩大 Full-SPE checkpoint sweep 或只靠增加训练步数来优化；已有 B128 600K、B128→800K、B256 190–600K 的证据足够说明它不是当前主线。
3. **原始 R-SMILES M500：**checkpoint sweep 已完成，不要先继续调参。先迁回 evaluation manifest、至少 550K 的 `sampling_metadata.json`/`diagnostics.json`/log、run config、monitor 和训练日志，确认它与 global dev 的 reaction blocks 是否完全配对，并分解 invalid、valid、true-unique 和轨迹重合。随后只对 450K/550K 中的较强点补第二个 seed，并与匹配的原始 Atom baseline 比较。
4. **K=1000/K=2000：**只有数据统计、没有训练；除非 M500 的后续改进停滞，否则不应同时扩展多个 merge 深度。
5. **最终 test：**在 dev 上冻结 checkpoint 和 sampler 后，才运行一次完整 `src-test` / `tgt-test`；不要用 full test 继续调 checkpoint。

## 7. 相关代码和配置索引

| 功能 | 文件 |
|---|---|
| SPE 数据构造、`--merges K` | [`scripts/preprocessing/preprocess_spe.py`](../scripts/preprocessing/preprocess_spe.py) |
| SPE/Atom 数据审计 | [`scripts/preprocessing/spe_stats.py`](../scripts/preprocessing/spe_stats.py) |
| 初版 global Full-SPE B128 | [`configs/retro_spe_600k.yaml`](../configs/retro_spe_600k.yaml) |
| Full-SPE B128 800K P1-A/P1-B | [`configs/retro_spe_p1a_800k.yaml`](../configs/retro_spe_p1a_800k.yaml), [`configs/retro_spe_p1b_800k.yaml`](../configs/retro_spe_p1b_800k.yaml) |
| 原始 R-SMILES Full/M500 新训练配置 | [`configs/retro_spe_full_rsmiles_600k_bs256.yaml`](../configs/retro_spe_full_rsmiles_600k_bs256.yaml), [`configs/retro_spe_m500_rsmiles_600k_bs256.yaml`](../configs/retro_spe_m500_rsmiles_600k_bs256.yaml) |
| 统一采样与正式评分 | [`scripts/eval.py`](../scripts/eval.py), [`scripts/score_#global#.py`](../scripts/score_%23global%23.py) |

本文件汇总本机可复算证据，以及用户迁回的外机聚合报告；后者均显式标为“已报告、待 artifacts 核验”。未把未运行的 P1-B、缺少配对数据的定性反馈，或未迁回的 checkpoint 包装成已验证结论。
