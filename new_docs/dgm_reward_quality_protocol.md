# DGM reward 质量审计与下一步校准协议

更新日期：2026-08-11  
状态：已完成只读审计工具、单元测试、现有多时间点数据的训练／held-out 检查，以及一个与其隔离的 200-reaction reward holdout；尚未训练新的 reward 或重新训练 guidance。

## 1. 为什么现在先检查 reward？

当前逐步引导网络已经能学会“正向模型给高分的终点应排在前面”：延长到 2,000 个优化步后，同一中间状态内的排序准确率明显提高。但最终逆合成的 Top-1 仍低于普通 Euler。

这留下一个最重要的疑问：正向模型认为“能回到输入产物”的候选，是否真的更常是数据集中的正确反应物？如果这个分数会频繁给错误反应物高分，guidance 即使把它学得再准确，也可能把错误方向推到前面。因此下一阶段先验证 reward 本身，而不是继续增加 guidance 训练步数。

## 2. 本次审计如何避免答案泄漏？

训练或推理时使用的正向 reward 只读取两样东西：输入产物和生成出的候选反应物。它不读取真实反应物。

本次审计额外读取真实反应物，但**只在模型输出已经生成之后**，把它规范化并与候选规范化后比较，得到“这个候选是否与数据集答案相同”的离线标签。该标签只用于衡量 reward 的好坏：

```text
候选反应物 + 输入产物 ──正向模型──> reward
候选反应物 + 真实反应物 ──仅离线比较──> 正确 / 错误标签
```

标签不会写回 guidance `.pt` 数据，不会在采样时读取，也不能用 test target 选择 reward、checkpoint 或引导强度。

新工具为 [audit_guidance_reward_quality.py](/root/autodl-tmp/edit_flows/scripts/audit_guidance_reward_quality.py)。它报告两类 AUC：

- **全局 AUC**：随机抽取一个正确候选和一个错误候选时，reward 给前者更高分的概率（并列算半分）。
- **同一中间状态组内 AUC**：只在同一产物、相同中间状态、相同时间、四条独立后继之间比较。这更直接回答 guidance 训练真正需要的问题：此时应该偏向哪条后继？

## 3. 当前 forward-beam reward 的实测质量

审计对象是五个时间点（10/30/50/70/90）、每个状态四条后继的现有数据。训练记录来自 1,000 个训练反应；held-out 记录来自 200 个 validation 反应。二者没有共享原始反应。所有 candidate 与 target 都先移除 atom map、再做 RDKit canonical SMILES 比较。

| 数据 | 终点数 | 正确终点比例 | 非法终点比例 | 全局 AUC | 同组 AUC | 至少有一个正确终点的组 | 同时含正确/错误终点的组 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train-1000 | 20,000 | 39.42% | 11.88% | 0.6798 | 0.6965 | 60.02% | 42.02% |
| held-out val-200 | 4,000 | 39.80% | 12.10% | **0.6971** | **0.7308** | 60.20% | 42.40% |

在 held-out 数据中，80.09% 的正确终点拿到正的 forward reward；但 **46.05% 的错误终点也拿到正 reward**。五个时间点的 AUC 介于 0.6831–0.7096，没有某一个时间点完全失效。

这给出两面结论：

1. reward 不是随机噪声。它在未参与生成的反应上仍能明显区分正确和错误，也能在同一中间状态的四条后继中提供可学习方向。
2. reward 远不是“正确性真值”。近一半错误终点仍能被正向模型重构为产物；而约 40% 的共享状态组根本没有正确后继，无法教会模型“正确比错误更好”。这正解释了训练内排序变好却不一定带来 Top-1 改善。

完整 JSON 结果（不提交 Git）位于：

```text
/root/autodl-tmp/dgm_guidance_runs/
├── reward_quality_shared_train1000_t10_30_50_70_90.json
└── reward_quality_shared_val200_t10_30_50_70_90.json
```

## 4. 下一步：一个可证伪的 reward 校准 pilot

下一步不是直接训练一个大型新模型，而是先做一个低风险、可解释的校准 baseline。它的目标是把已有的正向信息变成更接近“候选可能正确”的分数；推理或 reward 计算时仍只使用产物和候选，不使用真实反应物。

### 4.1 数据划分

1. 现有 train-1000 的 20,000 个终点可作为校准训练样本；真实反应物只用于离线标签。
2. 新建一个来自**训练 split 中未使用原始反应**的 200-reaction reward holdout（建议编号 1,000–1,199），以免复用当前 guidance validation 或方法开发集。
3. 该 holdout 必须用同一共享状态／四后继流程产生候选，再离线标注；不读取 `evaluation_v2` 的 dev/confirm/final target，更不读取 test target。

为安全选择这个连续反应块，生成器已增加显式的 `--start_product` 参数；它在按 20 条 augmentation 折叠后生效，且已有单元测试，避免把 20 个 SMILES 写法误当成 20 个反应。保存的记录使用原始训练文件中的**绝对反应编号**，因此离线审计能准确映射回对应 target，而不会把第 1,000 个块误当成新文件中的第 0 个块。

### 4.1.1 独立 holdout 已构造（2026-08-11）

实际使用训练 split 的原始反应块 **[1,000, 1,200)**：它与原有 guidance reward 训练数据的 [0, 1,000) 不重叠，也不使用 validation、开发集、确认集、最终集或 `src-test`。固定配置为 100 个 Euler 步、step 10/30/50/70/90 五个时间点、每个状态 4 条独立后续、batch 32、seed 42。

| 项目 | 结果 |
|---|---:|
| 原始反应／终点记录／共享状态组 | 200 / 4,000 / 1,000 |
| 轨迹构造耗时 | 69.57 s（57.49 records/s） |
| forward-beam reward 耗时 | 40.48 s（857 个去重有效候选） |
| 轨迹阶段峰值 CUDA allocated / reserved | 0.526 / 1.242 GiB |
| 原始 forward reward 全局 AUC | 0.7073 |
| 原始 forward reward 同组 AUC | 0.6836 |
| 正确终点比例／非法终点比例 | 43.35% / 13.15% |
| 有正确与错误候选同时出现的组比例 | 42.60% |
| 错误终点取得正 reward 的比例 | 42.14% |

因此该 holdout 支持“reward 有一定方向、但仍会给大量错误终点高分”的已有结论；它是后续校准器唯一允许用于通过／失败判断的数据，而不是用来反复挑选开发集参数。原始生成文件与 JSON 不提交 Git，SHA-256 分别为：

```text
reward_holdout_shared_train200_start1000_validity.pt  4b0fb8042a49814b3c4c2daa6d9f92613bd3d3d71a16322b252cc962fce0bdf9
reward_holdout_shared_train200_start1000_beam.pt      a3a2a30f69d8a4a2a9893c016b6bed25360f9b4e3990555e210508adadfc21f7
```

### 4.2 第一版校准器

第一版只允许使用实际可获得的、无需答案的信息：

- 正向 beam 是否重构到产物及其排名；
- 正向模型对“候选反应物生成输入产物”的 teacher-forced 对数似然；
- 候选是否合法，以及候选/产物的简单长度或原子数一致性特征。

先训练一个小型线性或两层校准器，而不是立即重写 Molecular Transformer 或 Edit Flows。这样可以先回答“已有信号的组合能否改善 reward”，同时控制算力和过拟合风险。模型输出必须是单个可审计分数，并记录输入特征、数据范围和 checkpoint。

#### 4.2.1 已冻结的 P1：结构化线性校准（已执行，不通过）

为避免在 200-reaction holdout 上反复挑选特征，P1 在实现和运行前固定为**带 L2 正则的线性逻辑回归**。训练标签是“候选终点是否等于训练数据的真实反应物”；它只在训练阶段读取，模型输入绝不含 target。P1 的输入恰为下列七项、在未来生成一个候选终点后都可得到的数值：

1. forward beam 的倒数排名（未重构为 0，rank 1 为 1）；
2. 是否被 forward beam 重构到；
3. 候选反应物能否被 RDKit 解析；
4. 候选与输入产物的 token 长度相对差；
5. 候选与输入产物的原子数相对差；
6. 候选中的 SMILES 碎片数；
7. 当前 Euler 时间。

固定训练范围是原始训练反应 0–999 的既有 20,000 条终点记录；固定评估范围是新建的训练反应 1,000–1,199 的 4,000 条记录。所有连续特征只用训练范围计算均值和方差，逻辑回归使用固定 L2 `0.01`，不会用 holdout 调整正则、特征、阈值或 epoch。

P1 先**不**加入 teacher-forced forward likelihood：它过去单独的 correctness AUC 仅约 0.564，且加入它会引入一次额外的正向模型计算。若 P1 不通过，才可把“在不改变其他特征和划分的情况下追加 likelihood”作为单独记录的 P2；两者不能混在同一次结果中声称成功。

校准输出是“候选正确的估计概率”，用于连续排序。由于概率天然都大于 0，原协议中“错误终点得正分”的检查在 P1 中定义为：**在训练范围预先选取一个阈值，使校准器选中的候选比例等于 raw forward-beam 的正 reward 比例；冻结该阈值后，holdout 上错误候选被选中的比例不得高于 raw forward-beam。** 这样比较的是相同候选预算，而不是把“数值大于零”的字面定义误用到概率。

#### 4.2.2 P1 结果（2026-08-11）：全局区分变好，但不适合本任务的局部排序

P1 用 20,000 条训练终点拟合 2,000 个固定优化步，CPU wall 为 **18.01 s**；binary cross entropy 从 0.6706 收敛到 0.5672。它随后只在独立的 4,000 条／200-reaction holdout 上评估一次。校准器、带 `calibrated_reward` 的候选副本和 summary 都保存在本机，不提交 Git；后者又由现有只读审计工具独立复核，且确认副本不含任何 label、原 `reward` 完全未改写。

| 指标 | raw forward-beam | P1 校准后 | 校准后 − raw |
|---|---:|---:|---:|
| 全局 correctness AUC | 0.7073 | **0.7655** | **+0.0583**，bootstrap 95% CI [+0.0322, +0.0820] |
| 同一共享状态组内 AUC | 0.6836 | 0.6843 | +0.0007，CI [−0.0402, +0.0411] |
| 固定候选预算下错误候选被选中比例 | **42.14%** | 42.28% | +0.13 pp，CI [−3.35, +3.33] pp |
| 同一终点池重排 Top-1 | **41.5%** | 40.0% | −1.5 pp，CI [−9.0, +5.5] pp |
| 同一终点池重排 Top-3 | **76.5%** | 73.0% | −3.5 pp，CI [−7.5, +0.5] pp |
| 同一终点池重排 Top-10 / Oracle | 83.0% / 83.0% | 83.0% / 83.0% | 0 / 0 |

这不是“校准器完全没学到东西”：它确实在跨所有产物的候选对上提升了 AUC。但 DGM action guidance 最需要的是**同一产物、同一中间状态下的几条后继谁更好**；该指标没有达到预注册的 +0.02，而且候选池内真实的 Top-1/Top-3 排序更差。因此 P1 **不通过**全部门槛，不重建 guidance 数据、不重训 guidance、不运行开发集或 test。

本次实验文件的 SHA-256：

```text
calibrator.pt                 f19627ad3909c6d2deb2bfc4873258b91f67b7f5de6dda49b6289f386f7429a9
holdout_calibrated.pt         3b915ee81fcf9b6ad8d33e6b7c81bff7f40e4efb769ea685ce8568107a544e7f
summary.json                  1711ca200a7f2f98775bc0bf5b6bfbd24e0d818d6a0b6e9a3553b6680027e222
audit_calibrated_reward.json  a33f0253a7cd6b119e00b67a6f6187fd532dbab97476f3cd92c899e99a8a754a
```

#### 4.2.3 唯一允许的后续校准假设 P2：追加 teacher-forced likelihood（已执行，不通过）

P2 在 P1 结果确认前已经写入 4.2.1，因此不属于事后扫描。它只改变一件事：用同一个冻结的正向 Molecular Transformer，为每个“候选反应物 → 输入产物”计算长度归一化 teacher-forced log likelihood，并把它作为第八个特征追加到 P1 的七个特征后。数据划分、候选记录、L2、逻辑回归、训练步数、阈值规则、bootstrap、通过门槛和同候选池重排均保持不变。

P2 的 likelihood 必须作为新字段附加到 `.pt` 副本中，不能覆盖 raw forward-beam reward；虽然该分数过去单独 AUC 较弱，它或许能在 beam rank 相同的候选之间提供连续的 tie-break 信息。P2 若仍未同时满足第 4.3 节全部门槛，就停止 reward 校准支线，不再增加特征、扫描正则或重跑开发集。

实际实现新增 `append_likelihood` 模式：它读取已有 beam reward 数据、仅新增长度归一化的 `forward_log_likelihood`，保留每条记录原有的 `reward`、`forward_beam_rank` 和 provenance。20,000 条训练记录／4,000 条 holdout 记录的特征附加分别耗时 **8.03 s / 2.45 s**，GPU batch 均为 16。holdout 上 likelihood 单独的全局／同组 AUC 为 **0.5596 / 0.5532**，符合它只能作为连续补充信号、不能替代 beam reward 的预期。

P2 的逻辑回归仍为 2,000 个固定步，CPU wall **20.48 s**，训练 BCE 从 0.6706 降到 0.5655。其 holdout 结果为：

| 指标 | raw forward-beam | P2 校准后 | 校准后 − raw |
|---|---:|---:|---:|
| 全局 correctness AUC | 0.7073 | **0.7598** | **+0.0526**，bootstrap 95% CI [+0.0262, +0.0784] |
| 同一共享状态组内 AUC | 0.6836 | **0.7093** | **+0.0257**，CI [−0.0256, +0.0757] |
| 固定候选预算下错误候选被选中比例 | 42.14% | **41.26%** | −0.88 pp，CI [−4.39, +2.53] pp |
| 同一终点池重排 Top-1 | **41.5%** | 41.0% | −0.5 pp，CI [−8.0, +6.5] pp |
| 同一终点池重排 Top-3 | **76.5%** | 73.0% | −3.5 pp，CI [−8.0, +1.0] pp |
| 同一终点池重排 Top-10 / Oracle | 83.0% / 83.0% | 83.0% / 83.0% | 0 / 0 |

P2 是比 P1 更有意义的诊断：它首次让 **同一共享状态内**的 AUC 点估计超过 +0.02，同时降低固定预算误报率。但它仍使终点池 Top-1/Top-3 下降，没有满足“Top-1 不低于 raw，且 Top-3/Top-10/coverage 至少一项提高”的硬门槛。因此 **P2 也拒绝**，不重建 guidance reward、不重训 guidance、不运行开发集或 test，reward 校准支线至此关闭。

P2 相关文件 SHA-256：

```text
train_shared_anchor1000_t10_30_50_70_90_beam_likelihood.pt  20163e5b24fd7cacb1dca2331d088f3371de105c2402f68772ab2e8816c6a243
reward_holdout_shared_train200_start1000_beam_likelihood.pt f0ff064afe8456f24cf351699ea05503c0e8a14039db08603cd49a96e612649c
calibrator.pt                                               4cf6b8d33b67ef3e8281e61e6b331d948fbd315794d7a813c78852a8cd2305f9
holdout_calibrated.pt                                       01eaae92616c39dc81a53ecbd7dc2ba4e28b242a12050da8be950388922b98f5
summary.json                                                9f5f62d482d19f9814377aace01dc610562047ad930b3b1c17be7ac54a854baa
audit_calibrated_reward.json                                55e16d0dc64b34c5e128649ecac98154dae05a524b3d4d943f9c1cb44fd54b5b
```

### 4.3 通过门槛

在新的 200-reaction reward holdout 上，与原始 forward-beam reward 同时比较。只有全部条件满足才允许用新 reward 重建 guidance 数据：

1. 全局 AUC 和同组 AUC 均比原始 reward 至少提高 **0.02**，并按原始反应／共享状态组 bootstrap 记录置信区间；
2. 正确终点的平均分高于错误终点，且错误终点拿到正分的比例不能上升；
3. 用同一候选池做终点重排时，Top-1 不低于原始 forward reward，且 Top-3、Top-10 或覆盖率至少一项提高；
4. 计算成本、非法候选处理和所有输入输出哈希完整记录；
5. 不根据 `evaluation_v2/dev_unique1000_aug20` 的结果挑特征、阈值或 checkpoint。

P1 和 P2 都未通过；reward 校准支线已经关闭。不能再把训练时长、beta、分支数、正则或更多 endpoint 特征混入补救，也不能以 P2 的局部 AUC 点估计为理由重训 guidance。只有未来一个有独立建模动机、在本协议外预先定义的新方法，才可重新提出 reward／credit-assignment 实验。

## 5. 当前可复现命令

```bash
conda activate ef

python scripts/audit_guidance_reward_quality.py \
  --data /root/autodl-tmp/dgm_guidance_runs/train_shared_anchor1000_t10_30_50_70_90_beam.pt \
  --targets_file "datasets/USPTO_50K_PtoR_aug20_#global#/train/tgt-train.txt" \
  --vocab_file "datasets/USPTO_50K_PtoR_aug20_#global#/example.vocab.src" \
  --augmentation 20 --score_field forward_beam_rank \
  --output_json /root/autodl-tmp/dgm_guidance_runs/reward_quality_shared_train1000_t10_30_50_70_90.json

python scripts/audit_guidance_reward_quality.py \
  --data /root/autodl-tmp/dgm_guidance_runs/val_shared_anchor200_t10_30_50_70_90_beam.pt \
  --targets_file "datasets/USPTO_50K_PtoR_aug20_#global#/val/tgt-val.txt" \
  --vocab_file "datasets/USPTO_50K_PtoR_aug20_#global#/example.vocab.src" \
  --augmentation 20 --score_field forward_beam_rank \
  --output_json /root/autodl-tmp/dgm_guidance_runs/reward_quality_shared_val200_t10_30_50_70_90.json
```

独立 holdout 的可复现构造命令如下。`--start_product 1000` 是**完整反应块**偏移，不是原始文本行偏移；审计时保持 `--target_start_product 0`（默认），因为记录中已经存储了绝对编号 1000–1199。

```bash
python scripts/generate_shared_anchor_guidance.py \
  --checkpoint new_checkpoints/checkpoint_step600000.pt \
  --products_file "datasets/USPTO_50K_PtoR_aug20_#global#/train/src-train.txt" \
  --output /root/autodl-tmp/dgm_guidance_runs/reward_holdout_shared_train200_start1000_validity.pt \
  --augmentation 20 --start_product 1000 --max_products 200 \
  --n_steps 100 --n_children 4 --anchor_steps 10 30 50 70 90 \
  --batch_size 32 --device cuda --seed 42

python scripts/generate_forward_guidance_data.py \
  --input_data /root/autodl-tmp/dgm_guidance_runs/reward_holdout_shared_train200_start1000_validity.pt \
  --output_data /root/autodl-tmp/dgm_guidance_runs/reward_holdout_shared_train200_start1000_beam.pt \
  --checkpoint new_checkpoints/MIT_mixed_augm_model_average_20.pt \
  --vocab_file "datasets/USPTO_50K_PtoR_aug20_#global#/example.vocab.src" \
  --reward_mode beam_reconstruction --forward_beam_size 5 \
  --canonicalize_source --batch_size 16 --device cuda

python scripts/audit_guidance_reward_quality.py \
  --data /root/autodl-tmp/dgm_guidance_runs/reward_holdout_shared_train200_start1000_beam.pt \
  --targets_file "datasets/USPTO_50K_PtoR_aug20_#global#/train/tgt-train.txt" \
  --vocab_file "datasets/USPTO_50K_PtoR_aug20_#global#/example.vocab.src" \
  --augmentation 20 --score_field forward_beam_rank \
  --output_json /root/autodl-tmp/dgm_guidance_runs/reward_quality_holdout_shared_train200_start1000.json
```

P1 的完整可复现命令（**已运行且不通过，不要把输出直接作为 guidance reward**）：

```bash
python scripts/train_reward_calibrator.py \
  --train_data /root/autodl-tmp/dgm_guidance_runs/train_shared_anchor1000_t10_30_50_70_90_beam.pt \
  --train_targets_file "datasets/USPTO_50K_PtoR_aug20_#global#/train/tgt-train.txt" \
  --holdout_data /root/autodl-tmp/dgm_guidance_runs/reward_holdout_shared_train200_start1000_beam.pt \
  --holdout_targets_file "datasets/USPTO_50K_PtoR_aug20_#global#/train/tgt-train.txt" \
  --vocab_file "datasets/USPTO_50K_PtoR_aug20_#global#/example.vocab.src" \
  --output_dir /root/autodl-tmp/dgm_guidance_runs/reward_calibration_p1_train1000_holdout_start1000 \
  --augmentation 20 --l2 0.01 --max_steps 2000 --learning_rate 0.05 \
  --bootstrap_samples 2000 --seed 42
```

P2 的复现先附加 likelihood（两个输出均为本地实验资产），再在同一配置下加一个显式开关：

```bash
python scripts/generate_forward_guidance_data.py \
  --input_data /root/autodl-tmp/dgm_guidance_runs/train_shared_anchor1000_t10_30_50_70_90_beam.pt \
  --output_data /root/autodl-tmp/dgm_guidance_runs/train_shared_anchor1000_t10_30_50_70_90_beam_likelihood.pt \
  --checkpoint new_checkpoints/MIT_mixed_augm_model_average_20.pt \
  --vocab_file "datasets/USPTO_50K_PtoR_aug20_#global#/example.vocab.src" \
  --reward_mode append_likelihood --batch_size 16 --device cuda

python scripts/train_reward_calibrator.py \
  --train_data /root/autodl-tmp/dgm_guidance_runs/train_shared_anchor1000_t10_30_50_70_90_beam_likelihood.pt \
  --train_targets_file "datasets/USPTO_50K_PtoR_aug20_#global#/train/tgt-train.txt" \
  --holdout_data /root/autodl-tmp/dgm_guidance_runs/reward_holdout_shared_train200_start1000_beam_likelihood.pt \
  --holdout_targets_file "datasets/USPTO_50K_PtoR_aug20_#global#/train/tgt-train.txt" \
  --vocab_file "datasets/USPTO_50K_PtoR_aug20_#global#/example.vocab.src" \
  --output_dir /root/autodl-tmp/dgm_guidance_runs/reward_calibration_p2_train1000_holdout_start1000 \
  --augmentation 20 --include_forward_log_likelihood \
  --l2 0.01 --max_steps 2000 --learning_rate 0.05 \
  --bootstrap_samples 2000 --seed 42
```
