# SPE 600k 实验记录与分析

记录日期：2026-08-18 UTC<br>
训练环境：conda `ef`，单卡 NVIDIA GeForce RTX 3090 24 GB，14 vCPU Intel Xeon Gold 6330<br>
当前状态：训练已完成；20% shuffled test 子集上的 SPE 对照和 original baseline 评估已完成；完整 test 尚未完成。

本文记录 SPE 分支从 50k pilot 到 600k 主训练的主要实验数据，并说明训练收敛、吞吐和采样结果的正确解读。评估结果均注明数据范围，避免把子集结果误认为完整 USPTO-50K test 结果。

## 1. 结论摘要

- SPE 600k 训练链路稳定完成 600,000 steps，没有 NaN/Inf 或 monitor anomaly。
- validation loss 从 step 100k 的 6.1660 降到 step 580k 的 4.5190，step 580k 是最佳 checkpoint；step 600k 回升到 4.7016。因此模型已经进入平台区，但不能据此断言已经达到全局最优或完全收敛。
- 本报告的主问题是 SPE 是否优于原 tokenizer，而不是比较两个 sampler。为此，使用同一组 1,001 个 reaction blocks、同一 ordinary Euler N=9 protocol，比较 SPE `checkpoint_step600000.pt` 和 `new_checkpoints/checkpoint_step600000.pt`。
- 原 tokenizer baseline 的 Top-1/3/5/10 为 **58.541% / 77.622% / 81.918% / 85.614%**，Oracle **90.909%**；SPE final 为 **51.349% / 70.330% / 74.825% / 77.323%**，Oracle **87.612%**。因此当前 600k SPE 在准确率上仍落后 baseline 约 7–8 pp，Oracle 落后 3.297 pp。
- SPE 的 ordinary Euler 采样速度约为 baseline 的 **2.42×**（58.73% 时间节省），true-unique 候选多 13.929 个/reaction，但 invalid 首候选高 10.185 pp、valid 候选少 18.570 个/reaction。速度和表面多样性收益尚未转化为更高准确率。
- R9K1M2/Euler N=9 的结果保留为 SPE 内部 sampler 对照，不作为判断 SPE 有效性的主证据。
- 这组结果只能作为 20% 子集上的正式预评估，不能替代完整 test。完整 test 有 5,007 个 reaction block、100,140 行；目前没有完整 test 的最终 Top-K 报告。
- batch 256/512 的吞吐探针只带来约 4.0%–6.1% 的 pairs/s 提升，却改变了有效 batch、学习率和梯度裁剪设置；因此 600k 主训练仍采用原始 batch 128，未把探针结果当作质量结论。

## 2. 实验谱系与数据口径

### 2.1 SPE 数据集

SPE 数据目录为 `datasets/USPTO_50K_PtoR_aug20_#global#_SPE`，是已经完成对齐和 `#global#` 表示的数据。文件行数如下：

| split | reaction block 数 | augmentation | 文件行数 |
|---|---:|---:|---:|
| train | 40,003 blocks（约） | 20 | 800,060 |
| val | 5,001 blocks（约） | 20 | 100,020 |
| test | 5,007 blocks | 20 | 100,140 |

train/val/test 的实际数据是按 augmentation 展开的，因此训练 loader 看到的是 800,060、100,020 和 100,140 对，而正式 scorer 会按 augmentation=20 聚合回 reaction block。

SPE vocabulary：3,035 个真实 token，加入 `<PAD>/<BOS>/<GAP>/<UNK>` 后为 3,039 个 model tokens。600k 模型参数量为 15,820,993。

### 2.2 50k pilot

50k pilot 的完整记录见 [`SPE_training_pilot_report.md`](SPE_training_pilot_report.md)。它的主要用途是验证数据、vocabulary、训练和采样链路，不应作为 SPE 最终质量结论：

- 训练完成 50,000 steps，validation loss 为 6.8347，耗时 3,580.74 s（约 59.7 min）。
- 当时的质量比较同时存在训练步数不一致（SPE 50k 对旧 baseline 600k）和采样 action-support bug 两个混杂因素。
- 因此 pilot 中的 Top-K/Oracle 数字不用于评价 600k SPE，也不与本文的 600k 子集结果直接拼接比较。

## 3. 600k 训练配置与运行结果

配置文件：[`retro_spe_600k.yaml`](../configs/retro_spe_600k.yaml)。核心配置为：

| 项目 | 值 |
|---|---:|
| device / topology | 单进程、单卡 CUDA |
| batch size | 128 |
| validation batch size | 128 |
| effective global batch | 128 |
| total steps | 600,000 |
| learning rate factor | 1.0 |
| warmup steps | 8,000 |
| max grad norm | 0.0（不裁剪） |
| train/sample scheduler | cubic |
| alignment | `opt` |
| sampling steps | 100 |
| data workers | 2 |
| validation | step 100k 开始，每 20k 一次，完整 validation |
| seed | 42 |

主要运行产物目录：

```text
checkpoints/retro_spe_600k/USPTO_50K_PtoR_aug20_#global#_SPE/2026-08-14_15-24-14/
```

最终 checkpoint：`checkpoint_step600000.pt`。<br>
最佳 validation checkpoint：`checkpoint_best.pt`，对应 step 580,000，精确 best validation loss 为 4.51897468189315。

训练摘要：

| 指标 | 值 |
|---|---:|
| completed steps | 600,000 / 600,000 |
| elapsed time | 40,901.27 s（约 11.36 h） |
| average speed | 14.6695 steps/s |
| 约等效训练样本遍数 | 约 96（600,000 × 128 / 800,060） |
| peak CUDA allocated | 3.50 GB |
| peak CUDA reserved | 6.58 GB |
| monitor records | 580 |
| monitor anomalies | 0 |
| model parameters | 15,820,993 |

训练曾从 step 20,000 checkpoint 恢复；恢复逻辑的 CUDA RNG 状态问题已修复并提交在 commit `21ec988`。最终训练 summary、checkpoint 和日志均位于上述运行目录。

### 3.1 Train loss 的解读

train monitor 是按间隔采样的 batch 估计，单点波动较大，不能用某一个 step 的 loss 判断收敛。按区间平均值看：

| step 区间 | monitor loss 均值 |
|---|---:|
| 20k–50k | 8.4599 |
| 50k–100k | 6.6655 |
| 100k–200k | 5.4433 |
| 200k–300k | 4.8515 |
| 300k–400k | 4.1360 |
| 400k–500k | 4.0957 |
| 500k–600k | 3.6695 |
| 550k–600k | 3.4005 |

step 600k 的单个 monitor batch 为 loss 3.0129；该值与 step 580k 的 2.0663 一样都只是单 batch 采样，不应与 validation loss 直接比较。整个过程没有非有限参数、梯度或 monitor anomaly。

### 3.2 Validation loss 曲线

完整 validation 记录如下：

| step | val loss | step | val loss |
|---:|---:|---:|---:|
| 100k | 6.1660 | 360k | 4.8688 |
| 120k | 5.8677 | 380k | 4.8910 |
| 140k | 5.6332 | 400k | 4.7651 |
| 160k | 5.5425 | 420k | 4.8741 |
| 180k | 5.3812 | 440k | 4.8708 |
| 200k | 5.2701 | 460k | 4.6211 |
| 220k | 5.3434 | 480k | 4.6703 |
| 240k | 5.1977 | 500k | 4.5192 |
| 260k | 5.0970 | 520k | 4.8102 |
| 280k | 5.1059 | 540k | 4.6758 |
| 300k | 5.0539 | 560k | 4.5541 |
| 320k | 4.9311 | 580k | **4.5190** |
| 340k | 4.9116 | 600k | 4.7016 |

必要说明：

1. 100k–580k 总体仍在下降，说明 600k 之前确实学到了更多；但 500k–580k 已基本平台化，且 600k 反弹约 0.183。
2. 这更适合描述为“后期平台区、最佳点出现在 580k”，而不是“严格完全收敛”。
3. 当前 Noam 学习率在 warmup=8k 时峰值约 `6.99e-4`，step 600k 约 `8.07e-5`，约为峰值的 11.5%。因此“后期学习率可能偏保守”是合理的待验证假设，但现有曲线本身不能证明提高 learning-rate factor 一定会提升 Top-K。

## 4. batch / 吞吐探针

以下是独立的约 1,000-step probe，不是完整训练质量实验。三种配置都保持 `batch_size × total_steps = 76.8M` 的样本预算近似一致，但 batch 256/512 同时改变了 learning-rate factor、warmup 和梯度裁剪，不能和 batch 128 的结果混为同一优化协议。

| 配置 | workers | steps/s | pairs/s | peak reserved | grad clip |
|---|---:|---:|---:|---:|---:|
| batch 128，factor 1.0 | 2 | 14.20（probe） | 1,817 | 约 5.07 GB | 无 |
| batch 512，factor 1.6，warmup 6k | 4 | 3.693 | 1,891 | 18.48 GB | 1.0 |
| batch 512，factor 1.6，warmup 6k | 2 | 3.754 | 1,922 | 18.48 GB | 1.0 |
| batch 256，factor 1.3，warmup 8k | 2 | 7.530 | 1,928 | 9.62 GB | 1.0 |

结论是：增大 batch 后 steps/s 下降，pairs/s 只提高约 4.0%–6.1%，显存压力显著增加。batch 256 的吞吐略好于 batch 512，但没有对应的 validation 或 test 证据。因此本次主训练选择 batch 128，保持与既有 SPE 训练协议的可比性；若要尝试更高 learning-rate factor，应先做短 continuation/validation probe，单独隔离 learning rate 变量。

## 5. 600k checkpoint 的采样评估

### 5.1 评估数据和 protocol

完整 test 为 100,140 行，即 5,007 个 reaction block × 20 augmentation。为避免一次完整采样耗时过长，先生成了一个确定性的 20% reaction-block 子集：

- SPE 子集目录：`datasets/USPTO_50K_PtoR_aug20_#global#_SPE/test_shuffled20pct_seed20260815/`
- 原 tokenizer matched 子集目录：`datasets/USPTO_50K_PtoR_aug20_#global#/test_shuffled20pct_seed20260815/`
- 来源：完整 test 的 5,007 个 reaction blocks
- 抽样：seed `20260815`，随机抽取 1,001 个 block
- 保持 block 完整：每个 reaction 的 20 条 augmentation 不拆分
- 两个子集都是 20,020 行（1,001 × 20），且使用完全相同的 `selected_original_reaction_indices`
- 两个表示的 100,140 行源/目标在去掉 token 空格后逐行一致；因此它们是同一批化学反应，只是分别使用 SPE 和原 tokenizer 表示
- SPE manifest：[`manifest.json`](../datasets/USPTO_50K_PtoR_aug20_%23global%23_SPE/test_shuffled20pct_seed20260815/manifest.json)
- baseline matched manifest：[`manifest.json`](../datasets/USPTO_50K_PtoR_aug20_%23global%23/test_shuffled20pct_seed20260815/manifest.json)

旧 checkpoint 不能直接读取 SPE tokenized 文件，因为两者词表不同。这里的“相同测试数据”因此按 reaction identity 和 augmentation block 对齐，而不是错误地把 SPE token 字符串直接喂给原 tokenizer 模型。

主比较严格使用同一 ordinary Euler protocol：

| 项目 | 设置 |
|---|---|
| device / batch | CUDA / 32 |
| random seed | 42 |
| augmentation | 20 |
| ordinary Euler | N=9，100 steps，cubic |
| SPE checkpoint | `checkpoints/retro_spe_600k/.../checkpoint_step600000.pt` + SPE vocab |
| 原 tokenizer checkpoint | `new_checkpoints/checkpoint_step600000.pt` + 原 tokenizer vocab |
| scorer | 项目正式 `#global#` scorer，Top-K 到 10，按 20 augmentation 聚合 |
| 输出 | 每组 180,180 条预测（20,020 输入行 × 9 samples） |

当前评估使用包含 action-support 修复的代码状态，metadata 记录 commit `21ec988`。修复的核心是允许合法的 `INS(pos=0)`，禁止改写/删除 BOS，并过滤 special/no-op token；两组模型使用同一套修复后的 sampler 逻辑。

### 5.2 主结果：SPE vs 原 tokenizer baseline

下面的 Top-K/Oracle 来自正式 scorer；invalid 和 unique/valid 候选数来自同一采样的 diagnostics。Oracle 是 1,001 个 reaction blocks 中任意候选命中目标的比例。

| 模型 / 表示 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle-any | Invalid（首候选） | mean true-unique / reaction | mean valid / reaction | 采样时间 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原 tokenizer，600k | **58.541%** | **77.622%** | **81.918%** | **85.614%** | **90.909% (910/1001)** | **12.008%** | 22.983 | **158.684** | 3,501.3 s（58.4 min） |
| SPE，600k final | 51.349% | 70.330% | 74.825% | 77.323% | 87.612% (877/1001) | 22.193% | **36.912** | 140.114 | **1,445.1 s（24.1 min）** |
| SPE − 原 tokenizer | −7.192 pp | −7.292 pp | −7.093 pp | −8.291 pp | −3.297 pp | +10.185 pp | +13.929 | −18.570 | 2.42× faster |

主要解读：

- **准确率：SPE 当前没有超过 baseline。** Top-1/3/5/10 分别低 7.192/7.292/7.093/8.291 pp，Oracle 低 3.297 pp。
- **效率：SPE 的收益明确。** 同样是 20,020 个输入行、9 条 Euler 轨迹、100 steps，SPE 采样耗时从 3,501.3 s 降到 1,445.1 s，约 2.42× 加速，时间减少 58.73%。
- **多样性：SPE 产生更多不同的 canonical 候选。** true-unique 多 13.929 个/reaction，但首候选 invalid 高 10.185 pp，valid 候选少 18.570 个/reaction；因此“更多 unique”主要伴随有效性损失，并没有转化为更高 Oracle。

这组控制变量结果回答的是 SPE 的有效性：SPE 600k 已证明序列压缩带来显著推理加速，但在当前训练和采样协议下尚未证明能保持原 tokenizer 的候选质量。

### 5.3 辅助结果：SPE 内部 Euler 与 R9K1M2

这不是 SPE vs baseline 的主结论，只用于说明 SPE checkpoint 在不同 sampler 下的行为。R9K1M2 相对 SPE Euler N=9：Top-1/3/5/10 分别为 +2.997/+1.898/+1.399/+1.198 pp，Oracle +0.300 pp，true-unique +1.190 个/reaction；采样时间增加约 24.3%，首候选 invalid 增加 0.614 pp。

因此 R9K1M2 适合作为后续 sampler 研究的辅助对照，但不能用它与 Euler 的差异替代 SPE 与原 tokenizer 的公平比较。

### 5.4 SPE final 与 best checkpoint 对照

为回答“应该使用 final 还是 best checkpoint”，同一子集上也评估了 step 580k 的 `checkpoint_best.pt`：

| checkpoint | sampler | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid（首候选） | true-unique / reaction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| best，580k | Euler N=9 | 50.649% | 70.629% | 74.825% | 77.223% | 87.113% (872/1001) | 22.807% | 36.208 |
| best，580k | R9K1M2 | 53.447% | 71.329% | 75.225% | 78.422% | **88.212% (883/1001)** | 23.801% | 38.168 |
| final，600k | Euler N=9 | **51.349%** | **70.330%** | **74.825%** | **77.323%** | 87.612% (877/1001) | **22.193%** | 36.912 |
| final，600k | R9K1M2 | **54.346%** | **72.228%** | **76.224%** | **78.521%** | 87.912% (880/1001) | **22.807%** | 38.102 |

在主比较使用的 ordinary Euler 上，final 的 Top-1/3/10 高于 best，Top-5 持平；R9K1M2 仅作为辅助 sampler 时，best 的 Oracle 比 final 高 0.300 pp。整体没有足够证据把 best 设为默认，final 更适合作为当前主报告 checkpoint，best 作为敏感性对照保留。

### 5.5 关于 unique rate 和 invalid rate

旧 scorer 的 legacy unique-rate 字段可能出现超过 100% 的数值（例如 110.856%），这是历史分母定义造成的，不能直接当作“候选有多少比例不同”。本文使用两个更直观的 diagnostics：

- `mean true-unique candidates / reaction`：canonicalize 后的真正不同候选数；
- `mean valid candidates / reaction`：去除 invalid 后的候选数。

在 SPE final checkpoint 上，R9K1M2 的 true-unique 为 38.102，高于 Euler 的 36.912；但 valid 候选数反而略低（138.500 对 140.114）。更重要的是，主比较中 SPE Euler 的 true-unique 为 36.912，虽然高于原 tokenizer 的 22.983，但 SPE 的 invalid 更高、valid 候选更少，说明不能只看 unique 数字判断 SPE 是否有效。

## 6. 评估产物索引

每个目录包含 `predictions.txt`、`sampling_metadata.json` 和 `diagnostics.json`：

- [final Euler N=9](../results/spe600k_subset20_seed20260815_final_euler_n9/)
- [final R9K1M2](../results/spe600k_subset20_seed20260815_final_r9k1m2/)
- [best Euler N=9](../results/spe600k_subset20_seed20260815_best_euler_n9/)
- [best R9K1M2](../results/spe600k_subset20_seed20260815_best_r9k1m2/)
- [原 tokenizer matched baseline Euler N=9](../results/original600k_subset20_seed20260815_euler_n9/)

训练摘要：[training_summary.json](../checkpoints/retro_spe_600k/USPTO_50K_PtoR_aug20_%23global%23_SPE/2026-08-14_15-24-14/training_summary.json)。<br>
训练日志：[resume_attempt_3.log](../checkpoints/retro_spe_600k/USPTO_50K_PtoR_aug20_%23global%23_SPE/2026-08-14_15-24-14/resume_attempt_3.log)。

## 7. 当前判断与后续建议

当前能下的结论是：

1. **训练稳定性：通过。** 600k 训练完成，validation 在后期平台区，未发现训练数值异常。
2. **SPE 是否已完全收敛：不能确认。** 580k 是 validation 最佳点，600k 轻微回升；后期学习率较低，所以仍存在“学习率偏保守”这一待验证假设。
3. **SPE 准确率：当前不如原 tokenizer baseline。** 在同一 reaction-block 子集、同一 ordinary Euler N=9 protocol 下，SPE Top-1/3/5/10 低约 7–8 pp，Oracle 低 3.297 pp，invalid 高 10.185 pp。
4. **SPE 效率：明显更好。** 同样的 9 条 Euler 轨迹，SPE 采样约 2.42× 加速；但 true-unique 增加伴随 valid 候选减少和 invalid 增加，当前不能把它解释为质量提升。
5. **final 还是 best：当前主结果使用 final。** final 的 ordinary Euler Top-K 整体不低于 best；best 作为敏感性对照保留。
6. **R9K1M2 仅是辅助 sampler 对照。** 它不能替代 SPE 与原 tokenizer 的公平比较。
7. **完整 test：尚未完成。** 之前启动过 full-test Euler，但因完整采样耗时较长在早期停止，部分输出不作为正式结果。本文所有正式数字都明确标注为 1,001 reaction blocks 的 20% 子集结果。

如果继续研究 SPE，优先级建议是：

- 先在固定子集上做短 continuation probe，保持 batch、数据和 sampler 不变，只比较 learning-rate factor 1.0 与一个温和增大值，并以 validation loss、Top-K、Oracle 和 invalid 同时判定；
- 如果研究目标是保留准确率，下一步应优先做固定 sampler 的 SPE continuation / learning-rate probe，先确认 SPE 模型质量是否仍受训练不足或后期学习率影响；
- 如果研究目标是推理效率，SPE 的 2.42× 加速值得保留，但必须把 invalid rate 作为首要修复目标，而不能只优化 unique；
- 需要对外或做最终论文比较时，再用完全相同 protocol 跑完整 5,007 reaction-block test，并至少汇报原 tokenizer baseline、SPE Euler；R9K1M2 单独作为 sampler ablation。

本报告不修改现有训练配置、checkpoint 或评估结果；当前工作区中的 checkpoints、probe 配置和子集数据仍属于本地实验产物，未在本次记录动作中强行纳入 Git。

## 8. SPE 后续验证计划、判断依据与止损标准

本节是后续工作的**预注册式计划**：先写清楚为什么做、如何比较、什么结果算有效，避免根据已经看见的 test 子集结果不断调整超参数。除已有的 600k 结果外，本节中的实验均为**待执行**，不能把占位符当作结果。

### 8.1 为什么不能直接把问题归因为“训练步数不足”

当前证据同时支持“训练协议尚未为 SPE 适配”和“表示方式可能与 Edit Flows 不兼容”两类假设：

| 观察到的事实 | 含义 | 对后续实验的影响 |
|---|---|---|
| SPE 的 Top-1 比原 tokenizer 低 7.192 pp，Oracle 低 3.297 pp，首候选 invalid 高 10.185 pp | 差距不是只看 unique 或单个 checkpoint 就能忽略的小波动 | 必须同时看 Top-K、Oracle 和 invalid，不能只优化速度或 unique |
| SPE 平均 aligned length 从 50.809 降至 12.057（约 4.21× 更短） | 固定 `batch_size=128` 的单次更新看到的状态位置显著减少 | 原 tokenizer 的 batch/学习率/warmup 不能默认就是 SPE 的最佳协议 |
| 但总编辑操作数只从 5,752,040 降到 3,856,561（约少 33%） | 不能据长度比例直接断定需要 4.21× 更多 steps；SPE 每个 token 也承载了更多结构 | 应用受限的 continuation 来检验训练预算，而不是直接做数倍步数的大训练 |
| model vocab 从 73 增至 3,039；每个位置的 INS/SUB/DEL 动作空间从 `2×73+1=147` 增至 `2×3039+1=6079`（约 41×） | 选对 fragment 的分类问题明显更难；序列变短不保证生成决策变简单 | 需要单独诊断“选错 mode”还是“mode 正确但选错 token”，并保留一个中等词表 SPE 对照 |
| SUB 占比从 7.840% 升至 37.115% | SPE 把部分细粒度变化压成 fragment-level substitution；错误 fragment 更可能破坏 SMILES 合法性 | invalid 的根因可能是表示/编辑机制，而不仅是学习率 |
| val/test OOV token rate 仅约 `4e-5` | 几乎可以排除“未见 token”是主因 | 不把扩大词表或处理 OOV 作为优先修复方向 |
| 600k 训练数值稳定，但后期 validation loss 在约 4.5–4.8 平台波动 | 仍有后期学习率偏保守的可能；但 validation 每次重采样时间 `t`，580k 与 600k 的小差异不能单独证明过拟合或完全收敛 | checkpoint / 学习率判断必须以固定生成评估为主，不能只选最低 val loss |
| 未裁剪 gradient norm 的中位数约 11.8、最小约 5.68、P99 约 36.1 | `max_grad_norm=1.0` 会在几乎全部更新上强力裁剪，不是轻微稳定化 | 初始 continuation 不加入 `max_grad_norm=1.0`，避免把 LR、batch、clip 混为一个实验 |
| batch 256/512 probe 的 pairs/s 仅比 batch 128 高约 4–6% | 扩大 batch 没有明显的吞吐理由，且会同时改变优化步数与梯度噪声 | 第一轮不改变 batch；不以 256/512 作为默认“改进” |

因此，当前最合理的顺序是：先确认是否存在可恢复的优化/采样问题；若没有，再用一次小范围的表示层对照检验“大词表 fragment 编辑”是否是主因。不会直接将 SPE 训练到数百万 steps，也不会同时改 batch、学习率、warmup、clip 和 sampler。

### 8.2 评估纪律：先建立 validation-dev，冻结现有 test 子集

当前 20% shuffled test 子集已经用于探索性分析，后续不应用它选择学习率、batch 或 tokenizer。正式流程如下：

1. 从 `val` 的 5,001 个 reaction blocks 中，以固定 seed 抽取约 1,000 个完整 block；每个 block 保留 20 条 augmentation。
2. 为原 tokenizer 和 SPE 各生成一个 matched 表示，并验证去掉 token 空格后的 src/tgt 化学字符串及 reaction indices 完全一致。
3. 在该 **validation-dev** 上先评估原 tokenizer 600k baseline 和当前 SPE 600k final，使用相同 ordinary Euler：`N=9`、100 steps、cubic、相同 scorer/augmentation/n-best；候选模型用两个 sampling seed 报告均值与范围。
4. 所有调参与模型选择只依据 validation-dev。只有通过预设门槛的候选，才在当前冻结的 20% test 子集上确认；完整 5,007-block test 留给最终结论。

这样可以避免把 test 集变成超参数搜索集，也能区分 checkpoint/sampler 随机性与真实改进。

本轮实际采用项目已经构造并审计过的
`datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/dev_unique1000_aug20/`
作为 validation-dev：它包含 1,000 个 reaction blocks，选择 seed 为 `20260811`，并隔离了原始 reaction index `[0, 600)`。随后用
[`scripts/project_reaction_split.py`](../scripts/project_reaction_split.py)
将完全相同的 reaction indices 投影到
`datasets/USPTO_50K_PtoR_aug20_#global#_SPE/evaluation_v2/dev_unique1000_aug20/`。
逐行去除 token 空格后，SPE 与原 tokenizer 的 20,000 条 source/target 字符串均一致；投影脚本、SPE split manifest 和数据已提交。

### 8.3 待执行任务与实验登记表

#### P0：不训练的诊断与采样敏感性检查

目的：区分模型本身不会预测、Euler 数值积分不够、以及合法性约束缺失三种原因。

- 在固定的真实中间状态上做 teacher-forced 诊断：分别报告 INS/DEL/SUB mode 准确率；在 mode 正确条件下报告 token Top-1/Top-5；并按 token 频率和 INS/SUB 分组。
- 对 invalid 产物按 RDKit/SMILES 失败类型、最终编辑 mode、fragment 频率和编辑位置统计，确认 invalid 是否集中来自 fragment substitution。
- 在 validation-dev 的固定小子集上比较 Euler 100 与 200 steps。SPE 100 steps 当前耗时约 24.1 min；若近似线性，200 steps 仍接近或快于原 tokenizer 100 steps 的 58.4 min，因此是公平的速度—质量折中检查。

| 实验 ID | 状态 | checkpoint / sampler | 关键设置 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid | true-unique | 时间 | 结论 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P0-base | 待执行 | original 600k / Euler | validation-dev，N=9，100 steps，seed 42/43 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | baseline 参照 |
| P0-spe100 | 待执行 | SPE final 600k / Euler | 同上，100 steps | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 当前 SPE 参照 |
| P0-spe200 | 待执行 | SPE final 600k / Euler | 同上，200 steps | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 判断积分步数是否是主因 |

P0 的判读规则：若 200 steps 仅带来很小的 Top-K/Oracle 变化，且 invalid 基本不降，则停止继续增加 Euler steps；问题主要在模型/表示，而非数值积分。若 200 steps 明显降低 invalid 并提升 Oracle，则将其作为 SPE 的速度匹配推理协议，但仍需与原 tokenizer 在实际时间预算下重新比较。

P0 已先完成 20 个 reaction blocks 的 smoke check。原 tokenizer 和 SPE 均在 `conda ef`、CUDA、当前 action-support 修复代码下成功生成并评分；scorer 正确识别 `20 reactions × 20 augmentations × 9 candidates` 的布局。smoke 仅用于链路检查，不用于质量结论：

| smoke | Top-1 | Top-3 | Oracle | 首候选 invalid | true-unique / reaction | valid / reaction | 采样时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| original，20 reactions | 45.0% | 55.0% | 85.0% | 8.25% | 23.70 | 165.35 | 68.4 s |
| SPE，20 reactions | 55.0% | 65.0% | 85.0% | 24.0% | 37.15 | 137.60 | 32.1 s |

smoke 期间曾错误地从 base Python 3.8 启动，触发 `tuple[...]` 类型错误；显式激活 `ef`（Python 3.10）后通过。该失败属于环境调用错误，不属于模型、数据或 checkpoint 错误。完整 validation-dev 的 seed 42 结果仍待填入下表。

#### P1：两条受限的训练 continuation

目的：单独回答“是否只是训练步数不够”以及“后期学习率是否过低”。两条实验必须从**同一个 600k final checkpoint**分叉，使用独立输出目录，不覆盖现有 600k run。

| 实验 ID | 状态 | 起点 → 终点 | batch | learning-rate factor | warmup | grad clip | 其他设置 | 预期回答的问题 |
|---|---|---|---:|---:|---:|---:|---|---|
| P1-A | 待执行 | 600k → 800k | 128 | 1.0 | 8k（原值） | 0.0 | 数据、seed、模型、sampler 不变 | 单纯多训练 200k steps 是否有效 |
| P1-B | 待执行 | 600k → 800k | 128 | 1.3 | 8k（原值） | 0.0 | 仅提高后期 LR；其余同 P1-A | 当前后期 LR 是否过于保守 |

两条 continuation 各约 200k steps；按本次 14.67 steps/s 的实测速度估算，各约 3.8 h。这里不改 batch、不加 `clip=1.0`、不改 sampler。现有 Noam scheduler 在 resume 时恢复的是 completed step；600k 后已经在 decay 区，单改 `warmup_steps` 并不会形成真正的 warm restart，因此不把 warmup 伪装成有效变量。

| 实验 ID | 状态 | val loss（固定诊断） | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid | true-unique | valid candidates | 采样时间 | 相对 P0-spe100 的判断 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P1-A | 待执行 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待分析 |
| P1-B | 待执行 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待分析 |

P1 的判读规则：候选至少应做到 **Top-1 提升 ≥2.0 pp，且同时满足 Oracle 提升 ≥1.0 pp 或 invalid 降低 ≥3.0 pp**；Top-10 不应下降超过 1.0 pp。只有达到此“有真实信号”的门槛，才值得继续优化训练协议或延长该分支。只改善 train/val loss、只增加 unique、或只提升 Top-1 但牺牲 Oracle/valid，不算通过。

#### P2：一次中等词表 SPE 表示对照（仅在 P1 未解决问题时执行）

目的：检验当前 3,035-token SPE 是否因动作空间过大、fragment substitution 过多而不适合 Edit Flows，而不是泛化地否定所有 SPE。

设计原则：只做一个中等词表候选（建议约 512–1,024 个真实 token），目标是使序列仍显著短于原 tokenizer、但减少 INS/SUB 的 token 选择空间。重新生成对齐数据，保持模型主体、训练数据划分和正式评估 protocol 不变；不同时再做模型增宽、语法约束解码或 batch 网格搜索。

| 实验 ID | 状态 | tokenizer / vocab | 训练协议 | 目标 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid | true-unique | 时间 | 结论 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P2-medium-SPE | 待执行 | 待定：约 512–1,024 token | 待定：依据 P1 的有效设置；其余保持可比 | 保留加速、降低 fragment 选择难度 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待分析 |

如果 P1 已明显改善并接近可行性门槛，P2 可以不执行；如果 P1 无效，P2 是本路线最后一个优先级高、能直接区分机制的表示层实验。更大模型、reaction-aware tokenizer、SMILES grammar-constrained decoding 都属于更高成本的新研究方向，不能在当前 SPE 分支没有证明基本可行前无限追加。

### 8.4 分阶段止损与继续标准

为了避免“只要略有改善就继续训练”的无止境搜索，预先采用三层标准。所有百分比比较均在 validation-dev 上以两个 sampling seed 的均值判断，并在冻结 test 子集确认。

| 决策层级 | 继续条件 | 停止条件 | 后续动作 |
|---|---|---|---|
| 超参分支（P1） | 至少一个 P1 候选满足：Top-1 `+≥2.0 pp`，且 Oracle `+≥1.0 pp` 或 invalid `−≥3.0 pp`；Top-10 不显著下降 | P1-A、P1-B 均不满足 | 停止继续扫描 LR、steps、warmup、batch、clip；不以“再多训一点”作为默认方案 |
| 当前 large-vocab SPE 是否值得继续研究（P2 后） | 相对 original baseline：Top-1 距离不超过 3.5 pp、Oracle 距离不超过 2.0 pp、invalid 高不超过 6.0 pp，并保留 `≥1.5×` 加速 | P2-medium-SPE 仍达不到上述任一关键门槛，或 invalid/Oracle 无实质改善 | 停止将“当前 SPE + Edit Flows”作为主线；原 tokenizer 保持默认方案 |
| 是否可替代原 tokenizer | Top-1 距离不超过 2.0 pp、Oracle 距离不超过 1.5 pp、invalid 高不超过 3.0 pp，且保留 `≥1.5×` 加速 | 未达到 | 可以保留为速度探索分支，但不能宣称为主方法或替代 baseline |

以当前 20% test 子集的数字作直观参考，第二层的“值得继续”大致对应：Top-1 `≥55%`、Oracle `≥89%`、invalid `≤18%`；第三层的“可替代”大致对应：Top-1 `≥56.5%`、Oracle `≥89.4%`、invalid `≤15%`。validation-dev 的实际 baseline 数字可能略有不同，最终以相对阈值为准。

### 8.5 后续结果分析模板

每完成一个实验，按以下顺序补充本报告，而不是只记录一个最好分数：

1. **可比性检查**：checkpoint 起点、数据 reaction indices、词表、sampler、steps、seed、scorer 是否与计划一致；若不同，明确写为探索性结果。
2. **主指标**：补齐 Top-1/3/5/10、Oracle、invalid、true-unique、valid candidates 和端到端采样时间。
3. **相对变化**：同时相对 P0-spe100 和 original baseline 报告 pp 差异；不把 unique 增加单独解释为质量提升。
4. **机制证据**：结合 mode/token 诊断和 invalid 分类，判断提升来自更好的 token 预测、更多有效候选，还是仅来自更激进的采样。
5. **决策**：明确写“进入下一阶段”“停止超参分支”“停止当前 SPE 主线”三者之一，并引用本节的对应门槛。

本节写入时尚未启动 P0/P1/P2。现有 600k checkpoint、原 tokenizer baseline、20% test 子集结果和所有历史产物均保持不变。
