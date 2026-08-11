# DGM reward 质量审计与下一步校准协议

更新日期：2026-08-11  
状态：已完成只读审计工具、单元测试和现有多时间点 guidance 数据的训练／held-out 检查；尚未训练新的 reward 或重新训练 guidance。

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

为安全选择这个连续反应块，生成器需要增加一个显式的“从第几个原始反应块开始”参数；它应在按 20 条 augmentation 折叠后生效，且必须有单元测试，避免把 20 个 SMILES 写法误当成 20 个反应。

### 4.2 第一版校准器

第一版只允许使用实际可获得的、无需答案的信息：

- 正向 beam 是否重构到产物及其排名；
- 正向模型对“候选反应物生成输入产物”的 teacher-forced 对数似然；
- 候选是否合法，以及候选/产物的简单长度或原子数一致性特征。

先训练一个小型线性或两层校准器，而不是立即重写 Molecular Transformer 或 Edit Flows。这样可以先回答“已有信号的组合能否改善 reward”，同时控制算力和过拟合风险。模型输出必须是单个可审计分数，并记录输入特征、数据范围和 checkpoint。

### 4.3 通过门槛

在新的 200-reaction reward holdout 上，与原始 forward-beam reward 同时比较。只有全部条件满足才允许用新 reward 重建 guidance 数据：

1. 全局 AUC 和同组 AUC 均比原始 reward 至少提高 **0.02**，并按原始反应／共享状态组 bootstrap 记录置信区间；
2. 正确终点的平均分高于错误终点，且错误终点拿到正分的比例不能上升；
3. 用同一候选池做终点重排时，Top-1 不低于原始 forward reward，且 Top-3、Top-10 或覆盖率至少一项提高；
4. 计算成本、非法候选处理和所有输入输出哈希完整记录；
5. 不根据 `evaluation_v2/dev_unique1000_aug20` 的结果挑特征、阈值或 checkpoint。

若校准 pilot 未通过，就记录负结果并停止这条 reward 校准支线；不能把训练时长、beta 或分支数混入补救。若通过，才重建训练 reward、重新训练 guidance，并在冻结配置后回到 1,000-reaction development protocol 做一次 ordinary-Euler 检验。

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
