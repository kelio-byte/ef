# Stage 1 总览：项目目标、当前进展与后续入口

## 1. 这个项目在做什么

本项目尝试将论文 [Edit Flows: Flow Matching with Edit Operations](https://arxiv.org/abs/2506.09018) 引入到**化学逆合成**任务中，即：

- 输入：产物 SMILES
- 输出：反应物 SMILES

目标不是继续做传统自回归 seq2seq，而是改成一种**非自回归编辑生成**范式：从产物出发，利用插入 / 删除 / 替换三种编辑操作，逐步把产物编辑成目标反应物。

项目的核心假设是：

- 对逆合成来说，`x0 = product, x1 = reactant`
- 用 Edit Flows 的 CTMC 编辑过程去建模 `product -> reactant`
- 如果模型学到正确的编辑速率，那么 Euler 采样应该能把产物编辑成正确反应物

从目前的 oracle 实验看，这个基本方向**不是死路**，因为 oracle 生成结果非常高，说明“方法本身可行，当前主要瓶颈在模型学不好速率”。

---

## 2. 项目的理论主线

如果只想快速理解理论，不必先读全部代码，建议优先看：

- [edit-flows.md](/data3/duanbh/desktop/edit-flows/docs/edit-flows.md)
- [retro-impl.md](/data3/duanbh/desktop/edit-flows/docs/retro-impl.md)

简化版理论如下。

### 2.1 Edit Flows 是什么

Edit Flows 把序列生成建模为序列空间上的**连续时间马尔可夫链**（CTMC）。状态是序列，跃迁是编辑操作：

- insert
- delete
- substitute

模型在每个位置预测：

- 三类编辑的总速率 `λ_ins / λ_sub / λ_del`
- 插入 token 分布 `Q_ins`
- 替换 token 分布 `Q_sub`

从而定义所有可能编辑的瞬时速率。

### 2.2 为什么要引入 Z 空间

直接在变长序列空间上训练不方便，因此论文引入带 `GAP` 的辅助对齐空间 `Z`。

在 `Z` 空间里：

- `z0` 和 `z1` 是对齐后的起点与终点
- 中间态 `zt` 通过逐位置的条件概率路径采样
- 模型仍然只在原始 `X` 空间上预测编辑速率

项目里这部分关键实现主要在：

- [alignment.py](/data3/duanbh/desktop/edit-flows/edit_flows/core/alignment.py)
- [z_space.py](/data3/duanbh/desktop/edit-flows/edit_flows/core/z_space.py)
- [trainer.py](/data3/duanbh/desktop/edit-flows/edit_flows/training/trainer.py)

### 2.3 逆合成里的具体建模

这里采用的是 **Copy Product coupling**：

- `x0 = 产物`
- `x1 = 反应物`

训练时，模型看到一个从产物到反应物的编辑任务；生成时，也是从产物出发进行 Euler 采样。

这个设计最早的迁移思路记录在：

- [retro-todo.md](/data3/duanbh/desktop/edit-flows/docs/retro-todo.md)

最终实现说明在：

- [retro-impl.md](/data3/duanbh/desktop/edit-flows/docs/retro-impl.md)

---

## 3. 当前代码结构

如果要快速定位代码，先看下面这些目录和文件。

### 3.1 核心库

- [edit_flows/core](/data3/duanbh/desktop/edit-flows/edit_flows/core)
  - 调度器、对齐、Z 空间映射、速率缩放
- [edit_flows/models](/data3/duanbh/desktop/edit-flows/edit_flows/models)
  - Transformer 模型
- [edit_flows/training](/data3/duanbh/desktop/edit-flows/edit_flows/training)
  - batch 构造、loss、train step
- [edit_flows/sampling](/data3/duanbh/desktop/edit-flows/edit_flows/sampling)
  - Euler 采样、编辑操作应用、oracle 采样
- [edit_flows/data](/data3/duanbh/desktop/edit-flows/edit_flows/data)
  - 数据集、词表、预对齐数据加载

### 3.2 训练与评测脚本

- [train_retro.py](/data3/duanbh/desktop/edit-flows/scripts/train_retro.py)
  - 逆合成训练入口
- [sample_retro.py](/data3/duanbh/desktop/edit-flows/scripts/sample_retro.py)
  - 批量采样入口
- [eval_retro.py](/data3/duanbh/desktop/edit-flows/scripts/eval_retro.py)
  - 采样 + 评测一键脚本
- [score.py](/data3/duanbh/desktop/edit-flows/scripts/score.py)
  - standard 数据集评测
- [score_#global#.py](/data3/duanbh/desktop/edit-flows/scripts/score_#global#.py)
  - `#global#` 数据集评测
- [oracle_sample.py](/data3/duanbh/desktop/edit-flows/scripts/oracle_sample.py)
  - 用 oracle 速率直接生成
- [oracle_loss_profile.py](/data3/duanbh/desktop/edit-flows/scripts/oracle_loss_profile.py)
  - 分析 oracle loss 分布
- [precompute_alignments.py](/data3/duanbh/desktop/edit-flows/scripts/precompute_alignments.py)
  - 预计算训练对齐

### 3.3 配置

- [retro.yaml](/data3/duanbh/desktop/edit-flows/configs/retro.yaml)
  - 当前主配置
- [retro-example.yaml](/data3/duanbh/desktop/edit-flows/configs/retro-example.yaml)
  - 示例配置

---

## 4. 目前已经完成了什么

### 4.1 从论文到逆合成实现的迁移

已经完成：

- 基于 Edit Flows 的逆合成训练/采样主链路
- `product -> reactant` 的 Copy Product 耦合
- 适配 OpenNMT 风格的 Transformer
- 训练脚本 / 采样脚本 / 评测脚本

详见：

- [retro-impl.md](/data3/duanbh/desktop/edit-flows/docs/retro-impl.md)

### 4.2 训练性能优化

因为训练时 CPU 预处理太慢，后续又做了一轮工程优化，包括：

- `sample_cond_zt` 从 one-hot 改成 Bernoulli 采样
- 对齐预计算
- `rm_gap_tokens` 向量化
- CPU 侧 prepare，再整体搬到 GPU

详见：

- [retro-improve.md](/data3/duanbh/desktop/edit-flows/docs/retro-improve.md)

这部分是**工程效率优化**，主要是把训练跑快、GPU 喂满，不直接解决生成质量问题。

### 4.3 评测链路已经打通

目前已经有：

- 批量生成
- standard / `#global#` 两套评测
- Top-K、invalid、unique rate 等指标

详见：

- [eval-finish.md](/data3/duanbh/desktop/edit-flows/docs/eval-finish.md)

注意：

- 现有生成结果文件在 `checkpoints/*/*/eval/predictions.txt`
- 不要随意重跑生成脚本，否则会覆盖已有结果

---

## 5. 目前最重要的实验结论

### 5.1 模型生成效果不理想，训练集子集上也不好

已经观察到：

- 测试集结果不理想
- 即使在 `train_subsets` 上做生成，结果仍明显偏低

这说明问题不是单纯“泛化差”，而可能是：

- 模型本身没学对速率
- 或训练目标 / 参数化对模型不够友好

相关结果与分析入口：

- [oracle-analysis.md](/data3/duanbh/desktop/edit-flows/docs/oracle-analysis.md)
- `train_subsets/eval/*/eval.log`

### 5.2 Oracle 生成结果非常高

这是目前整个项目最关键的诊断结果之一。

在知道 target 的前提下，用理论最优速率替代模型输出，得到：

- standard Top-1 约 93.2%
- `#global#` Top-1 约 93.4%

这说明：

1. **Edit Flows + Copy Product 这个方向本身是可行的**
2. 当前最主要的问题在于：**模型没有学出足够准确/尖锐的速率**

详见：

- [oracle-analysis.md](/data3/duanbh/desktop/edit-flows/docs/oracle-analysis.md)
- [oracle.py](/data3/duanbh/desktop/edit-flows/edit_flows/sampling/oracle.py)

### 5.3 Oracle loss 分析说明模型“总量级对了，但分布不够锐”

通过直接把理论最优速率代入 loss，可以观察到：

- oracle loss 远低于真实训练 loss
- 但模型的 `u_tot` 量级并不离谱

这意味着：

- 模型并非完全不会预测速率总量
- 真正的问题是：**正确编辑不够集中，无关编辑不够接近 0**

详见：

- [loss-analysis.md](/data3/duanbh/desktop/edit-flows/docs/loss-analysis.md)

---

## 6. 已经修复过的重要实现问题

在对论文、代码和实验现象交叉审查后，已经发现并修复过几类明确实现问题。

### 6.1 BOS 与采样一致性、已完成样本过度编辑

修复内容：

- 采样起点加上 BOS，和训练保持一致
- Euler 采样时，对已经 `t >= 1` 的样本停止继续编辑

详见：

- [fix-1.md](/data3/duanbh/desktop/edit-flows/docs/fix-1.md)

### 6.2 Pre-Norm 残差写法错误、词表辅助函数遗留问题

修复内容：

- `PreNormEncoderLayer` 改成标准 Pre-LN 残差结构
- 清理旧的 `+3` 词表辅助函数，和现在 4 个特殊 token 保持一致

详见：

- [fix-2.md](/data3/duanbh/desktop/edit-flows/docs/fix-2.md)

这两类修复属于“实现 correctness”层面，不是简单调参。

---

## 7. 最近新增的一个重要尝试：速率重参数化

当前训练目标里，时间系数

$$
k(t) = \frac{\dot{\kappa}_t}{1-\kappa_t}
$$

会导致模型同时承担两件事：

- 学“内容上哪里该编辑”
- 学“时间上速率要放大多少倍”

为了把两者解耦，现在新增了一个可配置的**速率重参数化**：

- 模型不直接预测真实速率 `v`
- 而是预测 base rate `v'`
- 训练时优化去掉常数项后的
  $$
  k(t)\left(u'_{\text{tot}} - CE'\right)
  $$
- 采样时再恢复真实速率
  $$
  v = k(t) v'
  $$

当前该功能已接入：

- 训练
- 采样
- `yaml` 配置开关

详见：

- [rate-reparam-finish.md](/data3/duanbh/desktop/edit-flows/docs/rate-reparam-finish.md)
- [rate_scale.py](/data3/duanbh/desktop/edit-flows/edit_flows/core/rate_scale.py)

配置开关在：

- [retro.yaml](/data3/duanbh/desktop/edit-flows/configs/retro.yaml)

当前主配置里已经是：

```yaml
use_rate_reparam: true
```

---

## 8. 当前项目的核心问题

如果只关心“现在最卡在哪里”，答案可以概括为三条。

### 8.1 方法本身不是主要问题

Oracle 结果已经证明：

- 只要速率对，生成就能很好

所以项目当前不应优先怀疑：

- Edit Flows 完全不适合逆合成
- Copy Product 路线完全错误

### 8.2 当前主要瓶颈是模型速率学习

症状是：

- `u_tot` 看起来不离谱
- 但真实生成质量差很多
- 模型输出比 oracle 分散得多

所以接下来的重点应放在：

- 让速率分布更易学
- 让正确编辑更集中
- 让时间尺度因素不要干扰内容建模

### 8.3 采样离散化本身也有下界误差

即便 oracle 也不是 100%，仍然会有 invalid SMILES。

这部分来自：

- Euler 离散化
- clamp 限制
- 有些单次编辑在有限 hazard 下仍可能“整条轨迹中一次都没触发”

所以有两类问题要分开看：

1. **模型问题**：预测速率不准
2. **方法误差下界**：Euler + clamp 带来的不可避免损失

不要把这两者混在一起。

---

## 9. 当前最推荐的阅读顺序

如果是第一次接手这个项目，建议按下面顺序看。

### 9.1 先理解总体目标

1. [stage-1.md](/data3/duanbh/desktop/edit-flows/docs/stage-1.md)
2. [retro-impl.md](/data3/duanbh/desktop/edit-flows/docs/retro-impl.md)
3. [eval-finish.md](/data3/duanbh/desktop/edit-flows/docs/eval-finish.md)

### 9.2 再看最关键的诊断结论

1. [oracle-analysis.md](/data3/duanbh/desktop/edit-flows/docs/oracle-analysis.md)
2. [loss-analysis.md](/data3/duanbh/desktop/edit-flows/docs/loss-analysis.md)

### 9.3 再看最近的结构性改动

1. [fix-1.md](/data3/duanbh/desktop/edit-flows/docs/fix-1.md)
2. [fix-2.md](/data3/duanbh/desktop/edit-flows/docs/fix-2.md)
3. [rate-reparam-finish.md](/data3/duanbh/desktop/edit-flows/docs/rate-reparam-finish.md)

### 9.4 最后再进代码

建议优先读：

1. [trainer.py](/data3/duanbh/desktop/edit-flows/edit_flows/training/trainer.py)
2. [loss.py](/data3/duanbh/desktop/edit-flows/edit_flows/training/loss.py)
3. [transformer.py](/data3/duanbh/desktop/edit-flows/edit_flows/models/transformer.py)
4. [euler.py](/data3/duanbh/desktop/edit-flows/edit_flows/sampling/euler.py)
5. [oracle.py](/data3/duanbh/desktop/edit-flows/edit_flows/sampling/oracle.py)

---

## 10. 当前阶段的建议计划

从目前状态看，比较合理的 Stage 1 后续计划是：

1. **先做小规模对比实验**
   - 重点比较 `use_rate_reparam=false/true`
   - 不要先上大规模长训

2. **优先看 train-subset 上的生成是否抬升**
   - 因为这里更快暴露“模型到底能不能学到”
   - 也更方便和 oracle gap 对比

3. **如果重参数化有帮助，再进一步调模型和训练目标**
   - rate head 参数化
   - 分布锐化
   - 模型容量 / dropout

4. **采样侧改进放第二阶段**
   - 如增大 `n_steps`
   - 调整 clamp
   - 改 scheduler

因为当前最主要的问题仍是**模型没学好**，而不是“采样步数不够”。

---

## 11. 当前需要特别注意的事项

1. 不要随意重跑会覆盖已有结果的生成脚本。
   - 尤其是 `checkpoints/*/*/eval/predictions.txt`

2. 阅读旧文档时注意时间顺序。
   - 一些早期文档（例如最初实现总结）可能还保留旧设定
   - 以较新的修复文档和实现代码为准

3. `retro.yaml` 现在已经包含最新的速率重参数化开关。
   - 做对比实验时应显式记录 `use_rate_reparam`

4. 项目当前的核心判断是：
   - **方法可行**
   - **模型是主要瓶颈**
   - **工程效率问题已基本处理**

---

## 12. 一句话总结

这个项目当前处于这样一个阶段：

- Edit Flows 逆合成实现已经打通
- 训练、采样、评测、oracle 诊断都已经具备
- 主要 correctness bug 已修复
- oracle 证明了方法本身可行
- 当前最重要的问题，是如何让模型真正学到足够准确、足够尖锐的编辑速率

如果只想继续推进项目，最值得优先关注的是：

- [oracle-analysis.md](/data3/duanbh/desktop/edit-flows/docs/oracle-analysis.md)
- [rate-reparam-finish.md](/data3/duanbh/desktop/edit-flows/docs/rate-reparam-finish.md)
- [trainer.py](/data3/duanbh/desktop/edit-flows/edit_flows/training/trainer.py)
- [loss.py](/data3/duanbh/desktop/edit-flows/edit_flows/training/loss.py)
