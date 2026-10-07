# Edit Flows 训练代码审计（基础修复状态）

> 初始审计日期：2026-08-03。范围仅包括代码、当前 checkpoint、现有训练数据和本地
> `PDF/2025--Edit Flows Flow Matching with Edit Operations.pdf` 的只读核对。本轮没有
> 修改训练代码、数据或 checkpoint。2026-08-06 已完成审计中“训练基础设施”阶段，
> 具体改动和测试见 [`training_tensorboard_and_fixes.md`](training_tensorboard_and_fixes.md)。

## 1. 结论摘要

### 2026-08-06 状态更新

Noam 第一次 update、数据 fail-fast、PAD alignment、RNG/checkpoint resume、validation
和 TensorBoard 已在新脚本路径中修复。`configs/retro.yaml` 和旧 checkpoint 仍保持冻结；
后续新训练应使用 `configs/retro_v2.yaml`。本文件下面的“明确代码问题”保留原始审计证据，
其当前修复状态以新文档的阶段清单为准。

当前训练不是“loss 主公式写错、checkpoint 完全不可用”的状态。`bregman_loss()` 与论文
Eq.23 的两个核心项一致：对所有可执行编辑率求和，并对通向目标对齐状态的编辑率施加
`κdot/(1-κ)` 加权负对数项。Z 空间对齐、token-wise mixture、去 GAP、插入/删除/替换
mask 和 `λ×Q` 参数化也能对应论文。

但如果准备下一次重训，不能原样启动 600k steps。优先需要处理：

1. 第一次优化器更新使用了 Adam 默认学习率 `1e-3`，之后才调用 Noam，把学习率降到
   `8.7346e-8`；调度顺序存在明确错误。
2. 没有 validation loop、最佳 checkpoint 选择或早停，只能使用最后一步模型。
3. 没有统一随机 seed，训练不可严格复现。
4. 数据文件用 `zip()` 读取，行数不一致会静默截断；on-the-fly alignment fallback 还会
   把 collate PAD 当作真实 token。
5. 当前 copy-product coupling 没有不可编辑的完整 product 条件；从条件生成角度，这是
   重要建模限制，不是简单推理参数能补上的问题。
6. 全量预对齐训练集的目标编辑中 91.20% 是插入，删除只有 0.92%。这符合“reactants
   通常比 product 更长”的数据特征，但会造成操作头训练极不平衡，必须在新训练实验中
   单独监控。

## 2. 论文与实现逐项对照

| 论文要素 | 当前实现 | 审计判断 |
|---|---|---|
| 用 alignment 构造 `(z0,z1)`，blank 不进入模型输入 | `alignment.py`、预对齐文件中的 `<GAP>`、`rm_gap_tokens()` | 一致 |
| `z_t` 每个位置按 `κ(t)` 在 `z0/z1` 间采样 | `sample_cond_zt()` | 一致 |
| 模型只观察去 blank 后的 `x_t` | `rm_gap_tokens()` 后传 Transformer | 一致 |
| 三类 rate：insert / substitute / delete | `rates_out(...,3)` | 一致 |
| insert/substitute 分解为 `λ×Q(token)` | `log_rate + log_softmax` | 一致 |
| Eq.23：总输出率减去目标编辑 log-rate | `bregman_loss()` | 主体一致 |
| `t ~ Uniform[0,1]` | `prepare_batch()` 中 `torch.rand(B,1)` | 一致 |
| 每位置独立、一步可并行发生多次编辑 | Euler/Euler-Beam sampler | 一致 |
| 论文 first-order `hλ`，附录给出冻结率区间概率 | 当前默认 `1-exp(-hλ)` | 合理的数值改进，但不是论文主实验的逐字实现 |
| 可选 CFG、Q sharpening、reverse rates、localized edits | 当前训练未实现 | 不能由现 checkpoint 直接获得 |

`use_rate_reparam=False` 时，模型直接学习完整 rate；loss 中目标 CE 项乘
`κdot/(1-κ)`，总率项不乘。这与 Eq.23 一致。`use_rate_reparam=True` 的另一分支把公共
scale 提到模型输出之外，目标的梯度等价，但当前 checkpoint 没有使用该分支。

## 3. 当前 checkpoint 与数据证据

`checkpoint_step600000.pt`：

- step = 600000，scheduler state `_step=600000`；
- `use_origin_mask=False`，权重中没有 `origin_embedding`；
- optimizer 保存学习率 `8.0687153e-5`；
- 所有浮点模型权重有限，最大绝对值约 7.029；
- cubic scheduler、batch 128、Noam warmup 8000、无 gradient clipping；
- checkpoint 文件时间晚于两份预对齐训练文件，且训练代码在文件存在时优先使用它们。

仓库中没有与该根目录 checkpoint 配套的 `train.log/config.yaml`，所以无法从日志绝对
证明当时走了哪条 data branch，也无法回看 loss 曲线；但文件时间与代码路径强烈支持它
使用了预对齐数据。

对现有四份训练文件的只读检查：

- raw src、raw tgt、aligned src、aligned tgt 均为 800,060 行；
- aligned src/tgt 每行长度完全相等；
- 去除 `<GAP>` 后逐行精确还原 raw src/tgt；
- 无未知 token；最长 aligned 序列加 BOS 不超过 256。

因此静默截断、fallback PAD alignment 和长度越界是潜在代码缺陷，但没有证据表明它们
污染了当前 checkpoint 的这批预对齐输入。

## 4. 全训练集编辑分布

按 800,060 对预对齐序列逐位置统计：

| 类型 | 次数 | 占目标编辑比例 |
|---|---:|---:|
| Insert | 4,188,917 | 91.201% |
| Substitute | 361,694 | 7.875% |
| Delete | 42,451 | 0.924% |
| Keep | 36,057,141 | 非编辑 |

平均每个 augmentation pair 有 5.741 个目标编辑；0-edit pair 为 0。约 15.95% pair 至少
有 10 个编辑，3.88% 至少有 20 个编辑。

这不是 alignment 损坏：最短编辑距离本身就主要需要向 product 中插入 reactant/reagent
片段。不过它带来两个研究问题：

- delete head 的正例极少，三类 rate 共用主干但输出监督严重失衡；
- 推理中 invalid 常与过插入/错误插入有关，继续只调 branch 数未必能解决模型级偏差。

不建议未经实验直接给 loss 加 class weight，因为这会改变 CTMC 的目标 rate。应先在
validation 上记录各操作 head 的校准、目标事件召回和错误编辑组成，再设计概率上自洽的
重加权或 coupling。

## 5. 明确代码问题

### 5.1 P0：Noam 第一次更新顺序

当前循环是：

```text
train_step() -> optimizer.step()
lr_scheduler.step()
```

Adam 构造时未指定 `lr`，所以第一次参数更新使用 PyTorch 默认 `1e-3`。更新结束后才把
学习率设为 Noam step 1 的 `8.7346e-8`。后续每次更新也使用“上一次 scheduler step”
留下的学习率。

影响判断：这是一处确定性训练 bug，但只造成第一个 update 异常；当前 600k checkpoint
权重有限且已有可用准确率，因此不能据此宣布 checkpoint 作废。未来重训必须在第一个
optimizer update 前初始化正确 lr，并加单元测试验证 update n 使用 Noam n 的 lr。

### 5.2 P0：fallback alignment 会处理 PAD

`RetroDataset + collate_fn` 先把不同长度序列 PAD 到 batch 最大长度，再把整行 tensor
交给 `_align_pair()`；后者按 tensor 长度做 DP，不剥离 PAD。不同 pair 的 alignment
结果还可能长度不同，随后 `torch.stack()` 失败。

当前 checkpoint 很可能走 `PreAlignedDataset + identity_align`，所以不受影响。未来代码
要么取消损坏的自动 fallback、强制先预对齐，要么在 DP 前按真实长度裁剪并重新 pad
alignment 输出。

### 5.3 P0：数据 `zip()` 静默截断

`RetroDataset`、`PreAlignedDataset` 和 `precompute_alignments.py` 都使用 `zip(f0,f1)`。
任一文件少一行都不会报错。现训练文件计数相等，但未来训练应在加载前 fail-fast 校验
行数、空行、aligned 长度、可逆性和词表覆盖。

### 5.4 P1：resume step 会重复一次 update

checkpoint 保存 `step=s` 后，resume 使用 `range(start_step,total_steps)`，会再次执行
step s。checkpoint 的“已完成 update 数”和循环 label 语义也不完全一致。应统一保存
`completed_steps`，resume 从下一步开始，并做 uninterrupted/resume 参数一致性测试。

### 5.5 P1：训练随机性没有被记录

脚本没有设置 Python、NumPy、PyTorch CPU/CUDA seed，也没有保存 DataLoader generator
状态或 sampler epoch。相同 config 无法严格复现。checkpoint 只保存模型、optimizer 和
Noam step，不保存 RNG state。

### 5.6 P1：没有 validation 和 checkpoint selection

训练只打印 train minibatch loss/u_tot，每 10k steps 保存一次并删除旧 checkpoint；没有
validation loss、Top-k、invalid 或 early stopping。这意味着 600k 是固定终点，不是经
validation 选择的最优点，也无法判断过拟合、操作头退化或何时开始回归。

### 5.7 P1：`κdot/(1-κ)` 固定截断

即使 `clamp_kappa=False`，`get_rate_scale()` 仍把完整 scale 截到 50。对 cubic scheduler，
大约从 `t≈0.9804` 开始发生截断，即均匀 t 训练样本约最后 1.96% 的区域。它是合理的
数值稳定选择，但偏离未截断理论目标；日志应明确记录，并在重训时监控 cap 命中率，而
不是误以为 `clamp_kappa=False` 表示完全不 clamp。

## 6. 建模层面的重要问题：product 条件是否会丢失

当前设置是 `x0=product, x1=reactants`，模型只输入当前可编辑状态 `x_t`。这能学习一条
从 product 分布到 reactant 分布的 edit flow，并且训练 coupling 会影响生成 coupling；
但论文也明确说明生成 coupling 不必等于训练 coupling。

对单步逆合成，我们真正需要的是对固定 product 的 `p(reactants | product)`。当前 product
没有作为独立、不可编辑的条件保存：当 token 被替换/删除后，模型只能从剩余 `x_t` 推断
原 product。不同 product 的中间状态如果碰撞，理论边际 rate 会对其目标进行平均。

这应视为“可工作的 copy-product 建模选择，但条件保证较弱”，而不是已证实 bug。最有
价值的下一训练分支是显式 conditional Edit Flow：

- 用独立 product encoder/cross-attention，或受保护的 product memory；
- edit state 仍可从 product copy 开始，以保留最少编辑优势；
- 每一步 rate/Q 都显式条件于完整 product；
- condition dropout 之后才有资格做论文中的 CFG。

这个方向会改变模型结构和 checkpoint，必须作为独立训练研究分支，与当前推理优化隔离。

## 7. 建议的重训优先级

### 阶段 A：只修训练基础设施

1. 修 Noam 初始化/调用顺序并新增 scheduler 单测；
2. 数据加载 fail-fast；禁用或修复 fallback alignment；
3. 固定并保存全部 RNG 状态；
4. 修 resume completed-step 语义并做连续/断点等价测试；
5. 新增 validation loss、操作分项、Top-k/invalid 和 checkpoint selection。

这些修改不改变模型目标，应先用极短 synthetic/小数据训练验证，再开正式训练。

### 阶段 B：建立可比较的小规模训练基线

- 冻结当前 checkpoint 和现有推理配置作为 baseline；
- 不先跑 600k，先跑相同 seed 的 10k～30k pilot；
- 比较 train/val loss、三类编辑召回、rate 校准、Top-1～10、invalid 和 wall time；
- 只有曲线与 resume/reproducibility 验证通过才扩大训练。

### 阶段 C：研究模型改动

优先做“显式 product conditioning vs copy-product”的单变量对照；之后才考虑 origin mask、
localized edits、reverse rates 或 guidance。不能把这些同时加入一次重训，否则无法归因。

## 8. 对当前工作的直接结论

- 当前 checkpoint 可以继续用于 Euler/Euler-Beam 推理研究；
- 不应声称训练已严格复现论文实验，因为架构、优化器、条件形式和数据任务都做了迁移；
- 下一次训练前必须先修基础设施，但本轮没有修改训练代码；
- 若目标是明显突破当前准确率，显式保留 product 条件可能比继续扩大 K/M 更有研究价值。
