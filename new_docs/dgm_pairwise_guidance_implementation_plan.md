# DGM Product-Internal Pairwise Guidance 落实文档

> 日期：2026-08-08
> 执行对象：后续 GPT-Luna / 项目维护者
> 状态：P0–P5、P5b、P5c 实现与 smoke 已完成；P5 正确实现但未通过收益门槛，新的 1k pilot 待做，P6 暂缓
> 研究路线：继续改进 learned action-level approximate DGM；本阶段不做 terminal reranker
> 校准，也不进入 exact Z-space 重写。

## 0. 本文档的执行目标

当前多终点数据已经为每个 product 提供 4 个独立 Euler terminal，但训练仍把 40,000 条记录
打散成普通样本，逐条拟合 Bregman action target。当前 10k guidance checkpoint 在原有诊断
口径下的 product 内 pair 排序只有 60.19%，learned guidance 在 validation-A/B 的 Top-1
变化方向相反。

本任务的目标是增加**同一 product 内的显式 pairwise-ranking 监督**：在相同 product、相同
中间状态和相同时间上，要求通向高 forward reward terminal 的动作集合得分高于通向低 reward
terminal 的动作集合。

阶段最终通过条件不是“loss 能下降”，而是：

1. 分组和 pairwise mechanics 全部通过测试；
2. 1k pilot 在两个种子上相对同协议 control 明显提升 held-out shared-anchor pairwise 指标，且
   Bregman 校准没有明显恶化；
3. 10k 模型通过相同 held-out 门槛；
4. 固定 ordinary Euler `β=0.10` 后，在 validation-A 和 validation-B **都不降低 Top-1**，
   并在每个 split 上至少改善 Top-3、Top-10 或 Oracle 中的一项；
5. 完整记录准确率、invalid/unique、wall、显存和 Git 提交。

如果第 2 或第 3 条不通过，停止扩大实验；如果第 4 条不通过，不接 Euler-Beam。

---

## 1. 必须先理解的现状

### 1.1 必读文件

执行前完整阅读：

- `new_docs/0808.md`：截至 2026-08-08 的可汇报结论；
- `new_docs/dgm.md`：完整 DGM 阶段记录；
- `edit_flows/guidance/data.py`：record schema、collate 和 dataset；
- `edit_flows/guidance/targets.py`：`x_t → terminal` action mask；
- `edit_flows/guidance/training.py`：当前 Bregman action loss；
- `edit_flows/guidance/model.py`：约 5.26M 参数 guidance model；
- `scripts/train_guidance.py`：训练循环、validation 和 checkpoint；
- `scripts/sample_retro.py`、`edit_flows/sampling/euler.py`：guidance 推理入口。

### 1.2 已冻结资产

```text
Base Edit Flows:
new_checkpoints/checkpoint_step600000.pt

Forward Molecular Transformer:
new_checkpoints/MIT_mixed_augm_model_average_20.pt

1k train guidance data:
/root/autodl-tmp/dgm_guidance_data/train_multiterm1000_beam.pt

10k train guidance data:
/root/autodl-tmp/dgm_guidance_data/train_multiterm10000_beam.pt

held-out validation guidance data:
/root/autodl-tmp/dgm_guidance_data/val_multiterm200_beam.pt

current 10k checkpoint:
/root/autodl-tmp/dgm_guidance_runs/multiterm_beam10k_5000/guidance_best.pt
```

大 `.pt` 数据和实验 checkpoint 不进入 Git；必须把路径、文件大小、可用时的 SHA256 和生成
配置写进实验报告。如果资产缺失，先检查是否迁移遗漏，不要立刻重新生成。

### 1.3 不允许改动的范围

本任务不修改：

- Edit Flows 训练代码、600k checkpoint 或 YAML；
- Molecular Transformer 权重；
- USPTO 数据文件和历史 `results/`；
- ordinary Euler 的基础概率/rate 公式；
- Euler-Beam；
- exact Z-space/GAP 表示；
- 现有 `per_position` 默认推理行为。

新功能必须由显式 CLI 参数启用。`pairwise_loss_weight=0` 时，同一 batch 上的总 loss 和当前
Bregman loss 必须数值一致；现有 guidance checkpoint 必须仍能由 `sample_retro.py` 加载。

---

## 2. 为什么不能直接比较现有四条记录

同一个 product 的四条记录分别具有不同 terminal、不同随机 `t` 和不同 `x_t`。如果简单比较：

```text
score(product, x_t^i, t_i, terminal_i)
```

与：

```text
score(product, x_t^j, t_j, terminal_j)
```

得分差异可能来自状态难度或时间，而不是 terminal 的 reward。这样的 pairwise loss 会把条件
混淆。

正确的第一版比较单位是 shared anchor。对同一 product 选择一条记录的 `(x_t^a, t_a)` 作为
anchor，只运行一次 guidance forward，然后分别构造从这个相同 anchor 通向四个 terminal 的
动作集合：

```text
H = H_theta(product, x_t^a, t_a)

A(a -> terminal_1)
A(a -> terminal_2)
A(a -> terminal_3)
A(a -> terminal_4)
```

在相同的 `H` 上计算四个动作集合得分，之后只比较 reward 不相等的 terminal pair。这样训练
问题才接近“当前状态下一步应该更偏向哪个未来”。

这仍是 action-level 近似：`A(a → terminal)` 是由完整终点差异构造的稀疏编辑集合，不是严格
Doob transform 的单步 rate ratio。代码和文档中必须继续使用 `approximate` 表述。

---

## 3. 数学定义和默认设计

### 3.1 Terminal 动作集合得分

guidance 输出三个正张量：

```text
H_ins: [B, L, V]
H_sub: [B, L, V]
H_del: [B, L, 1]
```

对 shared anchor `a` 和候选 terminal `j`，利用
`build_action_target_masks(anchor_state, terminal_j)` 得到三个布尔 mask。定义长度归一化的
动作集合 log-score：

```text
s(a, j) = sum_{u in A(a -> j)} log(clamp(H_u, min=eps)) / |A(a -> j)|
```

选择平均 `log H` 而不是 `H` 总和，原因是：

1. 防止编辑数更多的 terminal 仅因动作数量更大而得分更高；
2. 与推理时乘法重加权 `base × H^β` 的 log-space 形式一致；
3. 对极大的正权重更稳定。

默认 `eps=1e-8`。如果 `|A|=0`，当前模型没有显式 no-op head，不能凭空定义 no-op 分数；该
anchor-terminal 只从 pairwise loss 中跳过，但仍参与原 Bregman loss。必须记录 no-action 跳过率。

### 3.2 Pairwise loss

对同一 product、同一 anchor 下 reward 不相等的 terminal `i,j`，令高 reward 为 `h`、低
reward 为 `l`：

```text
L_pair = softplus(-(s(a,h) - s(a,l)) / tau)
```

第一版固定：

```text
tau = 1.0
reward_equal_tolerance = 1e-6
```

所有有效 unequal-reward pair 等权平均。第一版不要同时加入 margin、reward-difference weight、
hard-negative mining 或温度扫描；否则无法判断收益来自哪一项。可以记录 reward gap 分桶指标，
但不改变 loss 权重。

### 3.3 总损失

保留当前 action Bregman loss：

```text
L_total = L_bregman + lambda_pair * L_pair
```

1k pilot 只允许以下预先声明的三种设置：

```text
lambda_pair = 0.00   # 新 pipeline control
lambda_pair = 0.25
lambda_pair = 1.00
```

不扫描其它值。先用 seed 42 比较三者；只有最佳 pairwise 设置通过第一轮门槛后，才用 seed 43
复跑该设置和 `lambda=0` control。

### 3.4 Anchor 选择与效率

train batch 保持 64 records，即 16 个 product group × 每组 4 terminal。Bregman loss 使用全部
64 条原始 record。

pairwise 训练时每个 product 每步只选择 1 个 anchor，而不是把 4 个 anchor 全部展开：

```text
anchor_position = (global_step + stable_group_offset) % group_size
```

`stable_group_offset` 必须由 `source_index` 的确定性整数运算得到，不能使用 Python 进程随机
hash。这样多个 step 会轮换覆盖四个 anchor，同时把 action alignment 成本控制在可接受范围。

validation 时使用每组全部 4 个 anchor，以获得稳定、完整的 held-out 指标。必须分别记录 train
和 validation 的 pair 数、有效组比例、no-action 跳过率和耗时。

---

## 4. 代码实施阶段

### P0：Preflight 和基线冻结

执行：

1. `conda activate ef`，记录 `which python`、PyTorch/CUDA/GPU；
2. `git status --short`，不得覆盖用户未提交改动；
3. 确认上述 1k/10k/val 数据和两个 checkpoint 可读；
4. 对三个 guidance 数据做只读审计：group size min/max、每组唯一 `source_index`、record 数、
   reward 可变组比例；
5. 运行当前 targeted tests 并保存结果；
6. 用新的 shared-anchor evaluator（P3 完成后）评估现有 1k best、10k best/final，建立**新口径**
   reference。

重要：旧文档的 60.19%/62.96% 比较的是“每条记录自己的状态和 terminal”，与 shared-anchor
指标不是同一口径。不得直接声称新模型超过 60.19%；必须先用同一个新 evaluator 重算旧
checkpoint。

P0 通过条件：资产、数据分组和基线测试无新增问题。只读审计不需要修改数据。

### P1：实现 group-aware batch sampler

建议修改：`edit_flows/guidance/data.py`，必要时新增独立小模块
`edit_flows/guidance/grouping.py`，不要重写现有 dataset。

实现一个明确的 sampler，例如：

```text
ProductGroupBatchSampler
```

要求：

1. 按 `source_index` 建立 `source_index -> record indices`；
2. 默认要求每组正好 4 条；不符合时 fail fast，并列出异常 source；
3. 以 product group 为单位 shuffle，不能打散组内记录；
4. `batch_size=64` 表示 record 数，因此每 batch 16 groups；
5. 不允许一个 group 跨 batch；
6. seed 和 epoch 决定 shuffle，重复运行可复现；
7. validation 不 shuffle，顺序稳定；
8. `drop_last=False` 时最后一个 batch 可以少于 16 groups，但仍必须保留完整 group；
9. collate 保留 `source_index/sample_index/time_index`；
10. 当 `lambda_pair=0` 且未显式启用 grouped mode 时，旧 DataLoader 路径不变。

新增测试至少覆盖：完整组、不完整组报错、组不跨 batch、seed 复现、不同 epoch 顺序变化、最后
小 batch、batch size 不能被 group size 整除时报错。

P1 通过条件：新增 sampler 测试全过，旧 `tests/guidance/test_data.py` 无回归。

完成后创建独立 commit，例如：

```text
Add product-grouped guidance batches
```

### P2：实现 shared-anchor pairwise mechanics

建议修改：`edit_flows/guidance/training.py`，可把纯计算 helper 放到新的
`edit_flows/guidance/ranking.py`。不要把复杂逻辑塞进训练脚本。

建议提供两个可独立测试的函数：

```text
score_terminal_action_sets(...)
shared_anchor_pairwise_loss(...)
```

实现要求：

1. 由 `source_index` 分组，不依赖 batch 中恰好连续排列；
2. 同一 anchor 的 product/state/time 只取一次模型输出；
3. terminal action mask 必须全部从同一个 anchor state 构造；
4. score 使用 selected action 的 mean log-H；
5. equal reward 不构成 pair；
6. no-action terminal 跳过并计数；
7. 没有任何有效 pair 时返回可反向传播的零 loss，不能产生 NaN；
8. 不对 reward 或 action mask 反向传播；
9. 不修改模型 architecture 和 checkpoint state_dict；
10. `lambda_pair=0` 时，训练 step 不执行额外的 pairwise alignment，避免 control 无意义变慢；但 validation 或只读 evaluator 在显式请求 shared-anchor 指标时仍可执行该诊断。

必须返回并记录：

```text
loss_bregman
loss_pairwise
loss_total
pair_count
pair_accuracy_strict
pair_accuracy_tie_half
pair_tie_fraction
pair_margin_mean
valid_pair_group_fraction
no_action_candidate_fraction
reward_score_pearson
```

测试应构造人工 H 和 reward，验证：

- 高 reward 得分更高时 loss 更小、accuracy=1；
- 交换高低得分后 loss 增大；
- equal reward 被排除；
- 两个 terminal 动作数量不同但每个动作 H 相同，mean log-H 得分相同；
- 同一 terminal 的 score 只依赖同一 anchor H；
- batch 重排不改变结果；
- no-action、无 pair、极小/极大 H 均有限；
- pairwise loss 对 guidance 参数有非零梯度；
- `lambda=0` 的 `L_total` 与现有 `guidance_action_loss` 在同 batch 上严格一致。

P2 通过条件：所有人工排序、梯度、数值稳定和回归测试通过。

完成后创建独立 commit，例如：

```text
Add shared-anchor guidance ranking loss
```

### P3：训练、validation 与 checkpoint selection

修改 `scripts/train_guidance.py`，只增加兼容参数，不改变旧默认值：

```text
--group_size 4
--pairwise_loss_weight 0.0
--pairwise_temperature 1.0
--pairwise_equal_tolerance 1e-6
--pairwise_all_val_anchors
--checkpoint_selection pairwise_guarded
```

参数名字可以做小幅调整，但 config/checkpoint/终端帮助/文档必须一致。默认
`pairwise_loss_weight=0`，旧命令必须继续可运行。

checkpoint 选择不能继续只看 total loss，也不能看过 Top-k 后事后挑 step。采用预注册规则：

1. 每次 validation 同时报告 Bregman component 和 shared-anchor pairwise metrics；
2. 对 pairwise run，候选 checkpoint 必须满足：
   `val_bregman_loss <= 1.15 × 同 seed lambda=0 control 的最佳 val_bregman_loss`；
3. 在 eligible checkpoints 中最大化 `pair_accuracy_tie_half`；
4. accuracy 差小于 0.5 个百分点时，依次选择更高 Pearson、更低 Bregman loss、更早 step；
5. control 仍按最低 Bregman loss选择；
6. checkpoint 内保存 selection rule、所有 validation 指标、pairwise 超参和 reference control；
7. `sample_retro.py` 只依赖模型 config/state_dict，因此新 checkpoint 必须向后兼容加载。

由于 control 的 Bregman reference 在 control 完成后才知道，允许训练 pairwise 时通过
`--control_metrics_json` 读取固定 reference；不能在同一 run 中动态改变门槛。

建议新增只读脚本 `scripts/evaluate_guidance.py`，输入 data + checkpoint，输出 JSON，便于用同一
口径复核旧/new checkpoints。它不能训练或修改 checkpoint。

TensorBoard 至少记录：

```text
train/loss_bregman
train/loss_pairwise
train/loss_total
train/pair_accuracy_tie_half
train/pair_count
train/no_action_candidate_fraction
validation/以上同名指标
validation/reward_score_pearson
validation/eligible_for_selection
efficiency/steps_per_second
```

P3 通过条件：旧命令兼容；lambda=0 训练 smoke；pairwise checkpoint 能保存、加载、只读评估；
TensorBoard/JSON 指标齐全。

#### P3 当前实现记录（2026-08-08）

- 新增 `ProductGroupBatchSampler`，默认训练命令不启用，grouped 模式要求每组严格 4 条记录。
- 新增 `shared_anchor_pairwise_loss()` 和 `score_terminal_action_sets()`；mean-log-H、equal
  reward 跳过、no-action 统计、pairwise accuracy 和 Pearson 均有单元测试。
- `train_guidance.py` 新增 grouped/pairwise 参数、全 anchor validation、Bregman guardrail、
  TensorBoard 指标和 `summary.json`。
- 新增只读 `scripts/evaluate_guidance.py`，用于在相同 shared-anchor 口径下比较旧/new checkpoint。
- guidance/forward 定向回归为 **53 passed**；1-step CPU grouped pairwise smoke、guarded
  checkpoint smoke、旧默认路径 smoke 均通过；新 checkpoint 可由 `sample_retro.py` loader 读取。
- 目前没有 GPU pilot、Top-k 结果或 pairwise 方法准确率结论；P4 先做 GPU smoke，P5 才开始 1k
  预注册实验。

完成后创建独立 commit，例如：

```text
Train guidance with guarded pairwise selection
```

### P4：回归测试和 GPU smoke

先运行 CPU targeted tests：

```bash
conda activate ef
python -m pytest \
  tests/guidance/test_data.py \
  tests/guidance/test_targets.py \
  tests/guidance/test_training.py \
  tests/guidance/test_model.py \
  tests/guidance/test_sampling.py -q
```

再运行全部 guidance/forward tests：

```bash
python -m pytest tests/guidance tests/forward -q
```

随后运行全量测试并与实施前 baseline 比较。仓库当前历史上有 17 个与本任务无关的
`tests/sampling/test_beam.py` 旧 API/controlled model failure；不得为了让数字好看而修改或跳过，
只要求本任务不增加新 failure，并在报告中列出 exact failing tests。

GPU smoke 只取极小 record/step，验证：

- CUDA 上 loss、梯度、checkpoint 均有限；
- `lambda=0` 不执行额外 pairwise alignment；
- 一个 pairwise training step 后参数确实更新；
- checkpoint 可由 `scripts/sample_retro.py` 加载；
- `β=0` 仍与无 guidance 的 Euler 输出 byte-level 相同。

P4 通过条件：targeted tests 全过、无新增 full-suite failure、GPU smoke 全过。

#### P4 当前实现记录（2026-08-08）

- guidance/forward 定向回归：**53 passed**；全量 pytest：**285 passed, 17 failed**。17 个 failure
  均为既有 `tests/sampling/test_beam.py` 的 `EditCandidate(log_u_real)` API 和 controlled-model
  长度问题，本任务没有新增 failure。
- RTX 3090 CUDA 1-step pairwise smoke 通过，缩小模型 peak allocated/reserved 为
  **60.4/86.0 MB**，CUDA evaluator 也能输出有限指标。
- 用同一 seed、`n_steps=2`、20 条 augmentation 做实际 `sample_retro.py` 对照，baseline 与
  `guidance_beta=0` predictions SHA256 完全一致（`32c12c…f8b5d45`）；因此新 guidance 接口
  的 identity-limit 在真实 CUDA sampler 上成立。
- 该 smoke 只验证正确性和调用链，不提供 pairwise 准确率结论；现在进入 P5 1k pilot。

### P5：1k pilot，禁止直接上 10k

固定资产：

```text
train = train_multiterm1000_beam.pt      # 4,000 records
val   = val_multiterm200_beam.pt         # 800 records，独立 products
batch_size = 64 records = 16 groups
max_steps = 500
epochs = 8
learning_rate = 1e-4
val_interval = 100
model architecture = 完全沿用当前配置
```

先运行 seed 42 的三个预注册实验：

```text
lambda = 0.00 / 0.25 / 1.00
```

命令模板（P3 参数已实现；lambda=0 control 使用 `--checkpoint_selection validation_loss`，
pairwise run 使用 `pairwise_guarded` 并引用 control 的 `summary.json`）：

```bash
conda activate ef
python scripts/train_guidance.py \
  --train_data /root/autodl-tmp/dgm_guidance_data/train_multiterm1000_beam.pt \
  --val_data /root/autodl-tmp/dgm_guidance_data/val_multiterm200_beam.pt \
  --output_dir /root/autodl-tmp/dgm_guidance_runs/pairwise_pilot_lam025_seed42 \
  --device cuda --batch_size 64 --num_workers 2 \
  --epochs 8 --max_steps 500 --val_interval 100 --val_batches 0 \
  --log_interval 50 --learning_rate 1e-4 --weight_decay 1e-5 \
  --max_grad_norm 1.0 --background 1e-4 --background_loss_weight 0.01 \
  --model_vocab 73 --hidden_dim 256 --product_layers 2 --state_layers 4 \
  --num_heads 8 --dim_feedforward 1024 --max_seq_len 512 \
  --dropout 0.1 --attention_dropout 0.1 --seed 42 \
  --use_grouped_batches --group_size 4 --pairwise_loss_weight 0.25 \
  --pairwise_temperature 1.0 --pairwise_all_val_anchors \
  --checkpoint_selection pairwise_guarded \
  --control_metrics_json /path/to/lambda0_control/summary.json
```

先运行一次很短的 `max_steps=5` smoke，再运行 500 steps。不同 lambda 使用不同输出目录，禁止
`--overwrite` 历史结果。

seed 42 第一轮进入 seed 43 复核的门槛：

1. shared-anchor pair accuracy 相对 lambda=0 control 提高至少 **3.0 个百分点**；
2. reward-score Pearson 相对 control 不下降超过 0.03；
3. Bregman loss 不超过 control 的 1.15 倍；
4. valid pair group fraction 足够稳定，no-action 跳过率被报告且无异常激增；
5. 训练无 NaN，wall 不超过 control 的 2.5 倍。

若 0.25 和 1.0 都通过，按 pair accuracy、Pearson、Bregman、较小 lambda 的顺序选一个，不添加
新 lambda。然后用 seed 43 只复跑 control 和胜出设置。最终 pilot 通过门槛是两个 seed 的提升
方向一致，且平均 pair accuracy 提升至少 3 点。

若不通过：记录负结果，停止，不生成新数据、不跑 Top-k、不上 10k。

#### P5 当前实验记录（2026-08-08）

三组实验均使用 seed42、1k train/200 validation、500 steps、batch64、同一模型和
`per_position` 无关的 guidance 训练设置。shared-anchor 指标由完整 800-record validation
evaluator 复核：

| run | best step | best Bregman | shared-anchor pair acc | Pearson | wall | peak allocated/reserved |
|---|---:|---:|---:|---:|---:|---:|
| lambda=0 control | 100 | 0.79768 | **59.73%** | 0.1132 | 136.8s | 2.03/3.50GB |
| lambda=0.25 | 300 | 0.81301 | 58.63% | 0.1562 | 169.3s | 2.04/3.50GB |
| lambda=1.0 | 300 | 0.80533 | 57.93% | 0.1822 | 177.1s | 2.04/3.50GB |

两种 pairwise 权重都低于同 seed grouped control，未达到预注册的“至少提升 3 个百分点”门槛；
因此不运行 seed43、不生成 10k 新数据、不做 Top-k sampling。旧 checkpoint 采用同一 evaluator
的参考值为：1k old best **56.04%**、10k old best **55.39%**、10k old final **57.49%**。
这说明 grouped control 本身比旧打散训练的 checkpoint 更好，但当前 pairwise loss 没有带来
额外排序收益。P5 记为**正确实现但实验未通过**，下一步先做 loss/数据诊断，不把失败归因于
随机波动，也不通过事后换 seed 或 checkpoint 选择规则来挽救。

P5 完成后更新 `new_docs/dgm.md`、`new_docs/0808.md` 对应实验占位，并创建结果 commit。大型
checkpoint 不提交，只提交配置/summary JSON 和文档；若 summary 位于 `/root/autodl-tmp`，复制
前先确认文件不包含巨大 tensor 或敏感路径。

#### P5b：pairwise pilot 失败原因审计（2026-08-08）

P5 的 pairwise loss 名义上要求同一 product 组内共享 `(product, x_t, t)`，因此在继续扩大训练
前先审计冻结 guidance records 的真实结构。新增只读脚本
`scripts/audit_guidance_anchors.py`，只读取 `.pt` 记录，统计每组 state/time 的唯一数以及同一
时间的偶然配对；对应单元测试为 `tests/guidance/test_anchor_audit.py`。

validation-200 的可复现结果（JSON 保存于 `/root/autodl-tmp/dgm_guidance_runs/anchor_audit_val200.json`）：

| split | records/groups | group size | mean unique time | mean unique state | all states equal | all times equal |
|---|---:|---:|---:|---:|---:|---:|
| train-1k | 4,000/1,000 | 4 | 3.936 | 2.630 | 117/1,000 (11.7%) | 0/1,000 |
| train-10k | 40,000/10,000 | 4 | 3.9364 | 2.6092 | 1,176/10,000 (11.76%) | 0/10,000 |
| val-200 | 800/200 | 4 | 3.940 | 2.655 | 23/200 (11.5%) | 0/200 |

validation-200 中只有 12 个同时间 record pair，且仅 7 个 state 完全相同。原因在于旧生成脚本
先对每个独立 Euler terminal 采样，再调用 `sample_intermediate_states(product, terminal, t)`
为每条记录独立重建中间 state；`source_index` 只表示同一 product，并不表示共享 anchor。因而
当前 evaluator 的“shared-anchor pair accuracy”和 P5 的 pairwise loss 实际是在把不同条件状态
下的 terminal action mask 放进同一组比较，排序信号是反事实且有噪声。这个结论也与 P5 的结果
一致：pairwise lambda=0.25/1.0 分别为 58.63%/57.93%，低于 grouped control 的 59.73%。

辅助审计还发现：validation-200 的 unequal-reward pair 中 18.14% 因某一候选没有有效 action
而被跳过；有效 action-set 的平均 token-set Jaccard 为 0.347，说明候选之间也并非天然共享局部
动作。5 个 batch 的 Bregman/pairwise 梯度 cosine 均值约 0.158，范围 -0.221 到 0.909，不能把
失败简单归因于单一的梯度冲突。

P5b 结论：不是继续调 lambda 或换 seed，而是先修正数据定义。保留现有 `.pt` 和 checkpoint 不
覆盖；下一子阶段应新增隔离的 shared-anchor 数据生成路径：先采样一个公共 `x_t`，再从该状态
独立继续 Euler 得到多个 terminal，并在记录中保存 anchor provenance。只有新数据审计达到每组
时间/state 全相同，才重新进行 1k pilot；P6 的 10k、Top-k 和采样 A/B 继续暂停。

#### P5c：真实 shared-anchor continuation 实现与 smoke（2026-08-08）

为避免复制 Euler 主循环，`sample_euler` 新增可选 `start_time` 和 `initial_origin_mask` 参数；默认
不传参数时通过 byte-level 等价测试。新增隔离脚本 `scripts/generate_shared_anchor_guidance.py`：
每个 product 先执行一次 prefix 到固定 interior step，再将 exact state 批量复制为
`n_children` 行，用一次向量化 continuation 得到多个 terminal；旧数据生成脚本和历史 `.pt` 不
覆盖。`tests/guidance/test_shared_anchor_data.py` 验证配置边界。

真实 `checkpoint_step600000.pt` 的 CUDA smoke：4 products、`n_steps=4`、`anchor_time=0.5`、
`n_children=2`、batch=4，wall **0.799s**，peak allocated/reserved **234/247MB**。结构审计为
4/4 组 `state` 相同、4/4 组 `time` 相同，4/4 组的两个 terminal 不同。该 smoke 只验证 continuation
正确性和批量效率，不提供准确率结论。

随后发现 adaptive endpoint 可能使 `n_steps=100` 的轨迹包含 101 个增量，不能用
`anchor_index / actual_steps` 近似时间。新增 `get_euler_step_times()` 复现真实 step schedule，
并要求 trajectory/time 长度一致；修正后的 full-step smoke 的第 50 步时间为 **0.49999979**。
修正前生成的 `train_shared_anchor1000_validity.pt` 不进入任何 reward 或训练实验，必须覆盖重生成。

下一步实验顺序固定为：

1. 用该脚本在隔离目录生成 1k products、每组 4 children 的 validity 数据；
2. 用已有 `generate_forward_guidance_data.py` 附加 Molecular Transformer forward-beam reward；
3. 用 `audit_guidance_anchors.py` 验证所有组 `state/time` 唯一数为 1；
4. 在同一新数据上运行 grouped control 与 pairwise λ=0.25/1.0 的短 pilot；只有新 pilot 通过预注册
   +3pp 门槛，才恢复 P6 10k。

#### P5d：corrected shared-anchor 1k pilot（2026-08-08）

修正 adaptive-time 后重新生成的隔离数据为：train 1,000 products/4,000 records，validation
200 products/800 records；每组 4 children，`n_steps=100`、anchor step=50、真实
`anchor_time=0.49999979`，并用同一 Molecular Transformer forward-beam reward。两份数据均通过
anchor audit：所有组的 state/time 唯一数均为 1。生成 wall 为 train **121.6s**、validation
**25.3s**；forward reward wall 为 **106.6s/18.0s**。

在相同 500 steps、batch=64、seed=42、完整 validation 的三组训练结果如下。pairwise checkpoint
仍按 control Bregman 的 1.15× guard 选择；control 按 validation loss 选择。

| run | best step | Bregman | shared-anchor pair acc | Pearson | wall | peak allocated/reserved |
|---|---:|---:|---:|---:|---:|---:|
| lambda=0 control | 500 | 0.69165 | **55.15%** | 0.1135 | 118.0s | 2.03/3.44GB |
| lambda=0.25 | 100 | 0.76427 | **59.66%** | -0.0118 | 143.7s | 2.04/3.44GB |
| lambda=1.0 | 300 | 0.77967 | 53.65% | 0.0645 | 156.3s | 2.04/3.44GB |

lambda=0.25 的 pair accuracy 相对 control 提升 4.51pp 且 Bregman guard 通过，但 Pearson 下降
0.125，超过预注册允许的 0.03；lambda=1.0 的 pair accuracy 反而低于 control。因此 P5d
是**排序改善、校准失败的部分结果**，没有满足联合 gate，不运行 seed43、10k、Top-k 或 Euler
采样 A/B。该结果不能解释为 end-to-end 准确率提升：尚未将任何 guidance checkpoint 接入
ordinary Euler 做反应级评估。

诊断含义：真实 shared anchor 修复了 P5b 的数据语义问题，pairwise loss 确实能提高 held-out
rank accuracy，但当前 `mean(log H)` 排序目标会牺牲与 reward 的连续校准（尤其 lambda=.25）。
后续若继续，应明确选择“校准优先的 score-aware ranking”或“纯 rank 优先”研究目标；在目标未
重新定义前，不再盲目扫描 lambda。

### P6：10k 训练

只有 P5 通过才执行。使用：

```text
train_multiterm10000_beam.pt  # 40,000 records / 10,000 groups
val_multiterm200_beam.pt      # 同一冻结 held-out，仅用于训练诊断
batch 64
5,000 steps / 8 epochs
val_interval 500
```

只训练 P5 胜出的一个 lambda，先 seed 42。与旧 10k checkpoint 以及必要的 fresh lambda=0
grouped control 用**同一个 shared-anchor evaluator**比较。若 seed 42 通过，再用 seed 43 复核；
不运行第三个 seed，不扫描 dropout、学习率、网络宽度或 step 数。

10k 通过门槛：

1. 两个 seed 的 shared-anchor pair accuracy 均不低于对应 control；
2. 平均提升至少 2.0 个百分点；
3. Pearson 不系统下降，Bregman guardrail 通过；
4. 与 1k 的信号方向一致；
5. 记录 training wall、steps/s、peak allocated/reserved 和 validation wall。

未通过则停止，不靠改 checkpoint selection 或事后选择 final step 挽救结论。

### P7：固定 ordinary Euler A/B 准确率验证

只有 P6 通过才执行。禁止先接 Euler-Beam。固定：

```text
sampler = euler
n_samples = 3
n_steps = 100
batch_size = 64
seed = 42
guidance_beta = 0.10
guidance_rate_normalization = per_position
augmentation = 20
n_best = 10
```

validation-A 命令模板：

```bash
conda activate ef
python scripts/eval.py \
  --checkpoint new_checkpoints/checkpoint_step600000.pt \
  --products_file "datasets/USPTO_50K_PtoR_aug20_#global#/val/src-val.txt" \
  --targets "datasets/USPTO_50K_PtoR_aug20_#global#/val/tgt-val.txt" \
  --output_dir results/dgm_pairwise_valA \
  --sampler euler --n_samples 3 --n_steps 100 \
  --batch_size 64 --device cuda --seed 42 \
  --start_product 4000 --max_products 4000 --augmentation 20 \
  --guidance_checkpoint /path/to/pairwise_guidance_best.pt \
  --guidance_beta 0.10 --guidance_rate_normalization per_position \
  --n_best 10 --diagnostics
```

先加 `--dry_run` 评估命令推导和 `target_offset=200`，确认合理后才正式运行。validation-B 使用：

```text
--start_product 8000 --max_products 4000
```

并确认 target offset 自动为 400。不得覆盖现有 baseline predictions；优先复用已有、相同 SHA
和 metadata 的 baseline，若环境/代码变化导致不可比才重新运行。

必须报告：

- Top-1～10；
- Oracle；
- invalid rate、unique candidates、target rank/coverage；
- sampling wall、score wall、总 wall、peak GPU memory；
- base/guidance forward 次数；
- 与各自 split baseline 的逐项差值。

P7 综合通过门槛：

1. validation-A 和 B 的 Top-1 都不得低于各自 baseline；
2. A 和 B 各自至少在 Top-3、Top-10、Oracle 中改善一项；
3. invalid rate 不增加超过 0.5 个百分点；
4. 无概率、CUDA、长度或 scoring 错误；
5. 效率开销完整披露。当前旧 guidance 约慢 47%，新方案如果仍接近该开销可以作为研究结果，
   但不能隐瞒；若更慢需 profile，不能直接扩大数据。

不允许先看 A 后调 beta 再看 B。两块都固定 beta=0.10，避免 validation overfitting。

### P8：是否接 Euler-Beam

只有 P7 完整通过后，才能开一个新的隔离任务把 guidance 接到 Euler-Beam R9K1M2。该工作不在
本文档的代码范围内，因为它需要重新定义 branch/child 层的 guidance 调用、固定总候选预算和
效率比较。

P7 不通过时，结论应是“显式 pairwise 仍不足以让 learned guidance 稳定改善采样”，保留代码
作为研究开关，默认关闭，不继续堆叠到 Beam。

---

## 5. 实验记录模板

每次正式实验把下表追加到 `new_docs/dgm.md`，不要只写终端口头结论。

### 5.1 训练实验

| run | data | seed | lambda | steps | best step | val Bregman | pair acc | Pearson | no-action | wall | peak memory | conclusion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 占位 | 1k/10k | 42 | 0/0.25/1 | 500/5000 |  |  |  |  |  |  |  |  |

### 5.2 采样实验

| split/method | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Oracle | invalid | unique | sampling wall | total wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val-A baseline | 51.0 | 66.5 | 72.0 | 77.0 | 83.5 | 86.5 |  |  | 253.0s |  |
| val-A pairwise |  |  |  |  |  |  |  |  |  |  |
| val-B baseline | 58.5 | 73.0 | 77.5 | 85.5 | 88.5 | 91.0 |  |  | 264.4s |  |
| val-B pairwise |  |  |  |  |  |  |  |  |  |  |

每个结论必须写清楚：数据 split、reaction range、checkpoint、seed、candidate budget、beta、rate
normalization、aggregation/scoring mode。不能只写“提升了 X%”。

---

## 6. Git 和执行纪律

每个阶段遵守：

1. 修改前检查 `git status --short`；
2. 只用小范围 patch，不大规模重写；
3. 修改后先 targeted tests，再必要 smoke/实验；
4. 更新本计划和 `new_docs/dgm.md` 的状态、命令、结果、负结论；
5. 明确阶段进展后创建范围清晰的 commit，并 push 当前分支；
6. 不提交 checkpoint、guidance dataset、巨大 predictions；只提交代码、测试、小型 JSON/Markdown；
7. 不覆盖历史实验输出目录；
8. 不以长时间等待代替排查。先用 5 steps/小 batch 验证，再运行 500/5000 steps；
9. 不因实验失败而偷偷换 split、seed、beta 或 checkpoint selection；
10. 如果发现用户已有未提交改动与本任务重叠，停止并报告，不能覆盖。

建议 commit 序列：

```text
Add product-grouped guidance batches
Add shared-anchor guidance ranking loss
Train guidance with guarded pairwise selection
Record pairwise guidance pilot
Record 10k pairwise guidance experiment
Record pairwise Euler validation
```

没有达到对应阶段门槛时，也要提交代码测试或负结果文档，但不能把阶段标成“通过”。

---

## 7. 需要停止并汇报的实质性阻塞

只有以下情况停止等待用户决策：

1. 现有 guidance `.pt` 数据缺失且无法从 provenance 确认重生成范围；
2. group 的实际结构不是每 product 4 records，且修复需要改变已冻结的数据定义；
3. shared-anchor action mask 暴露出与当前 X-space action convention 冲突的正确性问题；
4. 实现必须修改基础 Edit Flows checkpoint、训练数据或 Euler 概率公式；
5. 两个研究目标发生冲突，例如必须在 pairwise、terminal reranking 或 exact Z-space 中重新选主线；
6. GPU/CUDA/依赖在合理排查后仍无法运行。

普通实验未提升、训练较慢或某个 lambda 失败不是权限阻塞：按本文 gate 记录负结果并停止该扩大
路径即可。

---

## 8. 最终交付定义

本文档任务完成时，仓库应具备：

1. 可复现的 product-group batch sampler；
2. 可独立测试的 shared-anchor terminal score 和 pairwise loss；
3. Bregman + pairwise 联合训练及 guarded checkpoint selection；
4. 同口径旧/new checkpoint evaluator；
5. 完整 targeted/full regression 记录；
6. 1k pilot 结论；通过时再有 10k 和 Euler validation-A/B 结论；
7. 每阶段 Markdown/JSON、清晰 Git commit 和远端同步；
8. 明确写出该方法是否通过默认推理门槛。

即使结果为负，只要正确实现、遵守预注册实验协议并得到可解释结论，也属于完成；不得用没有
实验依据的进一步重写来掩盖负结果。
