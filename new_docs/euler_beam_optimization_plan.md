# Euler-Beam 优化计划与进度记录

## 1. 文档用途

本文档是 `edit_flows/sampling/euler_beam.py` 后续优化的唯一主进度文档。

执行规则：

1. 按本文档中的优先级推进，除非后续实验表明需要调整顺序。
2. 每完成一项任务，立即在对应章节更新：
   - 状态；
   - 实际修改；
   - 测试结果；
   - 性能与准确率结果；
   - 遗留问题。
3. 若设计发生变化，保留原方案和变更原因，不只覆盖最终结论。
4. 每次影响采样行为或性能的修改，都运行第 3 节中的固定基准。

状态标记：

- `[ ]` 未开始
- `[-]` 进行中
- `[x]` 已完成
- `[!]` 受阻或结论不符合预期

---

## 2. 已确认的前提

### 2.1 Checkpoint 与 origin mask

当前固定基准使用：

```text
checkpoint_step600000.pt
```

已确认：

- `use_origin_mask: False`
- checkpoint 内没有 `origin_embedding` 权重

因此当前 Euler-Beam 优化不实现 origin-mask 状态追踪。`use_origin_mask=True`
不应被静默忽略，后续需要增加明确的参数检查。

### 2.2 Euler 与当前 Euler-Beam

普通 Euler：

- `n_samples=N` 表示同一输入的 N 条独立随机轨迹；
- 每条轨迹完整运行 `n_steps`；
- 轨迹之间不合并、不排序、不剪枝；
- N 条结果全部输出。

当前 Euler-Beam：

- 每次搜索初始化 `n_branches=K` 条随机 Euler 分支；
- 每条父分支每一步只产生一个随机后继；
- K 个父分支最多得到 K 个候选，因此当前 Top-K 剪枝没有真正发挥作用；
- 相同 token 序列会被合并，数量不足 K 时再复制高排名分支；
- 每次搜索最终只输出一条结果；
- `n_runs=N` 表示独立执行 N 次搜索并输出 N 条结果。

### 2.3 优化目标

核心目标：

1. 保证分支随机性、路径概率和排序语义正确。
2. 让每个父分支产生 M 个随机后继，形成真正的候选扩展和 Top-K 剪枝。
3. 合理利用多个路径汇聚到同一状态的共识。
4. 在不重复执行 Transformer forward 的前提下批量生成和应用 K×M 个候选。
5. 相比普通 Euler，提高 Top-k 质量或降低相同质量下的采样成本。

---

## 3. 固定性能与准确率基准

每次影响采样结果或性能的修改后，都执行以下四条命令。

### 3.1 Euler-Beam

```bash
python scripts/sample_retro.py \
    --checkpoint checkpoint_step600000.pt \
    --products_file "datasets/USPTO_50K_PtoR_aug20_#global#/test/src-test-tiny.txt" \
    --sampler euler_beam \
    --n_branches 5 \
    --n_runs 3 \
    --n_steps 100 \
    --batch_size 64 \
    --device cuda \
    --seed 42 \
    --output_dir results/bench_beam/

python 'scripts/score_#global#.py' \
    --predictions results/bench_beam/predictions.txt \
    --targets "datasets/USPTO_50K_PtoR_aug20_#global#/test/tgt-test-tiny.txt" \
    --augmentation 20 \
    --beam_size 3 \
    --n_best 5
```

### 3.2 普通 Euler

```bash
python scripts/sample_retro.py \
    --checkpoint checkpoint_step600000.pt \
    --products_file "datasets/USPTO_50K_PtoR_aug20_#global#/test/src-test-tiny.txt" \
    --sampler euler \
    --n_samples 3 \
    --n_steps 100 \
    --batch_size 16 \
    --device cuda \
    --seed 42 \
    --output_dir results/bench_euler/

python 'scripts/score_#global#.py' \
    --predictions results/bench_euler/predictions.txt \
    --targets "datasets/USPTO_50K_PtoR_aug20_#global#/test/tgt-test-tiny.txt" \
    --augmentation 20 \
    --beam_size 3 \
    --n_best 5
```

注意：Euler 的评分必须读取：

```text
results/bench_euler/predictions.txt
```

不能误读 `results/bench_beam/predictions.txt`。

### 3.3 当前固定基准结果（2026-07-31）

| 指标 | Euler-Beam | Euler |
|---|---:|---:|
| 采样总耗时 | 479.062 秒 | 81.144 秒 |
| 预测行数 | 3000 | 3000 |
| Top-1 | 30.000% | 56.000% |
| Top-2 | 54.000% | 68.000% |
| Top-3 | 60.000% | 74.000% |
| Invalid rank 1 | 14.400% | 22.400% |
| Invalid rank 2 | 14.300% | 21.100% |
| Invalid rank 3 | 16.400% | 21.400% |
| Unique Rates | 160.667% | 166.667% |
| 原样输出率 | 26.067% | 6.400% |
| 平均 token 编辑距离 | 4.894 | 8.894 |
| 中位 token 编辑距离 | 3 | 6 |

结论：Euler-Beam 的完整单路径概率排序明显偏向少编辑/不编辑轨迹。下一阶段不能
只优化单路径分数，需要通过 M 后继和状态合并估计最终状态的聚合概率质量。

### 3.4 每次记录的指标

| 指标 | Euler-Beam | Euler | 说明 |
|---|---:|---:|---|
| 修改版本/阶段 |  |  | 对应本文任务编号 |
| 采样总耗时 |  |  | 使用相同机器和 GPU |
| 预测行数 |  |  | 检查输出布局 |
| Top-1 |  |  | 主要质量指标 |
| Top-2 |  |  |  |
| Top-3 |  |  | `beam_size=3` 时的稳定输出上限 |
| Invalid SMILES |  |  | 各 rank |
| Unique Rates |  |  | 结果多样性 |
| 相同 seed 可复现 |  |  | 是/否 |
| 峰值 GPU 显存 |  |  | M 后继实现后重点关注 |

`score_#global#.py` 当前在 `beam_size=3` 时通常只打印到 Top-3，即使
`n_best=5`。若后续需要稳定观察 Top-5，需要单独修复评分脚本的打印限制。

---

## 4. 总体执行顺序

```text
0. 修复分支 seed
        ↓
1. 建立 Euler-Beam 测试和基线
        ↓
2. 修复完整动作概率和排序
        ↓
3. 每个父分支生成固定 M 个后继
        ↓
4. K×M 候选编辑和评分向量化
        ↓
5. 重构合并与概率质量
        ↓
6. 用 offspring budget 取代机械复制
        ↓
7. Profiling 与针对性性能优化
        ↓
8. Guided child、动态预算和多样性研究
```

正确性优先于性能。第 2 项完成前，不使用当前路径分数对大量候选做正式剪枝。

---

## 5. 任务 0：修复分支随机种子

状态：`[x] 已完成`

### 问题

旧实现虽然在 `_BranchState` 中保存了 `seed`，但实际每步执行：

```python
torch.manual_seed(base_seed + step)
```

所有分支共享全局 RNG，分裂产生的新 seed 没有生效，分支结果还依赖 batch
排列。

### 已完成修改

- `_sample_edit_actions()` 增加可选 `torch.Generator` 参数。
- Euler-Beam 为每条分支创建私有 generator。
- 当前分支随机流由 `branch.seed + step` 决定。
- 模型 forward 继续批量执行。
- 普通 Euler 不传 generator 时保持原有行为。

### 验证结果

- 相同 seed 和 step 可复现。
- 不同分支 seed 产生不同随机动作。
- 交换分支 batch 顺序，不改变对应分支结果。
- 分裂生成的新 seed 能影响后续采样。
- Euler 与编辑算子测试：`24 passed`。

### 后续改进

任务 3 引入 M 个 child 时，用稳定的 seed 混合函数替代简单的
`parent.seed + step + child_index`，降低碰撞和相关性风险。

---

## 6. 任务 1：建立 Euler-Beam 测试与初始基线

状态：`[x] 基础测试与固定基线完成；M 专属测试移至任务 3`

### 目标

为后续评分、M 后继和向量化重构建立回归保护。

### 计划测试

- [x] 相同 `base_seed` 完整输出可复现。
- [x] 不同 `base_seed` 能产生不同轨迹。
- [x] 分支 batch 顺序不影响各自随机结果。
- [x] `n_branches=1`（当前单后继）能完整运行。
- [ ] 候选数量满足 `K×M`。
- [ ] 相同状态能够正确合并。
- [ ] 剪枝后状态数不超过 K。
- [ ] 高分候选优先保留。
- [ ] 不同序列长度能够批量编辑。
- [x] `use_origin_mask=True` 明确报错。
- [x] 输出行数符合 `n_products × n_runs`。

### 基线记录

完成任务时在此填写第 3 节的全部指标。

结果：

- 新增完整 `sample_euler_beam()` 同 seed 可复现测试。
- 新增 `n_branches >= 1`、`n_steps >= 1` 和非空 batch 参数保护。
- `use_origin_mask=True` 现在明确抛出 `NotImplementedError`。
- Euler-Beam、Euler 与编辑算子测试共 `31 passed`。
- 固定基准均已完成，结果见下方和任务 2/2.5。
- M 候选数量、合并和 Top-K 测试将在任务 3 实现后完成。

---

## 7. 任务 2：修复完整动作概率与排序

状态：`[!] 实现与测试完成，但准确率下降，需分析评分偏置`

### 当前问题

1. 当前排序使用 `-path_log_p` 再降序排列，可能优先保留概率更低的路径。
2. `_step_log_p()` 只累计已触发事件，没有计算未触发事件概率。
3. delete/substitute 的竞争条件概率没有完整计入。
4. 逐位置 `.item()` 会造成大量 GPU→CPU 同步。

### 正确的一步概率

对每个有效位置，插入过程：

```text
发生 INS：
log(1 - exp(-h λins)) + log Qins(token)

未发生 INS：
-h λins
```

delete/substitute 竞争过程，令：

```text
λds = λdel + λsub
pds = 1 - exp(-h λds)
```

则：

```text
未发生 DEL/SUB：
-h λds

发生 DEL：
log(pds) + log(λdel / λds)

发生 SUB：
log(pds) + log(λsub / λds) + log Qsub(token)
```

插入过程和 delete/substitute 过程的 log-prob 相加，PAD 位置不参与。

### 实施要求

- [x] 实现数值稳定的 `log(1-exp(-x))`。
- [x] 完整计算发生与未发生事件概率。
- [x] 完整计算 DEL/SUB 条件概率。
- [x] 使用 tensor gather 取得 token log-prob。
- [x] 删除逐位置 Python 循环；每个分支仅在最终返回时 `.item()` 一次。
- [x] 修正排序方向。
- [x] 增加手工概率单元测试。
- [x] 增加极低/极高速率的数值稳定性测试。
- [x] 运行固定基准并记录结果。

结果：

- `_step_log_p()` 现在计算完整动作集合概率，包括所有 no-event survival 项。
- DEL/SUB 事件按总速率触发，再加入操作类型条件概率。
- `_branch_sort_key()` 改为优先选择更大的 `path_log_p`。
- 增加完整标量参考、no-event、极端速率和排序测试。
- Euler-Beam、Euler 与编辑算子测试共 `33 passed`。
- 修改后固定 CUDA 基准：479.062 秒；Top-1/2/3 为 30%/54%/60%。
- 相比旧基线 58%/68%/76% 明显下降，完整轨迹概率可能偏向少编辑路径。
- 旧入口连续切分后再拼接，最终产品行顺序仍然正确；旧 58% 不是产品错位造成的虚高。
- 旧入口的问题是同一产品的三个副本没有按预期获得 `42/1042/2042` 三组 run seed。
- 与普通 Euler 对比：原样输出率 26.1% vs 6.4%，平均编辑距离 4.89 vs 8.89。
- 结论：当前选择的是单条 trajectory MAP，天然偏向 no-event；需要 M 后继与合并 log-mass。

---

## 8. 任务 2.5：恢复独立 seed 的批量性能

状态：`[x] 已完成主要批量化；仍可继续 profiling`

### 性能回退

seed 修复前，用户记录的固定采样约为 2–3 分钟。seed 修复后，完整 tiny
Euler-Beam 基准耗时：

```text
4336.885 秒（72 分 16.9 秒）
```

该完整基线处于“seed 已修复、完整路径评分尚未修复”的版本。

### 根因

旧版每一步对整个 `(N,L)` batch 一次性调用 `_sample_edit_actions()`。seed 修复后，
为了给每条分支使用私有 `torch.Generator`，每一步对 N 条分支逐条调用，导致大量
小 CUDA random、bernoulli 和 multinomial kernel。

### 目标

- [x] 保留 branch seed 决定随机结果。
- [x] 保留 batch 排列无关性。
- [x] 不污染 PyTorch 全局 RNG。
- [x] 将随机动作恢复为少量批量 tensor kernel。
- [x] 为后续 `(K,M,L)` 随机动作张量提供基础。
- [x] 用短基准确认速度恢复后，再重新运行完整固定基准。

### 候选方案

优先评估无状态、branch-keyed RNG：随机数由
`(branch_seed, step, position, stream)` 决定并在 GPU 上批量生成。token 采样使用批量
CDF/inverse-CDF，避免逐分支 `multinomial`。若统计质量或性能不理想，再评估其他
可复现的批量 RNG 设计。

结果：

- 无状态 branch-keyed RNG 已实现，并通过基础均匀分布测试。
- `_step_log_p_batch()` 和 `_apply_edits_batch()` 一次处理所有分支。
- 去重 token、step score 和时间各自只批量同步到 CPU 一次。
- 修正每个产品三个 runs 的 seed 分配，并将三个 runs 合并到一次调用；原拼接后的产品输出顺序本身是正确的。
- 相关测试：`39 passed`。
- 64 行单批：164.18 秒 → 34.46 秒 → 32.70 秒。
- 完整 1000 行：4336.885 秒 → 479.062 秒，约 9.05× 加速。

---

### 评分归因消融（进入任务 3 前）

在当前 RNG、批量实现、seed 分配和输出布局完全相同的条件下，只改变评分与排序：

| 模式 | Top-1 | Top-2 | Top-3 | Invalid rank 1 | 原样输出率 | 平均编辑距离 |
|---|---:|---:|---:|---:|---:|---:|
| 完整单路径概率 | 30% | 54% | 60% | 14.4% | 26.07% | 4.89 |
| 旧 triggered-only + reverse | 52% | 64% | 72% | 25.6% | 0.57% | 13.23 |
| 普通 Euler | 56% | 68% | 74% | 22.4% | 6.40% | 8.89 |
| 旧归档基线（不同 RNG/seed） | 58% | 68% | 76% | 20.2% | 5.23% | 10.91 |

归因：

1. 在相同采样候选分布下，旧评分恢复了 22pp Top-1 和 12pp Top-3，评分目标是
   准确率下降的主因。
2. 旧评分只累计已触发事件，并用 `-path_log_p` 降序，因此越低概率、越多编辑的
   轨迹反而越优先；它是有效的激进编辑启发式，不是概率分数。
3. 完整单路径概率倾向 no-event，因为同一最终编辑结果的概率分散在大量不同轨迹上，
   而 no-event 是一条集中且高概率的单路径。
4. 旧 58% 指标的产品对应关系正确，不是虚假指标；52%→58% 的剩余差异只有 3/50
   个样本，主要来自 RNG 和 run seed 改变导致的采样方差。
5. 任务 3 应保留完整转移概率，但排序目标升级为合并后的状态概率质量；旧评分保留为
   `legacy_triggered_reverse` 消融模式，不作为概率解释。

---

## 9. 任务 3：每个父分支生成 M 个后继

状态：`[x] K×M、状态质量合并与受控 changed-state 先验完成；Top-K 多样性留待后续`

### 目标

把当前：

```text
K 个父分支 → K 个候选
```

改为：

```text
K 个父分支 × 每个 M 个 child
→ K×M 个候选
→ 合并
→ 排序
→ 保留 K 个
```

### 第一版设计

- [x] 新增参数：`n_children`，默认值为 1；正式实验显式指定 2 或 4。
- 每个 child 都按标准 Euler 转移分布独立采样。
- 不重复执行 Transformer forward：
  - 模型只处理 K 个父状态；
  - 模型输出用于采样 K×M 组动作。
- 第一版不加入强制编辑、greedy child 或条件采样。

### Seed 设计

- [x] 实现稳定的 64 位 seed 混合函数。
- [x] child seed 由 `(parent_seed, step, child_index)` 决定。
- [x] child seed 不依赖 batch 排列。
- [x] 增加碰撞和可复现性测试。

### 验收标准

- [x] `M=1` 能退化到单后继行为。
- [x] 扩展前模型 batch 规模是 K，而不是 K×M。
- [x] 扩展后候选数量是 K×M。
- [x] Top-K 剪枝第一次真正处理多于 K 个候选。
- [x] 按短实验筛选后运行完整候选基准；明确回归的 M=4 和无收益的 K=5 不做全量。

阶段结果（2026-07-31）：

- 使用等权 Monte Carlo child 质量 `parent_log_mass - log(M)`，相同 token 状态用
  `logsumexp` 合并；不把已用于采样的 action probability 再乘一次，避免概率平方偏置。
- 正式模式的等质量平局用稳定 seed 排序，轨迹 log-prob 只作诊断；旧启发式模式保持原排序。
- `M>1` 不再机械复制分支；`M=1` 暂保留兼容路径。
- 兼容性复核后补回独立的 M=1 trajectory 排序键；用 Git 基线 `520dd42` 在同一
  checkpoint、单产物、K=3、10 步配置下逐字节比较 predictions，结果完全一致。
- 针对性测试：`44 passed`；覆盖 M=1 兼容、K 与 K×M batch 规模、seed 碰撞/
  可复现性、batch 排列无关性和状态质量合并。
- 单产物 10 步 CUDA 烟雾：K=3,M=1 为 3.82 秒；K=3,M=4 为 4.23 秒（含加载）。
- 100 增广行短实验（5 个原始反应，K=3,n_runs=3,n_steps=100）：

| M | 时间 | Top-1/2/3 | 原样输出率 |
|---:|---:|---:|---:|
| 1 | 32.63 秒 | 60/80/80 | 1% |
| 2 | 27.16 秒 | 60/60/60 | 6% |
| 4 | 32.82 秒 | 0/20/20 | 42% |

- 结论：K×M 扩展的正确性和运行效率通过短验收，但纯概率质量随 M 增大明显偏向
  no-event 状态。暂不浪费时间运行全量 M=4；下一阶段先设计有概率依据的探索/进度
  校正，再用更大短集确认后决定是否运行固定全量基准。

补充评分消融（20 个原始反应，K=3,M=2,n_runs=3,n_steps=100）：

| 排序 | 时间 | Top-1/2/3 | Invalid rank 1/2/3 |
|---|---:|---:|---:|
| 状态概率质量 | 97.65 秒 | 40/50/55 | 14.0/14.0/18.0% |
| legacy aggressive | 126.11 秒 | 45/55/60 | 23.25/24.25/22.75% |

该消融确认多后继候选有效，但纯概率排序过于保守、legacy 又明显增加无效结构；后续
应采用受控的 changed-state/progress 先验，而不是在两种极端排序中直接二选一。

### Changed-state 固定先验实验

实现参数：`--euler_beam_changed_state_bonus`，默认 `0.0`。它只给当前 token 状态
不同于输入 product 的候选一个固定排序加分，不修改真实 `log_mass`，也不随编辑次数
持续增加，因此与 legacy 的“越激进越优先”不同。

短筛选（5 个原始反应）：

- K=3,M=2,bonus=0.5：Top-1/2/3 60/60/80，原样率 1.67%，invalid 11–15%。
- K=3,M=2,bonus=1.0：Top-1/2/3 60/60/80，原样率 0.33%，invalid 略升。
- M=4 在 bonus=0.5/1.0 下仍只有 0/0/20 和 20/20/20，淘汰 M=4。

中等复验（20 个原始反应，K=3,M=2）：

| 排序 | 时间 | Top-1/2/3 | Invalid rank 1/2/3 | 原样率 |
|---|---:|---:|---:|---:|
| 纯状态质量 | 97.65 秒 | 40/50/55 | 14.0/14.0/18.0% | 未单独记录 |
| changed bonus=0.5 | 95.81 秒 | 50/55/60 | 14.0/16.75/18.25% | 5.83% |
| legacy aggressive | 126.11 秒 | 45/55/60 | 23.25/24.25/22.75% | 接近 0% |

结论：选择 `K=3,M=2,bonus=0.5` 进入完整 50 反应基准；不运行已出现明确回归的 M=4
全量实验。默认 bonus 仍为 0，避免在完整基准前静默改变正式行为。

### Seed 跨 K / batch 修复与最终全量结果

复核发现入口旧公式 `seed + run*1000 + local_index*n_branches` 同时依赖 K，且
`local_index` 每个 batch 重置，导致改变 beam 宽度或 batch size 会改变/重复产品随机流。
现改为稳定混合 `(base_seed, global_product_index, run_index)`：

- 与 `n_branches` 无关；
- 与 batch 切分无关；
- 21 个 product/run 组合测试无碰撞；
- 相关针对性测试总计 `47 passed`。

公平中等对比（20 个原始反应，M=2,bonus=0.5）：

| K | 时间 | Top-1/2/3 | Invalid rank 1/2/3 |
|---:|---:|---:|---:|
| 3 | 95.18 秒 | 45/55/60 | 16.25/19.25/19.50% |
| 5 | 139.36 秒 | 45/55/55 | 11.50/14.25/14.25% |

K=5 没有提升准确率且慢约 46%，因此最终选择 K=3。

完整固定数据集候选结果（50 个原始反应，1000 增广输入）：

```text
n_branches=3, n_children=2, changed_state_bonus=0.5
n_runs=3, n_steps=100, batch_size=64, seed=42
时间：231.130 秒
Top-1/2/3：58% / 64% / 66%
Invalid rank 1/2/3：13.0% / 14.1% / 13.1%
原样输出率：7.0%
```

相较正式完整单路径概率基线 30/54/60，提升 28/10/6pp；Top-1 达到旧 58% 归档，
耗时相对 479.062 秒约加速 2.07×。Top-2/3 仍低于旧归档 68/76，说明下一阶段重点
应是保留更多互补状态，而不是继续增加 M 或 K。

---

## 10. 任务 4：K×M 候选编辑与评分向量化

状态：`[x] K×M 编辑/评分批量化与父 batch 复用完成`

### 提前完成的局部优化

- [x] 当前单分支 `_step_log_p()` 改为 `mask + gather + sum`。
- [x] 函数内部逐位置 `.item()` 降为最终返回时的一次同步。
- [x] 随后在任务 2 中升级为完整动作概率。
- [x] 增加标量实现等价性测试和无事件测试。

### 当前低效点

- [x] 随机动作已使用 branch-keyed 无状态 GPU batch RNG。
- [x] `_apply_edits_batch()` 已一次处理全部分支。
- [x] `_step_log_p_batch()` 已一次处理全部分支并批量同步。
- [x] 候选 token 已一次传到 CPU 后再构造 `_token_key()`。
- [x] `x_batch` 与 `x_br` 的重复构造已删除。

### 目标结构

```text
父状态：       (K, L)
动作：         (K, M, L)
展平动作：     (K×M, L)
候选序列：     (K×M, L_next)
候选分数：     (K×M,)
```

### 实施要求

- [x] K×M 个 substitution 一次应用。
- [x] `apply_ins_del_operations()` 一次处理 K×M 个候选。
- [x] 一次计算所有候选 step log-prob。
- [x] 一次将候选 token batch 传到 CPU 去重。
- [x] 合并 `x_batch` 与 `x_br` 的重复构造。
- [x] 保留分支独立、顺序无关的随机语义。

### 随机数策略

已实现基于 `(seed, step, child, position, stream)` 的无状态 GPU RNG；模型前向保持
父状态 K，动作、评分和编辑为 K×M batch。

结果（2026-07-31）：

- 删除父级 `x_br/lr_br/lip_br/lsp_br/t_vals` 的重复分配和逐行复制，直接复用模型
  输入及已屏蔽 PAD 的模型输出。
- 删除采样器中从未被后续消费的 `ins_probs/sub_probs` 指数张量和返回值。
- 对优化前 commit `efaad03`，分别在 M=1/M=2、真实 checkpoint、K=3、20 步下
  逐字节比较 predictions，结果完全一致。
- Euler/Euler-Beam/编辑算子针对性测试：`47 passed`。
- 100 增广行短性能（K=3,M=2,n_runs=3,n_steps=100）：26.924 秒 → 22.775 秒，
  加速约 15.4%；完整 predictions 逐字节一致。

---

## 11. 任务 5：重构合并与候选评分语义

状态：`[ ] 未开始`

### 合并 key

当前 checkpoint 不使用 origin mask。第一版合并状态至少包含：

```text
token sequence
discrete step
```

避免直接使用浮点 `t` 作为哈希 key。

### 状态分数

计划区分：

```text
best_path_log_p：到达该状态的最佳单路径概率
log_mass：不同有效路径的 logsumexp 概率质量
consensus_count：本轮到达该状态的候选数量
```

完全相同的 `(parent_state_id, action_signature)` 重复样本不能被误认为不同数学
路径并重复累加概率。

### 分阶段排序

第一阶段：

```text
主键：best_path_log_p
次键：consensus_count
```

概率质量定义验证完成后，再评估：

```text
主键：log_mass
次键：best_path_log_p
```

### 验收标准

- [ ] 同一结果的多条不同路径正确聚合。
- [ ] 完全重复动作不重复计算路径质量。
- [ ] 合并结果不依赖候选输入顺序。
- [ ] 排序具有稳定 tie-break。
- [ ] 运行固定基准。

结果：待填写。

---

## 12. 任务 6：用 offspring budget 取代机械复制

状态：`[ ] 未开始`

### 当前问题

当前补充分支时：

```python
parent_idx = len(branches) % len(branches)
```

非空时永远等于 0，因此总是复制排名第一的状态。相同状态被物理复制后还会重复
执行相同的模型 forward。

### 目标

只保存唯一逻辑状态，为每个状态分配下一步后继预算：

```text
state A → M_A 个 child
state B → M_B 个 child
state C → M_C 个 child
```

满足：

```text
Σ M_i = expansion_budget
```

### 实施阶段

1. 先平均分配，并保证每个保留状态至少一个 child。
2. 再研究按 `log_mass` 和 temperature 动态分配。
3. 评估是否需要设置最大单状态预算，防止过早塌缩。

结果：待填写。

---

## 13. 任务 7：Profiling 与性能优化

状态：`[ ] 未开始`

### 分项计时

- [ ] Transformer forward。
- [ ] 私有 RNG 动作采样。
- [ ] K×M 编辑应用。
- [ ] 路径评分。
- [ ] GPU→CPU 候选传输。
- [ ] Python 去重和排序。
- [ ] 总采样耗时。
- [ ] 峰值 GPU 显存。

### 参数组合

```text
K=5,  M=1
K=5,  M=2
K=5,  M=4
K=10, M=4
K=10, M=8
```

根据 profiling 决定是否实现：

- 无状态 GPU RNG；
- 分块生成 child；
- GPU 端序列哈希；
- 更紧凑的状态表示；
- mixed precision 后处理。

结果：待填写。

---

## 14. 任务 8：后续搜索创新

状态：`[ ] 未开始`

只有在标准 M 后继基线稳定后才开展：

- [ ] 显式 no-op child。
- [ ] greedy child + stochastic children。
- [ ] 条件至少发生一次编辑的 exploration child。
- [ ] proposal correction。
- [ ] stratified / antithetic sampling。
- [ ] diversity penalty。
- [ ] 动态 `n_children`。
- [ ] 基于不确定性的 expansion budget。
- [ ] canonical SMILES 级合并。

所有非标准 Euler proposal 都必须明确区分：

```text
目标转移概率 p
实际 proposal 概率 q
```

如果用于概率评分，需要加入相应校正；如果仅作为启发式搜索，也必须在实验记录中
明确标注。

---

## 15. 实验记录

### 实验 1：单分支 `_step_log_p()` 向量化

日期：2026-07-31

修改：

- 使用 `-expm1(-h·λ)` 稳定计算事件概率。
- 使用 `gather`、布尔 mask 和 tensor sum 累计 INS/SUB/DEL 贡献。
- 保持只累计已触发事件的原有评分定义。

测试：

```text
tests/sampling/test_euler_beam.py
tests/sampling/test_euler.py
tests/sampling/test_ops.py

26 passed
```

固定准确率基准：该修改理论上与旧评分等价，暂不改变搜索语义。

结论：

- 向量化结果与修改前标量实现一致。
- 无编辑时仍返回 `0.0`，保持现有行为。
- `_step_log_p()` 内部不再按编辑位置触发多次 GPU→CPU 同步。
- 当前仍是每个分支返回时同步一次；K×M 全批量评分留待任务 4 主体完成。

---

### 实验 0：分支 seed 修复

日期：2026-07-31

修改：

- 为 `_sample_edit_actions()` 增加可选 generator。
- Euler-Beam 使用分支私有 RNG。
- 移除 Euler-Beam 每步覆盖全局 RNG 的行为。

测试：

```text
branch RNG checks passed
24 passed
```

固定准确率基准：尚未运行。

结论：

- seed 字段和分裂 seed 已真正生效；
- 分支随机动作不再依赖 batch 排列；
- 下一步进入任务 1。

---

## 16. 决策与变更日志

### 2026-07-31

- 完整固定基准：Euler-Beam 479.062 秒、Top-1 30%；Euler 81.144 秒、Top-1 56%。
- 确认完整单路径概率导致少编辑偏置；不回退概率公式，进入 M 后继和状态质量聚合。
- 修正 `sample_retro.py` 的 run seed 分配，并合并三个 runs 的模型调用；撤回“旧预测产品错位”的判断。

- seed 修复后的逐分支 CUDA RNG 导致性能从约 2–3 分钟退化到 72 分钟。
- 终止完整评分版本基准于 3/16 batch，避免继续消耗约一小时。
- 在 M 后继前插入任务 2.5，先恢复 batch RNG 性能。

- 确认固定 checkpoint 不使用 origin mask。
- 确认先修评分，再实现 M 后继，最后优化高级搜索策略。
- 确认固定比较配置：
  - Euler-Beam：`n_branches=5, n_runs=3`
  - Euler：`n_samples=3`
  - 两者：`n_steps=100, seed=42`
- 修正 Euler 评分输入路径为 `results/bench_euler/predictions.txt`。


### 2026-07-31：58% 版本恢复快照

- 新增 `edit_flows/sampling/recover_euler_beam.py`，保留 58% 基线当时的逐分支私有 RNG、仅事件评分和反向路径排序语义。
- 新增 `scripts/recover_sample_retro.py`，保留当时 Euler-Beam 三段串行采样及 `seed + r * 1000` 的调用方式。
- 两个现行文件均未覆盖；恢复副本来自项目内自动保存文件，并补回当时已经完成的 `_step_log_p` 向量化。
- 已通过 `py_compile`、CLI 导入和旧排序/评分语义检查；归档 `results/bench_beam_pre_full_score/predictions.txt` 为 3000 行。
- 未重复运行耗时的完整基准；历史归档评分为 Top-1 58%、Top-2 68%、Top-3 76%。
