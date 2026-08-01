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

状态：`[x] 状态质量、最佳路径与本轮共识计数语义已分离并验证`

### 合并 key

当前 checkpoint 不使用 origin mask。候选只在同一离散 Euler step 内合并，因此 key 为：

```text
token sequence（外层循环已经固定 discrete step）
```

避免直接使用浮点 `t` 作为哈希 key。

### 状态分数

当前已区分：

```text
best_path_log_p：到达该状态的最佳单路径概率，仅诊断
log_mass：等权 Monte Carlo offspring 的聚合概率质量，正式主分数
consensus_count：本轮到达该状态的 child 数量，仅诊断
```

这里的 child 是独立 Monte Carlo 抽样，不是精确 action 枚举。完全相同 action 被多次
抽中时，频次本身就是目标状态概率的估计证据，应分别贡献 `parent_mass/M`；若未来改成
确定性 action 枚举，才需要按 `(parent_state_id, action_signature)` 去重避免重复计数。

### 分阶段排序

```text
主键：log_mass + changed_state_bonus * I[state != product]
平局：稳定 seed（不重复使用 action probability）
```

### 验收标准

- [x] 同一结果的多个独立 Monte Carlo child 用 `logsumexp` 正确聚合。
- [x] `best_path_log_p` 保留最大值，不再取决于代表分支 seed。
- [x] consensus count 只统计本轮 child，不复制父分支历史 count。
- [x] 合并结果不依赖候选输入顺序。
- [x] 排序具有稳定 tie-break。
- [x] 运行固定基准（复用任务 3 的 58/64/66 全量结果）。

结果（2026-07-31）：

- 新增正序/逆序候选合并测试，总计 `48 passed`。
- 对任务 4 commit `d7fc40c` 使用真实 checkpoint、K=3,M=2、30 步逐字节比较输出，
  predictions 完全一致；本任务只修复诊断语义，不改变正式搜索结果。

---

## 12. 任务 6：用 offspring budget 取代机械复制

状态：`[!] M>1 平均分配已完成；质量 anchor + 序列多样性槽短筛失败，动态预算仍待研究`

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

1. [x] 先平均分配，并保证每个保留状态至少一个 child。
2. [ ] 再研究按 `log_mass` 和 temperature 动态分配。
3. [ ] 评估是否需要设置最大单状态预算，防止过早塌缩。

当前结果与决策点（2026-07-31）：

- M>1 已只保存合并后的唯一 token 状态，每个状态下一步生成相同的 M 个 child；不足 K
  时不再机械复制。M=1 为逐字节兼容旧行为，仍保留旧复制路径。
- 当前 expansion 总量是 `active_unique_states × M`，会在状态塌缩时自动减少计算；尚未
  强制固定为 K×M budget。
- 现有证据显示纯 `log_mass` 在 M=4 时产生 42% 原样输出，而 changed-state 先验虽把
  Top-1 恢复到 58%，Top-2/3 仍只有 64/66。按质量动态增加高分状态 offspring 很可能
  进一步牺牲多样性；反向设置均匀/多样性 quota 则可能降低 Top-1。
- 用户选择先验证“一个质量 anchor + 其余探索槽”的折中策略；验证结果见下节。

### 用户补充全量消融：M=3 的 bonus 阈值

同一完整数据集、K=3,n_runs=3,n_steps=100 下：

| M | changed bonus | Top-1/2/3 | Invalid rank 1/2/3 | 原样率 | 唯一预测 |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.5 | 58/64/66 | 13.0/14.1/13.1% | 7.00% | 2152 |
| 3 | 0.5 | 34/52/56 | 12.5/13.3/13.3% | 16.73% | 2134 |
| 3 | 0.8 | 46/56/60 | 13.7/16.2/16.7% | 4.80% | 2177 |

M=3、bonus=0.8 越过 `log(2)` 阈值后确实消除了 no-event 偏置，但没有恢复 M=2
准确率；它与 M=2 只有 29.33% predictions 逐行一致。由于 binary changed bonus 对所有
已改变状态加分相同，原样率降至 4.8% 后继续增大 bonus 已无法改善不同 changed states
之间的排序，只可能进一步压低原状态并增加无效编辑。因此停止继续扫描 M=3 的固定
bonus，保留 M=2、bonus=0.5 为当前最佳配置。

### Anchor + diversity slots 短筛（2026-08-01）

候选选择固定保留最高质量 anchor，其余槽按“质量 + 多样性权重”贪心选择。第一次以
归一化 Levenshtein 距离实现，在首个 64 样本 batch 已耗时约 71 秒，明显不适合完整
实验，立即终止。随后改用近线性的 token-bigram Jaccard 距离，并先在 5 个反应（100
条增强输入）上与 mass-only 基线比较：

| 选择方法 | diversity weight | 耗时 | Top-1/2/3 | Invalid rank 1/2/3 |
|---|---:|---:|---:|---:|
| mass-only | 0 | 22.349 秒 | 60/60/80 | 16/19/21% |
| anchor + bigram | 0.1 | 25.573 秒 | 60/60/60 | 18/13/14% |
| anchor + bigram | 0.2 | 25.893 秒 | 60/60/60 | 18/13/14% |
| anchor + bigram | 0.4 | 26.106 秒 | 60/60/60 | 18/13/14% |

结论：三种非零权重没有表现出准确率梯度，Top-3 均下降 20 个百分点，同时运行时间
增加约 14–17%。这说明 token 序列表面差异并不是有效的反应路径多样性代理。按照
“短筛无正向信号则不跑全量”的原则，停止完整实验并撤回 `anchor_diverse` 实验代码；
正式实现仍采用 mass-only 排序，当前推荐配置保持 K=3,M=2,bonus=0.5。

下一步不应继续扫描该 diversity weight。若继续任务 6，应优先研究与采样动作/反应
中心相关的探索约束，或先完成任务 7 profiling，再决定动态 offspring budget 的实现
是否值得其额外开销。

---

## 13. 任务 7：Profiling 与性能优化

状态：`[x] 已完成 profiling、批量 key、inference mode 与 3090 TF32 优化`

### 分项计时

- [x] Transformer forward（首轮 cProfile）。
- [x] 私有 RNG 动作采样（首轮 cProfile）。
- [x] K×M 编辑应用（首轮 cProfile）。
- [x] 路径评分（首轮 cProfile）。
- [x] GPU→CPU 候选传输与 key 构造（首轮 cProfile，并完成优化）。
- [x] Python 去重和排序（首轮 cProfile）。
- [x] 总采样耗时（100 条短集）。
- [x] 峰值 GPU 显存。

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

首轮结果（2026-08-01，20 条输入、K=3,M=2,n_runs=1,n_steps=30）：

- 采样阶段 1.772 秒，其中 Transformer forward 累计 1.266 秒，约占 72%。
- Python 后处理热点依次为私有 RNG 0.125 秒、token key 0.078 秒、批量编辑
  0.071 秒、路径评分 0.026 秒；`_step_log_p_batch` 已不是主要瓶颈。
- M=1/2/3 在 100 条输入、100 步上的采样时间分别为 8.80/7.44/8.04 秒。
  成本没有随 M 线性增长，因为 Transformer 只对 parent forward，child 后处理较轻，
  且不同 M 会改变每步合并后的活跃唯一状态数。不能用 M 直接推算耗时。

完成的优化：把初始状态和每步所有候选一次性 `.cpu().tolist()`，再以纯 Python 整批
构造 token key，取代逐样本/逐 token Tensor 标量转换。相同 cProfile 中 key 构造从
0.078 秒降至 0.029 秒（约 63%）；100 条 M=2 短集从 7.44 秒降至 6.84 秒（单次约
8.1%，仍需把它视作短测信号而非稳定全量加速比）。24 项单元测试通过，30 步和
100 步真实 checkpoint predictions 均逐字节一致。

下一轮 profiling 应使用 CUDA event 或同步分段计时进一步区分 forward、GPU RNG、
编辑 kernel 和 CPU merge；当前不依据一次 cProfile 大范围重写 forward 路径。

同日以 PyTorch CPU+CUDA profiler 对相同 20 条短集复核（profiler 本身有较大开销，
不用于吞吐计时）：`addmm/GEMM` 占 self CUDA 约 58.6%，copy 6.8%，clamp 5.6%，
layer norm 5.2%，attention 的 bmm/baddbmm 合计约 9.6%，`nonzero` 约 1.7%。这再次
表明主要 GPU 成本在 Transformer 矩阵运算，编辑 kernel 和候选后处理并非当前首要
瓶颈。后续性能研究应优先评估推理精度/编译/模型 forward batch 利用率；任何改变
数值精度的方案都必须先做 predictions 与准确率回归。

低风险推理上下文优化（2026-08-01）：确认 `sample_retro.py` 已调用 `model.eval()`，
将 Euler-Beam 的 `@torch.no_grad()` 改为 `@torch.inference_mode()`。在 100 条输入、
K=3,M=2,n_runs=1,n_steps=100 下，修改前 3 次为 6.54/6.52/6.50 秒（中位 6.52），
修改后为 6.40/6.28/6.30 秒（中位 6.30），短测中位数约改善 3.4%。24 项单元测试
通过；30 步真实 checkpoint 回归以及三次 100 步短测 predictions 均与基线逐字节
一致。因此保留该修改。该优化不改变模型精度、随机数或搜索排序。

TF32 可选短筛（2026-08-01）：RTX 3090 / PyTorch 2.7.1 的默认 float32 matmul
precision 为 `highest`，矩阵乘 TF32 未启用。临时设为 `high` 后，100 条输入、
K=3,M=2,n_runs=1 的三次采样为 5.28/5.30/5.32 秒，中位 5.30 秒，相对当前
FP32+inference-mode 的 6.30 秒约快 15.9%，三次 TF32 predictions 完全一致。

扩大到 20 个反应（400 条增强输入）、n_runs=3 后，当前版本配对结果为：

| precision | 采样时间 | Top-1/2/3 | Invalid rank 1/2/3 |
|---|---:|---:|---:|
| highest (FP32) | 61.46 秒 | 45/55/60 | 16.25/19.25/19.50% |
| high (TF32) | 48.09 秒 | 45/55/60 | 16.25/19.50/19.75% |

TF32 约快 21.8%，与 FP32 有 1185/1200（98.75%）原始预测行一致，Top-1/2/3
完全相同，invalid 差异不超过 0.25 个百分点。旧的 `results/bench_beam` 与当前版本
只有 229/1200 行一致，确认属于不同代码/seed 基线，未用于最终配对结论。

实现 `--euler_beam_matmul_precision {highest,high}`：默认 `highest` 保持完全兼容，
仅 CUDA Euler-Beam 且用户显式传入 `high` 时启用 TF32；不影响 Euler 或其他 sampler。
24 项测试通过，默认模式真实 checkpoint predictions 与修改前逐字节一致。由于 20 个
反应仍不是完整测试集，暂不把 `high` 设为默认；下一步应做一次全量 TF32 采样并与
当前完整 FP32 配对评分，确认 58/64/66 基线是否保持。

完整配对实验（2026-08-01，50 个反应/1000 条增强输入，K=3,M=2,n_runs=3,
n_steps=100,bonus=0.5）：

| precision | 采样时间 | Top-1/2/3 | Invalid rank 1/2/3 | Unique |
|---|---:|---:|---:|---:|
| highest (FP32) | 162.9 秒 | 58/64/66 | 13.0/14.1/13.1% | 164.667% |
| high (TF32) | 123.8 秒 | 58/64/66 | 12.9/14.4/13.1% | 164.667% |

TF32 在完整配对中约快 24.0%，2979/3000（99.30%）原始预测行与 FP32 一致，
Top-1/2/3 和 Unique 完全相同，invalid 差异最多 0.3 个百分点。因此在 RTX 3090 上
将 `high` 确认为日常实验和参数搜索的推荐模式；最终论文数字或严格复现历史 FP32
时使用 `highest` 复核。为避免旧命令静默改变数值行为，CLI 默认仍保留 `highest`，
推荐实验命令必须显式写 `--euler_beam_matmul_precision high`。

峰值显存短测（20 条输入、K=3,M=2,n_runs=3,n_steps=30）中，FP32 和 TF32 均为
429.9 MiB allocated、1766 MiB reserved。TF32 提升来自 Tensor Core 矩阵吞吐，不会
降低 FP32 参数/激活的存储占用。任务 7 至此收口：保留批量 key、inference mode 和
可选 TF32；淘汰 BF16 与当前单次 CLI 下的 `torch.compile`。

BF16 autocast 短筛（2026-08-01）：仅以运行时 wrapper 包裹 Euler-Beam，不修改仓库。
第一个 batch 在模型 `_log_softplus` 内立即失败：autocast 使索引赋值的 destination 为
BF16、source 为 FP32，PyTorch 要求二者 dtype 一致。修复需要修改所有 sampler 共用
的模型 forward，实现范围会扩展到训练/通用模型代码；按当前任务边界不修改，并停止
BF16 实验。`torch.compile` 同样作用共享模型，且当前动态 shape 与首次编译成本未知，
在 TF32 已有稳定 16–22% 短测收益的情况下暂不优先引入。

完整 TF32 验证后对 `torch.compile(mode="reduce-overhead")` 做隔离短筛：仅在
`sample_retro.py` 中临时编译已加载的 Euler-Beam 推理模型，不修改模型定义。100 条
TF32 配置运行 73.3 秒仍未完成第一个 batch，而未编译 TF32 整次只需约 5.3 秒，遂
终止进程。首次编译成本相对当前完整采样约 123.8 秒也过高，且动态 batch/序列 shape
可能触发重编译。因此撤回临时 CLI 参数，不保留 `torch.compile` 实现。

---

## 14. 任务 8：后续搜索创新

状态：`[~] stochastic + 单次 no-op anchor 已完成完整验证`

只有在标准 M 后继基线稳定后才开展：

- [x] 显式 no-op child（单次 t≈0.9 启发式干预）。
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

### 启发式 child 第一阶段（2026-08-01）

初始实验尝试 child 0 使用标准 Euler 随机后继、child 1 使用 greedy-MAP。连续每步
greedy 会造成严重过度编辑；限制为仅在 `t≈0.9` 干预一次后取得完整集提升。但后续
动作诊断发现，5 反应短集中该步 899 个 parent 的 greedy-MAP 全部选择 no-op；将其
替换为显式 no-op 后，20 反应的 1200 条 predictions 逐字节一致。因此准确率提升的
真实来源是“一次性 no-op anchor”，而不是反应中心 greedy 编辑。

正式显式 no-op 完整复跑又与旧 greedy 完整输出 3000/3000 行逐字节一致，指标仍为
60/64/70；采样约 122.6 秒，略快于包含无效 greedy 计算的 124.6 秒。由此完成完整
方法归因，而非仅根据小样本推断。

正式实现据此简化为默认关闭的 `--euler_beam_child_policy stochastic_noop`，仅允许
M=2：child 0 始终是标准随机 Euler 后继；仅在 `t≈0.9` 将 child 1 的编辑 mask 清零，
保护一个父状态免受该步随机编辑破坏。删除未发挥作用的 greedy 动作构造器。

该 no-op child 不是从目标转移分布 p 随机采样，当前没有 proposal q 校正，其 mass
仅作为启发式搜索权重，不能解释为无偏状态概率。默认 `stochastic` 路径不变，真实
checkpoint 回归逐字节一致。

受控消融说明了干预预算的重要性：

| 干预策略 | 5 反应 Top-1/2/3 | Invalid rank 1/2/3 | 结论 |
|---|---:|---:|---|
| 每步强制一个编辑 | 0/0/0 | 58/55/57% | 严重过度编辑，淘汰 |
| 每步在 no-op/单编辑中 MAP | 0/20/40 | 42/39/26% | 后期仍连续编辑，淘汰 |
| 仅在 t≈0.9 保留 no-op | 60/60/80 | 12/22/19% | 通过短筛 |

扩大到 20 个反应后，单次干预由标准 TF32 的 45/55/60 提升至 50/55/65，耗时
48.09→49.49 秒。完整 50 反应配对结果：

| child policy | 时间 | Top-1/2/3 | Invalid rank 1/2/3 | Unique |
|---|---:|---:|---:|---:|
| stochastic | 123.8 秒 | 58/64/66 | 12.9/14.4/13.1% | 164.667% |
| stochastic_noop | 124.6 秒 | 60/64/70 | 12.5/14.5/13.4% | 165.333% |

完整结果 Top-1 提升 2、Top-3 提升 4 个百分点，额外时间约 0.6%，2786/3000
（92.87%）原始预测行保持一致。因此保留为显式启发式选项，但暂不改变默认 policy。
下一步应验证干预时刻 `0.8/0.9/0.95`，且必须先短筛，避免无依据参数扫描。

干预时刻短筛（5 个反应）：`t≈0.8` 为 40/60/60，明显退化；`t≈0.95` 为
60/60/80，但 rank-1 invalid 为 18%，劣于 `t≈0.9` 的 12%。因此保留已有完整验证的
固定 `t≈0.9`，不向 CLI 暴露新的 fraction 参数，也不继续做密集时刻扫描。

分支数短筛：保持 M=2、bonus=0.5 和单次 no-op 不变，K=5 在 5 个反应上得到
40/60/60，耗时 18.84 秒；K=3 为 60/60/80、13.38 秒。K=5 同时降低 Top-1/3
并增加约 41% 时间，因此停止完整实验，推荐配置保持 K=3。

反应中心 forced-edit 筛选：在 `t≈0.9` 统计 899 个 parent 的最佳单编辑相对 no-op
log-prob 增益，全部小于 0；中位数约 −2.30，95% 分位约 −1.89，类型为 INS/SUB/DEL
=807/90/2。以 −1.9 为置信阈值，仅让最高约 5% parent 的 child 1 从 no-op 切换为
forced edit。5 反应准确率不变且 invalid 略降，但扩大至 20 反应后，no-op anchor 的
50/55/65 下降为 50/55/60；耗时约增加 6%。因此撤回实验 policy，不继续扫描阈值。
结论是：模型在该时刻明确偏好 no-op，强制执行概率低于 no-op 的编辑会损害 Top-3。

多次 no-op 短筛：在已有 `t≈0.9` anchor 外再增加 `t≈0.8` no-op，指标降至
40/60/60；改为额外增加 `t≈0.95` no-op，准确率保持 60/60/80，但 invalid 变为
16/24/17%，劣于单次 anchor。因此 no-op 保护也不能重复累积，固定只在 0.9 干预一次。

no-op 搜索质量短筛：默认干预步 stochastic/no-op 各分配 0.5 mass。将 no-op mass
降至 0.25 时准确率仍为 60/60/80，仅 invalid 排名间变化；升至 0.75 时 Top-1 降至
40%。没有准确率证据支持新增参数，且过度偏重 no-op 会伤害 Top-1，因此撤回实验
接口并固定保持等权 0.5。

antithetic child 消融：实验策略令 M=2 的 child 1 在五个随机 stream 上使用 child 0
的互补随机数 `1-u`，并保留 `t≈0.9` no-op anchor。5 与 20 反应上准确率不降且
invalid 明显下降，但完整集结果为：

| policy | 时间 | Top-1/2/3 | Invalid rank 1/2/3 | Unique |
|---|---:|---:|---:|---:|
| stochastic_noop | 122.6 秒 | 60/64/70 | 12.5/14.5/13.4% | 165.333% |
| antithetic_noop | 129.1 秒 | 56/60/66 | 10.5/11.7/11.7% | 164.000% |

互补随机数降低 invalid，却使所有 Top-k 各下降 4 个百分点，Unique 也下降，且增加约
5.3% 时间。说明合法率改善不等同于目标反应覆盖改善。撤回 antithetic 实现，不保留
CLI policy，继续推荐独立 stochastic child + 单次 no-op anchor。

自适应 no-op 初筛：以当前步期望事件数
`μ_total=Σ h(λ_ins+λ_sub+λ_del)` 作为编辑风险。5 反应诊断中，t=0.7/0.8/0.85/
0.9/0.95/0.99 的 μ 中位数约为 0.18/0.23/0.27/0.30/0.38/0.53，趋势上升但同一
时刻分布很宽。随后只在 t=0.9 对 `μ_total≥0.3` 的 parent 启用 no-op，保护
448/899（49.8%）parent。与全 parent no-op 相比仅 3/300 行变化，Top-1/2/3 和
invalid 完全相同。该条件没有收益信号，却引入阈值和额外语义，因此停止扩大实验，
不增加分支状态或 CLI 参数，保持固定 t=0.9 全 parent 单次 anchor。

最终内部 branch 输出消融：实验接口允许每个 run 返回多个最终内部 branch，并按
branch rank 优先排列，保证三个独立 run 的 best 仍占 rank 1–3。5 反应结果中，
`n_runs=1,internal=3` 仅为 20/60/60；`n_runs=3,internal=2` 的 Top-1/2/3 保持
60/60/80，但 Top-4/5/6 没有新增命中，新增候选 invalid 高达 42–48%。说明当前内部
剪枝分数只足以可靠选择第一分支，低排名状态不适合直接作为最终候选。撤回多分支
返回 API，不扩大实验；继续使用三个独立 run 各输出一个 best。

独立 run 数短筛：最佳配置只把 `n_runs=3` 提高到 4，5 反应得到 Top-1/2/3/4
=60/60/80/80，没有新增命中；采样约 12.9→16.7 秒，增加约 30%。第四个独立 run
没有观察到边际准确率收益，因此停止完整实验，推荐保持 `n_runs=3`。

最终 path reranking 消融：最终 K=3 分支的 `log_mass` 与 `path_log_p/n_steps` 相关
系数约 0.92，但二者第一名仅 32/60 一致。实验只在最终输出选择加入
`weight×path_log_p/n_steps`，不改变中间剪枝。weight=0.1/0.5 在 5 反应上均保持
60/60/80；扩大 weight=0.1 至 20 反应后改变 77/1200 行，但仍为 50/55/65，invalid
仅由 15.5/21.0/19.5% 变为 15.5/20.75/20.5%。没有命中收益且 rank-3 invalid
上升，因此撤回最终 path 权重和 CLI 参数，不跑完整集。

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
