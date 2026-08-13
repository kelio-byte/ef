# Structured first-edit diversification：实现与 dev1000 结果

日期：2026-08-13

## 结论

当前版本不应替换 Euler N=9 或 R9K1M2。

它确实解决了“9 条轨迹首步可能重复”的问题，但使用“局部 `log(lambda)` 排名前 9 的方向全部强制保留”作为筛选规则，选入了很多低质量方向。结果是候选多样性明显增加，Top-K、Oracle 和 invalid rate 变差。

这个结果说明：我们需要筛选“有用且不同”的 child，而不是只筛选“不同”的 child。

## 1. 实现了什么

新增 sampler：`structured_diversification`，代码在
[`edit_flows/sampling/structured_diversification.py`](../edit_flows/sampling/structured_diversification.py)。它的流程是：

1. 对当前 product 在 `t=0` 做一次模型前向。
2. 将合法的 `(position, operation)` 按 `log(lambda)` 从高到低排序；操作为 `INS/SUB/DEL`。
3. 取前 9 个不同方向；`INS/SUB` 的 token 默认取对应 Q 分布的合法 argmax，`DEL` 不需要 token。
4. 强制执行这 9 个不同的首编辑。
5. 每个 child 从首编辑后的状态独立运行普通 Euler M=1；轨迹之间不再竞争、不再互相剪枝。
6. 固定每个 product 9 个输出，记录首编辑和最终重复情况。

命令入口：

```bash
conda activate ef
python scripts/eval.py \
  --checkpoint new_checkpoints/checkpoint_step600000.pt \
  --products_file datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/dev_unique1000_aug20/src.txt \
  --targets datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/dev_unique1000_aug20/tgt.txt \
  --output_dir results/structured_diversification_dev1000_seed42 \
  --sampler structured_diversification \
  --structured_n_trajectories 9 \
  --structured_token_selection argmax \
  --n_steps 100 --batch_size 64 --device cuda --seed 42 \
  --augmentation 20 --n_best 10 \
  --aggregation_mode legacy_best_rank --process_number 16
```

测试：`95 passed`；新增单测覆盖了不同首方向、无竞争和 token fallback。实现是 opt-in，普通 `euler` 与 `euler_beam` 路径未改为使用它。

## 2. 公平对比配置

三组均使用：

- checkpoint：`new_checkpoints/checkpoint_step600000.pt`
- 数据：`evaluation_v2/dev_unique1000_aug20`，1,000 个 reaction、20,000 个 augmented product 输入
- `n_steps=100`、cubic scheduler、CUDA、batch size 64、seed 42
- 每个输入 9 个候选，共 180,000 条预测
- 评分：`n_best=10`、`aggregation_mode=legacy_best_rank`
- 评分指标在同一 `score_#global#.py` 进程中重新计算

结果目录：

- [Euler N=9](../results/r9_vs_euler_dev1000_euler_n9_seed42)
- [R9K1M2](../results/r9_vs_euler_dev1000_r9k1m2_seed42)
- [Structured diversification](../results/structured_diversification_dev1000_seed42)

## 3. Top-K、覆盖与 invalid

| 指标 | Euler N=9 | R9K1M2 | Structured | Structured−Euler | Structured−R9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-1 | 56.8% | **58.7%** | 55.6% | −1.2 pp | −3.1 pp |
| Top-2 | 71.4% | **73.1%** | 67.2% | −4.2 pp | −5.9 pp |
| Top-3 | 76.5% | **77.4%** | 74.4% | −2.1 pp | −3.0 pp |
| Top-5 | 80.8% | **81.3%** | 79.2% | −1.6 pp | −2.1 pp |
| Top-10 | 84.8% | **85.4%** | 81.7% | −3.1 pp | −3.7 pp |
| Oracle-any | 90.5% | **90.8%** | 89.2% | −1.3 pp | −1.6 pp |
| Top-1 invalid | 11.820% | 12.295% | 13.180% | +1.360 pp | +0.885 pp |
| Top-5 invalid | 11.765% | 12.340% | 24.750% | +12.985 pp | +12.410 pp |
| Top-9 invalid | 11.615% | 12.450% | 35.010% | +23.395 pp | +22.560 pp |
| mean valid candidates / reaction | 158.465 | 157.599 | 135.235 | −23.230 | −22.364 |
| mean true unique / reaction | 22.215 | 23.237 | **50.064** | +27.849 | +26.827 |

Structured 的多样性提升是真实的，但没有转化为有效覆盖：Oracle-any 反而低于两个基线，target 平均最终 rank 从 R9K1M2 的 2.811 降到 3.581。

这里的 invalid rate 是评分脚本按候选 rank 统计的 invalid SMILES，不是最终唯一候选去重后的比例；所以它直接反映了把候选放进 Top-K 排序列表后的可用性损失。

## 4. 机制诊断

### 首编辑确实没有重复

Structured diagnostics：

- 20,000 个输入、每个 9 条轨迹；首方向重复槽位：`0`
- 首个具体 action 重复槽位：`0`
- 9 条轨迹最终平均只有 `8.0971` 个唯一结果，最终重复槽位占 `10.03%`
- 最终恰好 9 个唯一结果的 product：`8,258 / 20,000`
- 最终唯一数分布：3 个 2 条、4 个 37 条、5 个 202 条、6 个 1,007 条、7 个 3,538 条、8 个 6,956 条、9 个 8,258 条

### 选入的方向偏向低层次局部差异

9 个方向的操作构成为：

- `INS`：141,129 / 180,000 = 78.405%
- `SUB`：36,076 / 180,000 = 20.042%
- `DEL`：2,795 / 180,000 = 1.553%

被选方向的 `log(lambda)` 范围是 `[-18.918, -2.760]`，均值约 `-10.109`。这说明为了凑满 9 个方向，后几个方向已经处在很低的局部事件概率区间；它们能制造差异，但不一定是模型真正支持的高价值编辑。

因此当前规则实际做的是：

```text
保证不同首编辑  >  评估首编辑是否有用
```

而不是：

```text
在质量基本不下降的前提下增加不同首编辑
```

## 5. 与当前 R9K1M2 选择逻辑的关系

当前 R9K1M2 的每个 parent 生成两个 child 后：

- child 状态相同：合并 `log_mass`，两个 child 没有质量差异，seed 只决定代表者。
- 一个 child 改变、另一个不改变：`changed_state_bonus` 可以偏向改变的 child；当前实验值为 `0.5`。
- 两个 child 都改变但状态不同：主要按 `log_mass`，相同则按 seed 平局。

所以它不是完全“只看 seed”。但在两个 child 的局部概率质量接近、且状态不同的常见情形下，seed 确实承担了最终平局决策。

R9K1M2 仍然优于本次 Structured，因为它保留了局部概率质量和后续路径竞争；Structured 直接放弃了这些质量信息，只保留“首步方向必须不同”。这也是本实验最重要的诊断。

## 6. 效率

| 组别 | 采样耗时 | 相对 Euler N=9 | 相对 R9K1M2 |
| --- | ---: | ---: | ---: |
| Euler N=9 | 3,550.8 s | — | — |
| R9K1M2 | 3,004.7 s | −15.4% | — |
| Structured | 3,721.9 s | +4.8% | +23.9% |

Structured 只在每个 product 额外做一次 `t=0` 前向，但 9 条轨迹被强制从不同初始状态继续推进，且没有 R9K1M2 的共享路径/质量剪枝收益；当前实现没有达到效率目标。

## 7. 下一版筛选规则建议

不建议继续调当前规则的 seed、signature 或强制 9 路版本。优先级应改为：

1. **质量门槛**：先保留 Top-1 高质量方向；只有当候选的 `log(lambda) + log Q(token)` 高于相对 Top-1 的阈值时，才允许它作为 diversity child。
2. **质量优先、差异作为次级目标**：用 `quality_score + diversity_bonus` 做选择，且 diversity bonus 不能压过明显的质量差距。
3. **混合预算**：不要 9 条全部强制分散。保留少量（例如 2–3 条）结构化首编辑，其余使用原始 Euler/R9K1M2 采样，先验证是否能保住 Top-1。
4. **用短 rollout 估计 future value**：对首编辑候选做极短的 M=1 continuation，用后续路径 log probability 或轻量 validity 作为 tie-break；不要只用 `t=0` 的 `lambda`。

下一次独立实验应先做“2 条高质量结构化 + 7 条原始 R9K1M2”的 dev1000 小对比；如果 Top-1 不低于 R9K1M2，再扩大到完整 `src-test`。
