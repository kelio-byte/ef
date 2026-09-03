# Delayed Structured Diversification v2：实现、sanity check 与 dev1000 对比

日期：2026-08-13

## 结论

v2 修复了上一版在 `t=0` 强行分散首编辑、容易选入低质量方向的问题，Top-1 从 `55.6%` 提升到 `58.5%`，但仍没有超过 R9K1M2；Top-3/5/10 分别落后 R9K1M2 `1.7/1.5/1.4` 个百分点，Oracle 也低 `0.1` 个百分点（`90.7%` vs `90.8%）。同时，9 路最终重复率仍为 `21.72%`，采样耗时比 R9K1M2 高 `33.9%`。

按照本实验预先设定的停止规则：**Oracle 仍下降且 Top-K 明显低于 R9K1M2 时，停止继续调参**。因此，structured diversification 作为当前框架的默认策略不值得继续推进；保留为 opt-in 研究实现，R9K1M2 继续作为主 baseline。

## 1. 实现内容

新增 sampler：
[`edit_flows/sampling/structured_diversification_v2.py`](../edit_flows/sampling/structured_diversification_v2.py)。旧版
[`structured_diversification.py`](../edit_flows/sampling/structured_diversification.py)
保持不变。

每个 product 的流程是：

1. 先运行普通 Euler M=1，直到第一次实际采样到 INS/SUB/DEL 事件；如果一直没有事件，在最后一个数值步做一次有记录的 fallback trigger。
2. 在触发状态上按一步事件概率排序，固定保留 Top-1 mode，再从前 6 个高概率 mode 中按概率无放回抽取另外 2 个 mode。
3. 每个 mode 生成 3 个 completion：INS/SUB 取合法 Q 分布 Top-3 token；DEL 没有 token，使用 3 个不同的 continuation seed。
4. 得到 `3 × 3 = 9` 条轨迹。首编辑后所有轨迹独立使用普通 Euler M=1，不跨轨迹竞争、不互相剪枝。
5. 使用 product/trajectory 稳定 seed，并记录 trigger 时间、mode 候选及排名、completion、首分叉重复率和最终重复率。

CLI 参数：

```text
--sampler structured_diversification_v2
--structured_v2_k_mode 3
--structured_v2_k_completion 3
--structured_v2_mode_pool_size 6
```

## 2. Sanity check

主实验共处理 `20,000` 个 augmented product，每个 product 恰好输出 9 条轨迹，共 `180,000` 条预测。

| 检查项 | 结果 |
| --- | ---: |
| 实际采样事件触发 | `19,079 / 20,000 = 95.395%` |
| 最后一步 fallback | `921 / 20,000 = 4.605%` |
| trigger t 均值 / 最小值 / 最大值 | `0.549804 / 0.0 / 0.999999` |
| trigger t 分布 `[0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]` | `938, 4,855, 6,289, 4,945, 2,973` |
| 每个 product 选中的唯一 mode 数 | 全部为 3 |
| mode pool | 前 6 个 mode |
| 选中 mode rank | rank 1: 20,000；rank 2: 16,825；rank 3: 12,155；rank 4: 5,794；rank 5: 3,217；rank 6: 2,009 |
| 首个具体 action 有 9 个唯一值 | `18,973 / 20,000 = 94.865%` |
| 首分叉平均重复率 | `1.251%` |
| 最终平均唯一候选数 | `7.0451 / 9` |
| 最终重复率 | `21.721%` |

首分叉的 9 个 action 中，唯一数为 9/7/5/3 的 product 数分别为
`18,973/930/95/2`。因此，`3 × 3` 的结构确实大多产生了不同首 action，但不同首 action 并不保证最终状态不同。

60,000 个选中 mode 的操作构成为：`INS 50,396 (83.99%)`、`SUB 8,478 (14.13%)`、`DEL 1,126 (1.88%)`。

按 mode rank 的平均一步事件概率为：rank 1 `0.03094`、rank 2 `0.00757`、rank 3 `0.00329`、rank 4 `0.00183`、rank 5 `0.00118`、rank 6 `0.00084`。因此，虽然备选 mode 限制在前 6 名，后续被抽到的 mode 仍可能只有 Top-1 事件概率的很小一部分；这解释了“有差异但质量不稳定”的现象。

## 3. 公平对比配置

四组均使用相同的：

- checkpoint：`new_checkpoints/checkpoint_step600000.pt`
- 数据：`evaluation_v2/dev_unique1000_aug20`，1,000 个 reaction、20,000 个 augmented product
- `n_steps=100`、cubic scheduler、CUDA、batch size 64、seed 42
- 每个 product 9 条预测，共 180,000 条输出
- 评分：`n_best=10`、`aggregation_mode=legacy_best_rank`、`process_number=16`

结果目录：

- [Euler N=9](../results/r9_vs_euler_dev1000_euler_n9_seed42)
- [R9K1M2](../results/r9_vs_euler_dev1000_r9k1m2_seed42)
- [上一版 Structured](../results/structured_diversification_dev1000_seed42)
- [Structured v2](../results/structured_diversification_v2_dev1000_seed42)

### Top-K、Oracle 与 invalid

| 指标 | Euler N=9 | R9K1M2 | Structured v1 | Structured v2 |
| --- | ---: | ---: | ---: | ---: |
| Top-1 | 56.8% | **58.7%** | 55.6% | 58.5% |
| Top-3 | 76.5% | **77.4%** | 74.4% | 75.7% |
| Top-5 | 80.8% | **81.3%** | 79.2% | 79.8% |
| Top-10 | 84.8% | **85.4%** | 81.7% | 84.0% |
| Oracle-any | 90.5% | **90.8%** | 89.2% | 90.7% |
| Top-1 invalid | 11.820% | 12.295% | 13.180% | **11.000%** |
| Top-5 invalid | **11.765%** | 12.340% | 24.750% | 23.325% |
| Top-9 invalid | **11.615%** | 12.450% | 35.010% | 28.005% |
| 平均 valid candidates / reaction | **158.465** | 157.599 | 135.235 | 143.771 |
| 平均 true unique candidates / reaction | 22.215 | 23.237 | **50.064** | 42.700 |
| 采样耗时 | 3,550.8 s | **3,004.7 s** | 3,721.9 s | 4,022.0 s |

相对 R9K1M2，v2 的变化为：Top-1 `-0.2 pp`、Top-3 `-1.7 pp`、Top-5 `-1.5 pp`、Top-10 `-1.4 pp`、Oracle `-0.1 pp`；Top-1 invalid 改善 `1.295 pp`，但 Top-5/Top-9 invalid 分别恶化 `10.985/15.555 pp`。平均 true unique 候选多 `19.463` 个，但平均 valid 候选少 `13.828` 个。

相对上一版 Structured，v2 的 Top-1/3/5/10 提升为 `+2.9/+1.3/+0.6/+2.3 pp`，Oracle 提升 `+1.5 pp`，Top-9 invalid 从 `35.010%` 降到 `28.005%`；但仍远未达到 R9K1M2 的质量水平。

### 效率

| 组别 | 采样耗时 | 相对 Euler N=9 | 相对 R9K1M2 |
| --- | ---: | ---: | ---: |
| Euler N=9 | 3,550.8 s | — | — |
| R9K1M2 | 3,004.7 s | −15.4% | — |
| Structured v1 | 3,721.9 s | +4.8% | +23.9% |
| Structured v2 | 4,022.0 s | +13.3% | +33.9% |

v2 还需要在触发阶段逐步追踪 pending product，并额外执行 mode/token 分支；“不跨轨迹竞争”保证了实现简单和轨迹独立，但没有带来采样加速。

## 4. 机制判断

v2 的延迟触发和高概率 mode pool 确实修复了上一版最明显的问题：不再在 `t=0` 为凑满 9 路而直接选很低概率的 mode。结果也印证了这一点：Top-1 相比 v1 恢复 `2.9 pp`，Top-9 invalid 下降 `7.005 pp`。

但 `k_completion=3` 产生的局部差异在后续 Euler 中容易重新合并：首分叉平均重复率只有 `1.251%`，最终重复率却升到 `21.721%`。另一方面，Q Top-3 并没有为候选排序提供足够的质量保证，导致中后段 invalid 明显增加。换句话说，v2 主要增加了“路径不同”，却没有增加足够多的“有用路径”。

最关键的质量门槛没有通过：

- Oracle `90.7% < 90.8%`，没有改善候选覆盖；
- Top-3/5/10 低于 R9K1M2；
- 运行时间高于所有对照组，尤其比 R9K1M2 慢约 17 分钟。

因此不继续扫描 `k_mode`、`k_completion`、pool size、seed 或 trigger 阈值，也不把 v2 推到更大数据集。旧版和 v2 都保留为 opt-in 研究代码；默认实验继续使用 Euler N=9 与 R9K1M2。

## 5. 测试与复现

本次新增/修改了 CLI、metadata、sampler 和单测。核心回归测试：

```text
99 passed, 11 warnings
```

实验命令：

```bash
conda activate ef
python scripts/eval.py --overwrite \
  --checkpoint new_checkpoints/checkpoint_step600000.pt \
  --products_file datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/dev_unique1000_aug20/src.txt \
  --targets datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/dev_unique1000_aug20/tgt.txt \
  --output_dir results/structured_diversification_v2_dev1000_seed42 \
  --sampler structured_diversification_v2 \
  --structured_v2_k_mode 3 --structured_v2_k_completion 3 \
  --structured_v2_mode_pool_size 6 \
  --n_steps 100 --batch_size 64 --device cuda --seed 42 --n_best 10 \
  --aggregation_mode legacy_best_rank --process_number 16
```

详细采样诊断在
[`structured_diagnostics.json`](../results/structured_diversification_v2_dev1000_seed42/structured_diagnostics.json)，评分结果在
[`diagnostics.json`](../results/structured_diversification_v2_dev1000_seed42/diagnostics.json)，运行配置和 hash 在
[`sampling_metadata.json`](../results/structured_diversification_v2_dev1000_seed42/sampling_metadata.json)。
