# 新训练 checkpoint 的 tiny 回归评估

日期：2026-08-07

## 1. 目的与范围

A6000 上重新训练完成了新的 `step=600000` checkpoint。本次先在 tiny 上做冻结配置
回归，确认新模型能够被当前采样器加载，并把模型变化与采样代码变化分开。

- 新 checkpoint：`new_checkpoints/checkpoint_step600000.pt`
- 旧 checkpoint：`checkpoint_step600000.pt`
- 数据：`src-test-tiny.txt` / `tgt-test-tiny.txt`，1000 条增强输入，即 50 个完整反应
- 采样：Euler-Beam R9K1M2，即 `n_runs=9, n_branches=1, n_children=2`
- 其余参数：100 steps、batch64、seed42、full probability、`stochastic_noop`、
  `changed_state_bonus=0.5`、TF32 `high`、相同状态 forward sharing
- 评分：augmentation=20、Top-1～10、legacy best-rank aggregation、diagnostics

两份 checkpoint 的模型配置和目标函数一致；新 checkpoint 额外保存了训练复现、validation
和 TensorBoard 配置字段。采样使用 checkpoint 内的配置，不需要外部 `config.yaml`。

## 2. PyTorch 加载兼容修复

当前环境使用 PyTorch 2.13。2.6 之后 `torch.load` 默认采用 `weights_only=True`，会拒绝
项目 checkpoint 中合法的 NumPy 元数据。`scripts/sample_retro.py` 增加了显式
`weights_only=False`，并保留旧版 PyTorch 的 fallback。

代码 commit：`88a0f2e Fix checkpoint loading on modern PyTorch`

该修改只影响 checkpoint 反序列化，不改变模型、随机数、采样或评分逻辑。

## 3. 正确性与布局检查

每个配置均生成 `50 × 20 × 9 = 9000` 行预测，metadata 校验通过；最终 branch shortfall
为 0。新 checkpoint 的预测 SHA-256 为：

```text
8dd1b3eea19d92527db5f1fd0211de350570549bfef7c3ef864f0dc0fa44f615
```

旧 checkpoint 的预测 SHA-256 为：

```text
9748b25877b1595f39670fa09aebc133db513dad36eaabb97d1a42e3129daeb6
```

## 4. 结果

| checkpoint | Top-1 | Top-2 | Top-3 | Top-4 | Top-5 | Top-6 | Top-7 | Top-8 | Top-9 | Top-10 | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧 checkpoint | 60 | 70 | 80 | 84 | 84 | 84 | 84 | 84 | 84 | 84 | 98 |
| 新 checkpoint | 58 | 74 | 78 | 80 | 84 | 86 | 86 | 88 | 90 | 90 | 92 |
| 新−旧（百分点） | -2 | +4 | -2 | -4 | 0 | +2 | +2 | +4 | +6 | +6 | -6 |

| checkpoint | rank-1 invalid | mean valid candidates/reaction | mean true unique/reaction | Oracle coverage | sampling wall |
|---|---:|---:|---:|---:|---:|
| 旧 checkpoint | 22.3% | 140.08 | 31.00 | 98% | 114.72 s |
| 新 checkpoint | 22.3% | 142.32 | 30.08 | 92% | 113.49 s |

结果目录：

- [新 checkpoint 结果](../results/new_ckpt_tiny_r9k1m2_correct/)
- [旧 checkpoint 配对结果](../results/old_ckpt_tiny_r9k1m2_correct/)

## 5. 实验结论

1. 新 checkpoint 没有出现加载、shape、seed 或输出布局错误。
2. 新模型不是所有指标都下降：Top-2 和 Top-6～10 提升，Top-1/3 和 Oracle 下降。
3. 新模型的目标一旦被覆盖，平均最终排名更靠前（2.37）；但被覆盖的反应数减少，说明
   当前更像是“覆盖模式改变”，而不是简单的全局性能提升或下降。
4. tiny 只有 50 个完整反应，不能据此决定新 checkpoint 是否替换旧 checkpoint。随后已在
   validation-200 上完成参数消融，并冻结 R9K1M2/T=1.0 后在 mini-1001 上得到
   Top-1/3/10=`58.242/77.922/86.414%`、Oracle=`91.508%`；详见
   [`new_checkpoint_validation_parameter_sweep.md`](new_checkpoint_validation_parameter_sweep.md)。
5. 采样时间基本不变（约 114 秒）；本次 PyTorch 加载修复没有引入采样开销。

## 6. 过程中的参数纠正

第一次启动误用了 `n_runs=1, n_branches=9`，实际是 R1K9，而不是 R9K1；该结果保留在
`results/new_ckpt_tiny_r9k1m2/` 作为可追溯记录，不用于本节结论。随后已使用正确的
`n_runs=9, n_branches=1` 完成配对复测。

## 7. 下一步

validation-200 已完成参数筛选，mini-1001 已按冻结的 R9K1M2 配置完成；不要再使用 mini
target 调参。若要解释新旧模型差异，应在明确目标后运行旧 checkpoint 的同规模配对，或
检查训练过程中的 validation loss、最佳 checkpoint 与最终 checkpoint，而不是立即修改
Euler-Beam 搜索器。
