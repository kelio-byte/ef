# Session Handoff: visualize_trajectory.py 调试

## 问题

`visualize_trajectory.py` 跑 Euler 采样产出随机垃圾 token（`[Zn][Si][NH3+]`），而同样输入+同样模型在 `sample_retro.py` 里产出正常 SMILES。

也就是说，对相同的测试样本src-test-10.txt，`visualize_trajectory.py` 的结果是mismatch，而 `sample_retro.py` 的结果是100%ACC。

## 已做的修改

两个可视化脚本都已修改：
- 输出文件名加时间戳
- HTML 显示 checkpoint 路径
- trajectory 新增 `--n_samples`，用 RDKit canonical SMILES 比较结果
- trajectory 模型加载已对齐 sample_retro.py

## 当前状态

刚加了 debug 打印 `x_0` token IDs 和模型单次 forward 输出，等待看完整报错信息（`head -20` 截断了）。

## 关键文件

- `scripts/visualize_trajectory.py` — 主战场
- `scripts/visualize_first_step.py` — 已改好，能用
- `scripts/sample_retro.py` — 参考对比
- `edit_flows/sampling/euler.py` — `sample_euler` 函数

## 常用命令

```bash
# trajectory（当前测试用单产物）
python scripts/visualize_trajectory.py \
    --checkpoint checkpoint_step600000.pt \
    --products_file test_one_product.txt \
    --targets_file test_one_target.txt \
    --output_dir "visualizations/trajectory/" \
    --scheduler linear --n_steps 100 --n_samples 1 --device cuda

# 首步可视化
python scripts/visualize_first_step.py \
    --checkpoint checkpoint_step600000.pt \
    --products_file "datasets/USPTO_50K_PtoR_aug20_#global#/test/src-test.txt" \
    --targets_file "datasets/USPTO_50K_PtoR_aug20_#global#/test/tgt-test.txt" \
    --output_dir "visualizations/first_step/" \
    --time_grid "0,0.1" --scheduler linear \
    --deduplicate 20 --n_examples 10 --device cuda

# 标准评测
python scripts/sample_retro.py --checkpoint checkpoint_step600000.pt \
    --products_file test_one_product.txt --sampler euler \
    --n_samples 10 --n_steps 100 --device cuda --output_dir results/test/
python scripts/score_#global#.py --predictions results/test/predictions.txt \
    --targets test_one_target.txt --beam_size 10 --augmentation 1 --n_best 10
```

## 概念备忘

- `deduplicate=20`：每 20 行取一行（20 种 SMILES 排列 = 1 个产物）
- `example_ids`：dedup 后的产物索引，不是原始行号
- score.py 的 Acc 先过滤 invalid SMILES 再计算，跟 Invalid SMILES 率正交
