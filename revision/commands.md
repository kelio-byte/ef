# P0 实际运行记录

日期：2026-08-21
代码基线：`5be921f37749a2edf95a0394da35222b4b8c944e`（工作树包含本轮 P0 修改）
协议：普通 Euler、cubic、100 steps、每个 reaction 取第 1 条 augmentation、N=2、seed=42、batch size=4。

## Atom@600K

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONPATH=. \
/root/miniconda3/envs/ef/bin/python scripts/trajectory_correction_analysis.py \
  --checkpoint new_checkpoints/checkpoint_step600000.pt \
  --products_file 'datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/dev_unique1000_aug20/src.txt' \
  --targets_file 'datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/dev_unique1000_aug20/tgt.txt' \
  --vocab_file 'datasets/USPTO_50K_PtoR_aug20_#global#/example.vocab.src' \
  --output_dir revision/results/p0_smoke/atom_600k \
  --device cuda --n_steps 100 --n_samples 2 --batch_size 4 \
  --augmentation 20 --max_reactions 10 --seed 42 \
  --verify_no_record_change
```

## SPE-M500@490K

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONPATH=. \
/root/miniconda3/envs/ef/bin/python scripts/trajectory_correction_analysis.py \
  --checkpoint new_checkpoints/spe_m500_checkpoints/checkpoint_step490000.pt \
  --products_file 'datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/evaluation_v2/dev_unique1000_aug20/src.txt' \
  --targets_file 'datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/evaluation_v2/dev_unique1000_aug20/tgt.txt' \
  --vocab_file 'datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/example.vocab.src' \
  --output_dir revision/results/p0_smoke/m500_490k \
  --device cuda --n_steps 100 --n_samples 2 --batch_size 4 \
  --augmentation 20 --max_reactions 10 --seed 42 \
  --verify_no_record_change
```

## 文件哈希

| 文件 | SHA256 |
|---|---|
| Atom checkpoint | `6d87c69aa15eb3d04f1b397f8bd54efd60634b804f31dfabdb383388572e0273` |
| SPE-M500 checkpoint | `225156c3e0120bba2f019285670f43a5bbbada1f6e29b75b208255e0abbfefc5` |
| Atom eval src | `c20e337496f52bbeacc7e5870e3eefbb27a653800fa3d3b0fd8dbca8cfd098f6` |
| Atom eval tgt | `a9539ae3b7c6a53450944a994099469041403c984c7df95fb5a288e118c64d8c` |
| Atom vocab | `60237c2047ca3fbca24f2486f4c718d4241a8a80c56920f9f46269381cab3732` |
| SPE-M500 eval src | `54384b145933d85ce707f2eb5b7551e4ba3d3ab9197686ea5f1664e608fd8439` |
| SPE-M500 eval tgt | `1862b520154415e10af85bcdc2d466e741ede4643f58750f9c36f6d224fffdaf` |
| SPE-M500 vocab | `efb4e3ef2b62c57799287c53089f69abeaadfce1644679907f0fa241003d6483` |

第一次 M500 尝试误用了 `#global#_SPE/example.vocab.src`（3039 model tokens），而 checkpoint 需要 `#global#_SPE_m500/example.vocab.src`（572 model tokens），因此触发 CUDA index assert。该次没有写入有效结果；修正路径后重新运行并通过。

---

# 顺序无关 P1/P2 修正版（2026-08-22）

## P1 重分类

自然轨迹原始 JSONL 不重新采样，使用实际事件前后的 token edit distance 重分类：

```bash
PYTHONPATH=. /root/miniconda3/envs/ef/bin/python \
  scripts/reclassify_trajectory_correction.py \
  --atom_dir revision/results/natural/atom \
  --m500_dir revision/results/natural/m500 \
  --output_dir revision/results/natural/order_invariant_summary \
  --n_bootstrap 5000 --seed 20260822
```

## P2 新版干预

固定协议：每模型 1,000 reaction、N=9、100 steps、batch=32、augmentation=20、
普通 Euler/cubic、seed=`42/7/123`。control 模式为
`progress_compatible_first`；harmful 模式为 `force_harmful_completion_first`。
harmful 只改变同一首事件中的一个 INS/SUB token，并要求首事件后的距离比 control
恰好增加 1。

Atom 的 checkpoint/data/vocab：

```text
new_checkpoints/checkpoint_step600000.pt
datasets/USPTO_50K_PtoR_aug20_#global#/
```

SPE-M500 的 checkpoint/data/vocab：

```text
new_checkpoints/spe_m500_checkpoints/checkpoint_step490000.pt
datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/
```

实际运行脚本按上述参数依次执行 12 个组合，结果写入：

```text
revision/results/intervention_order_invariant_v3/{atom,m500}/
revision/logs/intervention_order_invariant_v3_*.log
```

汇总命令：

```bash
PYTHONPATH=. /root/miniconda3/envs/ef/bin/python \
  scripts/summarize_order_invariant_intervention.py \
  --root revision/results/intervention_order_invariant_v3 \
  --output_dir revision/results/intervention_order_invariant_v3_summary \
  --n_bootstrap 5000 --seed 20260822
```

审计结果：40 个相关 pytest 通过；200 个随机事件状态与原始
`apply_ins_del_operations` 一致；2,000 个单 token 距离公式和 1,000 个输出位置/距离
检查通过；完整实验中所有已应用干预的 `damage` 均为 `1`。

---

# P1/P2 实际运行记录

## P1 自然轨迹

统一参数：`max_reactions=1000`、`n_samples=9`、`n_steps=100`、`batch_size=32`、`augmentation=20`、`scheduler=cubic`、seed=`42/7/123`。每个 reaction block 只使用第一条 augmentation。实际输出目录：

```text
revision/results/natural/atom/seed_{42,7,123}/
revision/results/natural/m500/seed_{42,7,123}/
```

Atom@600K 的单个 seed 命令模板：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONPATH=. \
/root/miniconda3/envs/ef/bin/python scripts/trajectory_correction_analysis.py \
  --checkpoint new_checkpoints/checkpoint_step600000.pt \
  --products_file 'datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/dev_unique1000_aug20/src.txt' \
  --targets_file 'datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/dev_unique1000_aug20/tgt.txt' \
  --vocab_file 'datasets/USPTO_50K_PtoR_aug20_#global#/example.vocab.src' \
  --output_dir revision/results/natural/atom/seed_42 \
  --device cuda --n_steps 100 --n_samples 9 --batch_size 32 \
  --augmentation 20 --max_reactions 1000 --seed 42
```

M500 只替换 checkpoint、data/vocab 和 output：

```text
checkpoint: new_checkpoints/spe_m500_checkpoints/checkpoint_step490000.pt
data/vocab: datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/
output: revision/results/natural/m500/seed_{42,7,123}/
```

P1 汇总：

```bash
/root/miniconda3/envs/ef/bin/python scripts/summarize_trajectory_correction.py \
  --atom_dir revision/results/natural/atom \
  --m500_dir revision/results/natural/m500 \
  --output_dir revision/results/natural/summary \
  --n_bootstrap 5000 --seed 20260821
```

## P1：全部 20 个 augmentation 的稳健性复核

主 P1 只取每个 reaction block 的第 1 条 R-SMILES 写法。为验证该选择不会造成表示偏差，固定 seed=42，对每个 `augmentation_index=0…19` 分别运行一次；每个 run 都是 1,000 reaction、N=9、100 steps。输出按 view 分目录，避免在内存中同时保留 20 个 view 的完整轨迹。

Atom 命令模板（`VIEW` 为 `0…19`）：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONPATH=. \
/root/miniconda3/envs/ef/bin/python scripts/trajectory_correction_analysis.py \
  --checkpoint new_checkpoints/checkpoint_step600000.pt \
  --products_file 'datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/dev_unique1000_aug20/src.txt' \
  --targets_file 'datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/dev_unique1000_aug20/tgt.txt' \
  --vocab_file 'datasets/USPTO_50K_PtoR_aug20_#global#/example.vocab.src' \
  --output_dir "revision/results/augmentation_robustness/natural/atom/augmentation_${VIEW}" \
  --device cuda --n_steps 100 --n_samples 9 --batch_size 32 \
  --augmentation 20 --augmentation_index "$VIEW" --max_reactions 1000 --seed 42
```

M500 命令只替换 checkpoint、data/vocab 和 output：

```text
checkpoint: new_checkpoints/spe_m500_checkpoints/checkpoint_step490000.pt
data/vocab: datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/
output: revision/results/augmentation_robustness/natural/m500/augmentation_${VIEW}
```

完成后以 reaction（而不是 path 或 augmentation view）为 paired-bootstrap 单位汇总：

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/root/miniconda3/envs/ef/bin/python scripts/summarize_augmentation_robustness.py \
  --atom_dir revision/results/augmentation_robustness/natural/atom \
  --m500_dir revision/results/augmentation_robustness/natural/m500 \
  --output_dir revision/results/augmentation_robustness/summary \
  --augmentation 20 --n_bootstrap 5000 --seed 20260822
```

## P2 受控首 completion 干预

使用 `scripts/trajectory_correction_intervention.py`，与 P1 相同的 1,000 reaction、N=9、100 steps、batch=32 和三个 seed。每个模型分别运行：

```text
force_correct_completion_first
force_wrong_completion_first
```

实际输出目录：

```text
revision/results/intervention/atom/{force_correct_completion_first,force_wrong_completion_first}/seed_{42,7,123}/
revision/results/intervention/m500/{force_correct_completion_first,force_wrong_completion_first}/seed_{42,7,123}/
```

Atom 的命令模板：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONPATH=. \
/root/miniconda3/envs/ef/bin/python scripts/trajectory_correction_intervention.py \
  --checkpoint new_checkpoints/checkpoint_step600000.pt \
  --products_file 'datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/dev_unique1000_aug20/src.txt' \
  --targets_file 'datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/dev_unique1000_aug20/tgt.txt' \
  --vocab_file 'datasets/USPTO_50K_PtoR_aug20_#global#/example.vocab.src' \
  --output_dir revision/results/intervention/atom/force_wrong_completion_first/seed_42 \
  --mode force_wrong_completion_first --device cuda \
  --n_steps 100 --n_samples 9 --batch_size 32 \
  --augmentation 20 --max_reactions 1000 --seed 42
```

M500 只替换 checkpoint、data/vocab 和 output；`force_correct_completion_first`、seed 7/123 依次重复同一命令。

P2 汇总：

```bash
/root/miniconda3/envs/ef/bin/python scripts/summarize_trajectory_intervention.py \
  --atom_correct revision/results/intervention/atom/force_correct_completion_first \
  --atom_wrong revision/results/intervention/atom/force_wrong_completion_first \
  --m500_correct revision/results/intervention/m500/force_correct_completion_first \
  --m500_wrong revision/results/intervention/m500/force_wrong_completion_first \
  --output_dir revision/results/intervention/summary \
  --n_bootstrap 5000 --seed 20260821
```

P3 按停止规则未运行；没有继续训练、checkpoint sweep 或 sampler sweep。
