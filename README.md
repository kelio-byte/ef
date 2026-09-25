# EFRetro

只保留 **Product Memory + SPE-M500 + K=1、M 可配置分支采样** 正式方法；默认正式设置为 `n_runs=9, n_children=2`。数据与指定的 500K checkpoint 已包含，无需访问旧仓库；无软链接、guidance、oracle 模型或历史实验。

## 安装

已验证环境：Linux、Python 3.10、PyTorch 2.7.1+cu126、RTX 3090。依赖版本统一定义在 `pyproject.toml`；`requirements.txt` 是含预处理依赖的便捷安装入口。克隆前请先安装 Git 与 Git LFS。

```bash
git lfs install
git clone --branch efretro --single-branch https://github.com/kelio-byte/ef.git efretro
cd efretro
git lfs pull
conda create -n efretro python=3.10 -y
conda activate efretro
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/verify_assets.py
```

以下命令均在仓库根目录执行。采用 editable 安装，数据与 checkpoint 保留在此目录；不要安装到仍含旧版 `edit_flows` 的环境。

## 内容与方法

| 路径 | 必要内容 |
| --- | --- |
| `edit_flows/` | 模型、对齐/损失、正式采样与评分 |
| `scripts/` | 训练、推理、评测、预处理及资产校验入口 |
| `configs/train.yaml` | 当前正式训练配置 |
| `data/uspto50k_m500/` | 冻结的训练/验证/测试数据、词表、训练与验证对齐文件、dev1000 划分 |
| `data/SPE_ChEMBL.txt` | SPE 规则，使用前 500 次合并 |
| `saved_checkpoints/product_memory_m500_step500000.pt` | 指定 checkpoint 的原样副本，含恢复训练所需状态 |
| `assets.json` | 冻结数据与 checkpoint 的校验和 |

模型：10 层状态 Transformer，宽度 256、8 heads、FFN 2048；2 层静态 product-memory 编码器，在状态层 5、10 后交叉注意力融合；572 个 token（568+4），长度上限 96。训练采用 Levenshtein 对齐、`κ(t)=t³` 条件桥及 Bregman 损失；推理中产品记忆只编码一次。

正式结果采样设置：seed=42、batch=32、100 步、R=`n_runs=9`、K=1/M=2、Poisson 编辑、`full_probability`、changed-state bonus=0.5、matmul precision=`high`。R（`--n-runs`）控制独立完整运行数，也就是每个产品最终输出的候选数；M（`--n-children`）控制每一步从当前状态采样的子候选数，不改变输出行数。

对每条运行的当前父状态，每步独立采样 M 个候选，再按结果 token 状态分组；按 `log(该状态出现次数) + changed_state_bonus × 是否偏离原状态` 选一个。平分时选 child seed 较小的代表。**不分配 `1/M` 质量**：同一父状态下该因子对所有子候选相同，比较时会抵消；重复状态的计数已体现其经验采样质量。M 默认 2，可通过 `--n-children` 调整。改变 R 或 M 都会改变采样预算或结果，不应把非正式设置的指标与正式结果直接比较。

评分固定：每个反应 20 个增强视图 × 9 个候选，逆 global 表示 → RDKit 规范化 → 去重 → `legacy_best_rank` 聚合，alpha=1。不要改变输入顺序、batch 或精度后再声称严格复现。

## 推理与评测

```bash
# 完整 dev1000：1000 个原始反应，20000 行输入
python scripts/evaluate.py --split dev1000 --output outputs/dev1000

# 完整 test：5007 个原始反应
python scripts/evaluate.py --split test --output outputs/test

# 示例：每个产品采样 3 次（不是正式 n_runs=9 设置）
python scripts/evaluate.py --split dev1000 --n-runs 3 --output outputs/dev1000_n3

# 示例：每一步采样 4 个子候选 M；每个产品仍输出 n_runs 个结果
python scripts/evaluate.py --split dev1000 --n-runs 9 --n-children 4 --output outputs/dev1000_m4

# 快速连通性检查：8 个反应；不是 dev1000 性能
python scripts/evaluate.py --split dev1000 --max-products 160 --output outputs/smoke
```

输出为 `predictions.txt`、`sampling_metadata.json`、`metrics.json`；默认拒绝覆盖已有预测。`oracle_any` 仅是离线候选覆盖率，不参与推理。`--score-only` 可只重新评分。独立推理/评分入口：

```bash
python scripts/sample.py --products data/uspto50k_m500/dev1000/src.txt --n-runs 9 --n-children 2 --output outputs/sample
python scripts/score.py --predictions outputs/sample/predictions.txt --targets data/uspto50k_m500/dev1000/tgt.txt
```

`--products` 接收 **global 表示、SPE-M500 分词、空格分隔**的产品，不是任意原始 SMILES。独立推理每行默认输出 9 个候选，可通过 `--n-runs` 调整；评分要求按同一反应连续排列的 20 倍增强输入/目标。

## 一键从零训练并测试

```bash
python scripts/run_full.py --device cuda --n-runs 9 --n-children 2
```

使用当前配置从零训练 **600000 步**，然后以本次训练产生的 **500000 步 checkpoint** 对完整 test（5007 个反应）推理并评分；不会使用仓库自带的 checkpoint。正式设置为 `n_runs=9, n_children=2`，两者都可调节。训练文件直接保存在 `training_run/train_月-日/`，例如 `training_run/train_09-25/`；指标在其 `test_step500000/metrics.json`。同名目录已存在时不会覆盖，可用 `--run-name train_09-25_retry` 另起一轮。默认不截取测试集；`--max-products` 仅供连通性检查，不能作为正式性能。若要评估最终 600K 权重，可加 `--evaluate-step 600000`。完整训练与测试需要较长时间和足够磁盘空间。

## 手动训练与继续训练

```bash
python scripts/train.py --config configs/train.yaml --device cuda --save_dir training_runs

# 从随仓库提供的 500K checkpoint 继续到 600K
python scripts/train.py --config configs/train.yaml --device cuda \
  --checkpoint saved_checkpoints/product_memory_m500_step500000.pt --save_dir training_runs
```

配置为单 GPU、batch=256、seed=42、Adam + Noam（warmup=8000），训练至 600000 步，最多保留 20 个常规 checkpoint。正式方法使用其中的 500K checkpoint；Noam 不依赖总步数。程序保存优化器、学习率、随机数与数据位置；不要用不同 batch/设备拓扑做逐位一致的续训比较。输出目录为 `save_dir/数据集名/时间戳/`。每 1000 步写入 `training_monitor.jsonl`（速度、梯度、显存和非有限值检查），完成后生成 `training_summary.json`；TensorBoard 和普通训练日志照常保留。

## 数据与预处理

随仓库提供的 20 倍增强数据可直接训练：train=40003、val=5001、test=5007 个反应。dev1000 是验证集的冻结子集，不是测试集；索引与顺序见其 `manifest.json`。所有数据、词表和 checkpoint 约 **451 MiB**，以 `assets.json` 的 SHA-256 为准。

训练优先读取现成对齐文件；缺失时可重建，已有文件会跳过：

```bash
python scripts/precompute_alignments.py --data_dir data/uspto50k_m500 --splits train val --num_workers 8
```

若另有完成 global 转换的 atom-tokenized train/val/test，可重建 SPE 数据：

```bash
python scripts/preprocess_spe.py --source-dir /path/to/atom_global \
  --output-dir /path/to/rebuilt_m500 --codes data/SPE_ChEMBL.txt --merges 500
```

原始 USPTO 的历史清洗/global 转换流水线不在本仓库范围内；复现从已冻结、已校验的数据开始。不要覆盖正式词表或用不同数据重建的词表加载此 checkpoint。数据与 checkpoint 已设 Git LFS 属性；后续发布须同时提供真实资产，不能只有 LFS pointer。

## 已验证范围

2026-09-25 的完整 dev1000 结果是移除 `stochastic_noop` 之前的历史验证，当时正式设置为 `n_runs=9`：

| 实现 | Top-1 | Top-3 | Top-5 | Top-10 |
| --- | ---: | ---: | ---: | ---: |
| 旧仓库 | 61.9% | 80.7% | 84.2% | 88.0% |
| 本仓库 | 61.9% | 80.7% | 84.2% | 88.0% |

**180000 条预测逐字节一致，1000 个反应的排名一致**；候选覆盖率 90.4%，invalid@1=8.615%。数据与 checkpoint 校验和见 [`assets.json`](assets.json)。

2026-09-26 在 `dev_unique1000_aug20`、M=2 上以 R=5 对比了 no-op 开关：有 no-op 的 Top-1/Top-5 为 61.4%/84.5%，无 no-op 为 61.3%/84.5%。这说明该测试中 Top-5 持平、Top-1 仅差 1 个反应；它不是正式 `n_runs=9` 配置的重新验证。之后的 M 泛化将 M=2 设计为与该无 no-op 选择规则一致，但尚未重跑完整 dev1000。

M 实现检查：M=2 时单输入预测与泛化前的无 no-op 版本逐字节一致；M=4 时完成单反应（20 个增强视图）的 GPU 推理与评分冒烟测试，`sampling_metadata.json` 和 `metrics.json` 均记录 `n_children=4`。这只是连通性检查，不代表性能评估。

训练用完整模型、8 条样本、batch=2 检查了 5 步更新、跨 epoch 与中途续训，权重/优化器/学习率均与旧代码逐项一致。移除已弃用参数后又复测了同样的训练状态及 1440 条 GPU 预测，结果不变。另在禁止读取旧仓库的条件下，完成 GPU 推理及全量训练数据上正式 batch=256 的一次 GPU 更新。由冻结 global 字符串重建的 6 个 SPE 数据文件、词表、4 个对齐文件也逐字节一致。

一键脚本另完成端到端冒烟测试：从零训练 5 步，取新生成的 step 4 checkpoint，完成 160 条 test 输入（8 个反应）的推理与评分；小样本、未收敛模型的分数不代表正式性能。

此次没有重新训练 500000 步，也没有重跑完整 test。旧记录中相同 checkpoint 的完整 test Top-1/3/5/10 为 64.110% / 83.044% / 86.878% / 89.854%，仅作为后续完整测试的历史参考，不是本次实测值。

历史 dev1000 Top-1=63.4% 来自普通 Euler N9，**不是本仓库 K=1/M 分支采样**，不能混作同配置基准。不同 GPU、CUDA 或依赖版本可能产生数值差异；性能复现以指定 checkpoint、冻结输入与上述固定环境/设置为准。
