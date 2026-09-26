# EFRetro

EFRetro 是一个基于编辑流（Edit Flows）的逆合成模型复现项目：给定产物分子，模型逐步预测插入、替换和删除操作，生成反应物候选。仓库包含模型代码、SPE-M500 处理后的 USPTO-50K 数据、正式训练配置，以及一个 500K Product-memory checkpoint，可直接运行推理评测，也可从头训练。

## 方法概览

1. **数据表示**：反应物和产物以空格分隔的 SPE-M500 token 表示；每个反应保留 20 个增强视图。训练和验证的序列对齐文件已提供，训练使用最优编辑距离对齐。
2. **模型**：10 层 Transformer 编码当前编辑状态，隐藏维度 256、8 个注意力头。独立的 2 层 Transformer 编码产物记忆，并在状态网络第 5、10 层后通过交叉注意力融合。模型为每个位置输出插入/替换/删除速率，以及插入和替换 token 分布。
3. **训练**：使用 cubic 条件桥 `κ(t)=t³` 从源序列和目标序列构造中间状态；模型学习其编辑速率，目标为 Bregman loss。正式训练参数见 [`configs/train.yaml`](configs/train.yaml)。
4. **生成与候选选择**：每条轨迹执行 100 步 Poisson 编辑采样。每一步从当前状态采样 `M` 个子候选，再保留一个：按 `log(状态出现次数) + 0.5 × 是否偏离原状态` 选择，平分时取 child seed 较小的候选。
5. **评分**：候选经 global 表示还原、RDKit 规范化并清除原子映射；同一反应的增强视图结果去重聚合后计算 Top-k。评分实现及聚合规则见 [`edit_flows/scoring.py`](edit_flows/scoring.py)。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `edit_flows/` | 模型、数据读取、序列对齐、编辑流损失、采样和评分实现 |
| `scripts/` | 训练、推理与评测、对齐预计算、SPE 预处理及数据资产校验脚本 |
| `configs/train.yaml` | 正式模型和训练配置：单卡 batch 256，训练 600K 步，最多保留 20 个 checkpoint |
| `data/uspto50k_m500/` | 冻结的 USPTO-50K train/val/test 数据、词表、对齐文件和 dev1000 验证子集 |
| `data/SPE_ChEMBL.txt` | SPE-M500 合并规则 |
| `saved_checkpoints/product_memory_m500_step500000.pt` | 随仓库提供的 500K Product-memory checkpoint |
| `assets.json` | 数据、词表和 checkpoint 的文件大小及 SHA-256 清单 |

数据包含 train 40,003、val 5,001、test 5,007 个反应；每个反应有 20 个增强视图。dev1000 是从 val 冻结抽取的 1,000 个反应，不是 test。大文件通过 Git LFS 管理。

## 安装

推荐 Linux、Python 3.10 和 PyTorch 2.7.1。以下安装命令以 CUDA 12.6 为例；依赖版本定义在 [`pyproject.toml`](pyproject.toml)，[`requirements.txt`](requirements.txt) 同时安装可选的 SPE 预处理依赖。

```bash
git lfs install
git clone --branch efretro --single-branch https://github.com/kelio-byte/ef.git efretro
cd efretro
git lfs pull

conda create -n efretro python=3.10 -y
conda activate efretro
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python scripts/verify_assets.py
```

资产校验通过后，所有命令均从仓库根目录运行。CPU 也可运行小规模检查；正式完整训练和评测建议使用有足够显存及磁盘空间的 CUDA GPU。

## 运行

### 评测随仓库提供的 checkpoint

完整 dev1000（1,000 个反应、20,000 个增强输入）：

```bash
python scripts/evaluate.py --split dev1000 --output outputs/dev1000
```

完整 test（5,007 个反应）：

```bash
python scripts/evaluate.py --split test --output outputs/test
```

默认使用仓库中的 500K checkpoint，正式推理参数为 seed=42、100 步、`R=9`、`M=2`、batch size 32。`R`（`--n-runs`）是每个增强输入的独立完整轨迹数；`M`（`--n-children`）是每一步生成的子候选数。每个反应最终有 `20 × R` 条候选输入评分。输出目录包含 `predictions.txt`、`sampling_metadata.json` 和 `metrics.json`；脚本默认不覆盖已有预测。

例如，调整每步子候选数为 4（R 仍为 9）：

```bash
python scripts/evaluate.py --split dev1000 --n-runs 9 --n-children 4 --output outputs/dev1000_m4
```

### 从头训练并评测完整 test

```bash
python scripts/run_full.py --device cuda --n-runs 9 --n-children 2
```

该命令按 `configs/train.yaml` 从头训练 600K 步，再用新训练产生的 500K checkpoint 对完整 test 推理和评分；不会加载随仓库提供的 checkpoint。默认结果写入 `training_run/train_MM-DD/`，包括训练 checkpoint、日志和 `test_step500000/metrics.json`。目录已存在时脚本会停止而不覆盖；需要重跑时指定新的 `--run-name`。

如需单独训练或恢复训练，可使用：

```bash
# 从头训练
python scripts/train.py --config configs/train.yaml --device cuda --save_dir training_run

# 从仓库的 500K checkpoint 续训
python scripts/train.py --config configs/train.yaml --device cuda \
  --checkpoint saved_checkpoints/product_memory_m500_step500000.pt \
  --save_dir training_run
```

训练默认每 10K 步保存 checkpoint，保留最近 20 个；每 1K 步写入监控记录，验证和 TensorBoard 间隔见配置文件。续训会恢复 checkpoint 中的模型、优化器、调度器、数据位置和随机状态。若需要评估某个单独 checkpoint，可传给 `scripts/evaluate.py --checkpoint`。

### 独立生成候选

输入必须是本项目使用的 **SPE-M500 global token 序列**，不是原始 SMILES；每个 token 之间用空格分隔。

```bash
python scripts/sample.py \
  --products data/uspto50k_m500/dev1000/src.txt \
  --checkpoint saved_checkpoints/product_memory_m500_step500000.pt \
  --n-runs 9 --n-children 2 --output outputs/sample
```

独立评分还需要按相同反应顺序排列的目标文件；dev1000 和 test 的完整推理加评分建议直接使用 `scripts/evaluate.py`。

## 数据重建（可选）

训练会优先使用随仓库提供的对齐文件。文件缺失时可按正式最优对齐重建：

```bash
python scripts/precompute_alignments.py \
  --data_dir data/uspto50k_m500 --splits train val --num_workers 8
```

若已有完成 global 转换的 atom-tokenized 数据，可用 `scripts/preprocess_spe.py` 按 `data/SPE_ChEMBL.txt` 的前 500 次合并重建 SPE 数据。原始 USPTO 清洗及 global 转换不属于本项目的运行步骤；标准复现直接使用仓库中的冻结数据。不要替换正式词表后加载随仓库提供的 checkpoint。
