# First-Step Visualization 实现与使用说明

## 1. 概述

`scripts/visualize_first_step.py` 将模型第一步预测与 oracle 并排可视化为 HTML 表格，用于诊断模型在初始阶段（`x_t = x_0`）的编辑预测质量。

核心思路：列对齐 x₀ 的每个 token 位置，Oracle 和 Model 两套预测分别展示，速率用颜色编码，直观对比"模型预测"与"理论最优"的差异。

## 2. 表格结构

每个样例、每个 t 值生成一张表格，纵向排列如下：

| 行 | 标签 | 内容 |
|----|------|------|
| 1 | `x₀` | product token 序列，oracle 正位置通过上边框颜色标记 |
| 2 | `ORACLE` | 节标题 |
| 3 | `λ_ins` | oracle insert 速率（整数 raw count），背景着色 |
| 4 | `ins top5` | oracle insert top-5 + other token 分布，白底 |
| 5 | `λ_sub` | oracle substitute 速率，背景着色 |
| 6 | `sub top5` | oracle substitute top-5 + other token 分布，白底 |
| 7 | `λ_del` | oracle delete 速率，背景着色 |
| 8 | `MODEL` | 节标题 |
| 9–13 | （同上格式） | 模型预测，与 oracle 区域格式完全一致 |

### x₀ 行 oracle 编辑类型标记

| 上边框颜色 | 含义 |
|-----------|------|
| 蓝色 | oracle 需要 INSERT |
| 橙色 | oracle 需要 SUBSTITUTE |
| 红色 | oracle 需要 DELETE |
| 无 | 该位置不需要编辑 |
| 灰色背景 | BOS 位置（不参与编辑） |

### 速率颜色编码

速率通过 cell 背景色表示，从白到深红渐变：

| 速率范围 (log10) | 颜色 |
|------------------|------|
| ~ -8 | 白色 |
| -8 ~ -5 | 浅粉 |
| -5 ~ -2 | 粉色 |
| -2 ~ 0 | 红色 |
| 0 ~ 2+ | 深红 |

ORACLE 正位置通常显示为深红（整数计数 1–3），其余位置接近白色。

### Token 分布

每个 INS/SUB 的 token 行显示 top-5 + other（汇总剩余概率），格式为：
```
C  0.450
N  0.300
O  0.150
c  0.050
+  0.050
```

## 3. 速率说明

可视化中展示的是**原始速率**（raw rates），不包含时间系数 `k(t)`：

- **Oracle**：Z-space 编辑需求量除以 `k(t)` 后的整数计数（如 1.000, 2.000），便于跨 t 比较
- **Model**：模型原始输出 `v'`（未经过 `apply_rate_parameterization` 乘以 `k(t)`）

这保证 oracle 正位置速率是干净的整数值，且 oracle 与 model 在同一尺度下可比。

## 4. 脚本用法

### 4.1 参数

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--checkpoint` | ✓ | 模型 checkpoint 路径 | — |
| `--products_file` | ✓ | product SMILES 文件 | — |
| `--targets_file` | ✓ | target SMILES 文件 | — |
| `--output_dir` | ✓ | 输出目录 | — |
| `--vocab_file` | | vocab 文件路径 | 从 checkpoint config 推断 |
| `--scheduler` | | 采样 scheduler | checkpoint config 中的值 |
| `--time_grid` | | 逗号分隔的时间点 | `0,0.1` |
| `--deduplicate` | | 每 N 行取一行（适配 aug20） | 0 |
| `--max_lines` | | 限制读取行数 | 0（全部） |
| `--n_examples` | | 随机选取样例数 | 5 |
| `--example_ids` | | 指定具体样例索引（逗号分隔） | — |
| `--seed` | | 随机种子 | 42 |
| `--device` | | 计算设备 | `cpu` |

### 4.2 使用示例

```bash
PYTHONPATH=. python scripts/visualize_first_step.py \
    --checkpoint "checkpoints/USPTO_50K_PtoR_aug20_#global#/2026-06-08_17-20-39/checkpoint_step1680000.pt" \
    --products_file "analysis_subsets/USPTO_50K_PtoR_aug20_#global#/test_dedup_seed42_1000/src-test.txt" \
    --targets_file "analysis_subsets/USPTO_50K_PtoR_aug20_#global#/test_dedup_seed42_1000/tgt-test.txt" \
    --output_dir "analysis_outputs/first_step/test_dedup_seed42_1000/visualizations" \
    --time_grid "0,0.1" \
    --scheduler linear \
    --n_examples 10 \
    --device cuda
```

查看指定样例：

```bash
PYTHONPATH=. python scripts/visualize_first_step.py \
    ... \
    --example_ids 0,5,23,100 \
    --time_grid 0,0.05,0.1
```

### 4.3 输出

```
{output_dir}/
└── visualization.html    # 包含所有样例的独立 HTML 文件
```

HTML 文件包含：
- 页面顶部图例和导航（可点击跳转到各样例）
- 每个样例的完整表格（可横向滚动）
- 内嵌 CSS，无需外部依赖，浏览器直接打开即可

## 5. 与已有实验链路的关系

| 脚本 | 用途 |
|------|------|
| `first_step_forward_analysis.py` | 批量统计指标（Center Hit@k, Type Acc 等） |
| `first_event_impact_analysis.py` | 第一次真实编辑相关性与干预分析 |
| **`visualize_first_step.py`** | **逐样本可视化诊断，定性查看模型预测与 oracle 的差异** |

可视化脚本是实验 1（静态初始预测）的补充工具，不覆盖已有结果文件，输出写入独立目录。

## 6. 典型分析流程

1. 先跑 `first_step_forward_analysis.py` 拿到聚合指标
2. 对指标异常的样例（如 Center Hit 但 Full Correct 为 False），用 `--example_ids` 指定可视化
3. 在 HTML 中逐列对比：模型在 oracle 正位置上的 token 预测是否正确、在非正位置是否有假阳性速率
