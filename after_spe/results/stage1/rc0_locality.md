# Stage1 RC0：反应中心是否覆盖当前 M500 编辑入口

日期：2026-08-24

## 一句话结论

**RC0 通过，值得进入 RC1 true-center upper bound。** radius-1 中心邻域只占约 28.7% 的 M500 token，却覆盖 91.1% 的已有-token 编辑和 96.3% 的插入入口，说明它提供了稀疏且有信息量的位置先验。

这仍不是准确率收益证明：只有 B0/B1/B2 的同预算 GPU 推理能回答它是否提升 Top-k、Oracle 或 invalid。

## 实验设置

- 数据：改进后 global R-SMILES + SPE-M500 的 train aligned 数据；
- 抽样：固定 seed=42 的 1,000 个原始 reaction；
- augmentation：每个 reaction 的全部 20 个写法，共 20,000 个视图；
- 统计单位：reaction；micro 指标用于显示所有编辑事件覆盖，macro 指标先在 reaction 内聚合再平均；
- insertion：连续 `<GAP> → token` 合并为一个 insertion run，只评估它在 product 上的入口；
- radius：0 为中心本身，1/2 为 product 图上一跳/两跳邻域；
- 不运行模型、不查看 dev Top-k、不训练中心预测器。

## 核心数据

| 半径 | 中心邻域占全部 token | 已有-token 编辑 micro recall | 已有-token macro recall | INS-run micro recall | INS-run macro recall |
|---:|---:|---:|---:|---:|---:|
| 0 | 14.055% | 83.400% | 88.348% | 91.885% | 90.311% |
| **1** | **28.737%** | **91.088%** | **93.752%** | **96.343%** | **94.681%** |
| 2 | 41.904% | 96.722% | 97.751% | 98.647% | 97.817% |

radius-2 的覆盖更高，但 token 范围扩到 41.9%，先验明显变宽。预注册 sampler 的主方案仍使用连续权重：中心 `1.0`、radius-1 `0.5`、其他 `0`，而不是把 radius-2 全部视为中心。

## 20 个 augmentation 是否稳定

“任一 augmentation 完全覆盖”和“全部 augmentation 都完全覆盖”可以区分偶然写法与稳定定位：

| 半径 | 编辑类型 | 任一写法完全覆盖 | 20 个写法全部覆盖 |
|---:|---|---:|---:|
| 0 | 已有-token | 96.57% | 47.02% |
| 0 | INS run | 97.67% | 59.23% |
| **1** | **已有-token** | **98.99%** | **64.58%** |
| **1** | **INS run** | **98.07%** | **73.73%** |
| 2 | 已有-token | 99.70% | 81.84% |
| 2 | INS run | 99.39% | 87.63% |

结论不是“每一种 R-SMILES 写法都完美定位”。更准确的说法是：中心对绝大多数 reaction 都有很强的入口覆盖，但字符串遍历顺序仍会影响个别 augmentation 的 edit alignment。RC1 因此必须保留 20 augmentation 聚合和普通 Euler fallback。

## 事件规模与质量检查

- existing-token 事件：26,572，其中 SUB 26,191、DEL 381；
- insertion runs：19,581；
- 1,000/1,000 reaction、20,000/20,000 views 成功；
- SPE tokenization 逐 token 复现率 100%；
- 964 个 reaction 是单中心、30 个双中心、6 个三中心；本次没有 `>3`；
- 排除 5 个无立体 crosswalk fallback 和 5 个多义 key 后，radius-1 recall 变化小于 0.1 pp。

## 决策

RC0 满足两个结构条件：

1. **足够稀疏：**radius-1 只覆盖 28.7% token，没有退化为“几乎整条序列”；
2. **覆盖关键入口：**对 INS run 和已有-token 编辑均超过 90% micro recall。

因此下一步不是训练反应中心预测器，而是先做 RC1：在真实中心已知的 oracle 条件下，只重新分配每条轨迹第一次非空 Euler step 的位置 rate，并保持每种编辑 mode 的总 hazard 和 token completion 分布不变。

RC1 若不能优于普通 Euler 且不能优于同 product 的伪中心，立即停止中心 sampler；即使 RC0 的静态 recall 很高，也不训练 product-only predictor。
