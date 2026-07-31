# First-Step vs Trajectory 可视化对比分析

**生成时间**: 2026-07-28

**数据源**:
- `visualizations/first_step/visualization_20260728_152919.html` (20 样本 × 4 时间点 = 80 张表)
- `visualizations/trajectory/trajectory_20260728_152939.html` (19 样本, 每个 n_samples=3)

**测试对象**: 10 个产物 × 2 种 SMILES 排列 = 20 个样本 (从 5007 产物中随机选取, seed=42)

---

## 1. 总体统计

| 指标 | 数值 |
|------|------|
| 总样本数 | 20 (first_step) / 19 (trajectory, #22860 缺失) |
| Trajectory MATCH | 10/19 (52.6%) |
| Trajectory MISMATCH | 9/19 (47.4%) |
| First-step Center-Hit (t>0) 命中率 | 46/57 = 80.7% |
| First-step Full-Correct (t>0) 命中率 | 39/57 = 68.4% |

---

## 2. 逐产物详细分析

### 2.1 产物 204: `C(F)(F)F` 芳香杂环 — 苯甲基溴化

| 排列 | First-step (t>0) | Trajectory | 事件 |
|------|:---:|:---:|------|
| #4080 | CH:✓✓✓ FC:✓✓✓ | **MATCH** (1/3) | 4/4 ✓ — 逐步 INS C, C, O, Br |
| #4081 | CH:✓✓✓ FC:✓✓✓ | **MATCH** (2/3) | 4/4 ✓ — 逐步 INS C, Br, C, O |

**分析**: 两个排列表现一致且完美。模型在 t=0.1 就已经确定了正确的编辑位置和 token。Trajectory 中 4 个事件都在晚期 (t>0.58) 按正确顺序执行。这是一个"教科书级"的成功案例。

---

### 2.2 产物 712: `C=O → C-O` 还原 — 呋喃吡啶

| 排列 | First-step (t>0) | Trajectory | 事件 |
|------|:---:|:---:|------|
| #14240 | CH:✗✗✗ FC:✗✗✗ | **MISMATCH** (0/3) | 0/9 — 全部 INSERT 随机非化学 token (2, (, O) |
| #14241 | CH:✗✗✗ FC:✗✗✗ | **MISMATCH** (0/3) | 0/9 — 全部 INSERT 随机 token (2, (, c, O, Cl) |

**分析**: 这是最糟糕的案例。Oracle 要求 DELETE `=` — 单纯的删除操作。但模型从 first-step 开始就完全没找到正确位置 (Center-Hit 全 N)，Trajectory 中模型一直在 INSERT 随机 token 而不是 DELETE。事件发生在 t=0.49~0.99，且全部标记为错误。

**问题**: 模型在这个产物上对 DELETE 编辑类型几乎没有预测能力，倾向于 INSERT。

---

### 2.3 产物 839: `C#N → I·C([Cu])#N` — 氰基碘化/铜催化

| 排列 | First-step (t>0) | Trajectory | 事件 |
|------|:---:|:---:|------|
| #16780 | CH:✓✓✓ FC:✓✓✓ | **MISMATCH** (0/3) | 4/4 "correct" 但最终 MISMATCH |
| #16781 | CH:✓✗✗ FC:✓✗✗ | **MISMATCH** (0/3) | 2/4 correct, 使用了错误的 SUB token |

**分析**: **关键案例**。#16780 的 first-step 全部 Y，trajectory 4/4 事件标记为 "correct"，但最终 MISMATCH。这暴露了当前 event correctness 判定的问题：只检查编辑位置和类型 (INS/SUB/DEL)，不检查 token。模型在正确位置做了正确的编辑类型，但 INS 了错误的 token，最终 SMILES 不匹配。

---

### 2.4 产物 912: `C(O)=O → C(OC)=O` — 羧酸酯化

| 排列 | First-step (t>0) | Trajectory | 事件 |
|------|:---:|:---:|------|
| #18240 | CH:✓✓✓ FC:✓✓✓ | **MATCH** (3/3) | 1/1 ✓ — t=0.93 单次 INS `O` |
| #18241 | CH:✓✓✓ FC:✓✓✓ | **MATCH** (1/3) | 1/1 ✓ — t=0.97 单次 INS `O` |

**分析**: 最简单的编辑类型 — 单一位置插入单个 token。模型在 t=0.1 就完全确定了编辑，trajectory 只在很晚期 (t>0.9) 执行了这一次编辑。3/3 match on #18240 说明这种简单编辑非常稳定。

---

### 2.5 产物 1143: `C-C → C=C` 双键形成

| 排列 | First-step (t>0) | Trajectory | 事件 |
|------|:---:|:---:|------|
| #22860 | CH:✗✗✗ FC:✗✗✗ | **缺失** | — |
| #22861 | CH:✓✓✓ FC:✓✓✓ | **MISMATCH** (0/3) | 3/8 correct, 涉及复杂 SUB+DEL 协调 |

**分析**: #22860 两个排列对同一产物但结果截然不同 — #22861 的 first-step 全对但 trajectory 失败 (3/8 events correct)。编辑涉及将单键 `C-C` 变为双键 `C=C`，同时需要 SUB (将 `C` 替换为 `/` 和 `\`) 和 DEL，属于复杂的多步协调编辑。

---

### 2.6 产物 1828: `[n+][O-] → n + OOC(=O)Ar` — N-氧化到 N-烷基化

| 排列 | First-step (t>0) | Trajectory | 事件 |
|------|:---:|:---:|------|
| #36560 | CH:✓✓✗ FC:✓✓✗ | **MISMATCH** (0/3) | 3/18 correct, 大量错误 INSERT |
| #36561 | CH:✓✓✓ FC:✓✓✓ | **MISMATCH** (0/3) | 4/19 correct, 大量错误 INSERT+SUB |

**分析**: first-step 表现不错 (尤其是 #36561 全 Y)，但 trajectory 中事件数极多 (18-19 个)，且大部分是错误的 INSERT。这说明模型知道"要编辑什么位置"，但在执行时反复尝试错误 token，不断产生新事件来修正。这是典型的 **探索性编辑行为** — 模型在采样中"试来试去"但始终没碰对。

---

### 2.7 产物 2006: 环己酮衍生物 — 脱胺/羰基化

| 排列 | First-step (t>0) | Trajectory | 事件 |
|------|:---:|:---:|------|
| #40120 | CH:✓✓✓ FC:✗✗✗ | **MISMATCH** (0/3) | 0/1 — t=0.14 错误 DELETE `=` |
| #40121 | CH:✗✗✗ FC:✗✗✗ | **MISMATCH** (0/3) | 3/6 correct, 但 token 错误 |

**分析**: 典型的 token-level 失败。Center-Hit 命中但 Full-Correct 全 N — 模型找到了位置但选了错误的 token。Trajectory 中 #40120 只在 t=0.14 有一个错误的 DELETE 事件，之后就再也没有编辑。说明模型在第一步犯错后"放弃了"。

---

### 2.8 产物 2253: 酰胺键断裂 — `N → O`

| 排列 | First-step (t>0) | Trajectory | 事件 |
|------|:---:|:---:|------|
| #45060 | CH:✓✓✓ FC:✓✓✓ | **MATCH** (1/3) | 2/2 ✓ — t=0.09 INS `)`, t=0.89 INS `O` |
| #45061 | CH:✓✓✓ FC:✓✓✓ | **MATCH** (2/3) | 4/4 ✓ — 逐步 INS C, (, (, O |

**分析**: 完美案例。两个排列 first-step 全对，trajectory 全部事件正确。

---

### 2.9 产物 4467: `C=N-O → C=O·N-O` — 肟重排

| 排列 | First-step (t>0) | Trajectory | 事件 |
|------|:---:|:---:|------|
| #89340 | CH:✓✓✓ FC:✓✓✓ | **MATCH** (3/3) | 2/2 ✓ — t=0.31 INS `=`, t=0.49 INS `O` |
| #89341 | CH:✓✓✓ FC:✓✗✗ | **MATCH** (2/3) | 5/5 ✓ — 复杂的多步 SUB+INS |

**分析**: #89341 很有意思 — first-step Full-Correct 在 t=0.3,0.5 为 N，但 trajectory 5/5 events correct 且最终 MATCH。说明 first-step 的单位置预测不能完全反映多步采样的能力。模型在 t=0.3,0.5 时可能选了"第二好"的位置，但通过多步采样逐步修正。

---

### 2.10 产物 4837: 苯甲酮衍生物 — 酰氯形成

| 排列 | First-step (t>0) | Trajectory | 事件 |
|------|:---:|:---:|------|
| #96740 | CH:✗✗✗ FC:✗✗✗ | **MATCH** (1/3) | 2/2 ✓ — t=0.89 INS `)`, t=0.98 INS `Cl` |
| #96741 | CH:✗✗✗ FC:✗✗✗ | **MATCH** (1/3) | 2/2 ✓ — t=0.59 INS `)`, t=0.71 INS `)` |

**分析**: **最具启发性的案例**。First-step 在所有 t 值 (0.1, 0.3, 0.5) 都显示 Center-Hit=N 和 Full-Correct=N，但 trajectory 最终 MATCH。原因：编辑事件发生在 t=0.59~0.98，而我们 first-step 的 time_grid 只到 t=0.5。模型的编辑信号在 t>0.5 之后才显现。

**这暴露了 time_grid 的局限性** — `"0,0.1,0.3,0.5"` 覆盖不到晚期编辑。对于这种"晚期编辑"，first-step 分析会给出错误的负面信号。

---

## 3. 关键发现

### 3.1 First-step Center-Hit 与 Trajectory 结果的关系

| First-step (t>0) | Trajectory MATCH | Trajectory MISMATCH |
|:---|:---:|:---:|
| Center-Hit 全 Y | #4080, #4081, #18240, #18241, #45060, #45061, #89340, #89341 (8) | #16780, #22861, #36561, #40120 (4) |
| Center-Hit 部分 Y | — | #16781, #36560 (2) |
| Center-Hit 全 N | #96740, #96741 (2) | #14240, #14241, #40121 (3) |

**结论**: Center-Hit 全 Y 对 MATCH 的预测准确率为 8/12 = 66.7%。但有两例 Center-Hit 全 N 却 MATCH (#96740, #96741)，原因是编辑发生在 t>0.5。

### 3.2 Event Correctness 的盲区

Event correctness 只检查**位置+类型**，不检查 token。这导致：
- #16780: 4/4 events "correct" 但最终 MISMATCH (token 错了)
- #89341: 5/5 events correct 且 MATCH (token 对了)

**建议**: 在 event correctness 中增加 token 级别检查。

### 3.3 编辑类型偏好

| Oracle 主要编辑类型 | 成功案例 | 失败案例 |
|:---|:---|:---|
| 纯 INS (插入) | #18240, #18241, #45060, #45061, #89340, #96740, #96741 | #16780*, #36560, #36561 |
| 纯 DEL (删除) | — | #14240, #14241 |
| INS+SUB 混合 | #4080, #4081, #45061, #89341 | #16781, #22861 |
| DEL+SUB 混合 | — | #40120, #40121 |

**结论**: 模型在纯 DELETE 操作上表现最差 (#14240/#14241 全失败)。在纯 INSERT 上表现最好。这可能是因为 DELETE 需要模型精确判断"去掉什么"，而 INSERT 只需要判断"加什么"。

### 3.4 编辑事件的时间分布

通过 trajectory 数据观察到的规律：
- **早期编辑 (t<0.3)**: 多为错误编辑，模型在"试探" (#36560 #1 t=0.11, #40120 #1 t=0.14)
- **中期编辑 (t=0.3~0.7)**: 混合正确和错误
- **晚期编辑 (t>0.7)**: 正确率更高，尤其是简单的单 token INS (#18240 t=0.93, #18241 t=0.97)

**结论**: 成功案例的编辑集中在 t>0.5，尤其是单一 token 插入。这说明模型需要积累足够的"信心" (高 kappa 值) 才做出正确编辑。

### 3.5 SMILES 排列敏感性

同一产物的两个不同 SMILES 排列 (augmentation) 在大部分情况下表现一致：
- 产物 204, 712, 912, 1828, 2253, 4837: 两个排列结果相同
- 产物 839, 1143, 2006, 4467: 两个排列有差异

其中 #22860 (全 N) vs #22861 (全 Y) 是同一个产物 (1143)，但排列不同导致完全相反的 first-step 结果。这提示 **SMILES 的书写顺序确实影响模型的单步预测**。

---

## 4. 建议

1. **扩展 time_grid**: 当前 `"0,0.1,0.3,0.5"` 无法覆盖晚期编辑 (#96740 的编辑在 t=0.89~0.98)。建议改为 `"0,0.1,0.3,0.5,0.7,0.9"`。

2. **增加 event token 正确性检查**: 在 trajectory 的 event correctness 中增加 token 匹配 (#16780 的 4/4 "correct" 却 MISMATCH 说明当前判定不够)。

3. **关注 DELETE 操作**: 模型对纯 DELETE 的预测能力明显弱于 INSERT，可考虑在训练中增加 DELETE 的权重或专门分析。

4. **晚期采样策略**: 简单编辑 (单 token INS) 集中在 t>0.9，考虑是否可以增加后期步数的密度。
