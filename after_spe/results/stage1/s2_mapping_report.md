# Stage1 S2：图中心到 global R-SMILES / M500 token 的映射

日期：2026-08-24

## 结论

S2 通过。我们可以在不修改现有 tokenizer 的前提下，精确重放 SPE-M500，并把每个 fragment token 映射回它覆盖的 product atom。

正式检查使用 raw train 中固定 `seed=42` 随机抽取的 1,000 个 reaction；每个 reaction 保留完整 20 个 augmentation，共 20,000 个字符串视图。没有使用 dev 准确率，也没有运行模型推理。

| 检查项 | 结果 |
|---|---:|
| Reaction 成功映射 | **1,000 / 1,000** |
| Augmentation 视图成功映射 | **20,000 / 20,000** |
| SPE replay 与保存的 M500 token 逐 token 一致 | **20,000 / 20,000（100%）** |
| M500 token 总数 | 267,576 |
| token 自身直接包含 product atom | 93.655% |
| syntax-only token | 6.345% |
| syntax 按左右最近 atom 投影后可定位 | **100%** |
| 一个 token 同时覆盖多个 center component | 0% |

这说明后续 sampler 不需要对 fragment surface 做模糊字符串匹配：每个 token、SUB/DEL 位置和 INS boundary 都可以得到明确的 product 图位置。

## 如何实现

1. 使用与 SmilesPE `0.0.3` 完全相同的 atom-wise tokenizer；
2. 按 `SPE_ChEMBL.txt` 的前 500 条规则、相同优先级和相同重叠处理方式重放 merge；
3. atom token 初始携带 RDKit atom index，merge 后取两个子 token atom 集合的并集；
4. 对纯括号、点号等 syntax-only token，使用左右最近 atom 的并集；
5. INS 位置使用插入边界左右两侧最近 atom，而不是给新 fragment 的每个 token重复计数。

代码位于 `edit_flows/chem/spe_provenance.py`，并由 `tests/chem/test_spe_provenance.py` 验证 SmilesPE 重放、原子顺序、对称分子和 insertion boundary。

## 对称分子的处理

atom map 被移除后，对称 product 可能有多个同样合法的图同构。20,000 个视图中有 16,200 个存在多个同构，这并不是映射失败。代码保留所有对称等价位置的并集，避免 oracle center 取决于 RDKit 返回“第一个匹配”的偶然顺序。

只有 1 个 reaction 的 20 个视图达到预设 `max_isomorphisms=1024` 上限；它已在详情中标记。后续 RC1 应对此 reaction 做 fallback 或单独敏感性检查，不能假装已经穷举所有对称映射。

## 对 crosswalk fallback 的敏感性

抽样中 995 条使用严格立体匹配，5 条使用无立体 fallback；另有 5 条属于重复 canonical key 的多义 occurrence。排除所有 fallback 和多义项后剩 990 条，radius-1 结果几乎不变：

| 指标 | 全部 1,000 | 严格且唯一的 990 |
|---|---:|---:|
| radius-1 token 覆盖 | 28.737% | 28.664% |
| 已有-token 编辑 recall | 91.088% | 90.992% |
| insertion-run anchor recall | 96.343% | 96.324% |

所以局部性结论不是由这 10 条特殊 crosswalk 记录驱动。

## 产物

- `s2_mapping_examples.jsonl`：20 个 reaction 的可审计逐视图示例；
- `rc0_locality.json`：完整统计与固定抽样 indices；
- `results/after_spe_stage1/cache/rc0_reaction_details.jsonl`：1,000 个 reaction 的逐视图详情，SHA256 为 `3800da3f15d1cbed1f05d74ff23008358e7adfd317fe2d7c9594ac1021dc80e2`，因体积约 11 MB 保持 gitignored。

当前 Stage1 相关单测共 **17 passed**。
