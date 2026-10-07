# Stage1 S1：反应中心标签与数据对应结果

日期：2026-08-24

## 结论

S1 已通过：raw atom-mapped reaction 可以完整对应到当前 global R-SMILES 训练/验证数据，且不需要按行 `zip` 或截断。

- train 的 40,003 个 processed reaction block 全部找到 raw reaction；raw 多出的 5 条明确保留为 raw-only，不进入后续映射。
- val 的 5,001 个 processed block 全部匹配。
- 99% 以上使用保留立体化学的严格 key 匹配；剩余约 1% 是历史 global-alignment 写法造成的 `@/@@` 差异，只对严格匹配失败的记录使用无立体 fallback，并在 crosswalk 中逐条标记。
- 所有 45,009 条 raw train/val reaction 都能被 RDKit 解析，没有 duplicate atom-map failure。

因此可以继续 S2：把图中心投影到每个 global R-SMILES augmentation 和 M500 token。

## 1. Crosswalk

| Split | Raw reactions | Processed blocks | 严格立体匹配 | 无立体 fallback | 最终匹配 | Processed 未匹配 | Raw-only |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 40,008 | 40,003 | 39,650 | 353 | **40,003** | **0** | 5 |
| val | 5,001 | 5,001 | 4,951 | 50 | **5,001** | **0** | 0 |

严格匹配失败并不等于化学反应不同。检查显示，train 的 353 条和 val 的 50 条在移除 atom/bond stereo 后都能匹配；它们保留 `match_method=achiral_fallback`，后续会单独报告结果是否依赖这批记录。

Canonical reaction 本身有重复：train crosswalk 有 405 个 occurrence、val 有 12 个 occurrence 落在重复 key 组中。脚本按原始 occurrence 顺序对应，同时设置 `key_is_ambiguous=true`。S2/RC0 会报告包含和排除这些多义项的结果，不能把它们伪装成唯一映射。

train 中多出的 5 条 raw reaction 为：

| Raw index | Reaction ID | 结构特征 |
|---:|---|---|
| 1446 | US20100279990A1 | product 只有 1 个 atom；reactants 21 atoms，其中 20 个无 map |
| 20320 | US08470816B2 | product 只有 1 个 atom；reactants 13 atoms，其中 12 个无 map |
| 23859 | US04198522 | product 只有 1 个 atom；reactants 4 atoms，其中 3 个无 map |
| 25749 | US04772600 | product 只有 1 个 atom；reactants 21 atoms，其中 20 个无 map |
| 34872 | US20070259879A1 | 3-atom product；reactants 含 1 个无 map atom |

它们确实不存在于 processed train，而不是 crosswalk 漏配。仓库没有保留最初生成 global 数据集的完整过滤日志，所以不能进一步声称具体是哪条旧过滤规则；后续只记录为“历史 processed 数据未收录”，不猜测原因。

## 2. 图级中心分布

中心由以下可审计图变化组成：product-only/reactant-only bond、键属性变化、保留原子的电荷/氢/手性/芳香性变化、product-only atom，以及 reactant-only 新片段连接到保留 product atom 的 attachment anchor。RDKit sanitize 后把 aromatic bond 统一为 `AROMATIC` 再比较，避免单纯 Kekulé 写法制造中心。

| Split | 1 个中心 component | 2 个 | 3 个 | >3 个 | 多中心合计 |
|---|---:|---:|---:|---:|---:|
| train | 38,732 (96.81%) | 1,096 | 171 | 9 | 1,276 (3.19%) |
| val | 4,829 (96.56%) | 151 | 21 | 0 | 172 (3.44%) |

train 中只有 9/40,008（0.022%）超过三个 component，因此 RC1 使用“最多三个中心假设”不会影响绝大多数反应，但这 9 条仍保留在标签中，没有预先截断。

所有有效 raw reaction 至少检测到一个中心。train 中 37,194 条、val 中 4,653 条含 reactant-side map=0 atom；这与 attachment 类反应很常见一致。代码不会把 map=0 atom当成可预测的 product center，只把与它相连的保留 product atom标为 attachment anchor。

## 3. 数据泄漏与重复检查

raw train 与 raw val 有 45 个严格 canonical reaction key 重复；忽略 stereo 后为 55 个。这是历史数据 split 自身的重复，不是本脚本造成的。后续若训练 product-only center predictor，内部验证必须按 canonical reaction 分组切分，避免同一反应落入内部 train/validation 两侧；正式 dev 仍不得参与训练。

## 4. 产物与复现

代码：

- `edit_flows/chem/reaction_center.py`
- `scripts/build_reaction_center_labels.py`
- `tests/chem/test_reaction_center.py`

机器可读小报告：`s1_crosswalk_report.json`。

大体积逐反应文件位于 gitignored 的 `results/after_spe_stage1/cache/`：

- `reaction_centers_{train,val}.jsonl`
- `processed_keys_{train,val}.jsonl`
- `raw_to_processed_{train,val}.jsonl`

其路径和 SHA256 已写入 `s1_crosswalk_report.json`。当前化学/解析单测为 **11 passed**。
