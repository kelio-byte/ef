"""用途：规范化 SMILES、合并重复候选并计算目标排名。
输入：候选 SMILES 和目标 SMILES。
输出：候选排名及有效性统计。
"""

from functools import lru_cache
from rdkit import Chem, RDLogger
from edit_flows.data.global_smiles import inverse_global_align

RDLogger.DisableLog("rdApp.*")


@lru_cache(maxsize=8192)
def canonicalize_smiles_clear_map(smiles, return_max_frag=True):
    """作用：解析并规范化 SMILES，同时清除原子映射。输入：global SMILES 和是否返回最大组分。输出：规范 SMILES；无法解析时为空字符串。"""
    smiles = inverse_global_align(smiles)
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is not None:
        [
            atom.ClearProp("molAtomMapNumber")
            for atom in mol.GetAtoms()
            if atom.HasProp("molAtomMapNumber")
        ]
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            if return_max_frag:
                return ("", "")
            else:
                return ""
        if return_max_frag:
            sub_smi = smi.split(".")
            sub_mol = [Chem.MolFromSmiles(smiles, sanitize=True) for smiles in sub_smi]
            sub_mol_size = [
                (sub_smi[i], len(m.GetAtoms()))
                for (i, m) in enumerate(sub_mol)
                if m is not None
            ]
            if len(sub_mol_size) > 0:
                return (
                    smi,
                    canonicalize_smiles_clear_map(
                        sorted(sub_mol_size, key=lambda x: x[1], reverse=True)[0][0],
                        return_max_frag=False,
                    ),
                )
            else:
                return (smi, "")
        else:
            return smi
    elif return_max_frag:
        return ("", "")
    else:
        return ""


def _deduplicate_valid(candidates):
    """作用：按原顺序去除无效或重复候选。输入：规范化候选列表。输出：去重后的有效列表。"""
    deduplicated = []
    seen = set()
    for candidate in candidates:
        if candidate[0] == "" or candidate in seen:
            continue
        seen.add(candidate)
        deduplicated.append(candidate)
    return deduplicated


def compute_rank(prediction, alpha=1.0, beam_size=None):
    """作用：聚合增强视图中的候选排名。输入：各视图预测、alpha 和 beam 大小。输出：候选得分字典及各名次无效率。"""
    if not prediction or not prediction[0]:
        raise ValueError("prediction must contain at least one candidate")
    if beam_size is None:
        beam_size = len(prediction[0])
    valid_score = [
        [k for k in range(len(prediction[j]))] for j in range(len(prediction))
    ]
    invalid_rates = [0 for k in range(len(prediction[0]))]
    rank = {}
    highest = {}
    for j in range(len(prediction)):
        for k in range(len(prediction[j])):
            if prediction[j][k][0] == "":
                valid_score[j][k] = beam_size + 1
                invalid_rates[k] += 1
        de_error = [
            i[0]
            for i in sorted(
                list(zip(prediction[j], valid_score[j])), key=lambda x: x[1]
            )
            if i[0][0] != ""
        ]
        unique_prediction = _deduplicate_valid(de_error)
        for k, data in enumerate(unique_prediction):
            if data in rank:
                rank[data] += 1 / (alpha * k + 1)
            else:
                rank[data] = 1 / (alpha * k + 1)
            if data in highest:
                highest[data] = min(k, highest[data])
            else:
                highest[data] = k
    for key in rank.keys():
        rank[key] += highest[key] * -100000000.0
    return (rank, invalid_rates)
