from functools import lru_cache
from rdkit import Chem, RDLogger
from edit_flows.data.global_smiles import inverse_global_align

RDLogger.DisableLog("rdApp.*")


@lru_cache(maxsize=8192)
def canonicalize_smiles_clear_map(smiles, return_max_frag=True):
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
    """Remove invalid and repeated canonical candidates, preserving order."""
    deduplicated = []
    seen = set()
    for candidate in candidates:
        if candidate[0] == "" or candidate in seen:
            continue
        seen.add(candidate)
        deduplicated.append(candidate)
    return deduplicated


def compute_rank(prediction, alpha=1.0, beam_size=None):
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
