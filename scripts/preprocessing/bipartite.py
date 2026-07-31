import json
from rdkit import Chem
from rdkit.Chem import Draw

import pickle

from tqdm import trange

def get_feature(atom):
    self_smiles = atom.GetSmarts().replace('@', '')
    neighbors = []
    for bond in atom.GetBonds():
        bond_type = int(bond.GetBondType())
        neighbor = bond.GetOtherAtom(atom)
        if neighbor.GetAtomMapNum() == 0:
            neighbors.append((0, neighbor.GetAtomicNum(), bond_type))
        else:
            neighbors.append((neighbor.GetAtomMapNum(), bond_type))
    neighbors.sort()
    neighbors.append(self_smiles)
    return neighbors


def get_bipartite_links(mols):
    feature_dict = dict()
    active = set()
    active.add(0)
    for mol in mols:
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            atom_feature = get_feature(atom)
            if map_num == 0:
                continue
            if any(map(lambda i: i.GetAtomMapNum() == 0, atom.GetNeighbors())):
                active.add(map_num)
            if map_num in feature_dict:
                if atom_feature != feature_dict[map_num]:
                    active.add(map_num)
            else:
                feature_dict[map_num] = atom_feature

    # print(active)
    flag = True
    while flag:
        flag = False
        for mol in mols:
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() not in active:
                    continue
                for bond in atom.GetBonds():
                    neighbor = bond.GetOtherAtom(atom)
                    if bond.GetBondType() != Chem.BondType.SINGLE and neighbor.GetAtomMapNum() not in active:
                        active.add(neighbor.GetAtomMapNum())
                        flag = True

    # print(active)
    for atom in mols[0].GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num in active:
            continue
        if atom.GetDegree() == 0:
            continue
        if all(map(lambda i: (i.GetAtomMapNum() in active), atom.GetNeighbors())):
            active.add(map_num)

    # print(active)
    links = []
    for atom in mols[0].GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num not in active:
            continue
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomMapNum() not in active:
                links.append((map_num, neighbor.GetAtomMapNum()))

    return links, active


def add_atom(mol, atomic_num: int, isotope: int, atom_map_num: int=0):
    idx = mol.AddAtom(Chem.Atom(atomic_num))
    atom = mol.GetAtomWithIdx(idx)
    atom.SetIsotope(isotope)
    atom.SetAtomMapNum(atom_map_num)
    return idx


def flip_bond_direction(bond_dir):
    if bond_dir == Chem.BondDir.ENDUPRIGHT:
        return Chem.BondDir.ENDDOWNRIGHT
    if bond_dir == Chem.BondDir.ENDDOWNRIGHT:
        return Chem.BondDir.ENDUPRIGHT
    return bond_dir

def remove_bond(mol, idx_s: int, idx_t: int):
    atom_s = mol.GetAtomWithIdx(idx_s)
    atom_t = mol.GetAtomWithIdx(idx_t)
    if atom_s.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
        neighbor_list = [n.GetIdx() for n in atom_s.GetNeighbors()]
        if (len(neighbor_list) + neighbor_list.index(idx_t)) % 2 == 0:
            atom_s.InvertChirality()
    if atom_t.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
        neighbor_list = [n.GetIdx() for n in atom_t.GetNeighbors()]
        if (len(neighbor_list) + neighbor_list.index(idx_s)) % 2 == 0:
            atom_t.InvertChirality()
    bond = mol.GetBondBetweenAtoms(idx_s, idx_t)
    bond_dir = bond.GetBondDir()
    if bond.GetBeginAtomIdx() != idx_s:
        bond_dir = flip_bond_direction(bond_dir)
    mol.RemoveBond(idx_s, idx_t)
    return bond_dir
    

def get_a_part(mol, links, active, get_active=True):
    mol = Chem.RWMol(mol)
    idx_map = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
    old_atoms_len = len(mol.GetAtoms())
    for i, (s, t) in enumerate(links):
        if not get_active:
            s, t = t, s
        bond_dir = remove_bond(mol, idx_map[s], idx_map[t])
        new_idx = add_atom(mol, 85, i+1, t)
        bond_idx = mol.AddBond(idx_map[s], new_idx, Chem.BondType.SINGLE) - 1
        mol.GetBondWithIdx(bond_idx).SetBondDir(bond_dir)
    mol.BeginBatchEdit()
    for atom in mol.GetAtoms():
        need_remove = atom.GetAtomMapNum() in active
        if get_active:
            need_remove = not need_remove
        if need_remove and atom.GetIdx()<old_atoms_len:
            mol.RemoveAtom(atom.GetIdx())
    mol.CommitBatchEdit()
    mol.UpdatePropertyCache(strict=False)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    return mol

def get_full(mol, links, active):
    mol = Chem.RWMol(mol)
    idx_map = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
    old_atoms_len = len(mol.GetAtoms())
    for i, (s, t) in enumerate(links):
        bond_dir = remove_bond(mol, idx_map[s], idx_map[t])
        new_idx = add_atom(mol, 84, i+1, 0)
        s_bond_idx = mol.AddBond(idx_map[s], new_idx, Chem.BondType.SINGLE) - 1
        t_bond_idx = mol.AddBond(idx_map[t], new_idx, Chem.BondType.SINGLE) - 1
        mol.GetBondWithIdx(s_bond_idx).SetBondDir(bond_dir)
        bond_dir = flip_bond_direction(bond_dir)
        mol.GetBondWithIdx(t_bond_idx).SetBondDir(bond_dir)
    mol.UpdatePropertyCache(strict=False)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    return mol

def clear_map_num(smiles: str, get_num=False):
    mol = Chem.MolFromSmiles(smiles)
    num_list = []
    for atom in mol.GetAtoms():
        num_list.append(atom.GetAtomMapNum())
        atom.SetAtomMapNum(0)
    
    smiles = Chem.MolToSmiles(mol, canonical=False, allHsExplicit=False)
    if get_num:
        return smiles, num_list
    return smiles
def sort_by_first_map_num(smiles: str):
    return '.'.join(
        sorted(
            smiles.split('.'), 
            key = lambda mol_smiles: Chem.MolFromSmiles(mol_smiles).GetAtomWithIdx(0).GetAtomMapNum()
        )
    )
def atom_align_for_full(S: str, T: str):
    S = Chem.MolFromSmiles(S)
    T = Chem.MolFromSmiles(T)
    S_atom_map_nums = {atom.GetAtomMapNum(): atom.GetIdx()+1 for atom in S.GetAtoms()}
    S_atom_map_nums.pop(0, None)
    max_num = max(S_atom_map_nums.keys())

    for atom in T.GetAtoms():
        if atom.GetAtomicNum() == 84:
            map_num = max(S_atom_map_nums[i.GetAtomMapNum()] for i in atom.GetNeighbors())
            atom.SetAtomMapNum(map_num)

    for atom in T.GetAtoms():
        if atom.GetAtomicNum() == 84:
            continue
        map_num = atom.GetAtomMapNum()
        if map_num != 0:
            atom.SetAtomMapNum(S_atom_map_nums[map_num])
        else:
            atom.SetAtomMapNum(max_num+1)

    T = Chem.MolToSmiles(T, canonical=True, allHsExplicit=True) # , ignoreAtomMapNumbers=False

    return clear_map_num(sort_by_first_map_num(T))

def fix_map(mols):
    intersection = None
    map_sets = []
    for mol in mols:
        map_list = [atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum()!=0]
        map_set = set(map_list)
        if len(map_list) != len(map_set):
            raise TypeError
        map_sets.append(map_set)
        if intersection is None:
            intersection = {*map_set}
        else:
            intersection.intersection_update(map_set)
    for mol, map_set in zip(mols, map_sets):
        map_set = map_set - intersection
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() in map_set:
                atom.SetAtomMapNum(0)

def process_reaction(smiles):
    mols = list(map(lambda x: Chem.MolFromSmiles(x), smiles))
    fix_map(mols)
    links, active = get_bipartite_links(mols)
    mols_res = (
        [get_a_part(mols[0], links, active, False)] 
        + [get_a_part(mol, links, active, True) for mol in mols]
    )
    smiles_res = list(map(lambda x: Chem.MolToSmiles(x), mols_res))
    return smiles_res

def process_reaction_full(smiles):
    mols = list(map(lambda x: Chem.MolFromSmiles(x), smiles))
    fix_map(mols)
    links, active = get_bipartite_links(mols)
    mols_res = (
        [get_full(mols[0], links, active)] 
        + [get_a_part(mol, links, active, True) for mol in mols]
    )
    smiles_res = list(map(lambda x: Chem.MolToSmiles(x), mols_res))
    return smiles_res

def flip_mol_direction(mol, atom, visited=None, last_idx=None):
    if visited is None:
        visited = set()
    idx = atom.GetIdx()
    if idx in visited:
        return
    visited.add(idx)
    for bond in atom.GetBonds():
        if bond.GetOtherAtom(atom).GetIdx() == last_idx:
            continue
        if bond.GetBondType() == Chem.BondType.DOUBLE or bond.GetBondDir() != Chem.BondDir.NONE:
            bond.SetBondDir(flip_bond_direction(bond.GetBondDir()))
            flip_mol_direction(mol, bond.GetOtherAtom(atom), visited, idx)

def glue_two_parts(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RWMol(mol)
    link_dict = dict()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum()!=85 or atom.GetIsotope()>100 or atom.GetIsotope()==0:
            continue
        link_id = atom.GetIsotope()
        if link_id not in link_dict:
            link_dict[link_id] = [atom.GetIdx()]
        else:
            link_dict[link_id].append(atom.GetIdx())
    for link in link_dict.values():
        assert len(link)==2
        s, t = link
        s_next = mol.GetAtomWithIdx(s).GetNeighbors()
        assert len(s_next)==1
        s_next = s_next[0].GetIdx()
        t_next = mol.GetAtomWithIdx(t).GetNeighbors()
        assert len(t_next)==1
        t_next = t_next[0].GetIdx()

        dir_from_s = remove_bond(mol, s, s_next)
        dir_from_t = remove_bond(mol, t, t_next)
        if dir_from_s != Chem.BondDir.NONE and dir_from_t != Chem.BondDir.NONE:
            if dir_from_s == dir_from_t:
                flip_mol_direction(mol, mol.GetAtomWithIdx(s_next))
                dir_from_s = flip_bond_direction(dir_from_s)
        final_dir = Chem.BondDir.NONE
        if dir_from_s != Chem.BondDir.NONE:
            final_dir = flip_bond_direction(dir_from_s)
        if dir_from_t != Chem.BondDir.NONE:
            final_dir = dir_from_t
        bond_idx = mol.AddBond(s_next, t_next, Chem.BondType.SINGLE) - 1
        mol.GetBondWithIdx(bond_idx).SetBondDir(final_dir)
    mol.BeginBatchEdit()
    for s, t in link_dict.values():
        mol.RemoveAtom(s)
        mol.RemoveAtom(t)
    mol.CommitBatchEdit()
    mol.UpdatePropertyCache(strict=False)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    return mol

def split(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RWMol(mol)
    mid = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 84:
            neighbors = atom.GetNeighbors()
            assert len(neighbors) == 2
            s, t = neighbors[0], neighbors[1]
            mid.append((atom, s, t))
    for atom, s, t in mid:
        s_next = add_atom(mol, 85, atom.GetIsotope(), t.GetAtomMapNum())
        t_next = add_atom(mol, 85, atom.GetIsotope(), s.GetAtomMapNum())
        s, t = s.GetIdx(), t.GetIdx()
        s_bond_dir = remove_bond(mol, s, atom.GetIdx())
        t_bond_dir = remove_bond(mol, t, atom.GetIdx())
        s_bond = mol.AddBond(s, s_next, Chem.BondType.SINGLE) - 1
        mol.GetBondWithIdx(s_bond).SetBondDir(s_bond_dir)
        t_bond = mol.AddBond(t, t_next, Chem.BondType.SINGLE) - 1
        mol.GetBondWithIdx(t_bond).SetBondDir(t_bond_dir)
    mol.BeginBatchEdit()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 84:
            mol.RemoveAtom(atom.GetIdx())
    mol.CommitBatchEdit()
    return mol