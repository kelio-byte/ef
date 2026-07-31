from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

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

def get_canonical(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol, canonical=True)
    return '.'.join(sorted(smiles.split('.')))

import re
smiles_regex = re.compile('(\(|\)|\.)|((-|=|#|\\\\|\/)?(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p)([-=#\\\\\/\%0-9]*[0-9])?)')
loop_idx_regex = re.compile('(-|=|#|\\\\|\/)?(\%[0-9]{2}|[0-9])')
def tokenize(smiles: str):
    smiles, num_list = clear_map_num(smiles, get_num=True)
    num_list.reverse()

    tokens = []
    for token in smiles_regex.findall(smiles):
        if token[0] != '':
            tokens.append(token[0])
            continue
        loop_idx_and_bond = loop_idx_regex.findall(token[4])
        atom_prop = {
            'map_num': num_list.pop(),
            'smiles': token[3],
            'bond_type': token[2],
            'loop_idx': [
                int(idx[1:]) if idx[0]=='%' else int(idx)
                for _, idx in loop_idx_and_bond
            ],
            'loop_bond': [bond for bond, _ in loop_idx_and_bond]
        }
        tokens.append(atom_prop)
    
    return tokens

def remove_brace(tokens: list, res: list, top):
    near_parentheses = False
    while top>=0 and tokens[top] != '{':
        if tokens[top] == ')':
            res.append(')')
            near_parentheses = True
            top -= 1
            continue
        if tokens[top] == '}':
            if not near_parentheses:
                res.append(')')
            top, res = remove_brace(tokens, res, top - 1)
            res.append('.')
            if not near_parentheses:
                res.append('(')
            top -= 1
            continue
        res.append(tokens[top])
        near_parentheses = False
        top -= 1
    return top, res

def token_to_str(tokens):
    if type(tokens)==str:
        return tokens
    s = tokens['bond_type'] + tokens['smiles']
    for bond, idx in zip(tokens['loop_bond'], tokens['loop_idx']):
        s += bond
        if idx > 9:
            s += '%'
        s += str(idx)
    return s

def sort_by_first_map_num(smiles: str):
    return '.'.join(
        sorted(
            smiles.split('.'), 
            key = lambda mol_smiles: Chem.MolFromSmiles(mol_smiles).GetAtomWithIdx(0).GetAtomMapNum()
        )
    )
def atom_align(S: str, T: str):
    S = Chem.MolFromSmiles(S)
    T = Chem.MolFromSmiles(T)
    S_atom_map_nums = {atom.GetAtomMapNum(): atom.GetIdx()+1 for atom in S.GetAtoms()}
    S_neighbors = {atom.GetAtomMapNum(): set(i.GetAtomMapNum() for i in atom.GetNeighbors()) for atom in S.GetAtoms()}
    S_atom_map_nums.pop(0, None)
    S_neighbors.pop(0, None)
    max_num = max(S_atom_map_nums.keys())

    new_atom_map_nums = {}
    for atom in T.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num not in S_neighbors:
            continue
        S_neighbor = S_neighbors[map_num]
        T_neighbor = set(i.GetAtomMapNum() for i in atom.GetNeighbors())
        link_atom = S_neighbor - T_neighbor
        link_atom.discard(0)
        if len(link_atom) == 0:
            continue
        link_atom = link_atom.pop()
        current_atom = S_atom_map_nums[map_num]
        link_atom = S_atom_map_nums[link_atom]
        if current_atom < link_atom:
            neighbor_map_num = link_atom * 2 - 1
        else:
            neighbor_map_num = current_atom * 2 + 1
        for i in atom.GetNeighbors():
            if i.GetAtomMapNum() == 0:
                new_atom_map_nums[i.GetIdx()] = neighbor_map_num
    
    queue = []
    for atom in T.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num in S_neighbors:
            atom.SetAtomMapNum(S_atom_map_nums[map_num] * 2)
            continue
        new_map_num = new_atom_map_nums.get(atom.GetIdx())
        if new_map_num is not None:
            atom.SetAtomMapNum(new_map_num)
            queue.append((atom, new_map_num))
    queue_i = 0
    while queue_i < len(queue):
        current, map_num = queue[queue_i]
        for atom in current.GetNeighbors():
            if atom.GetAtomMapNum() == 0:
                atom.SetAtomMapNum(map_num)
                queue.append((atom, map_num))
        queue_i += 1
    for atom in T.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_num*2+2)
    
    T = Chem.MolToSmiles(T, canonical=True, allHsExplicit=True) # , ignoreAtomMapNumbers=False

    return sort_by_first_map_num(T)

def flip_idx(idx: int, idx_set: set):
    if idx in idx_set:
        idx_set.remove(idx)
    else:
        idx_set.add(idx)
    return idx_set
def update_idx(tokens, idx_set):
    max_idx = 0
    for token in tokens:
        if type(token) == dict:
            for idx in token['loop_idx']:
                max_idx = max(max_idx, idx)
    idx = 0
    idx_list = []
    for _ in range(max_idx):
        idx += 1
        while idx in idx_set:
            idx += 1
        idx_list.append(idx)
    for token in tokens:
        if type(token) == dict:
            token['loop_idx'] = list(map(
                lambda idx: idx_list[idx-1],
                token['loop_idx']
            ))
def get_next(tokens):
    for token in tokens:
        if type(token) == dict:
            return token['map_num']
    return -1
def global_align(S, T):
    T = atom_align(S, T)
    tokens = tokenize(T)

    tokens.append('.')
    mol_list = []
    mol = []
    for token in tokens:
        if token == '.':
            mol_list.append(mol)
            mol = []
        else:
            mol.append(token)
    mol_list.append([{'map_num': float('inf')}])
    mol_list.reverse()

    idx_set: set[int] = set()
    
    res = []
    stack = []
    while True:
        if len(stack)!=0 and len(stack[-1])==0:
            res.append('}')
            stack.pop()
            continue
        if len(stack)==0 or get_next(stack[-1])>get_next(mol_list[-1]):
            if len(mol_list) == 1:
                break
            res.append('{')
            mol = mol_list.pop()
            update_idx(mol, idx_set)
            stack.append(mol)
        token = stack[-1].pop(0)
        if type(token) == dict:
            for idx in token['loop_idx']:
                flip_idx(idx, idx_set)
        res.append(token)

    res.append(')')
    _, res = remove_brace(res, [], len(res)-1)
    res = res[1: -1]
    res.reverse()
    return ''.join(token_to_str(token) for token in res)


dot_bracket_regex = re.compile('(\(|\)|\.|[^\(\)\.]+)')
def inverse_global_align(smiles: str):
    smiles = '.' + smiles + ')'
    stack = []
    level = []
    res = []
    for c in dot_bracket_regex.findall(smiles):
        if c == '.':
            stack.append('')
            level.append(0)
            continue
        if c == '(':
            level[-1] += 1
        if c == ')':
            while len(level) != 0 and level[-1] == 0:
                res.append(stack.pop())
                level.pop()
            if len(level) == 0:
                break
            level[-1] -= 1
        if c == ')':
            if stack[-1][-1] == '(':
                stack[-1] = stack[-1][:-1]
            else:
                stack[-1] += c
        else:
            stack[-1] += c
    res.reverse()
    return '.'.join(res)


def mol_shuffle(smiles: str, rootedAtAtom=-1, canonical_or_random=True):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True, doRandom=canonical_or_random, allHsExplicit=True, rootedAtAtom=rootedAtAtom)
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=False, allHsExplicit=True)
    return smiles


if __name__ == '__main__':
    S = '[C:43]([O:44][H:78])([H:75])([H:76])[H:77].[F:1][C:2]1=[C:3]([H:45])[C:4]([H:46])=[C:5]([C:6]([O:7][C:8]2=[C:9]([H:48])[C:10]([H:49])=[C:11]([C:12]([N:13]([C:14]([C:15]([C:16]([S:17][C:18]([H:56])([H:57])[H:58])([H:54])[H:55])([H:52])[H:53])([C:19](=[O:22])[O:42][H:74])[H:51])[H:50])=[O:23])[C:24]([C:25]3=[C:26]([H:62])[C:27]([H:63])=[C:28]([F:29])[C:30]([H:64])=[C:31]3[H:65])=[C:32]2[H:66])([C:33]([N:34]2[C:35]([H:69])=[C:36]([H:70])[N:37]=[C:38]2[H:71])([H:67])[H:68])[H:47])[C:39]([H:72])=[C:40]1[H:73].[Na+:41].[O-:20][C:21]([H:59])([H:60])[H:61]'
    T = '[C:43]([O:44][H:78])([H:75])([H:76])[H:77].[F:1][C:2]1=[C:3]([H:45])[C:4]([H:46])=[C:5]([C:6]([O:7][C:8]2=[C:9]([H:48])[C:10]([H:49])=[C:11]([C:12]([N:13]([C:14]([C:15]([C:16]([S:17][C:18]([H:56])([H:57])[H:58])([H:54])[H:55])([H:52])[H:53])([C:19]([O:20][C:21]([H:59])([H:60])[H:61])([O-:22])[O:42][H:74])[H:51])[H:50])=[O:23])[C:24]([C:25]3=[C:26]([H:62])[C:27]([H:63])=[C:28]([F:29])[C:30]([H:64])=[C:31]3[H:65])=[C:32]2[H:66])([C:33]([N:34]2[C:35]([H:69])=[C:36]([H:70])[N:37]=[C:38]2[H:71])([H:67])[H:68])[H:47])[C:39]([H:72])=[C:40]1[H:73].[Na+:41]'
    # S = '[Cl:1][H:46].[Li+:2].[O:3]([C:32]([C:31]([C:30]1=[N:37][N:23]([C:22]([C:21]2=[C:20]([H:60])[N:19]=[C:18]([N:17]([C:16]([C:15]3=[C:6]([H:52])[C:7]4=[C:8]([C:9]([H:53])=[C:10]([H:54])[C:11]([H:55])=[C:12]4[H:56])[C:13]([H:57])=[C:14]3[H:58])=[O:40])[H:59])[C:39]([H:75])=[C:38]2[H:74])([H:61])[H:62])[C:24]2=[C:25]1[C:26]([H:63])=[C:27]([H:64])[C:28]([H:65])=[C:29]2[H:66])([H:67])[H:68])([O:33][C:34]([C:35]([H:71])([H:72])[H:73])([H:69])[H:70])[O-:36])[H:47].[O:41]1[C:42]([H:76])([H:77])[C:43]([H:78])([H:79])[C:44]([H:80])([H:81])[C:45]1([H:82])[H:83].[O:4]([H:48])[H:49].[O:5]([H:50])[H:51]'
    # T = '[C:6]1([H:52])=[C:7]2[C:8](=[C:13]([H:57])[C:14]([H:58])=[C:15]1[C:16]([N:17]([C:18]1=[N:19][C:20]([H:60])=[C:21]([C:22]([N:23]3[C:24]4=[C:29]([H:66])[C:28]([H:65])=[C:27]([H:64])[C:26]([H:63])=[C:25]4[C:30]([C:31]([C:32]([O:33][C:34]([C:35]([H:71])([H:72])[H:73])([H:69])[H:70])=[O:36])([H:67])[H:68])=[N:37]3)([H:61])[H:62])[C:38]([H:74])=[C:39]1[H:75])[H:59])=[O:40])[C:9]([H:53])=[C:10]([H:54])[C:11]([H:55])=[C:12]2[H:56].[Cl:1][H:46].[Li+:2].[O-:3][H:47].[O:41]1[C:42]([H:76])([H:77])[C:43]([H:78])([H:79])[C:44]([H:80])([H:81])[C:45]1([H:82])[H:83].[O:4]([H:48])[H:49].[O:5]([H:50])[H:51]'
    # S = '[c:1]1(-[c:8]2[cH:9][cH:10][c:11]([O:12][c:13]3[cH:14][cH:15][cH:16][c:17]([CH2:18][C:19]([O:20][CH2:21][CH3:22])=[O:23])[cH:24]3)[c:25]([CH2:26][N:27]3[C:28](=[O:29])[O:30][C@H:31]([c:32]4[cH:33][cH:34][cH:35][cH:36][cH:37]4)[C@@H:38]3[CH3:39])[cH:40]2)[c:2]([CH3:3])[n:4][o:5][c:6]1[CH3:7]'
    # T = 'Br[c:1]1[c:2]([CH3:3])[n:4][o:5][c:6]1[CH3:7].CC1(C)OB([c:8]2[cH:9][cH:10][c:11]([O:12][c:13]3[cH:14][cH:15][cH:16][c:17]([CH2:18][C:19]([O:20][CH2:21][CH3:22])=[O:23])[cH:24]3)[c:25]([CH2:26][N:27]3[C:28](=[O:29])[O:30][C@H:31]([c:32]4[cH:33][cH:34][cH:35][cH:36][cH:37]4)[C@@H:38]3[CH3:39])[cH:40]2)OC1(C)C'
    # S_ = mol_shuffle(S, rootedAtAtom=0, canonical_or_random=False)
    # print(S_)
    # print(clear_map_num(S_))
    # S = mol_shuffle(clear_map_num(S), rootedAtAtom=0, canonical_or_random=False)
    # print(clear_map_num(S))
    # exit(0)

    # S = Chem.MolToSmiles(Chem.MolFromSmiles(S), canonical=True, doRandom=False, allHsExplicit=True)
    # T = Chem.MolToSmiles(Chem.MolFromSmiles(T), canonical=True, doRandom=False, allHsExplicit=True)
    S = mol_shuffle(S, rootedAtAtom=0, canonical_or_random=False)
    T = mol_shuffle(T, rootedAtAtom=0, canonical_or_random=False)
    print(S)
    print(T)
    T_ = atom_align(S, T)
    print(T_)
    print(clear_map_num(S))
    T = global_align(S, T)
    print(T)
    T = inverse_global_align(T)
    print(T)
    print(clear_map_num(T))