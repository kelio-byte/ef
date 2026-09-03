"""Exact SmilesPE tokenization with product-atom provenance.

SmilesPE 0.0.3 merges atom-wise SMILES tokens.  This module mirrors its
deterministic, dropout-free merge loop while carrying the set of RDKit atom
indices covered by every token.  It is an audit utility; it does not retrain
or alter the tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

from rdkit import Chem
from SmilesPE.tokenizer import atomwise_tokenizer


_ATOM_TOKEN = re.compile(
    r"^(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\*)$"
)


@dataclass(frozen=True)
class ProvenanceToken:
    surface: str
    atom_indices: frozenset[int]


@dataclass(frozen=True)
class ProductAtomMapping:
    atom_map_to_processed_indices: dict[int, tuple[int, ...]]
    isomorphism_count: int
    isomorphism_limit_reached: bool
    used_chirality: bool
    raw_atom_count: int
    processed_atom_count: int


def is_atom_token(token: str) -> bool:
    return bool(_ATOM_TOKEN.fullmatch(token))


def atomwise_with_provenance(smiles: str) -> list[ProvenanceToken]:
    """Atom-tokenize exactly as SmilesPE and attach RDKit encounter indices."""
    surfaces = atomwise_tokenizer(smiles)
    if "".join(surfaces) != smiles:
        raise ValueError(
            "SmilesPE atomwise tokenizer did not cover the full SMILES: "
            f"{smiles!r}"
        )
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse product SMILES: {smiles!r}")
    result: list[ProvenanceToken] = []
    atom_index = 0
    for surface in surfaces:
        if is_atom_token(surface):
            result.append(ProvenanceToken(surface, frozenset({atom_index})))
            atom_index += 1
        else:
            result.append(ProvenanceToken(surface, frozenset()))
    if atom_index != molecule.GetNumAtoms():
        raise ValueError(
            "atom-token/RDKit atom-count mismatch: "
            f"tokens={atom_index}, rdkit={molecule.GetNumAtoms()}, smiles={smiles!r}"
        )
    return result


def load_spe_codes(path: Path, *, merges: int) -> dict[tuple[str, str], int]:
    """Load merge ranks with the duplicate semantics of SmilesPE 0.0.3."""
    if merges < -1:
        raise ValueError("merges must be -1 or non-negative")
    pairs: list[tuple[str, str]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle):
            if merges != -1 and line_number >= merges:
                break
            fields = line.strip().split()
            if len(fields) != 2:
                raise ValueError(
                    f"invalid SPE code at {path}:{line_number + 1}: {line!r}"
                )
            pairs.append((fields[0], fields[1]))
    # SmilesPE iterates the reversed list before constructing a dict, making
    # the first occurrence of a duplicate pair authoritative.
    return {
        pair: rank for rank, pair in reversed(list(enumerate(pairs)))
    }


def replay_spe_merges(
    tokens: Sequence[ProvenanceToken],
    codes: dict[tuple[str, str], int],
) -> list[ProvenanceToken]:
    """Replay deterministic standard BPE while unioning atom provenance."""
    word = list(tokens)
    while len(word) > 1:
        ranked_pairs = [
            (codes[pair], index, pair)
            for index, pair in enumerate(
                zip(
                    (token.surface for token in word),
                    (token.surface for token in word[1:]),
                )
            )
            if pair in codes
        ]
        if not ranked_pairs:
            break
        bigram = min(ranked_pairs)[2]
        positions = [
            index for rank, index, pair in ranked_pairs if pair == bigram
        ]
        merged: list[ProvenanceToken] = []
        cursor = 0
        for position in positions:
            if position < cursor:
                continue
            merged.extend(word[cursor:position])
            first, second = word[position : position + 2]
            merged.append(
                ProvenanceToken(
                    first.surface + second.surface,
                    first.atom_indices | second.atom_indices,
                )
            )
            cursor = position + 2
        merged.extend(word[cursor:])
        word = merged
    return word


def tokenize_with_provenance(
    smiles: str,
    codes: dict[tuple[str, str], int],
) -> list[ProvenanceToken]:
    tokens = replay_spe_merges(atomwise_with_provenance(smiles), codes)
    if "".join(token.surface for token in tokens) != smiles:
        raise AssertionError("provenance merge replay was not lossless")
    return tokens


def _map_free_copy(molecule: Chem.Mol) -> Chem.Mol:
    result = Chem.Mol(molecule)
    for atom in result.GetAtoms():
        atom.SetAtomMapNum(0)
    return result


def map_raw_product_atoms(
    raw_mapped_product: str,
    processed_product: str,
    *,
    max_isomorphisms: int = 1024,
) -> ProductAtomMapping:
    """Map raw atom-map IDs to all equivalent processed-product atom indices.

    Multiple graph isomorphisms are retained as a union.  This is important
    for symmetric products: after atom maps are removed, choosing one
    arbitrary symmetry-equivalent atom would make an oracle center depend on
    RDKit match order rather than chemistry.
    """
    if max_isomorphisms <= 0:
        raise ValueError("max_isomorphisms must be positive")
    raw = Chem.MolFromSmiles(raw_mapped_product)
    processed = Chem.MolFromSmiles(processed_product)
    if raw is None or processed is None:
        raise ValueError("RDKit could not parse raw or processed product")
    if raw.GetNumAtoms() != processed.GetNumAtoms():
        raise ValueError(
            "raw/processed product atom counts differ: "
            f"{raw.GetNumAtoms()} != {processed.GetNumAtoms()}"
        )
    raw_map_by_index = {
        atom.GetIdx(): atom.GetAtomMapNum()
        for atom in raw.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    query = _map_free_copy(raw)
    matches = processed.GetSubstructMatches(
        query,
        uniquify=False,
        useChirality=True,
        maxMatches=max_isomorphisms,
    )
    used_chirality = True
    if not matches:
        matches = processed.GetSubstructMatches(
            query,
            uniquify=False,
            useChirality=False,
            maxMatches=max_isomorphisms,
        )
        used_chirality = False
    full_matches = [match for match in matches if len(match) == raw.GetNumAtoms()]
    if not full_matches:
        raise ValueError("raw and processed products are not graph-isomorphic")

    candidates: dict[int, set[int]] = {
        map_number: set() for map_number in raw_map_by_index.values()
    }
    for match in full_matches:
        for raw_index, map_number in raw_map_by_index.items():
            candidates[map_number].add(match[raw_index])
    return ProductAtomMapping(
        atom_map_to_processed_indices={
            map_number: tuple(sorted(indices))
            for map_number, indices in sorted(candidates.items())
        },
        isomorphism_count=len(full_matches),
        isomorphism_limit_reached=len(full_matches) >= max_isomorphisms,
        used_chirality=used_chirality,
        raw_atom_count=raw.GetNumAtoms(),
        processed_atom_count=processed.GetNumAtoms(),
    )


def project_syntax_tokens(
    tokens: Sequence[ProvenanceToken],
) -> list[frozenset[int]]:
    """Give syntax-only tokens their nearest left/right atom provenance."""
    left: list[frozenset[int]] = []
    nearest = frozenset()
    for token in tokens:
        if token.atom_indices:
            nearest = token.atom_indices
        left.append(nearest)
    right: list[frozenset[int]] = [frozenset()] * len(tokens)
    nearest = frozenset()
    for index in range(len(tokens) - 1, -1, -1):
        if tokens[index].atom_indices:
            nearest = tokens[index].atom_indices
        right[index] = nearest
    return [
        token.atom_indices or (left[index] | right[index])
        for index, token in enumerate(tokens)
    ]


def insertion_anchor_atoms(
    tokens: Sequence[ProvenanceToken], anchor: int
) -> frozenset[int]:
    """Return atoms next to the boundary before token ``anchor``.

    ``anchor`` ranges from 0 (before the first token) through ``len(tokens)``
    (after the final token).
    """
    if anchor < 0 or anchor > len(tokens):
        raise IndexError("insertion anchor is outside token boundaries")
    projected = project_syntax_tokens(tokens)
    atoms = frozenset()
    if anchor > 0:
        atoms |= projected[anchor - 1]
    if anchor < len(tokens):
        atoms |= projected[anchor]
    return atoms


def graph_distances(
    molecule: Chem.Mol, center_atom_indices: Iterable[int]
) -> dict[int, int]:
    centers = sorted(set(center_atom_indices))
    if not centers:
        return {}
    distances: dict[int, int] = {}
    frontier = centers
    for center in centers:
        if center < 0 or center >= molecule.GetNumAtoms():
            raise IndexError("center atom index is outside the product molecule")
        distances[center] = 0
    while frontier:
        next_frontier: list[int] = []
        for atom_index in frontier:
            atom = molecule.GetAtomWithIdx(atom_index)
            for neighbor in atom.GetNeighbors():
                neighbor_index = neighbor.GetIdx()
                if neighbor_index not in distances:
                    distances[neighbor_index] = distances[atom_index] + 1
                    next_frontier.append(neighbor_index)
        frontier = next_frontier
    return distances


def minimum_distance(
    atom_indices: Iterable[int], distances: dict[int, int]
) -> int | None:
    values = [distances[index] for index in atom_indices if index in distances]
    return min(values) if values else None


def radial_score(distance: int | None) -> float:
    if distance == 0:
        return 1.0
    if distance == 1:
        return 0.5
    return 0.0


def component_mode_position_scores(
    smiles: str,
    tokens: Sequence[ProvenanceToken],
    center_atom_indices: Iterable[int],
) -> list[list[float]]:
    """Build BOS-inclusive INS/SUB/DEL scores for an initial Euler state.

    Row 0 is BOS: only its INS action is meaningful and corresponds to the
    boundary before the first molecular token. Row i>0 corresponds to M500
    token i-1; insertion occurs after that token.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse product SMILES: {smiles!r}")
    distances = graph_distances(molecule, center_atom_indices)
    token_atoms = project_syntax_tokens(tokens)
    scores = [
        [
            radial_score(
                minimum_distance(insertion_anchor_atoms(tokens, 0), distances)
            ),
            0.0,
            0.0,
        ]
    ]
    for token_index, atoms in enumerate(token_atoms):
        existing_score = radial_score(minimum_distance(atoms, distances))
        insert_score = radial_score(
            minimum_distance(
                insertion_anchor_atoms(tokens, token_index + 1), distances
            )
        )
        scores.append([insert_score, existing_score, existing_score])
    return scores
