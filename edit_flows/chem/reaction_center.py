"""Reaction-center labels from atom-mapped reaction SMILES.

The functions in this module are deliberately independent of the model and
tokenizer.  They compare the mapped product graph (the retrosynthesis input)
with the mapped reactant graph (the target) and return JSON-serializable
labels.  Ground-truth reactants must never be passed to a deployable sampler;
these labels are for training, auditing, and the explicitly oracle-only RC1
experiment.
"""

from __future__ import annotations

from collections import defaultdict, deque
import hashlib
from typing import Any, Iterable

from rdkit import Chem


def split_reaction_smiles(reaction_smiles: str) -> tuple[str, str, str]:
    """Return ``(reactants, reagents, product)`` from a reaction SMILES."""
    fields = reaction_smiles.strip().split(">")
    if len(fields) != 3:
        raise ValueError(
            "reaction SMILES must contain exactly reactants>reagents>product"
        )
    reactants, reagents, product = fields
    if not reactants or not product:
        raise ValueError("reaction SMILES has an empty reactant or product field")
    return reactants, reagents, product


def _parse_smiles(smiles: str, *, role: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse {role} SMILES")
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def canonicalize_map_free_smiles(
    smiles: str, *, isomeric_smiles: bool = True
) -> str:
    """Canonicalize a possibly mapped, multi-component SMILES without maps."""
    molecule = _parse_smiles(smiles, role="canonicalization")
    return _canonicalize_map_free_molecule(
        molecule, isomeric_smiles=isomeric_smiles
    )


def _canonicalize_map_free_molecule(
    molecule: Chem.Mol, *, isomeric_smiles: bool = True
) -> str:
    molecule = Chem.Mol(molecule)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    if not isomeric_smiles:
        Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=isomeric_smiles,
        allHsExplicit=False,
    )


def canonical_reaction_key(product: str, reactants: str) -> str:
    """Stable map-free key used to crosswalk raw and processed reactions."""
    canonical_product = canonicalize_map_free_smiles(product)
    canonical_reactants = canonicalize_map_free_smiles(reactants)
    return f"{canonical_product}>>{canonical_reactants}"


def canonical_reaction_key_achiral(product: str, reactants: str) -> str:
    """Map-free reaction key with atom and bond stereochemistry removed."""
    canonical_product = canonicalize_map_free_smiles(
        product, isomeric_smiles=False
    )
    canonical_reactants = canonicalize_map_free_smiles(
        reactants, isomeric_smiles=False
    )
    return f"{canonical_product}>>{canonical_reactants}"


def reaction_key_sha256(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _mapped_atoms(
    molecule: Chem.Mol,
) -> tuple[dict[int, Chem.Atom], list[int], list[int]]:
    by_map: dict[int, Chem.Atom] = {}
    duplicate_maps: set[int] = set()
    zero_indices: list[int] = []
    for atom in molecule.GetAtoms():
        map_number = atom.GetAtomMapNum()
        if map_number <= 0:
            zero_indices.append(atom.GetIdx())
            continue
        if map_number in by_map:
            duplicate_maps.add(map_number)
        else:
            by_map[map_number] = atom
    return by_map, sorted(duplicate_maps), zero_indices


def _bond_signature(bond: Chem.Bond) -> dict[str, Any]:
    # Sanitized RDKit molecules perceive aromatic rings consistently even if
    # the input used a Kekule spelling.  Treat all aromatic bonds as one type.
    aromatic = bool(bond.GetIsAromatic())
    return {
        "type": "AROMATIC" if aromatic else str(bond.GetBondType()),
        "aromatic": aromatic,
        "conjugated": True if aromatic else bool(bond.GetIsConjugated()),
        "stereo": str(bond.GetStereo()),
    }


def _mapped_bonds(molecule: Chem.Mol) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for bond in molecule.GetBonds():
        first = bond.GetBeginAtom().GetAtomMapNum()
        second = bond.GetEndAtom().GetAtomMapNum()
        if first <= 0 or second <= 0:
            continue
        key = tuple(sorted((first, second)))
        result[key] = _bond_signature(bond)
    return result


def _atom_signature(atom: Chem.Atom) -> dict[str, Any]:
    return {
        "atomic_number": atom.GetAtomicNum(),
        "isotope": atom.GetIsotope(),
        "formal_charge": atom.GetFormalCharge(),
        "total_h": atom.GetTotalNumHs(includeNeighbors=True),
        "explicit_h": atom.GetNumExplicitHs(),
        "chirality": str(atom.GetChiralTag()),
        "aromatic": bool(atom.GetIsAromatic()),
        "radical_electrons": atom.GetNumRadicalElectrons(),
    }


def _changed_fields(
    first: dict[str, Any], second: dict[str, Any]
) -> list[str]:
    return [name for name in first if first[name] != second[name]]


class _DisjointSet:
    def __init__(self, values: Iterable[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[max(first_root, second_root)] = min(first_root, second_root)


def _product_adjacency(molecule: Chem.Mol) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for atom in molecule.GetAtoms():
        map_number = atom.GetAtomMapNum()
        if map_number > 0:
            adjacency.setdefault(map_number, set())
    for bond in molecule.GetBonds():
        first = bond.GetBeginAtom().GetAtomMapNum()
        second = bond.GetEndAtom().GetAtomMapNum()
        if first > 0 and second > 0:
            adjacency[first].add(second)
            adjacency[second].add(first)
    return dict(adjacency)


def _radius_maps(
    seeds: Iterable[int], adjacency: dict[int, set[int]], radius: int
) -> list[int]:
    distances = {seed: 0 for seed in seeds}
    queue = deque(distances)
    while queue:
        current = queue.popleft()
        if distances[current] >= radius:
            continue
        for neighbor in adjacency.get(current, ()):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return sorted(distances)


def _center_components(
    *,
    product: Chem.Mol,
    center_atom_maps: set[int],
    changed_bonds: list[dict[str, Any]],
    atom_changes: list[dict[str, Any]],
    attachments: list[dict[str, Any]],
    product_only_atom_maps: list[int],
) -> list[dict[str, Any]]:
    if not center_atom_maps:
        return []
    adjacency = _product_adjacency(product)
    disjoint = _DisjointSet(center_atom_maps)
    for first in center_atom_maps:
        for second in adjacency.get(first, ()):
            if second in center_atom_maps:
                disjoint.union(first, second)
    # A reactant-only bond has no corresponding product edge.  Keep its two
    # endpoints as one chemical event even when they are disconnected in the
    # product graph.
    for event in changed_bonds:
        first, second = event["atom_maps"]
        if first in center_atom_maps and second in center_atom_maps:
            disjoint.union(first, second)

    grouped: dict[int, set[int]] = defaultdict(set)
    for map_number in center_atom_maps:
        grouped[disjoint.find(map_number)].add(map_number)

    product_only_set = set(product_only_atom_maps)
    components: list[dict[str, Any]] = []
    for atom_maps_set in grouped.values():
        atom_maps = sorted(atom_maps_set)
        atom_map_lookup = set(atom_maps)
        bond_events = [
            event
            for event in changed_bonds
            if atom_map_lookup.intersection(event["atom_maps"])
        ]
        atom_events = [
            event for event in atom_changes if event["atom_map"] in atom_map_lookup
        ]
        attachment_events = [
            event
            for event in attachments
            if event["product_atom_map"] in atom_map_lookup
        ]
        types = set()
        types.update(event["kind"] for event in bond_events)
        if atom_events:
            types.add("atom_property_change")
        if attachment_events:
            types.add("attachment")
        if product_only_set.intersection(atom_map_lookup):
            types.add("product_only_atom")
        components.append(
            {
                "atom_maps": atom_maps,
                "bond_map_pairs": sorted(
                    [event["atom_maps"] for event in bond_events]
                ),
                "center_types": sorted(types),
                "component_size": len(atom_maps),
                "radius_1_atom_maps": _radius_maps(atom_maps, adjacency, 1),
                "radius_2_atom_maps": _radius_maps(atom_maps, adjacency, 2),
                "changed_event_count": (
                    len(bond_events)
                    + len(atom_events)
                    + len(attachment_events)
                    + len(product_only_set.intersection(atom_map_lookup))
                ),
                "has_product_bond_change": any(
                    event["kind"] in {"product_only_bond", "bond_property_change"}
                    for event in bond_events
                ),
            }
        )

    components.sort(
        key=lambda component: (
            not component["has_product_bond_change"],
            -component["changed_event_count"],
            component["atom_maps"],
        )
    )
    for component_id, component in enumerate(components):
        component["component_id"] = component_id
    return components


def extract_reaction_center(
    reaction_smiles: str,
    *,
    reaction_id: str | None = None,
    reaction_class: str | None = None,
    raw_index: int | None = None,
) -> dict[str, Any]:
    """Extract graph changes from one atom-mapped reaction.

    The returned object is JSON serializable.  Parse and mapping failures are
    returned as ``status != 'ok'`` records so a dataset build can count them
    instead of silently dropping rows.
    """
    base: dict[str, Any] = {
        "reaction_id": reaction_id,
        "reaction_class": reaction_class,
        "raw_index": raw_index,
    }
    try:
        reactants_smiles, reagents_smiles, product_smiles = split_reaction_smiles(
            reaction_smiles
        )
        reactants = _parse_smiles(reactants_smiles, role="reactant")
        product = _parse_smiles(product_smiles, role="product")
        reaction_key = (
            f"{_canonicalize_map_free_molecule(product)}>>"
            f"{_canonicalize_map_free_molecule(reactants)}"
        )
        achiral_reaction_key = (
            f"{_canonicalize_map_free_molecule(product, isomeric_smiles=False)}>>"
            f"{_canonicalize_map_free_molecule(reactants, isomeric_smiles=False)}"
        )
    except Exception as exc:
        return {
            **base,
            "status": "parse_error",
            "error": str(exc),
        }

    product_atoms, product_duplicates, product_zero = _mapped_atoms(product)
    reactant_atoms, reactant_duplicates, reactant_zero = _mapped_atoms(reactants)
    if product_duplicates or reactant_duplicates:
        return {
            **base,
            "status": "duplicate_atom_map",
            "reaction_key": reaction_key,
            "reaction_key_sha256": reaction_key_sha256(reaction_key),
            "achiral_reaction_key": achiral_reaction_key,
            "achiral_reaction_key_sha256": reaction_key_sha256(
                achiral_reaction_key
            ),
            "duplicate_product_maps": product_duplicates,
            "duplicate_reactant_maps": reactant_duplicates,
            "zero_map_product_atom_count": len(product_zero),
            "zero_map_reactant_atom_count": len(reactant_zero),
        }

    product_maps = set(product_atoms)
    reactant_maps = set(reactant_atoms)
    retained_maps = product_maps & reactant_maps
    product_only_maps = sorted(product_maps - reactant_maps)
    reactant_only_maps = sorted(reactant_maps - product_maps)

    product_bonds = _mapped_bonds(product)
    reactant_bonds = _mapped_bonds(reactants)
    changed_bonds: list[dict[str, Any]] = []
    for atom_maps in sorted(set(product_bonds) | set(reactant_bonds)):
        product_signature = product_bonds.get(atom_maps)
        reactant_signature = reactant_bonds.get(atom_maps)
        if product_signature is None:
            kind = "reactant_only_bond"
        elif reactant_signature is None:
            kind = "product_only_bond"
        elif product_signature != reactant_signature:
            kind = "bond_property_change"
        else:
            continue
        changed_bonds.append(
            {
                "atom_maps": list(atom_maps),
                "kind": kind,
                "product": product_signature,
                "reactants": reactant_signature,
                "changed_fields": (
                    _changed_fields(product_signature, reactant_signature)
                    if product_signature is not None
                    and reactant_signature is not None
                    else []
                ),
            }
        )

    atom_changes: list[dict[str, Any]] = []
    for map_number in sorted(retained_maps):
        product_signature = _atom_signature(product_atoms[map_number])
        reactant_signature = _atom_signature(reactant_atoms[map_number])
        changed = _changed_fields(product_signature, reactant_signature)
        if changed:
            atom_changes.append(
                {
                    "atom_map": map_number,
                    "changed_fields": changed,
                    "product": product_signature,
                    "reactants": reactant_signature,
                }
            )

    attachments: list[dict[str, Any]] = []
    seen_attachments: set[tuple[int, str]] = set()
    for atom in reactants.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        is_reactant_only = atom_map <= 0 or atom_map in reactant_maps - product_maps
        if not is_reactant_only:
            continue
        external_id = f"map:{atom_map}" if atom_map > 0 else f"idx:{atom.GetIdx()}"
        for neighbor in atom.GetNeighbors():
            neighbor_map = neighbor.GetAtomMapNum()
            if neighbor_map not in retained_maps:
                continue
            key = (neighbor_map, external_id)
            if key in seen_attachments:
                continue
            seen_attachments.add(key)
            attachments.append(
                {
                    "product_atom_map": neighbor_map,
                    "reactant_only_atom": external_id,
                }
            )
    attachments.sort(
        key=lambda item: (item["product_atom_map"], item["reactant_only_atom"])
    )

    center_atom_maps: set[int] = set(product_only_maps)
    for event in changed_bonds:
        center_atom_maps.update(
            map_number
            for map_number in event["atom_maps"]
            if map_number in product_maps
        )
    center_atom_maps.update(event["atom_map"] for event in atom_changes)
    center_atom_maps.update(event["product_atom_map"] for event in attachments)

    components = _center_components(
        product=product,
        center_atom_maps=center_atom_maps,
        changed_bonds=changed_bonds,
        atom_changes=atom_changes,
        attachments=attachments,
        product_only_atom_maps=product_only_maps,
    )
    return {
        **base,
        "status": "ok",
        "reaction_key": reaction_key,
        "reaction_key_sha256": reaction_key_sha256(reaction_key),
        "achiral_reaction_key": achiral_reaction_key,
        "achiral_reaction_key_sha256": reaction_key_sha256(
            achiral_reaction_key
        ),
        "reagent_field_nonempty": bool(reagents_smiles),
        "product_atom_count": product.GetNumAtoms(),
        "reactant_atom_count": reactants.GetNumAtoms(),
        "mapped_product_atom_count": len(product_maps),
        "mapped_reactant_atom_count": len(reactant_maps),
        "zero_map_product_atom_count": len(product_zero),
        "zero_map_reactant_atom_count": len(reactant_zero),
        "retained_atom_maps": sorted(retained_maps),
        "product_only_atom_maps": product_only_maps,
        "reactant_only_atom_maps": reactant_only_maps,
        "center_atom_maps": sorted(center_atom_maps),
        "changed_bonds": changed_bonds,
        "atom_changes": atom_changes,
        "attachments": attachments,
        "center_components": components,
        "center_component_count": len(components),
        "has_center": bool(components),
    }
