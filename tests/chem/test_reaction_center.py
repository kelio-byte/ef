from edit_flows.chem.reaction_center import (
    canonical_reaction_key,
    canonical_reaction_key_achiral,
    extract_reaction_center,
    split_reaction_smiles,
)


def _record(reactants: str, product: str):
    return extract_reaction_center(f"{reactants}>>{product}")


def test_split_reaction_smiles_preserves_reagent_field():
    assert split_reaction_smiles("[CH3:1]O>NaOH>[CH3:1]Cl") == (
        "[CH3:1]O",
        "NaOH",
        "[CH3:1]Cl",
    )


def test_canonical_key_ignores_maps_and_component_order():
    first = canonical_reaction_key("[CH3:1][OH:2]", "[Cl-:3].[CH3:1][OH:2]")
    second = canonical_reaction_key("[CH3:8][OH:9]", "[CH3:8][OH:9].[Cl-:4]")
    assert first == second


def test_achiral_key_ignores_stereo_but_isomeric_key_does_not():
    first = canonical_reaction_key("N[C@H:1](C)O", "N[C@H:1](C)Cl")
    second = canonical_reaction_key("N[C@@H:1](C)O", "N[C@@H:1](C)Cl")
    assert first != second
    assert canonical_reaction_key_achiral(
        "N[C@H:1](C)O", "N[C@H:1](C)Cl"
    ) == canonical_reaction_key_achiral(
        "N[C@@H:1](C)O", "N[C@@H:1](C)Cl"
    )


def test_single_product_bond_break_is_one_center_component():
    result = _record("[CH3:1].[OH:2]", "[CH3:1][OH:2]")
    assert result["status"] == "ok"
    assert result["changed_bonds"][0]["kind"] == "product_only_bond"
    assert result["center_atom_maps"] == [1, 2]
    assert result["center_component_count"] == 1
    assert result["center_components"][0]["has_product_bond_change"]


def test_bond_order_change_is_labeled():
    result = _record("[CH2:1]=[CH2:2]", "[CH3:1][CH3:2]")
    assert result["status"] == "ok"
    event = result["changed_bonds"][0]
    assert event["kind"] == "bond_property_change"
    assert "type" in event["changed_fields"]


def test_atom_charge_change_is_labeled():
    result = _record("[NH3+:1]", "[NH2:1]")
    assert result["status"] == "ok"
    assert result["atom_changes"][0]["atom_map"] == 1
    assert "formal_charge" in result["atom_changes"][0]["changed_fields"]


def test_reactant_only_fragment_marks_product_attachment_atom():
    result = _record("[CH3:1][NH:2][CH3:3]", "[CH3:1][NH2:2]")
    assert result["status"] == "ok"
    assert 3 in result["reactant_only_atom_maps"]
    assert any(
        event["product_atom_map"] == 2 for event in result["attachments"]
    )


def test_separated_bond_changes_remain_two_components():
    result = _record(
        "[CH3:1].[CH3:2].[CH3:3].[CH3:4]",
        "[CH3:1][CH3:2].[CH3:3][CH3:4]",
    )
    assert result["status"] == "ok"
    assert result["center_component_count"] == 2


def test_duplicate_map_returns_auditable_failure():
    result = _record("[CH3:1].[OH:1]", "[CH3:1][OH:2]")
    assert result["status"] == "duplicate_atom_map"
    assert result["duplicate_reactant_maps"] == [1]


def test_unmapped_attachment_is_retained_as_an_anchor():
    result = _record("[CH3:1]O", "[CH4:1]")
    assert result["status"] == "ok"
    assert result["zero_map_reactant_atom_count"] == 1
    assert result["attachments"] == [
        {"product_atom_map": 1, "reactant_only_atom": "idx:1"}
    ]


def test_invalid_smiles_returns_parse_error():
    result = extract_reaction_center("not_smiles>>[CH4:1]")
    assert result["status"] == "parse_error"
