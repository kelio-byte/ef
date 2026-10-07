from pathlib import Path

from SmilesPE.tokenizer import SPE_Tokenizer

from edit_flows.chem.spe_provenance import (
    ProvenanceToken,
    atomwise_with_provenance,
    component_mode_position_scores,
    insertion_anchor_atoms,
    load_spe_codes,
    map_raw_product_atoms,
    project_syntax_tokens,
    replay_spe_merges,
    tokenize_with_provenance,
)


CODES_PATH = Path("scripts/preprocessing/SPE_ChEMBL.txt")


def test_atomwise_provenance_follows_smiles_atom_encounter_order():
    tokens = atomwise_with_provenance("C(=O)N[C@@H](C)Cl")
    atoms = [token.atom_indices for token in tokens if token.atom_indices]
    assert atoms == [frozenset({index}) for index in range(6)]
    assert "".join(token.surface for token in tokens) == "C(=O)N[C@@H](C)Cl"


def test_replay_merges_all_non_overlapping_occurrences():
    tokens = [
        ProvenanceToken("C", frozenset({0})),
        ProvenanceToken("C", frozenset({1})),
        ProvenanceToken("C", frozenset({2})),
    ]
    merged = replay_spe_merges(tokens, {("C", "C"): 0})
    assert [token.surface for token in merged] == ["CC", "C"]
    assert merged[0].atom_indices == frozenset({0, 1})


def test_real_m500_replay_matches_smilespe():
    smiles = "c1c(NC(N[C@@H](CCCCN)C(OC)=O)=O)c(O)c(C(C)(C)C)cc1OC"
    codes = load_spe_codes(CODES_PATH, merges=500)
    actual = tokenize_with_provenance(smiles, codes)
    with CODES_PATH.open() as handle:
        expected = SPE_Tokenizer(handle, merges=500).tokenize(
            smiles, dropout=0
        ).split()
    assert [token.surface for token in actual] == expected
    assert "".join(token.surface for token in actual) == smiles


def test_raw_product_mapping_retains_all_symmetric_images():
    mapping = map_raw_product_atoms("[CH3:1][CH2:2][CH3:3]", "CCC")
    assert mapping.isomorphism_count == 2
    assert mapping.atom_map_to_processed_indices[2] == (1,)
    assert mapping.atom_map_to_processed_indices[1] == (0, 2)
    assert mapping.atom_map_to_processed_indices[3] == (0, 2)


def test_syntax_projection_and_boundary_anchor_use_nearest_atoms():
    tokens = [
        ProvenanceToken("C", frozenset({0})),
        ProvenanceToken("(", frozenset()),
        ProvenanceToken("=O", frozenset({1})),
        ProvenanceToken(")", frozenset()),
        ProvenanceToken("N", frozenset({2})),
    ]
    projected = project_syntax_tokens(tokens)
    assert projected[1] == frozenset({0, 1})
    assert projected[3] == frozenset({1, 2})
    assert insertion_anchor_atoms(tokens, 0) == frozenset({0})
    assert insertion_anchor_atoms(tokens, 4) == frozenset({1, 2})


def test_component_scores_include_bos_and_separate_insert_anchors():
    tokens = atomwise_with_provenance("CCO")
    scores = component_mode_position_scores("CCO", tokens, {1})
    assert len(scores) == 4
    assert scores[0] == [0.5, 0.0, 0.0]
    assert scores[1] == [1.0, 0.5, 0.5]
    assert scores[2] == [1.0, 1.0, 1.0]
    assert scores[3] == [0.5, 0.5, 0.5]
