from scripts.audit_reaction_center_locality import _extract_edits


def test_extract_edits_groups_insertions_and_keeps_source_positions():
    edits = _extract_edits(
        ["A", "<GAP>", "<GAP>", "B", "C"],
        ["A", "X", "Y", "D", "<GAP>"],
        source_length=3,
    )
    assert edits == [
        {
            "kind": "insertion_run",
            "anchor": 1,
            "aligned_begin": 1,
            "run_length": 2,
        },
        {"kind": "existing_token", "mode": "SUB", "index": 1},
        {"kind": "existing_token", "mode": "DEL", "index": 2},
    ]
