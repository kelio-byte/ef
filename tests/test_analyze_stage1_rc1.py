from edit_flows.analysis.trajectory_correction import token_edit_distance
from scripts.analyze_stage1_rc1 import _apply_first_event, _token_distance


def test_fast_token_distance_matches_existing_torch_implementation():
    pairs = [
        ([1, 10, 11], [1, 10, 11]),
        ([1, 10, 11], [1, 10, 12, 13]),
        ([1, 10, 11, 12], [1, 13, 11]),
    ]
    for left, right in pairs:
        assert _token_distance(left, right) == token_edit_distance(left, right)


def test_apply_first_event_matches_ins_sub_del_semantics():
    source = [1, 10, 11, 12]
    output, n_effective = _apply_first_event(
        source,
        [
            {"mode": "INS", "position": 0, "token_id": 7},
            {"mode": "SUB", "position": 1, "token_id": 20},
            {"mode": "DEL", "position": 2, "token_id": None},
        ],
        max_seq_len=32,
    )
    assert output == [1, 7, 20, 12]
    assert n_effective == 3


def test_apply_first_event_treats_ins_del_as_replacement():
    source = [1, 10, 11]
    output, n_effective = _apply_first_event(
        source,
        [
            {"mode": "SUB", "position": 1, "token_id": 20},
            {"mode": "INS", "position": 1, "token_id": 30},
            {"mode": "DEL", "position": 1, "token_id": None},
        ],
        max_seq_len=32,
    )
    # The substitution happens first, then replacement overwrites it.
    assert output == [1, 30, 11]
    assert n_effective == 1
