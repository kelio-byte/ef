import pytest
import torch

from edit_flows.guidance.zspace import (
    ZSpaceMappingError,
    apply_z_transition,
    edit_to_z_candidates,
    z_transition_to_edit,
)


def test_z_transition_maps_substitute_and_operation_channel():
    old = torch.tensor([1, 10, 11, 0])
    new = torch.tensor([1, 10, 12, 0])
    edit = z_transition_to_edit(old, new, vocab_size=32)
    assert edit.operation == "substitute"
    assert edit.z_position == 2
    assert edit.x_position == 2
    assert edit.token == 12
    assert edit.action_channel(32) == 44
    torch.testing.assert_close(apply_z_transition(old, new, vocab_size=32), new)


def test_gap_transition_exposes_unique_insertion_boundary():
    old = torch.tensor([1, 10, 2, 11, 0])
    new = torch.tensor([1, 10, 12, 11, 0])
    edit = z_transition_to_edit(old, new, vocab_size=32, require_unique=True)
    assert edit.operation == "insert"
    assert edit.x_position == 1  # insert after token 10
    assert edit.token == 12
    assert not edit.ambiguous
    candidates = edit_to_z_candidates(
        old, "insert", x_position=1, token=12, vocab_size=32,
    )
    assert len(candidates) == 1
    assert candidates[0] == edit


def test_contiguous_gaps_prove_insertion_mapping_is_not_bijective():
    old = torch.tensor([1, 10, 2, 2, 11, 0])
    new = old.clone()
    new[2] = 12
    edit = z_transition_to_edit(old, new, vocab_size=32)
    assert edit.ambiguous
    candidates = edit_to_z_candidates(
        old, "insert", x_position=1, token=12, vocab_size=32,
    )
    assert len(candidates) == 2
    with pytest.raises(ZSpaceMappingError, match="contiguous GAP"):
        z_transition_to_edit(old, new, vocab_size=32, require_unique=True)


def test_delete_is_unique_and_bos_or_multi_edit_is_rejected():
    old = torch.tensor([1, 10, 11, 0])
    new = torch.tensor([1, 10, 2, 0])
    edit = z_transition_to_edit(old, new, vocab_size=32)
    assert edit.operation == "delete"
    assert edit.x_position == 2
    assert edit.token == -1
    assert len(edit_to_z_candidates(old, "delete", 2, vocab_size=32)) == 1

    bos_new = old.clone()
    bos_new[0] = 12
    with pytest.raises(ZSpaceMappingError, match="BOS"):
        z_transition_to_edit(old, bos_new, vocab_size=32)

    multi = old.clone()
    multi[1:3] = torch.tensor([12, 13])
    with pytest.raises(ZSpaceMappingError, match="exactly one coordinate"):
        z_transition_to_edit(old, multi, vocab_size=32)
