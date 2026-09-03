import torch

from edit_flows.guidance.targets import (
    build_action_target_masks,
    make_action_reward_targets,
)
from edit_flows.utils.tokens import BOS_TOKEN, GAP_TOKEN, PAD_TOKEN


def test_action_target_masks_cover_insert_substitute_delete():
    state = torch.tensor([[BOS_TOKEN, 4, 5, PAD_TOKEN]])
    terminal = torch.tensor([[BOS_TOKEN, 4, 6, 7]])
    insert, substitute, delete = build_action_target_masks(
        state, terminal, vocab_size=16,
    )
    # One deterministic optimal alignment inserts 6 after 4 and substitutes
    # the final current token with 7.  (Equal-cost alignments are allowed to
    # choose a different but still valid edit decomposition.)
    assert bool(substitute[0, 2, 7])
    assert bool(insert[0, 1, 6])
    assert not substitute[0, 1].any()
    assert not delete[0].any()

    state = torch.tensor([[BOS_TOKEN, 4, 5]])
    terminal = torch.tensor([[BOS_TOKEN, 4, PAD_TOKEN]])
    _, _, delete = build_action_target_masks(state, terminal, vocab_size=16)
    assert bool(delete[0, 2, 0])


def test_action_target_masks_map_gap_insertion_after_preceding_token():
    # This pair has a single inserted token between 4 and 5.
    state = torch.tensor([[BOS_TOKEN, 4, 5]])
    terminal = torch.tensor([[BOS_TOKEN, 4, 6, 5]])
    insert, _, _ = build_action_target_masks(state, terminal, vocab_size=16)
    assert bool(insert[0, 1, 6])


def test_action_reward_targets_use_positive_background_and_reward():
    insert = torch.zeros((2, 3, 8), dtype=torch.bool)
    substitute = torch.zeros_like(insert)
    delete = torch.zeros((2, 3, 1), dtype=torch.bool)
    insert[0, 1, 4] = True
    delete[1, 2, 0] = True
    targets = make_action_reward_targets(
        insert, substitute, delete, torch.tensor([1.0, 0.0]), background=1e-3,
    )
    assert targets[0][0, 1, 4].item() == torch.tensor(1.001).item()
    assert targets[0][0, 0, 4].item() == torch.tensor(0.001).item()
    assert targets[2][1, 2, 0].item() == torch.tensor(0.001).item()
    assert torch.isfinite(torch.cat([x.reshape(-1) for x in targets])).all()
