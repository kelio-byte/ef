import importlib.util
from pathlib import Path
import re

import pytest
import torch

from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN


ROOT = Path(__file__).parents[1]


def _load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


trajectory = _load_script("visualize_trajectory.py")
first_step = _load_script("visualize_first_step.py")


@pytest.mark.parametrize(
    ("text", "expected"),
    [("True", True), ("yes", True), ("1", True),
     ("False", False), ("no", False), ("0", False)],
)
def test_trajectory_boolean_cli_values(text, expected):
    assert trajectory._str_to_bool(text) is expected


def test_trajectory_boolean_cli_rejects_unknown_value():
    with pytest.raises(Exception, match="expected a boolean"):
        trajectory._str_to_bool("maybe")


def _actions(length, *, ins=None, sub=None, delete=None):
    ins_mask = torch.zeros(length, dtype=torch.bool)
    sub_mask = torch.zeros(length, dtype=torch.bool)
    del_mask = torch.zeros(length, dtype=torch.bool)
    ins_tokens = torch.zeros(length, dtype=torch.long)
    sub_tokens = torch.zeros(length, dtype=torch.long)
    if ins is not None:
        position, token = ins
        ins_mask[position] = True
        ins_tokens[position] = token
    if sub is not None:
        position, token = sub
        sub_mask[position] = True
        sub_tokens[position] = token
    if delete is not None:
        del_mask[delete] = True
    return {
        "ins_mask": ins_mask,
        "sub_mask": sub_mask,
        "del_mask": del_mask,
        "ins_tokens": ins_tokens,
        "sub_tokens": sub_tokens,
    }


def test_sequence_ladder_lists_every_post_edit_state_and_operation():
    id2token = {BOS_TOKEN: "<bos>", 3: "C", 4: ")", 5: "O"}
    events = [
        {
            "x_t": torch.tensor([BOS_TOKEN, 3, PAD_TOKEN]),
            "x_next": torch.tensor([BOS_TOKEN, 3, 4, PAD_TOKEN]),
            "actions": _actions(3, ins=(1, 4)),
        },
        {
            "x_t": torch.tensor([BOS_TOKEN, 3, 4, PAD_TOKEN]),
            "x_next": torch.tensor([BOS_TOKEN, 5, 4, 3, PAD_TOKEN]),
            "actions": _actions(4, sub=(1, 5), ins=(2, 3)),
        },
    ]

    html = trajectory._build_sequence_ladder(
        "C", "O ) C", events, id2token,
    )

    assert len(re.findall(r"\+?Edit \d+", html)) == 2
    assert "C )" in html
    assert "O ) C" in html
    assert "+) after pos 1" in html
    assert "C→O @pos 1" in html
    assert "+C after pos 2" in html
    edit_labels = list(re.finditer(r"\+?Edit [12]", html))
    assert html.index("Product") < edit_labels[0].start()
    assert edit_labels[1].start() < html.index("Target")


def _state_event(step, *tokens):
    return {
        "step_idx": step,
        "x_next": torch.tensor([BOS_TOKEN, *tokens, PAD_TOKEN]),
    }


def test_reconvergence_detection_requires_prior_state_divergence():
    initial = torch.tensor([BOS_TOKEN, 3, PAD_TOKEN])
    paths = [
        [
            _state_event(0, 4),
            _state_event(2, 6),
            _state_event(3, 7),
            _state_event(4, 9),
        ],
        [
            _state_event(0, 5),
            _state_event(2, 6),
            _state_event(3, 8),
            _state_event(4, 9),
        ],
        [],
    ]

    episodes = trajectory._find_reconvergence_episodes(initial, paths, 6)

    pair_episodes = [
        episode for episode in episodes
        if (episode["left_path"], episode["right_path"]) == (0, 1)
    ]
    assert [
        (episode["divergence_step"], episode["reconvergence_step"])
        for episode in pair_episodes
    ] == [(0, 2), (3, 4)]
    assert all(episode["right_path"] != 2 for episode in episodes)


def test_cross_example_collisions_are_reported_separately():
    initial_states = [
        torch.tensor([BOS_TOKEN, 3, PAD_TOKEN]),
        torch.tensor([BOS_TOKEN, 4, PAD_TOKEN]),
    ]
    grouped_events = [
        [[_state_event(1, 6)]],
        [[_state_event(1, 6)]],
    ]

    collisions = trajectory._find_cross_example_collisions(
        initial_states, grouped_events, 3,
    )

    assert len(collisions) == 1
    assert collisions[0]["step"] == 1
    assert collisions[0]["state"] == (6,)
    assert collisions[0]["members"] == [(0, 0), (1, 0)]


def test_matching_path_view_excludes_mismatches_and_preserves_indices():
    grouped_events = [
        [[_state_event(0, 4)], [_state_event(0, 5)], [_state_event(0, 6)]],
        [[_state_event(0, 7)]],
    ]

    matched_events, matched_indices = trajectory._matching_path_view(
        grouped_events, [[False, True, True], [False]],
    )

    assert matched_indices == [[1, 2], []]
    assert matched_events == [[grouped_events[0][1], grouped_events[0][2]], []]


def test_overview_reconvergence_uses_only_target_matching_paths():
    initial_states = [torch.tensor([BOS_TOKEN, 3, PAD_TOKEN])]
    def overview_event(step, *tokens):
        event = _state_event(step, *tokens)
        event["actions"] = _actions(len(tokens) + 2)
        event["x_t"] = event["x_next"].clone()
        return event

    grouped_events = [[
        [overview_event(0, 4), overview_event(1, 6)],
        [overview_event(0, 5), overview_event(1, 6)],
        [overview_event(0, 8), overview_event(1, 6)],
    ]]
    grouped_finals = [torch.tensor([
        [BOS_TOKEN, 6, PAD_TOKEN],
        [BOS_TOKEN, 6, PAD_TOKEN],
        [BOS_TOKEN, 6, PAD_TOKEN],
    ])]

    html = trajectory._build_trajectory_overview(
        example_ids=[7],
        product_strs=["C"],
        target_strs=["O"],
        initial_states=initial_states,
        grouped_events=grouped_events,
        grouped_finals=grouped_finals,
        path_correctness=[[True, False, True]],
        id2token={BOS_TOKEN: "<bos>", 3: "C", 4: "N", 5: "F", 6: "O", 8: "S"},
        n_steps=2,
    )

    assert "Path #1 vs Path #3" in html
    assert "Path #1 vs Path #2" not in html
    assert "Target-matching paths only" in html


def test_overview_skips_reconvergence_when_no_path_matches_target():
    initial = torch.tensor([BOS_TOKEN, 3, PAD_TOKEN])
    grouped_events = [[[], []]]
    grouped_finals = [torch.stack([initial, initial])]

    html = trajectory._build_trajectory_overview(
        example_ids=[2],
        product_strs=["C"],
        target_strs=["O"],
        initial_states=[initial],
        grouped_events=grouped_events,
        grouped_finals=grouped_finals,
        path_correctness=[[False, False]],
        id2token={BOS_TOKEN: "<bos>", 3: "C"},
        n_steps=2,
    )

    assert "analysis skipped: no path matches Target" in html


def test_first_step_navigation_has_one_description_per_link():
    html = first_step._build_index_html(
        "unused", [7], [0.0, 0.1], ["C C"], ["C O"],
    )
    assert html.count("Ex#7 t=") == 2
    assert html.count("C C → C O") == 2
