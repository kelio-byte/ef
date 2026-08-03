import importlib.util
from pathlib import Path

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

    assert html.count("+Edit ") == 2
    assert "C )" in html
    assert "O ) C" in html
    assert "+) after pos 1" in html
    assert "C→O @pos 1" in html
    assert "+C after pos 2" in html
    assert html.index("Product") < html.index("+Edit 1")
    assert html.index("+Edit 2") < html.index("Target")


def test_first_step_navigation_has_one_description_per_link():
    html = first_step._build_index_html(
        "unused", [7], [0.0, 0.1], ["C C"], ["C O"],
    )
    assert html.count("Ex#7 t=") == 2
    assert html.count("C C → C O") == 2
