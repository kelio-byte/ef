"""Compact trajectory diagnostics for the fragment-level correction study.

The production sampler remains responsible for generation.  This module only
interprets compact event traces and applies the same global-SMILES/RDKit
canonicalization used by the formal SPE evaluator.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from edit_flows.sampling.oracle import _align_pair
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN

try:
    from scripts.preprocessing.global_align import inverse_global_align
except ModuleNotFoundError:  # pragma: no cover - supports direct script usage
    from preprocessing.global_align import inverse_global_align

from rdkit import Chem
from rdkit import RDLogger


RDLogger.logger().setLevel(RDLogger.CRITICAL)


def decode_token_ids(
    token_ids: Sequence[int],
    id2token: Mapping[int, str],
) -> str:
    """Join model tokens into the compact SMILES string used by evaluation."""
    return "".join(
        id2token[int(token_id)]
        for token_id in token_ids
        if int(token_id) not in (PAD_TOKEN, BOS_TOKEN)
    )


def canonicalize_global_smiles(smiles: str) -> str | None:
    """Inverse global alignment and return an RDKit canonical SMILES."""
    compact = "".join(smiles.strip().split())
    if not compact:
        return None
    try:
        molecule = Chem.MolFromSmiles(inverse_global_align(compact))
    except Exception:
        return None
    if molecule is None:
        return None
    try:
        return Chem.MolToSmiles(molecule, isomericSmiles=True) or None
    except Exception:
        return None


def canonicalize_token_ids(
    token_ids: Sequence[int],
    id2token: Mapping[int, str],
) -> str | None:
    return canonicalize_global_smiles(decode_token_ids(token_ids, id2token))


def token_edit_distance(
    left: Sequence[int],
    right: Sequence[int],
) -> int:
    """Levenshtein distance after removing BOS/PAD structural positions."""
    left_ids = [
        int(value) for value in left
        if int(value) not in (PAD_TOKEN, BOS_TOKEN)
    ]
    right_ids = [
        int(value) for value in right
        if int(value) not in (PAD_TOKEN, BOS_TOKEN)
    ]
    _, _, distance = _align_pair(
        torch.tensor(left_ids, dtype=torch.long),
        torch.tensor(right_ids, dtype=torch.long),
    )
    return int(distance)


def _oracle_rows_by_position(event: Mapping) -> dict[int, Mapping]:
    return {
        int(row["position"]): row
        for row in event.get("oracle", [])
    }


def action_is_oracle_consistent(
    action: Mapping,
    oracle_row: Mapping | None,
) -> bool:
    """Check position, operation type, and completion token together."""
    if oracle_row is None:
        return False
    if int(action["position"]) != int(oracle_row["position"]):
        return False
    action_type = str(action["type"])
    oracle_types = set(oracle_row.get("types", []))
    token = action.get("token")
    if action_type == "ins":
        return "ins" in oracle_types and int(token) in set(
            oracle_row.get("ins_token_support", [])
        )
    if action_type == "sub":
        return "sub" in oracle_types and int(token) in set(
            oracle_row.get("sub_token_support", [])
        )
    if action_type == "del":
        return "del" in oracle_types
    if action_type == "replace":
        return (
            {"ins", "del"}.issubset(oracle_types)
            and int(token) in set(oracle_row.get("ins_token_support", []))
        )
    return False


def event_is_fully_oracle_consistent(event: Mapping) -> bool:
    """Return whether every sampled action in an event is oracle-supported."""
    actions = list(event.get("actions", []))
    if not actions or not bool(event.get("oracle_available", False)):
        return False
    oracle_rows = _oracle_rows_by_position(event)
    return all(
        action_is_oracle_consistent(
            action,
            oracle_rows.get(int(action["position"])),
        )
        for action in actions
    )


def classify_event_progress(
    event: Mapping,
    event_summary: Mapping,
) -> dict:
    """Classify local progress without choosing one edit-order alignment.

    The oracle-alignment label is useful for diagnostics, but a single
    Levenshtein traceback can reject an equally valid alternative order.  The
    order-invariant check below only asks how much the *actual* event changes
    token edit distance.  For an event containing ``k`` effective operations,
    a decrease of exactly ``k`` means that the event is compatible with at
    least one shortest path.  Partial decrease, no change, and increase are
    reported separately rather than collapsed into "wrong".
    """
    actions = list(event.get("actions", []))
    n_actions = int(event_summary.get("n_actions", len(actions)))
    before = int(event_summary.get("edit_distance_before", 0))
    after = int(event_summary.get("edit_distance_after", before))
    distance_delta = before - after
    if n_actions <= 0:
        category = "no_event"
    elif distance_delta == n_actions:
        category = "full_progress"
    elif distance_delta > 0:
        category = "partial_progress"
    elif distance_delta == 0:
        category = "neutral"
    else:
        category = "harmful"
    return {
        "category": category,
        "n_actions": n_actions,
        "edit_distance_before": before,
        "edit_distance_after": after,
        "distance_delta": distance_delta,
    }


def first_event_index(events: Sequence[Mapping]) -> int | None:
    for index, event in enumerate(events):
        if event.get("actions"):
            return index
    return None


def summarize_trace(
    events: Sequence[Mapping],
    final_ids: Sequence[int],
    target_ids: Sequence[int],
    id2token: Mapping[int, str],
) -> dict:
    """Compute the P0/P1 compact statistics for one sampled trajectory."""
    first_index = first_event_index(events)
    target_canonical = canonicalize_token_ids(target_ids, id2token)
    final_canonical = canonicalize_token_ids(final_ids, id2token)
    final_hit = bool(
        target_canonical is not None
        and final_canonical is not None
        and target_canonical == final_canonical
    )

    event_rows = []
    for event in events:
        before = event.get("x_t", [])
        after = event.get("x_next", [])
        row = {
            "step_idx": int(event.get("step_idx", -1)),
            "t": float(event.get("t", 0.0)),
            "n_actions": len(event.get("actions", [])),
            "n_ins": sum(a.get("type") == "ins" for a in event.get("actions", [])),
            "n_sub": sum(a.get("type") == "sub" for a in event.get("actions", [])),
            "n_del": sum(a.get("type") == "del" for a in event.get("actions", [])),
            "edit_distance_before": token_edit_distance(before, target_ids),
            "edit_distance_after": token_edit_distance(after, target_ids),
            "fully_oracle_consistent": event_is_fully_oracle_consistent(event),
        }
        event_rows.append(row)

    first_event = events[first_index] if first_index is not None else None
    first_consistent = (
        event_is_fully_oracle_consistent(first_event)
        if first_event is not None else False
    )
    later_events = (
        event_rows[first_index + 1:]
        if first_index is not None else []
    )
    later_sub_del = sum(
        row["n_sub"] + row["n_del"] for row in later_events
    )
    distance_rebound = sum(
        row["edit_distance_after"] < row["edit_distance_before"]
        for row in later_events
    )

    return {
        "first_event_found": first_index is not None,
        "first_event_index": first_index,
        "first_event_fully_oracle_consistent": first_consistent,
        "first_event_off_oracle": bool(first_event is not None and not first_consistent),
        "n_events": len(event_rows),
        "later_sub_del_actions": int(later_sub_del),
        "later_distance_decrease_events": int(distance_rebound),
        "final_canonical": final_canonical,
        "target_canonical": target_canonical,
        "final_valid": final_canonical is not None,
        "final_hit": final_hit,
        "events": event_rows,
    }


def aggregate_trace_summaries(rows: Sequence[Mapping]) -> dict:
    """Aggregate per-trajectory summaries without treating paths as reactions."""
    rows = list(rows)
    first_rows = [row for row in rows if row.get("first_event_found")]
    off_rows = [row for row in first_rows if row.get("first_event_off_oracle")]
    clean_rows = [
        row for row in first_rows
        if row.get("first_event_fully_oracle_consistent")
    ]

    def mean(values: Sequence[bool | int | float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "n_trajectories": len(rows),
        "n_first_event": len(first_rows),
        "first_off_oracle_rate": mean(
            [row.get("first_event_off_oracle", False) for row in first_rows]
        ),
        "first_fully_oracle_rate": mean(
            [row.get("first_event_fully_oracle_consistent", False) for row in first_rows]
        ),
        "clean_path_success": mean([row.get("final_hit", False) for row in clean_rows]),
        "natural_recovery_rate": mean([row.get("final_hit", False) for row in off_rows]),
        "final_hit_rate": mean([row.get("final_hit", False) for row in rows]),
        "final_valid_rate": mean([row.get("final_valid", False) for row in rows]),
        "mean_later_sub_del_after_first_off": mean(
            [row.get("later_sub_del_actions", 0) for row in off_rows]
        ),
        "mean_later_distance_decrease_events_after_first_off": mean(
            [row.get("later_distance_decrease_events", 0) for row in off_rows]
        ),
    }
