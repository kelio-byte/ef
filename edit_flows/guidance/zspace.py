"""Explicit mappings between aligned Z-space transitions and Edit actions.

The current Edit Flows sampler operates on variable-length X-space sequences,
while the DGM theorem is easiest to state on fixed-coordinate states.  This
module is deliberately an isolated DG-0 artifact: it does not change Euler or
Euler-Beam.  It makes the boundary conditions and non-uniqueness of the
mapping executable so a future sampler cannot silently claim an exact
Z-space posterior when an X-space insertion has multiple aligned realizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor

from edit_flows.utils.tokens import BOS_TOKEN, GAP_TOKEN, PAD_TOKEN


Operation = Literal["insert", "substitute", "delete"]


class ZSpaceMappingError(ValueError):
    """Raised when a fixed-coordinate Z transition is not a valid edit."""


def compose_edit_action_log_weights(
    log_rates: Tensor,
    log_insert_probs: Tensor,
    log_substitute_probs: Tensor,
) -> Tensor:
    """Combine Edit Flows outputs into ``position × operation/token`` weights.

    ``log_rates`` must already use the caller's desired time/rate
    parameterization.  The output channel order is exactly the one used by
    :func:`edit_flows.core.z_space.make_ut_mask_from_z`: insertion tokens,
    substitution tokens, then one deletion channel.  No normalization or
    guidance is applied here, so this is safe to use for diagnostics and for a
    future fixed-coordinate sampler.
    """
    if log_rates.ndim != 3 or log_rates.shape[-1] != 3:
        raise ValueError("log_rates must have shape [batch, length, 3]")
    if log_insert_probs.ndim != 3 or log_substitute_probs.ndim != 3:
        raise ValueError("token log-probabilities must have shape [batch, length, vocab]")
    if log_insert_probs.shape != log_substitute_probs.shape:
        raise ValueError("insert and substitute shapes must match")
    if log_rates.shape[:2] != log_insert_probs.shape[:2]:
        raise ValueError("rate and token-probability batch/length dimensions differ")
    insert = log_rates[..., 0:1] + log_insert_probs
    substitute = log_rates[..., 1:2] + log_substitute_probs
    delete = log_rates[..., 2:3]
    return torch.cat((insert, substitute, delete), dim=-1)


@dataclass(frozen=True)
class ZEdit:
    """A one-coordinate Z transition and its X-space edit interpretation.

    ``x_position`` follows ``apply_ins_del_operations``: substitution/deletion
    address an existing X token, while insertion addresses the anchor token
    *after which* the new token is inserted.  ``ambiguous`` is true when a
    contiguous GAP run contains multiple coordinates representing the same X
    insertion boundary.
    """

    operation: Operation
    z_position: int
    x_position: int
    token: int
    old_token: int
    new_token: int
    ambiguous: bool

    def action_channel(self, vocab_size: int) -> int:
        """Return the Edit Flows operation channel used by ``make_ut_mask``."""
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.operation == "insert":
            return self.token
        if self.operation == "substitute":
            return self.token + vocab_size
        return 2 * vocab_size


def _as_state(value: Tensor | Sequence[int], name: str) -> Tensor:
    if isinstance(value, Tensor):
        state = value
    else:
        state = torch.as_tensor(value, dtype=torch.long)
    if state.ndim != 1:
        raise ZSpaceMappingError(
            f"{name} must be a rank-1 state, got shape {tuple(state.shape)}"
        )
    if state.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        state = state.to(dtype=torch.long)
    return state


def validate_z_state(
    z_state: Tensor | Sequence[int],
    *,
    vocab_size: int | None = None,
    bos_token: int = BOS_TOKEN,
    gap_token: int = GAP_TOKEN,
    pad_token: int = PAD_TOKEN,
) -> int:
    """Validate one padded aligned state and return its active length.

    PAD is storage-only and must form one trailing suffix.  BOS is structural
    and must be present at column zero.  GAP is allowed only in the active
    prefix, where it represents a fixed Z coordinate rather than an X token.
    """
    z = _as_state(z_state, "z_state")
    if z.numel() == 0:
        raise ZSpaceMappingError("z_state must contain BOS_TOKEN")
    pad_positions = torch.nonzero(z == pad_token, as_tuple=False).flatten()
    active_len = int(pad_positions[0].item()) if pad_positions.numel() else z.numel()
    if active_len < 1 or int(z[0].item()) != bos_token:
        raise ZSpaceMappingError("z_state must begin with BOS_TOKEN")
    if pad_positions.numel() and (z[active_len:] != pad_token).any():
        raise ZSpaceMappingError("PAD_TOKEN must form a trailing suffix")
    if vocab_size is not None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        active = z[:active_len]
        if ((active < 0) | (active >= vocab_size)).any():
            raise ZSpaceMappingError(
                f"active Z tokens must lie in [0, {vocab_size})"
            )
    return active_len


def _transition_pair(
    z_old: Tensor | Sequence[int],
    z_new: Tensor | Sequence[int],
) -> tuple[Tensor, Tensor]:
    old = _as_state(z_old, "z_old")
    new = _as_state(z_new, "z_new")
    if old.shape != new.shape:
        raise ZSpaceMappingError(
            f"z_old and z_new must have equal shape, got {tuple(old.shape)} "
            f"and {tuple(new.shape)}"
        )
    return old, new


def z_transition_to_edit(
    z_old: Tensor | Sequence[int],
    z_new: Tensor | Sequence[int],
    *,
    vocab_size: int | None = None,
    bos_token: int = BOS_TOKEN,
    gap_token: int = GAP_TOKEN,
    pad_token: int = PAD_TOKEN,
    require_unique: bool = False,
) -> ZEdit:
    """Convert one valid one-coordinate Z transition into an X edit.

    The transition must preserve the fixed aligned support and change exactly
    one active coordinate.  A GAP→token change is an insertion, token→GAP a
    deletion, and token→token a substitution.  In a contiguous GAP run, the
    insertion has multiple equivalent aligned representations; callers that
    require a bijection can set ``require_unique=True``.
    """
    old, new = _transition_pair(z_old, z_new)
    old_len = validate_z_state(
        old, vocab_size=vocab_size, bos_token=bos_token,
        gap_token=gap_token, pad_token=pad_token,
    )
    new_len = validate_z_state(
        new, vocab_size=vocab_size, bos_token=bos_token,
        gap_token=gap_token, pad_token=pad_token,
    )
    if old_len != new_len:
        raise ZSpaceMappingError("a fixed-coordinate transition cannot change PAD support")
    diff = torch.nonzero(old != new, as_tuple=False).flatten()
    if diff.numel() != 1:
        raise ZSpaceMappingError(
            "a Z transition must change exactly one coordinate; "
            f"got {int(diff.numel())}"
        )
    z_position = int(diff.item())
    old_token = int(old[z_position].item())
    new_token = int(new[z_position].item())
    if z_position == 0 or old_token == bos_token or new_token == bos_token:
        raise ZSpaceMappingError("BOS_TOKEN is structural and cannot be edited")
    if old_token == pad_token or new_token == pad_token:
        raise ZSpaceMappingError("PAD_TOKEN is storage-only and cannot be edited")
    if old_token == gap_token and new_token != gap_token:
        operation: Operation = "insert"
        token = new_token
    elif old_token != gap_token and new_token == gap_token:
        operation = "delete"
        token = -1
    elif old_token != gap_token and new_token != gap_token:
        operation = "substitute"
        token = new_token
    else:
        raise ZSpaceMappingError("GAP_TOKEN→GAP_TOKEN is not an edit")

    before = old[:z_position]
    n_x_before = int(((before != gap_token) & (before != pad_token)).sum().item())
    if operation == "insert":
        # X insertions are anchored after an existing token.  BOS anchors the
        # first molecule token, hence the -1 offset with a lower bound of 0.
        x_position = max(n_x_before - 1, 0)
        left_gap = z_position > 1 and int(old[z_position - 1].item()) == gap_token
        right_gap = (
            z_position + 1 < old_len
            and int(old[z_position + 1].item()) == gap_token
        )
        ambiguous = left_gap or right_gap
    else:
        x_position = n_x_before
        ambiguous = False

    if require_unique and ambiguous:
        raise ZSpaceMappingError(
            "insertion lies in a contiguous GAP run and has no unique X mapping"
        )
    return ZEdit(
        operation=operation,
        z_position=z_position,
        x_position=x_position,
        token=token,
        old_token=old_token,
        new_token=new_token,
        ambiguous=ambiguous,
    )


def apply_z_transition(
    z_old: Tensor,
    z_new: Tensor,
    *,
    require_unique: bool = False,
    **kwargs,
) -> Tensor:
    """Return ``z_old`` after validating and applying one Z transition."""
    edit = z_transition_to_edit(
        z_old, z_new, require_unique=require_unique, **kwargs,
    )
    result = z_old.clone()
    result[edit.z_position] = edit.new_token
    return result


def edit_to_z_candidates(
    z_state: Tensor | Sequence[int],
    operation: Operation,
    x_position: int,
    token: int = -1,
    *,
    vocab_size: int | None = None,
    bos_token: int = BOS_TOKEN,
    gap_token: int = GAP_TOKEN,
    pad_token: int = PAD_TOKEN,
) -> list[ZEdit]:
    """Return all aligned Z transitions implementing one X-space edit.

    A list rather than one result is intentional: multiple entries are the
    concrete witness that variable-length insertion is not always bijective.
    ``token`` is required for insert/substitute and ignored for delete.
    """
    if operation not in {"insert", "substitute", "delete"}:
        raise ValueError(f"unsupported operation: {operation}")
    if x_position < 0:
        raise ValueError("x_position must be non-negative")
    z = _as_state(z_state, "z_state")
    active_len = validate_z_state(
        z, vocab_size=vocab_size, bos_token=bos_token,
        gap_token=gap_token, pad_token=pad_token,
    )
    if operation in {"insert", "substitute"}:
        if token < 0 or (vocab_size is not None and token >= vocab_size):
            raise ValueError("token is outside the configured vocabulary")
        if token in {bos_token, gap_token, pad_token}:
            raise ZSpaceMappingError(
                "insert/substitute token must be an ordinary molecule token"
            )

    candidates: list[ZEdit] = []
    n_x_before = 0
    for pos in range(active_len):
        old_token = int(z[pos].item())
        if old_token == gap_token:
            if operation == "insert" and max(n_x_before - 1, 0) == x_position:
                z_next = z.clone()
                z_next[pos] = token
                candidates.append(
                    z_transition_to_edit(
                        z, z_next, vocab_size=vocab_size, bos_token=bos_token,
                        gap_token=gap_token, pad_token=pad_token,
                    )
                )
            continue
        if pos == 0:
            n_x_before += 1  # BOS is an X anchor but never editable.
            continue
        current_x_position = n_x_before
        if current_x_position == x_position:
            if operation == "delete":
                z_next = z.clone()
                z_next[pos] = gap_token
                candidates.append(
                    z_transition_to_edit(
                        z, z_next, vocab_size=vocab_size, bos_token=bos_token,
                        gap_token=gap_token, pad_token=pad_token,
                    )
                )
            elif operation == "substitute" and token != old_token:
                z_next = z.clone()
                z_next[pos] = token
                candidates.append(
                    z_transition_to_edit(
                        z, z_next, vocab_size=vocab_size, bos_token=bos_token,
                        gap_token=gap_token, pad_token=pad_token,
                    )
                )
        n_x_before += 1
    return candidates
