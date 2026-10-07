"""Action-level supervision targets for the first guidance adapter.

The guidance network emits one positive weight per edit action.  A scalar
terminal reward broadcast to every action cannot identify a preferred child,
so this module derives sparse action targets from the sampled intermediate
state and its terminal sample using the same optimal alignment primitive as
the Edit Flows training path.  This remains an action-level approximation; the
strict fixed-coordinate Z-space mapping is reserved for DGM stage 8.
"""

from __future__ import annotations

import torch
from torch import Tensor

from edit_flows.core.alignment import opt_align_xs_to_zs
from edit_flows.utils.tokens import GAP_TOKEN, PAD_TOKEN


def build_action_target_masks(
    state_tokens: Tensor,
    terminal_tokens: Tensor,
    *,
    vocab_size: int,
    pad_token: int = PAD_TOKEN,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build sparse insert/substitute/delete masks in current X coordinates.

    Inputs include BOS and use ``pad_token`` for trailing storage padding.  The
    returned masks have shapes ``[B, L_state, V]``, ``[B, L_state, V]`` and
    ``[B, L_state, 1]``.  An insertion at a gap is attached to the preceding
    current token, matching ``apply_ins_del_operations``'s “insert after
    position” convention.  Multiple aligned gaps at one position set multiple
    insert-token entries.
    """
    if state_tokens.ndim != 2 or terminal_tokens.ndim != 2:
        raise ValueError("state_tokens and terminal_tokens must be rank-2")
    if state_tokens.shape[0] != terminal_tokens.shape[0]:
        raise ValueError("state and terminal batches must have equal size")
    if state_tokens.shape[1] < 1 or terminal_tokens.shape[1] < 1:
        raise ValueError("state and terminal sequences must be non-empty")
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")

    batch_size, state_length = state_tokens.shape
    insert_mask = torch.zeros(
        (batch_size, state_length, vocab_size),
        dtype=torch.bool, device=state_tokens.device,
    )
    substitute_mask = torch.zeros_like(insert_mask)
    delete_mask = torch.zeros(
        (batch_size, state_length, 1),
        dtype=torch.bool, device=state_tokens.device,
    )
    aligned_state, aligned_terminal = opt_align_xs_to_zs(
        state_tokens.to(dtype=torch.long), terminal_tokens.to(dtype=torch.long),
    )

    for row in range(batch_size):
        x_position = 0
        z_length = int(
            ((aligned_state[row] != pad_token) |
             (aligned_terminal[row] != pad_token)).sum().item()
        )
        for z_position in range(z_length):
            source = int(aligned_state[row, z_position].item())
            target = int(aligned_terminal[row, z_position].item())
            source_is_gap = source == GAP_TOKEN
            target_is_gap = target == GAP_TOKEN
            source_is_pad = source == pad_token
            target_is_pad = target == pad_token

            if source_is_pad:
                continue
            if source_is_gap:
                if not target_is_gap and not target_is_pad:
                    if not 0 <= target < vocab_size:
                        raise ValueError(
                            f"terminal token {target} outside vocab_size={vocab_size}"
                        )
                    # apply_ins_del_operations inserts after the current
                    # position, so a gap before the next token belongs after
                    # the preceding current token.  Clamp to the last current
                    # slot for a leading gap.
                    insert_position = max(x_position - 1, 0)
                    if insert_position < state_length:
                        insert_mask[row, insert_position, target] = True
                continue

            if x_position >= state_length:
                raise RuntimeError("alignment produced an out-of-range X position")
            if target_is_gap:
                delete_mask[row, x_position, 0] = True
            elif not target_is_pad and target != source:
                if not 0 <= target < vocab_size:
                    raise ValueError(
                        f"terminal token {target} outside vocab_size={vocab_size}"
                    )
                substitute_mask[row, x_position, target] = True
            x_position += 1

    return insert_mask, substitute_mask, delete_mask


def make_action_reward_targets(
    insert_mask: Tensor,
    substitute_mask: Tensor,
    delete_mask: Tensor,
    reward: Tensor,
    *,
    background: float = 1e-4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert sparse action masks and scalar rewards into positive targets."""
    if background <= 0 or not torch.isfinite(torch.tensor(background)):
        raise ValueError("background must be finite and positive")
    if insert_mask.ndim != 3 or substitute_mask.shape != insert_mask.shape:
        raise ValueError("insert/substitute masks must have equal rank-3 shapes")
    if delete_mask.ndim != 3 or delete_mask.shape[:2] != insert_mask.shape[:2]:
        raise ValueError("delete mask must match the first two mask dimensions")
    reward = reward.to(device=insert_mask.device, dtype=torch.float32)
    if reward.ndim != 1 or reward.shape[0] != insert_mask.shape[0]:
        raise ValueError("reward must have one scalar per batch row")
    if not torch.isfinite(reward).all() or (reward < 0).any():
        raise ValueError("reward must contain finite non-negative values")

    def targets(mask: Tensor) -> Tensor:
        result = torch.full(
            mask.shape, float(background), dtype=torch.float32,
            device=mask.device,
        )
        return result + mask.to(dtype=result.dtype) * reward.reshape(
            reward.shape[0], *([1] * (mask.ndim - 1)),
        )

    return targets(insert_mask), targets(substitute_mask), targets(delete_mask)
