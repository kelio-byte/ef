"""Optional, read-only trajectory diagnostics for sampler experiments."""

import math

import torch

from edit_flows.sampling.ops import (
    legal_token_log_probs,
)
from edit_flows.sampling.branch_sampler_helpers import _total_edit_hazard
from edit_flows.utils.tokens import PAD_TOKEN


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


class TrajectoryDiagnostics:
    """Collect legal hazards and state-aligned intensity changes without edits."""

    def __init__(self, tracked_trajectories=16):
        if tracked_trajectories < 1:
            raise ValueError("tracked_trajectories must be positive")
        self.tracked_trajectories = tracked_trajectories
        self.device = None
        self.buckets = [self._new_bucket() for _ in range(4)]
        self.history = {}

    @staticmethod
    def _new_bucket():
        return {
            "hazard_samples": [],
            "event_counts": [],
            "event_observations": 0,
            "unchanged_count": 0,
            "trajectory_steps": 0,
            "intensity_changes": {gap: [] for gap in (1, 2, 4)},
        }

    def begin_batch(self):
        """Keep aggregate results while preventing histories crossing batches."""
        self.history.clear()

    @staticmethod
    def _quantiles(chunks):
        chunks = [chunk for chunk in chunks if chunk.numel()]
        if not chunks:
            return {"p50": None, "p90": None, "count": 0}
        values = torch.cat([chunk.reshape(-1) for chunk in chunks]).float()
        quantiles = torch.quantile(
            values,
            torch.tensor([0.50, 0.90], device=values.device),
        ).detach().cpu().tolist()
        return {"p50": quantiles[0], "p90": quantiles[1], "count": values.numel()}

    @torch.inference_mode()
    def record_step(
        self,
        *,
        step,
        parent_ids,
        parent_keys,
        selected_keys,
        x_batch,
        log_rates,
        log_ins_probs,
        log_sub_probs,
        actions,
        n_children,
    ):
        bucket_index = min(step // 25, 3)
        bucket = self.buckets[bucket_index]
        self.device = x_batch.device
        parent_count = len(parent_ids)
        if parent_count == 0:
            return

        # The checkpoint emits finite log-softmax values at non-padding rows;
        # legal token renormalization therefore has support at each legal edit
        # position. These position masks match the sampler's INS/SUB/DEL sites.
        hazard = _total_edit_hazard(x_batch, log_rates).squeeze(-1)
        # A rotating systematic sample bounds host/device memory while covering
        # every row over four adjacent steps.
        hazard_sample = hazard[step % 4 :: 4].detach()
        if hazard_sample.numel():
            bucket["hazard_samples"].append(hazard_sample)

        event = (
            actions["ins_mask"] | actions["sub_mask"] | actions["del_mask"]
        ).any(dim=1)
        no_event = (~event).reshape(parent_count, n_children).all(dim=1)
        bucket["event_counts"].append(no_event.sum().detach())
        bucket["event_observations"] += parent_count

        unchanged = [
            previous == selected
            for previous, selected in zip(parent_keys, selected_keys)
        ]
        bucket["unchanged_count"] += sum(unchanged)
        bucket["trajectory_steps"] += parent_count

        n_tracked = min(parent_count, self.tracked_trajectories)
        if n_tracked == 1:
            rows = [0]
        else:
            rows = [
                round(index * (parent_count - 1) / (n_tracked - 1))
                for index in range(n_tracked)
            ]
        tracked = torch.tensor(rows, device=x_batch.device, dtype=torch.long)
        tracked_x = x_batch.index_select(0, tracked)
        tracked_rates = torch.exp(log_rates.index_select(0, tracked))
        tracked_ins_positions, tracked_sub_positions = edit_position_masks(tracked_x)
        ins_legal, ins_norm = legal_token_log_probs(
            log_ins_probs.index_select(0, tracked)
        )
        sub_legal, sub_norm = legal_token_log_probs(
            log_sub_probs.index_select(0, tracked),
            current_tokens=tracked_x,
        )
        tracked_lambda_ins = tracked_rates[:, :, 0] * (
            tracked_ins_positions & torch.isfinite(ins_norm)
        ).to(tracked_rates.dtype)
        tracked_lambda_sub = tracked_rates[:, :, 1] * (
            tracked_sub_positions & torch.isfinite(sub_norm)
        ).to(tracked_rates.dtype)
        tracked_lambda_del = tracked_rates[:, :, 2] * tracked_sub_positions
        fields = (
            tracked_lambda_ins.unsqueeze(-1) * torch.exp(ins_legal),
            tracked_lambda_sub.unsqueeze(-1) * torch.exp(sub_legal),
            tracked_lambda_del,
        )

        for local, row in enumerate(rows):
            branch_id = parent_ids[row]
            key = parent_keys[row]
            current = tuple(field[local].detach().clone() for field in fields)
            history = self.history.get(branch_id, [])
            if history and history[-1][1] != key:
                history = []
            for previous_step, _, previous in history:
                gap = step - previous_step
                if gap not in (1, 2, 4):
                    continue
                differences = []
                magnitudes = []
                for new, old in zip(current, previous):
                    if new.shape[0] < old.shape[0]:
                        padding = (
                            (0, old.shape[0] - new.shape[0])
                            if new.ndim == 1
                            else (0, 0, 0, old.shape[0] - new.shape[0])
                        )
                        new = torch.nn.functional.pad(
                            new, padding
                        )
                    elif old.shape[0] < new.shape[0]:
                        padding = (
                            (0, new.shape[0] - old.shape[0])
                            if old.ndim == 1
                            else (0, 0, 0, new.shape[0] - old.shape[0])
                        )
                        old = torch.nn.functional.pad(
                            old, padding
                        )
                    differences.append((new - old).abs().sum())
                    magnitudes.append(old.abs().sum())
                relative_change = torch.stack(differences).sum() / (
                    torch.stack(magnitudes).sum() + 1e-8
                )
                bucket["intensity_changes"][gap].append(relative_change.detach())
            history.append((step, key, current))
            self.history[branch_id] = history[-5:]

    def report(self):
        result = []
        for index, bucket in enumerate(self.buckets):
            event_count = (
                torch.stack(bucket["event_counts"]).sum().item()
                if bucket["event_counts"]
                else 0
            )
            error = {
                str(gap): self._quantiles(bucket["intensity_changes"][gap])
                for gap in (1, 2, 4)
            }
            event_observations = bucket["event_observations"]
            trajectory_steps = bucket["trajectory_steps"]
            result.append(
                {
                    "time_range": [index / 4, (index + 1) / 4],
                    "trajectory_steps": trajectory_steps,
                    "hazard": self._quantiles(bucket["hazard_samples"]),
                    "both_no_event_fraction": (
                        event_count / event_observations if event_observations else None
                    ),
                    "selected_unchanged_fraction": (
                        bucket["unchanged_count"] / trajectory_steps
                        if trajectory_steps
                        else None
                    ),
                    "ideal_nfe_savings_ceiling": bucket["unchanged_count"],
                    "ideal_nfe_savings_ceiling_fraction": (
                        bucket["unchanged_count"] / trajectory_steps
                        if trajectory_steps
                        else None
                    ),
                    "intensity_change": error,
                }
            )
        return result
