from typing import Dict, List, Optional

import torch
from torch import Tensor
from tqdm import tqdm

from edit_flows.core.rate_scale import apply_rate_parameterization, get_rate_scale
from edit_flows.core.scheduler import KappaScheduler
from edit_flows.sampling.ops import apply_ins_del_operations
from edit_flows.utils.tokens import PAD_TOKEN, BOS_TOKEN


def _event_probability(mu: Tensor, mode: str) -> Tensor:
    if mode == "poisson":
        return 1 - torch.exp(-mu)
    if mode == "linear":
        return torch.clamp(mu, min=0.0, max=1.0)
    raise ValueError(f"Unsupported event_prob_mode: {mode}")


def _sample_edit_actions(
    x_t: Tensor,
    log_rates: Tensor,
    log_ins_probs: Tensor,
    log_sub_probs: Tensor,
    adapt_h: Tensor,
    pad_token: int,
    event_prob_mode: str = "poisson",
) -> dict:
    device = x_t.device
    x_pad_mask = x_t == pad_token

    rates = torch.exp(log_rates)
    ins_probs = torch.exp(log_ins_probs)
    sub_probs = torch.exp(log_sub_probs)

    lambda_ins = rates[:, :, 0]
    lambda_sub = rates[:, :, 1]
    lambda_del = rates[:, :, 2]

    ins_prob = _event_probability(adapt_h * lambda_ins, event_prob_mode)
    del_sub_prob = _event_probability(
        adapt_h * (lambda_sub + lambda_del), event_prob_mode,
    )

    ins_mask = torch.rand_like(lambda_ins) < ins_prob
    del_sub_mask = torch.rand_like(lambda_sub) < del_sub_prob

    prob_del = torch.where(
        del_sub_mask,
        lambda_del / (lambda_sub + lambda_del + 1e-8),
        torch.zeros_like(lambda_del),
    )
    del_mask = torch.bernoulli(prob_del).bool()
    sub_mask = del_sub_mask & ~del_mask

    non_pad_mask = ~x_pad_mask
    ins_tokens = torch.full(
        ins_probs.shape[:2], pad_token, dtype=torch.long, device=device,
    )
    sub_tokens = torch.full(
        sub_probs.shape[:2], pad_token, dtype=torch.long, device=device,
    )

    if non_pad_mask.any():
        ins_sampled = torch.multinomial(
            ins_probs[non_pad_mask], num_samples=1, replacement=True,
        ).squeeze(-1)
        sub_sampled = torch.multinomial(
            sub_probs[non_pad_mask], num_samples=1, replacement=True,
        ).squeeze(-1)
        ins_tokens[non_pad_mask] = ins_sampled
        sub_tokens[non_pad_mask] = sub_sampled

    ins_mask = ins_mask & non_pad_mask
    del_mask = del_mask & non_pad_mask
    sub_mask = sub_mask & non_pad_mask
    return {
        "rates": rates,
        "ins_mask": ins_mask,
        "del_mask": del_mask,
        "sub_mask": sub_mask,
        "ins_tokens": ins_tokens,
        "sub_tokens": sub_tokens,
        "ins_probs": ins_probs,
        "sub_probs": sub_probs,
    }


def _first_true(mask: Tensor) -> int:
    idx = torch.nonzero(mask, as_tuple=False)
    return int(idx[0].item()) if idx.numel() > 0 else -1


def _extract_first_event_summary(
    sample_idx: int,
    x_t: Tensor,
    actions: dict,
    t: Tensor,
    oracle: Optional[dict] = None,
    pad_token: int = PAD_TOKEN,
) -> dict:
    ins_mask = actions["ins_mask"][sample_idx]
    del_mask = actions["del_mask"][sample_idx]
    sub_mask = actions["sub_mask"][sample_idx]
    any_mask = ins_mask | del_mask | sub_mask

    event_positions = torch.nonzero(any_mask, as_tuple=False).squeeze(-1).tolist()
    pos = event_positions[0] if event_positions else -1
    event_types = []
    event_tokens = []
    for event_pos in event_positions:
        if bool(sub_mask[event_pos]):
            event_types.append("sub")
            event_tokens.append(int(actions["sub_tokens"][sample_idx, event_pos].item()))
        elif bool(del_mask[event_pos]) and bool(ins_mask[event_pos]):
            event_types.append("replace")
            event_tokens.append(int(actions["ins_tokens"][sample_idx, event_pos].item()))
        elif bool(del_mask[event_pos]):
            event_types.append("del")
            event_tokens.append(-1)
        else:
            event_types.append("ins")
            event_tokens.append(int(actions["ins_tokens"][sample_idx, event_pos].item()))

    summary = {
        "sample_idx": sample_idx,
        "first_event_step_idx": -1,
        "first_event_t": float(t[sample_idx, 0].item()),
        "n_first_events": len(event_positions),
        "event_positions": event_positions,
        "anchor_pos": pos,
        "anchor_type": event_types[0] if event_types else None,
        "anchor_token": event_tokens[0] if event_tokens else None,
        "event_types": event_types,
        "event_tokens": event_tokens,
        "source_tokens": x_t[sample_idx].detach().cpu().tolist(),
        "center_hit": False,
        "type_correct": False,
        "token_correct": False,
        "event_set_correct": False,
        "oracle_anchor_type": None,
        "oracle_anchor_token": None,
    }

    if oracle is None or pos < 0:
        return summary

    oracle_pos_mask = oracle["pos_mask"][sample_idx]
    oracle_ins_mask = oracle["ins_mask"][sample_idx]
    oracle_sub_mask = oracle["sub_mask"][sample_idx]
    oracle_del_mask = oracle["del_mask"][sample_idx]
    summary["center_hit"] = bool(oracle_pos_mask[pos].item())

    if bool(oracle_ins_mask[pos].item()):
        summary["oracle_anchor_type"] = "ins"
        summary["oracle_anchor_token"] = int(oracle["ins_token"][sample_idx, pos].item())
    elif bool(oracle_sub_mask[pos].item()):
        summary["oracle_anchor_type"] = "sub"
        summary["oracle_anchor_token"] = int(oracle["sub_token"][sample_idx, pos].item())
    elif bool(oracle_del_mask[pos].item()):
        summary["oracle_anchor_type"] = "del"
        summary["oracle_anchor_token"] = -1

    summary["type_correct"] = summary["anchor_type"] == summary["oracle_anchor_type"]
    if summary["anchor_type"] in {"del"} and summary["type_correct"]:
        summary["token_correct"] = True
    elif summary["anchor_type"] in {"ins", "replace"} and summary["oracle_anchor_type"] == "ins":
        summary["token_correct"] = summary["anchor_token"] == summary["oracle_anchor_token"]
    elif summary["anchor_type"] == "sub" and summary["oracle_anchor_type"] == "sub":
        summary["token_correct"] = summary["anchor_token"] == summary["oracle_anchor_token"]

    oracle_positions = torch.nonzero(oracle_pos_mask, as_tuple=False).squeeze(-1).tolist()
    summary["event_set_correct"] = event_positions == oracle_positions
    return summary


def _override_with_anchor_event(
    sample_idx: int,
    actions: dict,
    anchor: Optional[dict],
) -> None:
    actions["ins_mask"][sample_idx].fill_(False)
    actions["del_mask"][sample_idx].fill_(False)
    actions["sub_mask"][sample_idx].fill_(False)
    if anchor is None or anchor.get("pos", -1) < 0:
        return

    pos = int(anchor["pos"])
    event_type = anchor["type"]
    token = int(anchor.get("token", PAD_TOKEN))
    if event_type == "ins":
        actions["ins_mask"][sample_idx, pos] = True
        actions["ins_tokens"][sample_idx, pos] = token
    elif event_type == "sub":
        actions["sub_mask"][sample_idx, pos] = True
        actions["sub_tokens"][sample_idx, pos] = token
    elif event_type == "del":
        actions["del_mask"][sample_idx, pos] = True
    elif event_type == "replace":
        actions["ins_mask"][sample_idx, pos] = True
        actions["del_mask"][sample_idx, pos] = True
        actions["ins_tokens"][sample_idx, pos] = token
    else:
        raise ValueError(f"Unsupported intervention event type: {event_type}")


def _select_oracle_anchor(
    oracle: dict,
    sample_idx: int,
) -> Optional[dict]:
    pos_mask = oracle["pos_mask"][sample_idx]
    positions = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1)
    if positions.numel() == 0:
        return None
    pos = int(positions[0].item())
    if bool(oracle["ins_mask"][sample_idx, pos].item()):
        return {
            "pos": pos,
            "type": "ins",
            "token": int(oracle["ins_token"][sample_idx, pos].item()),
        }
    if bool(oracle["sub_mask"][sample_idx, pos].item()):
        return {
            "pos": pos,
            "type": "sub",
            "token": int(oracle["sub_token"][sample_idx, pos].item()),
        }
    if bool(oracle["del_mask"][sample_idx, pos].item()):
        return {"pos": pos, "type": "del", "token": -1}
    return None


def _select_wrong_anchor(
    sample_idx: int,
    actions: dict,
    oracle: dict,
    x_t: Tensor,
    pad_token: int,
) -> Optional[dict]:
    rates = actions["rates"][sample_idx]
    pad_mask = x_t[sample_idx] == pad_token
    oracle_pos_mask = oracle["pos_mask"][sample_idx]

    pos_scores = rates.sum(dim=-1).clone()
    pos_scores[pad_mask] = float("-inf")
    if pos_scores.numel() > 0:
        pos_scores[0] = float("-inf")

    ranked_positions = torch.argsort(pos_scores, descending=True)
    for pos_tensor in ranked_positions:
        pos = int(pos_tensor.item())
        if pos_scores[pos].item() == float("-inf"):
            continue
        if bool(oracle_pos_mask[pos].item()):
            continue
        type_idx = int(torch.argmax(rates[pos]).item())
        if type_idx == 0:
            token = int(torch.argmax(actions["ins_probs"][sample_idx, pos]).item())
            return {"pos": pos, "type": "ins", "token": token}
        if type_idx == 1:
            token = int(torch.argmax(actions["sub_probs"][sample_idx, pos]).item())
            return {"pos": pos, "type": "sub", "token": token}
        return {"pos": pos, "type": "del", "token": -1}
    return None


def get_adaptive_h(
    h: float,
    t: Tensor,
    scheduler: KappaScheduler,
) -> Tensor:
    coeff = (1 - scheduler(t)) / scheduler.derivative(t)
    h_adapt = torch.minimum(h * torch.ones_like(t), coeff)
    return h_adapt


def _compute_model_time(
    t: Tensor,
    sample_scheduler: KappaScheduler,
    time_input: str,
    train_scheduler: Optional[KappaScheduler] = None,
) -> Tensor:
    """Compute the time value to pass to the model.

    When time_input == "kappa", the model expects kappa(t) regardless of scheduler.
    When time_input == "t", the model expects raw t.  If the sampling scheduler
    differs from the training scheduler, we map: compute kappa under the sampling
    scheduler, then invert under the training scheduler to get the equivalent t.
    """
    if time_input not in {"t", "kappa"}:
        raise ValueError(f"Unsupported time_input: {time_input}")

    if time_input == "kappa":
        return sample_scheduler(t)

    # time_input == "t"
    if train_scheduler is None or sample_scheduler.name == train_scheduler.name:
        return t

    kappa = sample_scheduler(t)
    return train_scheduler.inverse(kappa)


@torch.no_grad()
def sample_euler(
    model,
    x_0: Tensor,
    scheduler: KappaScheduler,
    n_steps: int = 100,
    max_seq_len: int = 512,
    pad_token: int = PAD_TOKEN,
    bos_token: int = BOS_TOKEN,
    record_trajectory: bool = False,
    verbose: bool = False,
    use_rate_reparam: bool = False,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
    time_input: str = "t",
    train_scheduler: Optional[KappaScheduler] = None,
    event_prob_mode: str = "poisson",
    record_first_events: bool = False,
    record_all_events: bool = False,
    x_1: Optional[Tensor] = None,
    vocab_size: Optional[int] = None,
    use_origin_mask: bool = False,
) -> tuple:
    from edit_flows.analysis.first_step import extract_oracle_event_set
    from edit_flows.sampling.oracle import compute_oracle_model_output

    device = next(model.parameters()).device
    batch_size = x_0.shape[0]

    x_t = x_0.to(device)
    if use_origin_mask:
        origin_mask = torch.ones_like(x_t, dtype=torch.bool, device=device)
    else:
        origin_mask = None
    t = torch.zeros(batch_size, 1, device=device)
    default_h = 1.0 / n_steps

    trajectory: List[Tensor] = []
    first_events: List[Optional[dict]] = [None for _ in range(batch_size)]
    all_events: List[List[dict]] = [[] for _ in range(batch_size)] if record_all_events else []
    if record_trajectory:
        trajectory.append(x_t.cpu().clone())

    if verbose:
        pbar = tqdm(total=n_steps, desc="Euler Sampling")
    while (t < 1.0).any():
        recorded_event_samples: List[int] = []
        x_pad_mask = x_t == pad_token

        t_model = _compute_model_time(t, scheduler, time_input, train_scheduler)

        log_rates, log_ins_probs, log_sub_probs = model(
            x_t, t_model, x_pad_mask, origin_mask=origin_mask,
        )

        # When use_rate_reparam=False, the model predicts raw rates v that
        # internally bake in k_train(t_model).  If we are sampling with a
        # different scheduler, rescale to the correct k_sample(t).
        if not use_rate_reparam and train_scheduler is not None and \
           scheduler.name != train_scheduler.name:
            k_sample = get_rate_scale(t, scheduler,
                                      clamp_kappa=clamp_kappa, clamp_max=clamp_max)
            k_train = get_rate_scale(t_model, train_scheduler,
                                     clamp_kappa=clamp_kappa, clamp_max=clamp_max)
            # clamp_min prevents log(0) when k_train=0 (e.g. cubic deriv=0 at t=0).
            # Use 1e-2 rather than 1e-12 to cap the correction factor at ~100x,
            # avoiding massive rate inflation at early t where the training
            # scheduler had near-zero weight.
            log_correction = torch.log(
                k_sample / k_train.clamp_min(1e-2)
            ).unsqueeze(1)
            log_rates = log_rates + log_correction

        # Snapshot raw rates before k(t) parameterization (for visualization)
        log_rates_raw = log_rates.detach().clone()

        log_rates = apply_rate_parameterization(
            log_rates, t, scheduler, use_rate_reparam=use_rate_reparam,
            clamp_kappa=clamp_kappa, clamp_max=clamp_max,
        )

        adapt_h = get_adaptive_h(default_h, t, scheduler)
        actions = _sample_edit_actions(
            x_t, log_rates, log_ins_probs, log_sub_probs, adapt_h,
            pad_token=pad_token, event_prob_mode=event_prob_mode,
        )

        done = (t >= 1.0).squeeze(-1)
        if done.any():
            actions["ins_mask"][done] = False
            actions["del_mask"][done] = False
            actions["sub_mask"][done] = False

        if record_first_events or record_all_events:
            oracle_out = None
            oracle = None
            if x_1 is not None and vocab_size is not None:
                oracle_out = compute_oracle_model_output(
                    x_t, x_1.to(device), t, scheduler, vocab_size,
                    pad_token=pad_token, bos_token=bos_token,
                )
                oracle = extract_oracle_event_set(
                    oracle_out[0], oracle_out[1], oracle_out[2], x_t,
                )

            event_mask = actions["ins_mask"] | actions["del_mask"] | actions["sub_mask"]

            if record_first_events:
                for sample_idx in range(batch_size):
                    if first_events[sample_idx] is None and bool(event_mask[sample_idx].any().item()):
                        summary = _extract_first_event_summary(
                            sample_idx=sample_idx,
                            x_t=x_t,
                            actions=actions,
                            t=t,
                            oracle=oracle,
                            pad_token=pad_token,
                        )
                        summary["first_event_step_idx"] = int((t[sample_idx, 0] * n_steps).item())
                        first_events[sample_idx] = summary

            if record_all_events:
                for sample_idx in range(batch_size):
                    if done[sample_idx]:
                        continue
                    if bool(event_mask[sample_idx].any().item()):
                        event = {
                            "step_idx": int((t[sample_idx, 0] * n_steps).item()),
                            "t": float(t[sample_idx, 0].item()),
                            "x_t": x_t[sample_idx].cpu().clone(),
                            "origin_mask": origin_mask[sample_idx].cpu().clone() if use_origin_mask else None,
                            "log_rates": log_rates[sample_idx].detach().cpu().clone(),
                            "log_rates_raw": log_rates_raw[sample_idx].detach().cpu().clone(),
                            "log_ins_probs": log_ins_probs[sample_idx].detach().cpu().clone(),
                            "log_sub_probs": log_sub_probs[sample_idx].detach().cpu().clone(),
                            "oracle_log_rates": oracle_out[0][sample_idx].detach().cpu().clone() if oracle_out is not None else None,
                            "oracle_log_ins_probs": oracle_out[1][sample_idx].detach().cpu().clone() if oracle_out is not None else None,
                            "oracle_log_sub_probs": oracle_out[2][sample_idx].detach().cpu().clone() if oracle_out is not None else None,
                            "actions": {
                                "ins_mask": actions["ins_mask"][sample_idx].cpu().clone(),
                                "del_mask": actions["del_mask"][sample_idx].cpu().clone(),
                                "sub_mask": actions["sub_mask"][sample_idx].cpu().clone(),
                                "ins_tokens": actions["ins_tokens"][sample_idx].cpu().clone(),
                                "sub_tokens": actions["sub_tokens"][sample_idx].cpu().clone(),
                            },
                        }
                        if oracle is not None:
                            event["oracle_event"] = {
                                "pos_mask": oracle["pos_mask"][sample_idx].cpu().clone(),
                                "ins_mask": oracle["ins_mask"][sample_idx].cpu().clone(),
                                "sub_mask": oracle["sub_mask"][sample_idx].cpu().clone(),
                                "del_mask": oracle["del_mask"][sample_idx].cpu().clone(),
                            }
                        all_events[sample_idx].append(event)
                        recorded_event_samples.append(sample_idx)

        x_t[actions["sub_mask"]] = actions["sub_tokens"][actions["sub_mask"]]
        if use_origin_mask:
            origin_markers = torch.where(
                x_pad_mask,
                torch.full_like(x_t, 2),
                origin_mask.long(),
            )
            origin_markers[actions["sub_mask"]] = 0
            origin_ins = torch.zeros_like(actions["ins_tokens"], dtype=torch.long)
            origin_markers = apply_ins_del_operations(
                origin_markers, actions["ins_mask"], actions["del_mask"], origin_ins,
                max_seq_len=max_seq_len, pad_token=2,
            )
            origin_mask = origin_markers == 1
        x_t = apply_ins_del_operations(
            x_t, actions["ins_mask"], actions["del_mask"], actions["ins_tokens"],
            max_seq_len=max_seq_len, pad_token=pad_token,
        )
        # Store the exact post-edit state for diagnostic reconstruction.  This
        # runs only when event recording is enabled and does not alter normal
        # sampling, RNG consumption, or model calls.
        for sample_idx in recorded_event_samples:
            all_events[sample_idx][-1]["x_next"] = (
                x_t[sample_idx].cpu().clone()
            )

        t = t + adapt_h
        if record_trajectory:
            trajectory.append(x_t.cpu().clone())
        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()
    if record_all_events:
        return x_t, trajectory, all_events
    if record_first_events:
        return x_t, trajectory, first_events
    return x_t, trajectory


@torch.no_grad()
def sample_euler_with_first_step_intervention(
    model,
    x_0: Tensor,
    x_1: Tensor,
    scheduler: KappaScheduler,
    vocab_size: int,
    mode: str = "normal",
    n_steps: int = 100,
    max_seq_len: int = 512,
    pad_token: int = PAD_TOKEN,
    bos_token: int = BOS_TOKEN,
    use_rate_reparam: bool = False,
    clamp_kappa: bool = False,
    clamp_max: float = 50.0,
    time_input: str = "t",
    train_scheduler: Optional[KappaScheduler] = None,
    event_prob_mode: str = "poisson",
    record_first_events: bool = False,
) -> tuple:
    from edit_flows.analysis.first_step import extract_oracle_event_set
    from edit_flows.sampling.oracle import compute_oracle_model_output

    if mode not in {"normal", "force_correct_first", "force_wrong_first"}:
        raise ValueError(f"Unsupported intervention mode: {mode}")

    device = next(model.parameters()).device
    batch_size = x_0.shape[0]
    x_t = x_0.to(device)
    x_1 = x_1.to(device)
    t = torch.zeros(batch_size, 1, device=device)
    default_h = 1.0 / n_steps

    first_events: List[Optional[dict]] = [None for _ in range(batch_size)]
    intervention_applied = torch.zeros(batch_size, dtype=torch.bool, device=device)

    while (t < 1.0).any():
        x_pad_mask = x_t == pad_token
        t_model = _compute_model_time(t, scheduler, time_input, train_scheduler)
        log_rates, log_ins_probs, log_sub_probs = model(x_t, t_model, x_pad_mask)

        if not use_rate_reparam and train_scheduler is not None and \
           scheduler.name != train_scheduler.name:
            k_sample = get_rate_scale(
                t, scheduler, clamp_kappa=clamp_kappa, clamp_max=clamp_max,
            )
            k_train = get_rate_scale(
                t_model, train_scheduler, clamp_kappa=clamp_kappa, clamp_max=clamp_max,
            )
            log_correction = torch.log(
                k_sample / k_train.clamp_min(1e-2)
            ).unsqueeze(1)
            log_rates = log_rates + log_correction

        log_rates = apply_rate_parameterization(
            log_rates, t, scheduler, use_rate_reparam=use_rate_reparam,
            clamp_kappa=clamp_kappa, clamp_max=clamp_max,
        )

        adapt_h = get_adaptive_h(default_h, t, scheduler)
        actions = _sample_edit_actions(
            x_t, log_rates, log_ins_probs, log_sub_probs, adapt_h,
            pad_token=pad_token, event_prob_mode=event_prob_mode,
        )

        oracle_out = compute_oracle_model_output(
            x_t, x_1, t, scheduler, vocab_size,
            pad_token=pad_token, bos_token=bos_token,
        )
        oracle = extract_oracle_event_set(
            oracle_out[0], oracle_out[1], oracle_out[2], x_t,
        )

        event_mask = actions["ins_mask"] | actions["del_mask"] | actions["sub_mask"]
        for sample_idx in range(batch_size):
            if intervention_applied[sample_idx]:
                continue
            if not bool(event_mask[sample_idx].any().item()):
                continue
            if mode == "force_correct_first":
                _override_with_anchor_event(
                    sample_idx, actions, _select_oracle_anchor(oracle, sample_idx),
                )
            elif mode == "force_wrong_first":
                _override_with_anchor_event(
                    sample_idx,
                    actions,
                    _select_wrong_anchor(sample_idx, actions, oracle, x_t, pad_token),
                )
            intervention_applied[sample_idx] = True

        event_mask = actions["ins_mask"] | actions["del_mask"] | actions["sub_mask"]
        if record_first_events:
            for sample_idx in range(batch_size):
                if first_events[sample_idx] is None and bool(event_mask[sample_idx].any().item()):
                    summary = _extract_first_event_summary(
                        sample_idx=sample_idx,
                        x_t=x_t,
                        actions=actions,
                        t=t,
                        oracle=oracle,
                        pad_token=pad_token,
                    )
                    summary["first_event_step_idx"] = int((t[sample_idx, 0] * n_steps).item())
                    summary["intervention_mode"] = mode
                    first_events[sample_idx] = summary

        x_t[actions["sub_mask"]] = actions["sub_tokens"][actions["sub_mask"]]
        x_t = apply_ins_del_operations(
            x_t, actions["ins_mask"], actions["del_mask"], actions["ins_tokens"],
            max_seq_len=max_seq_len, pad_token=pad_token,
        )
        t = t + adapt_h

    if record_first_events:
        return x_t, first_events
    return x_t, None


@torch.no_grad()
def sample_euler_oracle(
    x_0: Tensor,
    x_1: Tensor,
    scheduler: KappaScheduler,
    vocab_size: int,
    n_steps: int = 100,
    max_seq_len: int = 512,
    pad_token: int = PAD_TOKEN,
    bos_token: int = BOS_TOKEN,
    record_trajectory: bool = False,
    record_edit_distances: bool = False,
    verbose: bool = False,
    use_rate_reparam: bool = False,
    event_prob_mode: str = "poisson",
) -> tuple:
    """Euler sampling with oracle (optimal) rates instead of model predictions.

    At each step, dynamically aligns x_t with x_1, computes the theoretically
    optimal edit rates, and samples edits accordingly.

    Args:
        record_edit_distances: if True, also returns (ts_list, dists_list) where
            each is a list of (B,) tensors recording t and edit distance at
            each step (including step 0).

    Returns:
        (x_t, trajectory) if record_edit_distances=False, else
        (x_t, trajectory, (ts_list, dists_list))
    """
    from edit_flows.sampling.oracle import compute_oracle_model_output

    device = x_0.device
    batch_size = x_0.shape[0]

    x_t = x_0
    x_1 = x_1.to(device)
    t = torch.zeros(batch_size, 1, device=device)
    default_h = 1.0 / n_steps

    trajectory: List[Tensor] = []
    ts_list: List[Tensor] = []
    dists_list: List[Tensor] = []

    if record_edit_distances:
        ts_list.append(t.squeeze(-1).cpu().clone())
        # Compute initial edit distances from x_0 to x_1
        init_dists = _compute_batch_edit_dists(x_t, x_1, pad_token)
        dists_list.append(torch.tensor(init_dists, dtype=torch.float))

    if record_trajectory:
        trajectory.append(x_t.cpu().clone())

    if verbose:
        pbar = tqdm(total=n_steps, desc="Oracle Euler")
    while (t < 1.0).any():
        x_pad_mask = x_t == pad_token

        log_rates, log_ins_probs, log_sub_probs, edit_dists = \
            compute_oracle_model_output(
                x_t, x_1, t, scheduler, vocab_size,
                pad_token=pad_token, bos_token=bos_token,
            )

        if record_edit_distances:
            ts_list.append(t.squeeze(-1).cpu().clone())
            dists_list.append(torch.tensor(edit_dists, dtype=torch.float))

        rates = torch.exp(log_rates)
        ins_probs = torch.exp(log_ins_probs)
        sub_probs = torch.exp(log_sub_probs)

        lambda_ins = rates[:, :, 0]
        lambda_sub = rates[:, :, 1]
        lambda_del = rates[:, :, 2]

        adapt_h = get_adaptive_h(default_h, t, scheduler)

        ins_prob = _event_probability(adapt_h * lambda_ins, event_prob_mode)
        del_sub_prob = _event_probability(
            adapt_h * (lambda_sub + lambda_del), event_prob_mode,
        )

        ins_mask = torch.rand_like(lambda_ins) < ins_prob
        del_sub_mask = torch.rand_like(lambda_sub) < del_sub_prob

        done = (t >= 1.0).squeeze(-1)
        if done.any():
            ins_mask[done] = False
            del_sub_mask[done] = False

        prob_del = torch.where(
            del_sub_mask,
            lambda_del / (lambda_sub + lambda_del + 1e-8),
            torch.zeros_like(lambda_del),
        )
        del_mask = torch.bernoulli(prob_del).bool()
        sub_mask = del_sub_mask & ~del_mask

        non_pad_mask = ~x_pad_mask
        ins_tokens = torch.full(
            ins_probs.shape[:2], pad_token, dtype=torch.long, device=device,
        )
        sub_tokens = torch.full(
            sub_probs.shape[:2], pad_token, dtype=torch.long, device=device,
        )

        if non_pad_mask.any():
            ins_sampled = torch.multinomial(
                ins_probs[non_pad_mask], num_samples=1, replacement=True,
            ).squeeze(-1)
            sub_sampled = torch.multinomial(
                sub_probs[non_pad_mask], num_samples=1, replacement=True,
            ).squeeze(-1)
            ins_tokens[non_pad_mask] = ins_sampled
            sub_tokens[non_pad_mask] = sub_sampled

        x_t[sub_mask] = sub_tokens[sub_mask]
        x_t = apply_ins_del_operations(
            x_t, ins_mask, del_mask, ins_tokens,
            max_seq_len=max_seq_len, pad_token=pad_token,
        )

        t = t + adapt_h
        if record_trajectory:
            trajectory.append(x_t.cpu().clone())
        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()
    if record_edit_distances:
        final_dists = _compute_batch_edit_dists(x_t, x_1, pad_token)
        dists_list.append(torch.tensor(final_dists, dtype=torch.float))
        return x_t, trajectory, (ts_list, dists_list)
    return x_t, trajectory

def _compute_batch_edit_dists(
    x_t,
    x_1,
    pad_token=PAD_TOKEN,
):
    """Compute Levenshtein edit distance from each x_t to x_1."""
    from edit_flows.sampling.oracle import _align_pair
    B = x_t.shape[0]
    dists = []
    for b in range(B):
        xt_b = x_t[b][x_t[b] != pad_token]
        x1_b = x_1[b][x_1[b] != pad_token]
        _, _, dist = _align_pair(xt_b, x1_b)
        dists.append(dist)
    return dists
