#!/usr/bin/env python
"""Generate shared-anchor guidance data with event-conditioned edit proposals.

Natural Euler numerical steps are mostly no-op at ``n_steps=100``.  This
generator instead uses the frozen base model's instantaneous valid atomic edit
rates to draw exactly one state-changing edit for every child at a shared
anchor.  Each child then resumes ordinary Euler at the normal adaptive end of
that numerical step and receives the same terminal reward pipeline as other
guidance data.

This is an offline proposal-data generator, not a replacement for ordinary
Euler sampling.  It never reads reaction targets and leaves the existing
natural shared-anchor generator untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from edit_flows.guidance.data import make_guidance_record, save_guidance_dataset
from edit_flows.guidance.rewards import retro_tokenized_validity_reward
from edit_flows.sampling.euler import (
    get_euler_step_times,
    sample_euler,
    sample_event_conditioned_euler_transition,
)
from edit_flows.utils.tokens import BOS_TOKEN, GAP_TOKEN, PAD_TOKEN, UNK_TOKEN

try:  # Works both as ``python scripts/foo.py`` and as a test import.
    from scripts.generate_guidance_data import (
        _decode_rows,
        _load_model,
        _make_initial_batch,
        _mix_seed,
        _read_original_products,
        _trim_pad,
    )
    from scripts.generate_shared_anchor_guidance import (
        resolve_anchor_steps,
        select_original_products,
        shared_anchor_group_index,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI execution
    from generate_guidance_data import (
        _decode_rows,
        _load_model,
        _make_initial_batch,
        _mix_seed,
        _read_original_products,
        _trim_pad,
    )
    from generate_shared_anchor_guidance import (
        resolve_anchor_steps,
        select_original_products,
        shared_anchor_group_index,
    )


_OPERATION_NAMES = ("insert", "substitute", "delete")
_FORBIDDEN_PROPOSAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, GAP_TOKEN, UNK_TOKEN)


def _set_torch_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _validate_event_actions(actions: dict[str, torch.Tensor]) -> None:
    """Defend the data contract before serializing any proposal record."""

    counts = (
        actions["ins_mask"].sum(dim=1)
        + actions["sub_mask"].sum(dim=1)
        + actions["del_mask"].sum(dim=1)
    )
    if not torch.equal(counts, torch.ones_like(counts)):
        raise RuntimeError(
            "event-conditioned proposal must contain exactly one action per row"
        )
    operation = actions["operation"]
    token = actions["token"]
    if ((operation < 0) | (operation >= len(_OPERATION_NAMES))).any():
        raise RuntimeError("event-conditioned proposal emitted an invalid operation")
    token_actions = operation < 2
    for forbidden in _FORBIDDEN_PROPOSAL_TOKENS:
        if (token[token_actions] == forbidden).any():
            raise RuntimeError("event-conditioned proposal emitted a structural token")


def generate(args: argparse.Namespace) -> dict:
    anchor_indices = resolve_anchor_steps(args)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {output}; pass --overwrite to replace it"
        )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    (
        checkpoint_data, cfg, model, token2id, id2token,
        sample_scheduler, train_scheduler, use_origin_mask,
    ) = _load_model(
        args.checkpoint, device, args.data_dir, args.vocab_file,
    )
    if use_origin_mask:
        raise ValueError(
            "event-conditioned shared-anchor generation currently requires "
            "use_origin_mask=False; the checkpoint must provide an origin-mask "
            "continuation state"
        )
    all_products = _read_original_products(args.products_file, args.augmentation)
    products, selection_start, selection_end = select_original_products(
        all_products,
        start_product=args.start_product,
        max_products=args.max_products,
    )
    unk_id = token2id.get("<unk>", UNK_TOKEN)
    product_ids = [
        [token2id.get(token, unk_id) for token in product.split()]
        for product in products
    ]
    step_times = get_euler_step_times(args.n_steps, sample_scheduler)
    anchor_specs: list[tuple[int, float]] = []
    for anchor_index in anchor_indices:
        if anchor_index >= len(step_times) - 1:
            raise RuntimeError(
                f"Euler produced only {len(step_times) - 1} steps, cannot use "
                f"anchor step {anchor_index}"
            )
        anchor_specs.append((anchor_index, float(step_times[anchor_index])))
    anchor_count = len(anchor_specs)

    records: list[dict] = []
    operation_counts = {name: 0 for name in _OPERATION_NAMES}
    started = time.perf_counter()
    total_batches = (len(products) + args.batch_size - 1) // args.batch_size
    for batch_index, batch_start in enumerate(
        range(0, len(products), args.batch_size), start=1,
    ):
        batch_products = products[batch_start:batch_start + args.batch_size]
        batch_ids = product_ids[batch_start:batch_start + args.batch_size]
        source_batch_start = selection_start + batch_start

        # One ordinary prefix creates the shared state.  Its RNG stream is
        # separate from both conditional proposals and terminal rollouts.
        _set_torch_seed(_mix_seed(args.seed, source_batch_start, 0), device)
        x0_cpu = _make_initial_batch(batch_ids, 1)
        _, trajectory = sample_euler(
            model,
            x0_cpu.to(device),
            sample_scheduler,
            n_steps=args.n_steps,
            max_seq_len=cfg["max_seq_len"],
            use_rate_reparam=cfg.get("use_rate_reparam", False),
            clamp_kappa=cfg.get("clamp_kappa", False),
            clamp_max=cfg.get("clamp_max", 50.0),
            time_input=cfg.get("time_input", "t"),
            train_scheduler=train_scheduler,
            record_trajectory=True,
            use_origin_mask=False,
        )
        if len(trajectory) - 1 != len(step_times) - 1:
            raise RuntimeError(
                "trajectory/time schedule length mismatch: "
                f"trajectory={len(trajectory) - 1}, schedule={len(step_times) - 1}"
            )

        for anchor_ordinal, (anchor_index, anchor_time) in enumerate(anchor_specs):
            anchor_state = trajectory[anchor_index].to(device)
            child_state = anchor_state.repeat_interleave(args.n_children, dim=0)

            proposal_seed = _mix_seed(
                args.seed, source_batch_start, anchor_index + 200003,
            )
            _set_torch_seed(proposal_seed, device)
            transition_states, rollout_start_times, actions = (
                sample_event_conditioned_euler_transition(
                    model,
                    child_state,
                    sample_scheduler,
                    n_steps=args.n_steps,
                    max_seq_len=cfg["max_seq_len"],
                    use_rate_reparam=cfg.get("use_rate_reparam", False),
                    clamp_kappa=cfg.get("clamp_kappa", False),
                    clamp_max=cfg.get("clamp_max", 50.0),
                    time_input=cfg.get("time_input", "t"),
                    train_scheduler=train_scheduler,
                    start_time=anchor_time,
                )
            )
            _validate_event_actions(actions)

            rollout_seed = _mix_seed(
                args.seed, source_batch_start, anchor_index + 300003,
            )
            _set_torch_seed(rollout_seed, device)
            terminal_states, _ = sample_euler(
                model,
                transition_states,
                sample_scheduler,
                n_steps=args.n_steps,
                max_seq_len=cfg["max_seq_len"],
                use_rate_reparam=cfg.get("use_rate_reparam", False),
                clamp_kappa=cfg.get("clamp_kappa", False),
                clamp_max=cfg.get("clamp_max", 50.0),
                time_input=cfg.get("time_input", "t"),
                train_scheduler=train_scheduler,
                use_origin_mask=False,
                start_time=rollout_start_times,
            )
            rewards = retro_tokenized_validity_reward(
                _decode_rows(terminal_states, id2token),
            )

            for product_offset, _ in enumerate(batch_products):
                product_index = source_batch_start + product_offset
                source_index = shared_anchor_group_index(
                    product_index, anchor_ordinal, anchor_count,
                )
                product_row = x0_cpu[product_offset]
                state_row = anchor_state[product_offset].detach().cpu()
                for child_index in range(args.n_children):
                    row_index = product_offset * args.n_children + child_index
                    operation_index = int(actions["operation"][row_index].item())
                    operation_name = _OPERATION_NAMES[operation_index]
                    operation_counts[operation_name] += 1
                    transition_tokens = _trim_pad(
                        transition_states[row_index].detach().cpu().tolist(),
                    )
                    if transition_tokens == _trim_pad(state_row.tolist()):
                        raise RuntimeError(
                            "event-conditioned proposal did not change its state"
                        )
                    record = make_guidance_record(
                        product_tokens=_trim_pad(product_row.tolist()),
                        state_tokens=_trim_pad(state_row.tolist()),
                        terminal_tokens=_trim_pad(
                            terminal_states[row_index].detach().cpu().tolist(),
                        ),
                        transition_tokens=transition_tokens,
                        time_step=anchor_time,
                        reward=float(rewards[row_index].item()),
                        source_index=source_index,
                        sample_index=child_index,
                        time_index=anchor_index,
                        sample_seed=_mix_seed(
                            args.seed, source_index, child_index + 1,
                        ),
                        coupling_seed=_mix_seed(
                            args.seed, source_index, 100000 + child_index + 1,
                        ),
                    )
                    record.update({
                        "product_index": product_index,
                        "anchor_ordinal": anchor_ordinal,
                        "proposal_operation": operation_name,
                        "proposal_position": int(
                            actions["position"][row_index].item(),
                        ),
                        "proposal_token": int(actions["token"][row_index].item()),
                        "proposal_log_rate": float(
                            actions["log_rate"][row_index].item(),
                        ),
                        "proposal_log_probability": float(
                            actions["log_probability"][row_index].item(),
                        ),
                        "proposal_batch_seed": proposal_seed,
                        "rollout_batch_seed": rollout_seed,
                        "rollout_start_time": float(
                            rollout_start_times[row_index, 0].item(),
                        ),
                    })
                    records.append(record)

        if batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0:
            elapsed = time.perf_counter() - started
            rate = batch_index / max(elapsed, 1e-9)
            eta = (total_batches - batch_index) / max(rate, 1e-9)
            print(
                f"Event-conditioned batches {batch_index}/{total_batches} | "
                f"products {min(batch_start + args.batch_size, len(products))}/"
                f"{len(products)} | anchors {anchor_count} | "
                f"elapsed {elapsed:.1f}s | ETA {eta:.1f}s",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    )
    metadata = {
        "schema_version": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "products_file": str(Path(args.products_file).resolve()),
        "source_product_count": len(all_products),
        "selection_start_product": selection_start,
        "selection_end_product_exclusive": selection_end,
        "product_count": len(products),
        "record_count": len(records),
        "augmentation": args.augmentation,
        "n_steps": args.n_steps,
        "n_children": args.n_children,
        "anchor_steps": [index for index, _ in anchor_specs],
        "anchor_times": [time_value for _, time_value in anchor_specs],
        "anchor_count": anchor_count,
        "anchor_group_count": len(products) * anchor_count,
        "source_index_semantics": "unique product_index/anchor_ordinal pair",
        "shared_anchor": True,
        "record_first_transition": True,
        "proposal_sampler": "event_conditioned_atomic_rate",
        "proposal_condition": "one valid state-changing atomic edit",
        "proposal_forbidden_token_ids": list(_FORBIDDEN_PROPOSAL_TOKENS),
        "proposal_time_semantics": "anchor_time before forced atomic edit",
        "rollout_time_semantics": (
            "normal adaptive Euler endpoint of the proposal numerical step"
        ),
        "rng_scope": (
            "product-batch prefix, event-conditioned proposal, and ordinary "
            "terminal rollout seeds"
        ),
        "sampler": "euler_shared_anchor_event_conditioned_continuation",
        "reward": "rdkit_validity",
        "sample_scheduler": sample_scheduler.name,
        "train_scheduler": train_scheduler.name,
        "model_vocab": checkpoint_data.get("model_vocab", len(token2id)),
        "proposal_operation_counts": operation_counts,
        "generation_wall_seconds": wall,
        "generation_records_per_second": len(records) / max(wall, 1e-9),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "batch_count": total_batches,
    }
    save_guidance_dataset(output, records, metadata=metadata)
    return {
        "output": str(output),
        "products": len(products),
        "records": len(records),
        "selection_start_product": selection_start,
        "selection_end_product_exclusive": selection_end,
        "anchor_steps": [index for index, _ in anchor_specs],
        "anchor_times": [time_value for _, time_value in anchor_specs],
        "proposal_operation_counts": operation_counts,
        "reward_mean": float(torch.tensor([record["reward"] for record in records]).mean()),
        "wall_seconds": wall,
        "records_per_second": len(records) / max(wall, 1e-9),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--products_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--vocab_file", default=None)
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument(
        "--start_product", type=int, default=0,
        help="First original reaction block after collapsing augmentation rows.",
    )
    parser.add_argument("--max_products", type=int, default=None)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--n_children", type=int, default=4)
    parser.add_argument(
        "--anchor_time", type=float, default=None,
        help="One interior anchor time in (0, 1); defaults to 0.5 when omitted.",
    )
    parser.add_argument(
        "--anchor_steps", type=int, nargs="+", default=None,
        help="One or more interior Euler step indices, e.g. --anchor_steps 10 30 50.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    print(generate(parser.parse_args()))


if __name__ == "__main__":
    main()
