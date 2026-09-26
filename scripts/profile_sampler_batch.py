"""Profile one representative pilot batch; profiling time is never a benchmark."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from edit_flows.core.scheduler import CubicScheduler
from edit_flows.inference import (
    CHECKPOINT,
    DATA,
    load_model,
    make_batch,
)
from edit_flows.sampling import branch_sampler
from edit_flows.sampling.branch_sampler_helpers import _mix_child_seed
from edit_flows.utils.tokens import BOS_TOKEN, PAD_TOKEN, UNK_TOKEN


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_median_length_batch(products, batch_size):
    batches = [
        products[start : start + batch_size]
        for start in range(0, len(products), batch_size)
        if len(products[start : start + batch_size]) == batch_size
    ]
    if not batches:
        raise ValueError("Products file must contain at least one full batch")
    ranked = sorted(
        range(len(batches)),
        key=lambda index: sum(
            len(product.split()) for product in batches[index]
        ),
    )
    return ranked[len(ranked) // 2]


def profiled(function, component, labels, quarterly=True, initial_call=False):
    calls = 0

    def wrapped(*args, **kwargs):
        nonlocal calls
        call = calls
        calls += 1
        if initial_call and call == 0:
            name = f"{component}_initial"
        else:
            step = call - int(initial_call)
            bucket = min(max(step, 0) // 25, 3) if quarterly else None
            suffix = f"_q{bucket + 1}" if bucket is not None else ""
            name = f"{component}{suffix}"
        labels.append(name)
        with record_function(name):
            return function(*args, **kwargs)

    return wrapped


def event_row(event):
    return {
        "name": event.key,
        "calls": event.count,
        "cpu_total_ms": event.cpu_time_total / 1000,
        "cpu_self_ms": event.self_cpu_time_total / 1000,
        "cuda_total_ms": getattr(event, "device_time_total", 0) / 1000,
        "cuda_self_ms": getattr(event, "self_device_time_total", 0) / 1000,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path)
    parser.add_argument("--batch-index", type=int)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--vocab", type=Path, default=DATA / "example.vocab.src")
    args = parser.parse_args()
    if args.batch_index is not None and args.batch_index < 0:
        parser.error("--batch-index must be nonnegative")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    model, cfg, token2id = load_model(args.checkpoint, args.vocab, device)
    products = args.products.read_text().splitlines()
    batch_index = (
        args.batch_index
        if args.batch_index is not None
        else choose_median_length_batch(products, 32)
    )
    start = batch_index * 32
    batch = products[start : start + 32]
    if len(batch) != 32:
        parser.error("--batch-index must select a complete batch")

    token_ids = [
        [token2id.get(token, UNK_TOKEN) for token in product.split()]
        for product in batch
    ]
    x_unique = make_batch(token_ids, device)
    x_0 = x_unique.repeat_interleave(9, dim=0)
    mask = x_unique == PAD_TOKEN
    memory_mask = mask.repeat_interleave(9, dim=0)
    seeds = [
        _mix_child_seed(42, start + i, run + 1)
        for i in range(32)
        for run in range(9)
    ]
    kwargs = {
        "product_memory_padding_mask": memory_mask,
        "n_steps": 100,
        "max_seq_len": cfg["max_seq_len"],
        "n_children": 2,
        "changed_state_bonus": 0.5,
    }
    scheduler = CubicScheduler()
    id2token = {index: token for token, index in token2id.items()}
    prediction_path = args.output.with_suffix(".predictions.txt")

    def run_batch(write_predictions):
        with torch.inference_mode():
            with record_function("product_encoding"):
                memory = model.encode_product(x_unique, mask).repeat_interleave(
                    9, dim=0
                )
            result = branch_sampler.sample_branches(
                model,
                x_0,
                scheduler,
                seeds,
                product_memory=memory,
                **kwargs,
            )
            with record_function("prediction_decode_write"):
                rows = result.cpu().tolist()
                lines = [
                    " ".join(
                        id2token.get(token, "<UNK>")
                        for token in row
                        if token not in (PAD_TOKEN, BOS_TOKEN)
                    )
                    for row in rows
                ]
                if write_predictions:
                    prediction_path.write_text("\n".join(lines) + "\n")
        return lines

    torch.cuda.reset_peak_memory_stats(device)
    run_batch(write_predictions=False)
    torch.cuda.synchronize(device)

    labels = []
    model.forward = profiled(model.forward, "transformer", labels)
    wrapped_functions = {
        "_token_keys_batch": ("state_keys", True, True),
        "_sample_actions_per_branch": ("sample_actions", True, False),
        "_apply_edits_batch": ("apply_edits", True, False),
        "get_adaptive_h": ("adaptive_h", True, False),
        "_adaptive_h_to_list": ("adaptive_h_to_host", True, False),
    }
    for attribute, (name, quarterly, initial_call) in wrapped_functions.items():
        setattr(
            branch_sampler,
            attribute,
            profiled(
                getattr(branch_sampler, attribute),
                name,
                labels,
                quarterly=quarterly,
                initial_call=initial_call,
            ),
        )

    selection_cpu_seconds = 0.0
    selection_calls = 0
    original_select = branch_sampler._select_k1m_child

    def timed_select(*positional, **named):
        nonlocal selection_cpu_seconds, selection_calls
        started = time.perf_counter()
        selected = original_select(*positional, **named)
        selection_cpu_seconds += time.perf_counter() - started
        selection_calls += 1
        return selected

    # Measure the high-frequency Python decision separately to avoid adding
    # tens of thousands of profiler scopes to the trace.
    branch_sampler._select_k1m_child = timed_select

    torch.cuda.synchronize(device)
    profile_started = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as measured:
        lines = run_batch(write_predictions=True)
        torch.cuda.synchronize(device)
    profile_wall_seconds = time.perf_counter() - profile_started

    named_components = set(labels) | {"product_encoding", "prediction_decode_write"}
    component_events = {}
    average_events = list(measured.key_averages())
    for event in average_events:
        if event.key not in named_components:
            continue
        row = event_row(event)
        previous = component_events.get(event.key)
        # CUDA profiler output can contain a device-only duplicate for a named
        # CPU scope. Keep the CPU scope, which has the correctly paired device
        # time and avoids counting the duplicate twice.
        if previous is None or row["cpu_total_ms"] > previous["cpu_total_ms"]:
            component_events[event.key] = row
    quartiles = []
    component_names = (
        "transformer",
        "state_keys",
        "sample_actions",
        "apply_edits",
        "adaptive_h",
        "adaptive_h_to_host",
    )
    sync_names = {"state_keys", "adaptive_h_to_host", "apply_edits"}
    for index in range(4):
        suffix = f"_q{index + 1}"
        rows = [
            component_events[f"{name}{suffix}"]
            for name in component_names
            if f"{name}{suffix}" in component_events
        ]
        cuda_ms = sum(row["cuda_total_ms"] for row in rows)
        cpu_ms = sum(row["cpu_total_ms"] for row in rows)
        sync_cpu_ms = sum(
            component_events[f"{name}{suffix}"]["cpu_total_ms"]
            for name in sync_names
            if f"{name}{suffix}" in component_events
        )
        transformer = component_events.get(f"transformer{suffix}", {})
        quartiles.append(
            {
                "time_range": [index / 4, (index + 1) / 4],
                "instrumented_cuda_ms": cuda_ms,
                "transformer_cuda_share_percent": (
                    100 * transformer.get("cuda_total_ms", 0) / cuda_ms
                    if cuda_ms
                    else None
                ),
                "instrumented_cpu_ms": cpu_ms,
                "synchronization_scope_cpu_ms": sync_cpu_ms,
                "synchronization_scope_cpu_share_percent": (
                    100 * sync_cpu_ms / cpu_ms if cpu_ms else None
                ),
            }
        )

    reference_match = None
    reference_hash = None
    if args.reference_predictions is not None:
        reference = args.reference_predictions.read_text().splitlines()
        expected = reference[start * 9 : (start + len(batch)) * 9]
        reference_match = lines == expected
        reference_hash = sha256(args.reference_predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    top_cpu_operators = sorted(
        (event_row(event) for event in average_events),
        key=lambda row: row["cpu_self_ms"],
        reverse=True,
    )[:20]
    report = {
        "products": str(args.products),
        "products_sha256": sha256(args.products),
        "checkpoint_sha256": sha256(args.checkpoint),
        "reference_predictions": (
            str(args.reference_predictions) if args.reference_predictions else None
        ),
        "reference_predictions_sha256": reference_hash,
        "reference_batch_exact_match": reference_match,
        "profiled_predictions": str(prediction_path),
        "profiled_predictions_sha256": sha256(prediction_path),
        "batch_index": batch_index,
        "batch_size": len(batch),
        "batch_mean_product_tokens": sum(map(len, token_ids)) / len(token_ids),
        "n_runs": 9,
        "n_children": 2,
        "n_steps": 100,
        "gpu": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "profile_wall_seconds_with_profiler_overhead": profile_wall_seconds,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "components": sorted(
            component_events.values(),
            key=lambda row: row["cpu_total_ms"],
            reverse=True,
        ),
        "selection_python_cpu_ms": 1000 * selection_cpu_seconds,
        "selection_calls": selection_calls,
        "quartiles": quartiles,
        "top_cpu_operators": top_cpu_operators,
        "notes": [
            "Profiling overhead makes profile_wall_seconds unsuitable for speed comparisons.",
            "Transformer and synchronization shares use instrumented sampler component scopes; CPU and CUDA time can overlap.",
            "Child selection Python time is measured separately to avoid profiler-scope overhead.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"quartiles": quartiles, "reference_batch_exact_match": reference_match}, indent=2))


if __name__ == "__main__":
    main()
