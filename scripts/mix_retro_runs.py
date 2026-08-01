#!/usr/bin/env python
"""Build heterogeneous-run prediction files without resampling.

Each input file must use the retrosynthesis layout
``(reaction, augmentation, run)`` with consecutive runs.  The output keeps the
same reaction/augmentation order and selects one requested run from one source
file for each output position.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_prediction_file(path):
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8")
    return text.splitlines(), _sha256_bytes(raw)


def validate_and_load_sources(
    prediction_specs,
    augmentation,
    input_beam_size,
):
    if augmentation <= 0:
        raise ValueError(f"augmentation must be > 0, got {augmentation}")
    if input_beam_size <= 0:
        raise ValueError(
            f"input_beam_size must be > 0, got {input_beam_size}"
        )
    if not prediction_specs:
        raise ValueError("at least one prediction source is required")

    sources = {}
    expected_count = None
    input_block = augmentation * input_beam_size
    for label, path in prediction_specs:
        if label in sources:
            raise ValueError(f"duplicate prediction source label: {label}")
        lines, sha256 = read_prediction_file(path)
        line_count = len(lines)
        if line_count == 0:
            raise ValueError(f"prediction source {label!r} is empty: {path}")
        if line_count % input_block != 0:
            raise ValueError(
                f"prediction source {label!r} has {line_count} lines, which "
                f"is not divisible by augmentation * input_beam_size "
                f"({augmentation} * {input_beam_size} = {input_block})"
            )
        if expected_count is None:
            expected_count = line_count
        elif line_count != expected_count:
            raise ValueError(
                "prediction sources have different line counts: expected "
                f"{expected_count}, got {line_count} for {label!r}"
            )
        sources[label] = {
            "path": os.path.abspath(path),
            "lines": lines,
            "line_count": line_count,
            "sha256": sha256,
        }
    return sources, expected_count


def parse_run_sources(run_source_specs, source_labels, input_beam_size):
    if not run_source_specs:
        raise ValueError("at least one --run_source is required")

    parsed = []
    for spec in run_source_specs:
        if ":" not in spec:
            raise ValueError(
                f"invalid run source {spec!r}; expected LABEL:RUN"
            )
        label, run_text = spec.rsplit(":", 1)
        if label not in source_labels:
            raise ValueError(
                f"unknown prediction source label {label!r} in {spec!r}"
            )
        try:
            run = int(run_text)
        except ValueError as exc:
            raise ValueError(
                f"invalid run index {run_text!r} in {spec!r}"
            ) from exc
        if not 1 <= run <= input_beam_size:
            raise ValueError(
                f"run index must be in [1, {input_beam_size}], got {run}"
            )
        parsed.append({"label": label, "run": run})
    return parsed


def mix_prediction_lines(sources, run_sources, input_beam_size):
    line_counts = {source["line_count"] for source in sources.values()}
    if len(line_counts) != 1:
        raise ValueError("all prediction sources must have the same line count")
    line_count = next(iter(line_counts))
    if line_count % input_beam_size != 0:
        raise ValueError(
            f"line count {line_count} is not divisible by input_beam_size "
            f"{input_beam_size}"
        )

    output = []
    group_count = line_count // input_beam_size
    for group_index in range(group_count):
        group_start = group_index * input_beam_size
        for source in run_sources:
            line_index = group_start + source["run"] - 1
            output.append(sources[source["label"]]["lines"][line_index])
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description="Mix aligned retrosynthesis runs from existing predictions"
    )
    parser.add_argument(
        "--prediction_file",
        nargs=2,
        action="append",
        required=True,
        metavar=("LABEL", "PATH"),
        help="Labeled input; repeat for every prediction file",
    )
    parser.add_argument(
        "--run_source",
        action="append",
        required=True,
        metavar="LABEL:RUN",
        help=(
            "Source label and 1-based input run for one output position; "
            "repeat in desired output order"
        ),
    )
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--input_beam_size", type=int, default=3)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing predictions.txt and mixing_metadata.json",
    )
    return parser


def main():
    args = build_parser().parse_args()
    sources, input_line_count = validate_and_load_sources(
        args.prediction_file,
        augmentation=args.augmentation,
        input_beam_size=args.input_beam_size,
    )
    run_sources = parse_run_sources(
        args.run_source,
        source_labels=set(sources),
        input_beam_size=args.input_beam_size,
    )
    output_lines = mix_prediction_lines(
        sources,
        run_sources,
        input_beam_size=args.input_beam_size,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    prediction_path = os.path.join(args.output_dir, "predictions.txt")
    metadata_path = os.path.join(args.output_dir, "mixing_metadata.json")
    existing = [
        path for path in (prediction_path, metadata_path) if os.path.exists(path)
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite existing output; choose a new output_dir "
            f"or pass --overwrite: {existing}"
        )

    output_bytes = ("\n".join(output_lines) + "\n").encode("utf-8")
    with open(prediction_path, "wb") as f:
        f.write(output_bytes)

    output_beam_size = len(run_sources)
    reaction_count = input_line_count // (
        args.augmentation * args.input_beam_size
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "layout": "reaction-major, augmentation-major, run-minor",
        "augmentation": args.augmentation,
        "input_beam_size": args.input_beam_size,
        "output_beam_size": output_beam_size,
        "reaction_count": reaction_count,
        "input_line_count_per_source": input_line_count,
        "output_line_count": len(output_lines),
        "run_sources": run_sources,
        "sources": {
            label: {
                "path": source["path"],
                "line_count": source["line_count"],
                "sha256": source["sha256"],
            }
            for label, source in sources.items()
        },
        "output_sha256": _sha256_bytes(output_bytes),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")

    print(
        f"Mixed {reaction_count} reactions x {args.augmentation} "
        f"augmentations x {output_beam_size} output runs"
    )
    print(f"Run sources: {args.run_source}")
    print(f"Predictions: {prediction_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
