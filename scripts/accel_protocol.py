"""Prepare a fixed dev1000 pilot split and score held-out reactions.

The held-out score reuses predictions from one full dev1000 sampling run.
It does not invoke the model again.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data/uspto50k_m500/dev1000"
AUGMENTATION = 20
PILOT_REACTIONS = 200
SPLIT_SEED = 20260926


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_reactions(lines, indices, lines_per_reaction):
    return [
        line
        for reaction in indices
        for line in lines[
            reaction * lines_per_reaction : (reaction + 1) * lines_per_reaction
        ]
    ]


def write_lines(path, lines):
    path.write_text("\n".join(lines) + "\n")


def prepare_split(output):
    src_path = DEV / "src.txt"
    tgt_path = DEV / "tgt.txt"
    src = src_path.read_text().splitlines()
    tgt = tgt_path.read_text().splitlines()
    if len(src) != len(tgt) or len(src) != 1000 * AUGMENTATION:
        raise ValueError("Expected the frozen dev1000 with 20 views per reaction")

    pilot = sorted(random.Random(SPLIT_SEED).sample(range(1000), PILOT_REACTIONS))
    pilot_set = set(pilot)
    heldout = [reaction for reaction in range(1000) if reaction not in pilot_set]
    output.mkdir(parents=True, exist_ok=False)
    for name, indices in (("pilot", pilot), ("heldout", heldout)):
        directory = output / name
        directory.mkdir()
        write_lines(directory / "src.txt", select_reactions(src, indices, AUGMENTATION))
        write_lines(directory / "tgt.txt", select_reactions(tgt, indices, AUGMENTATION))

    manifest = {
        "augmentation": AUGMENTATION,
        "split_seed": SPLIT_SEED,
        "source_sha256": sha256(src_path),
        "target_sha256": sha256(tgt_path),
        "pilot_reaction_indices": pilot,
        "heldout_reaction_indices": heldout,
        "pilot_source_sha256": sha256(output / "pilot/src.txt"),
        "heldout_source_sha256": sha256(output / "heldout/src.txt"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(pilot)} pilot and {len(heldout)} held-out reactions in {output}")


def score_heldout(split_dir, full_output, output, workers):
    from score import score

    manifest = json.loads((split_dir / "manifest.json").read_text())
    metadata = json.loads((full_output / "sampling_metadata.json").read_text())
    predictions = (full_output / "predictions.txt").read_text().splitlines()
    augmentation = manifest["augmentation"]
    n_runs = metadata["n_runs"]
    lines_per_reaction = augmentation * n_runs
    if metadata["n_products"] != 1000 * augmentation:
        raise ValueError("Expected a complete dev1000 prediction run")
    if len(predictions) != 1000 * lines_per_reaction:
        raise ValueError("Prediction line count does not match dev1000 and n_runs")
    if metadata["products_sha256"] != manifest["source_sha256"]:
        raise ValueError("Full predictions used a different product input")
    if sha256(full_output / "predictions.txt") != metadata["predictions_sha256"]:
        raise ValueError("Full predictions do not match their recorded hash")

    heldout = manifest["heldout_reaction_indices"]
    output.mkdir(parents=True, exist_ok=False)
    prediction_file = output / "predictions.txt"
    write_lines(
        prediction_file,
        select_reactions(predictions, heldout, lines_per_reaction),
    )
    subset_metadata = dict(metadata)
    subset_metadata.pop("seconds", None)
    subset_metadata.pop("performance", None)
    subset_metadata.update(
        n_products=len(heldout) * augmentation,
        products_sha256=sha256(split_dir / "heldout/src.txt"),
        predictions_sha256=sha256(prediction_file),
        parent_predictions_sha256=metadata["predictions_sha256"],
        subset="heldout800",
        scoring_only=True,
    )
    (output / "sampling_metadata.json").write_text(
        json.dumps(subset_metadata, indent=2) + "\n"
    )
    score(prediction_file, split_dir / "heldout/tgt.txt", workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    heldout = sub.add_parser("score-heldout")
    heldout.add_argument("--split-dir", type=Path, required=True)
    heldout.add_argument("--full-output", type=Path, required=True)
    heldout.add_argument("--output", type=Path, required=True)
    heldout.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_split(args.output)
    else:
        score_heldout(args.split_dir, args.full_output, args.output, args.workers)


if __name__ == "__main__":
    main()
