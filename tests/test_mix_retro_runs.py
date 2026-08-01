import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mix_retro_runs import (  # noqa: E402
    mix_prediction_lines,
    parse_run_sources,
    validate_and_load_sources,
)


def write_predictions(path, prefix, groups=4, beam_size=3):
    lines = [
        f"{prefix}_group{group}_run{run}"
        for group in range(groups)
        for run in range(1, beam_size + 1)
    ]
    path.write_text("\n".join(lines) + "\n")
    return lines


def test_mix_prediction_lines_preserves_groups_and_requested_order(tmp_path):
    noop_path = tmp_path / "noop.txt"
    stochastic_path = tmp_path / "stochastic.txt"
    noop_lines = write_predictions(noop_path, "noop")
    stochastic_lines = write_predictions(stochastic_path, "stochastic")
    sources, line_count = validate_and_load_sources(
        [("noop", str(noop_path)), ("stochastic", str(stochastic_path))],
        augmentation=2,
        input_beam_size=3,
    )
    run_sources = parse_run_sources(
        ["noop:1", "noop:2", "stochastic:3"],
        source_labels=set(sources),
        input_beam_size=3,
    )

    output = mix_prediction_lines(sources, run_sources, input_beam_size=3)

    assert line_count == 12
    assert output == [
        value
        for group in range(4)
        for value in (
            noop_lines[group * 3],
            noop_lines[group * 3 + 1],
            stochastic_lines[group * 3 + 2],
        )
    ]


def test_source_validation_rejects_misaligned_or_mismatched_files(tmp_path):
    good_path = tmp_path / "good.txt"
    short_path = tmp_path / "short.txt"
    write_predictions(good_path, "good")
    short_path.write_text("one\ntwo\n")

    with pytest.raises(ValueError, match="not divisible"):
        validate_and_load_sources(
            [("short", str(short_path))],
            augmentation=2,
            input_beam_size=3,
        )

    other_path = tmp_path / "other.txt"
    write_predictions(other_path, "other", groups=2)
    with pytest.raises(ValueError, match="different line counts"):
        validate_and_load_sources(
            [("good", str(good_path)), ("other", str(other_path))],
            augmentation=2,
            input_beam_size=3,
        )


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("missing_separator", "LABEL:RUN"),
        ("unknown:1", "unknown prediction source"),
        ("noop:zero", "invalid run index"),
        ("noop:0", "run index must"),
        ("noop:4", "run index must"),
    ],
)
def test_parse_run_sources_rejects_invalid_specs(spec, message):
    with pytest.raises(ValueError, match=message):
        parse_run_sources(
            [spec],
            source_labels={"noop"},
            input_beam_size=3,
        )


def test_cli_writes_predictions_and_auditable_metadata(tmp_path):
    noop_path = tmp_path / "noop.txt"
    stochastic_path = tmp_path / "stochastic.txt"
    write_predictions(noop_path, "noop")
    write_predictions(stochastic_path, "stochastic")
    output_dir = tmp_path / "mixed"

    command = [
        sys.executable,
        str(SCRIPTS_DIR / "mix_retro_runs.py"),
        "--prediction_file", "noop", str(noop_path),
        "--prediction_file", "stochastic", str(stochastic_path),
        "--run_source", "noop:1",
        "--run_source", "stochastic:2",
        "--run_source", "noop:3",
        "--augmentation", "2",
        "--input_beam_size", "3",
        "--output_dir", str(output_dir),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    predictions = (output_dir / "predictions.txt").read_text().splitlines()
    metadata = json.loads((output_dir / "mixing_metadata.json").read_text())
    assert len(predictions) == 12
    assert metadata["reaction_count"] == 2
    assert metadata["output_beam_size"] == 3
    assert metadata["output_line_count"] == 12
    assert metadata["run_sources"] == [
        {"label": "noop", "run": 1},
        {"label": "stochastic", "run": 2},
        {"label": "noop", "run": 3},
    ]

    rerun = subprocess.run(command, capture_output=True, text=True)
    assert rerun.returncode != 0
    assert "refusing to overwrite" in rerun.stderr
