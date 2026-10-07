#!/usr/bin/env python3
"""Summarize multi-component reactants in a target SMILES file.

The project target files are usually whitespace-tokenized ``#global#`` SMILES,
where ``.`` is a standalone token.  The parser also accepts ordinary
un-tokenized SMILES such as ``CCO.CN``.  Besides line-level counts, an optional
augmentation size reports statistics over complete contiguous augmentation
blocks (for example, 20 lines per USPTO reaction).
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _parse_record(raw_line: str, line_number: int) -> Dict[str, Any]:
    """Parse one input line without changing the original SMILES text."""
    text = raw_line.rstrip("\r\n")
    stripped = text.strip()
    if not stripped:
        return {
            "line_number": line_number,
            "text": text,
            "nonempty": False,
            "dot_count": 0,
            "fragment_count": 0,
        }

    tokens = stripped.split()
    # In #global# files the separator is a standalone token.  The fallback
    # keeps the utility useful for ordinary (un-tokenized) SMILES files.
    dot_count = tokens.count(".")
    if dot_count == 0 and "." in stripped and len(tokens) == 1:
        dot_count = stripped.count(".")

    return {
        "line_number": line_number,
        "text": text,
        "nonempty": True,
        "dot_count": dot_count,
        "fragment_count": dot_count + 1,
    }


def _read_records(path: Path, max_lines: int = 0) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if max_lines > 0 and len(records) >= max_lines:
                break
            records.append(_parse_record(raw_line, line_number))
    return records


def _percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _counter_to_json(counter: Counter[int]) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _line_statistics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    nonempty = [row for row in rows if row["nonempty"]]
    dot_rows = [row for row in nonempty if row["dot_count"] > 0]
    fragment_hist = Counter(int(row["fragment_count"]) for row in nonempty)
    dot_fragment_hist = Counter(
        int(row["fragment_count"]) for row in dot_rows
    )
    return {
        "total_lines": len(rows),
        "blank_lines": len(rows) - len(nonempty),
        "nonempty_lines": len(nonempty),
        "lines_with_dot": len(dot_rows),
        "lines_without_dot": len(nonempty) - len(dot_rows),
        "line_dot_percentage": _percentage(len(dot_rows), len(nonempty)),
        "total_dot_tokens": sum(int(row["dot_count"]) for row in nonempty),
        "mean_fragments_per_nonempty_line": (
            sum(int(row["fragment_count"]) for row in nonempty) / len(nonempty)
            if nonempty else 0.0
        ),
        "max_fragments": max(
            (int(row["fragment_count"]) for row in nonempty),
            default=0,
        ),
        "fragment_histogram_all_lines": _counter_to_json(fragment_hist),
        "fragment_histogram_dot_lines": _counter_to_json(dot_fragment_hist),
    }


def _augmentation_statistics(
    records: List[Dict[str, Any]], augmentation: int,
) -> Optional[Dict[str, Any]]:
    if augmentation <= 1:
        return None

    complete_count = len(records) // augmentation
    remainder = len(records) % augmentation
    complete_blocks = [
        records[start:start + augmentation]
        for start in range(0, complete_count * augmentation, augmentation)
    ]

    any_dot = 0
    all_nonempty_dot = 0
    block_dot_hist = Counter()
    block_fragment_hist = Counter()
    block_dot_fraction_sum = 0.0
    for block in complete_blocks:
        nonempty = [row for row in block if row["nonempty"]]
        dot_count = sum(row["dot_count"] > 0 for row in nonempty)
        block_dot_hist[int(dot_count)] += 1
        fragment_counts = {int(row["fragment_count"]) for row in nonempty}
        if len(fragment_counts) == 1:
            block_fragment_hist[str(next(iter(fragment_counts)))] += 1
        else:
            block_fragment_hist["mixed"] += 1
        if dot_count > 0:
            any_dot += 1
        if len(nonempty) == augmentation and dot_count == augmentation:
            all_nonempty_dot += 1
        block_dot_fraction_sum += _percentage(dot_count, len(nonempty))

    return {
        "augmentation": augmentation,
        "complete_blocks": complete_count,
        "partial_tail_lines": remainder,
        "blocks_with_any_dot": any_dot,
        "blocks_with_any_dot_percentage": _percentage(any_dot, complete_count),
        "blocks_all_lines_with_dot": all_nonempty_dot,
        "blocks_all_lines_with_dot_percentage": _percentage(
            all_nonempty_dot, complete_count,
        ),
        "mean_dot_line_percentage_per_complete_block": (
            block_dot_fraction_sum / complete_count if complete_count else 0.0
        ),
        "block_dot_line_count_histogram": _counter_to_json(block_dot_hist),
        "block_fragment_count_histogram": dict(
            sorted(block_fragment_hist.items(), key=lambda item: item[0])
        ),
        "interpretation": (
            "Blocks are contiguous groups of augmentation lines; this assumes "
            "the input file preserves the dataset's augmentation ordering."
        ),
    }


def _build_summary(
    path: Path,
    records: List[Dict[str, Any]],
    augmentation: int,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "file": str(path),
        "line_statistics": _line_statistics(records),
    }
    augmentation_stats = _augmentation_statistics(records, augmentation)
    if augmentation_stats is not None:
        summary["augmentation_statistics"] = augmentation_stats
    return summary


def _print_histogram(title: str, histogram: Dict[str, int]) -> None:
    print(title)
    if not histogram:
        print("  (none)")
        return
    for key, value in histogram.items():
        print(f"  {key} fragment(s): {value}")


def _print_summary(summary: Dict[str, Any]) -> None:
    line_stats = summary["line_statistics"]
    print(f"File: {summary['file']}")
    print("\nLine-level statistics (non-empty records):")
    print(f"  total lines:              {line_stats['total_lines']}")
    print(f"  blank lines:              {line_stats['blank_lines']}")
    print(f"  non-empty lines:          {line_stats['nonempty_lines']}")
    print(f"  lines containing '.':     {line_stats['lines_with_dot']}")
    print(f"  lines without '.':        {line_stats['lines_without_dot']}")
    print(f"  dot-line percentage:      {line_stats['line_dot_percentage']:.4f}%")
    print(f"  total '.' separators:     {line_stats['total_dot_tokens']}")
    print(
        "  mean fragments/line:      "
        f"{line_stats['mean_fragments_per_nonempty_line']:.4f}"
    )
    print(f"  maximum fragments/line:   {line_stats['max_fragments']}")
    _print_histogram(
        "\nFragment histogram (all non-empty lines):",
        line_stats["fragment_histogram_all_lines"],
    )
    _print_histogram(
        "\nFragment histogram (lines containing '.'): ",
        line_stats["fragment_histogram_dot_lines"],
    )

    augmentation_stats = summary.get("augmentation_statistics")
    if augmentation_stats is not None:
        print("\nAugmentation-block statistics:")
        print(f"  augmentation size:       {augmentation_stats['augmentation']}")
        print(f"  complete blocks:          {augmentation_stats['complete_blocks']}")
        print(f"  partial tail lines:       {augmentation_stats['partial_tail_lines']}")
        print(
            "  blocks with any '.':      "
            f"{augmentation_stats['blocks_with_any_dot']} "
            f"({augmentation_stats['blocks_with_any_dot_percentage']:.4f}%)"
        )
        print(
            "  blocks all lines with '.': "
            f"{augmentation_stats['blocks_all_lines_with_dot']} "
            f"({augmentation_stats['blocks_all_lines_with_dot_percentage']:.4f}%)"
        )
        print(
            "  mean dot-line percentage/block: "
            f"{augmentation_stats['mean_dot_line_percentage_per_complete_block']:.4f}%"
        )
        _print_histogram(
            "\nBlock histogram (number of dot-containing lines per block):",
            augmentation_stats["block_dot_line_count_histogram"],
        )
        print("\nBlock histogram (reactant fragment count):")
        for key, value in augmentation_stats["block_fragment_count_histogram"].items():
            print(f"  {key} fragment(s): {value}")


def _print_examples(records: List[Dict[str, Any]], limit: int) -> None:
    if limit <= 0:
        return
    print(f"\nFirst {limit} lines containing '.':")
    shown = 0
    for row in records:
        if not row["nonempty"] or row["dot_count"] == 0:
            continue
        text = " ".join(row["text"].split())
        if len(text) > 180:
            text = text[:177] + "..."
        print(
            f"  line {row['line_number']}: "
            f"{row['fragment_count']} fragments; {text}"
        )
        shown += 1
        if shown >= limit:
            break
    if shown == 0:
        print("  (none)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count multi-component reactants (SMILES containing '.') in a "
            "tokenized or ordinary target file."
        )
    )
    parser.add_argument("--targets_file", required=True, help="Target SMILES file")
    parser.add_argument(
        "--augmentation", type=int, default=20,
        help="Contiguous augmentation lines per reaction; 0 disables block stats (default: 20)",
    )
    parser.add_argument(
        "--max_lines", type=int, default=0,
        help="Only read this many lines; 0 reads the complete file (default: 0)",
    )
    parser.add_argument(
        "--show_examples", type=int, default=0,
        help="Print the first N dot-containing examples (default: 0)",
    )
    parser.add_argument(
        "--json_out", type=str, default=None,
        help="Optional path for a machine-readable JSON summary",
    )
    args = parser.parse_args()

    if args.augmentation < 0:
        parser.error("--augmentation must be >= 0")
    if args.max_lines < 0:
        parser.error("--max_lines must be >= 0")
    if args.show_examples < 0:
        parser.error("--show_examples must be >= 0")

    path = Path(args.targets_file)
    if not path.is_file():
        parser.error(f"targets file does not exist: {path}")

    records = _read_records(path, max_lines=args.max_lines)
    summary = _build_summary(path, records, args.augmentation)
    _print_summary(summary)
    _print_examples(records, args.show_examples)

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote JSON summary: {output_path}")


if __name__ == "__main__":
    main()
