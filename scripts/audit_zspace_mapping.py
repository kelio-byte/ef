#!/usr/bin/env python
"""Audit whether aligned Z edits have a unique variable-length X mapping.

This is a read-only DG-0 diagnostic.  It intentionally works on token strings
so it can audit pre-aligned files without loading a checkpoint or installing a
second tokenizer environment.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def audit(aligned_src: Path, aligned_tgt: Path, max_lines: int = 0) -> dict:
    counts: Counter[str] = Counter()
    examples: list[dict] = []
    with aligned_src.open() as src_file, aligned_tgt.open() as tgt_file:
        for line_no, (src_line, tgt_line) in enumerate(
            zip(src_file, tgt_file), start=1,
        ):
            if max_lines > 0 and line_no > max_lines:
                break
            old = ["<BOS>", *src_line.strip().split()]
            new = ["<BOS>", *tgt_line.strip().split()]
            counts["pairs"] += 1
            if len(old) != len(new):
                counts["length_mismatch"] += 1
                continue
            changed = [
                index for index, (old_token, new_token)
                in enumerate(zip(old, new))
                if old_token != new_token
            ]
            counts["changed_coordinates"] += len(changed)
            if len(changed) == 1:
                counts["single_coordinate_pairs"] += 1
            for index in changed:
                old_token, new_token = old[index], new[index]
                if old_token == "<GAP>" and new_token != "<GAP>":
                    counts["insert"] += 1
                    ambiguous = (
                        (index > 1 and old[index - 1] == "<GAP>")
                        or (
                            index + 1 < len(old)
                            and old[index + 1] == "<GAP>"
                        )
                    )
                    if ambiguous:
                        counts["ambiguous_insert"] += 1
                        if len(examples) < 3:
                            examples.append({
                                "line": line_no,
                                "z_position": index,
                                "context": old[max(0, index - 2):index + 3],
                            })
                elif old_token != "<GAP>" and new_token == "<GAP>":
                    counts["delete"] += 1
                elif old_token != "<GAP>" and new_token != "<GAP>":
                    counts["substitute"] += 1
                else:
                    counts["invalid_transition"] += 1
    result = dict(counts)
    result["ambiguous_insert_rate"] = (
        counts["ambiguous_insert"] / counts["insert"]
        if counts["insert"] else 0.0
    )
    result["single_coordinate_rate"] = (
        counts["single_coordinate_pairs"] / counts["pairs"]
        if counts["pairs"] else 0.0
    )
    result["examples"] = examples
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned_src", required=True, type=Path)
    parser.add_argument("--aligned_tgt", required=True, type=Path)
    parser.add_argument(
        "--max_lines", type=int, default=0,
        help="limit rows for a smoke test; 0 means all rows",
    )
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()
    if args.max_lines < 0:
        parser.error("--max_lines must be non-negative")
    result = audit(args.aligned_src, args.aligned_tgt, args.max_lines)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
