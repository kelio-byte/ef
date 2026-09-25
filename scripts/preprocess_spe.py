#!/usr/bin/env python
"""用途：将原子级 SMILES 数据转换为 SPE token 数据。
输入：源数据分割文件、SPE 词对表和合并次数。
输出：转换后的各分割文件、训练词表及预处理元数据。
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Iterator, Sequence


DEFAULT_SOURCE_DIR = Path("data/atom_global")
DEFAULT_OUTPUT_DIR = Path("data/uspto50k_m500")
DEFAULT_CODES_PATH = Path("data/SPE_ChEMBL.txt")
SPLITS = ("train", "val", "test")


_WORKER_TOKENIZER = None
_WORKER_CACHE_RESET_INTERVAL = 0
_WORKER_PAIR_COUNT = 0


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """作用：分块计算文件校验和。输入：文件路径和读取块大小。输出：SHA-256 字符串。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_smiles(tokenized_line: str) -> str:
    """作用：还原带空格展示的 SMILES。输入：tokenized 文件行。输出：移除分隔空格后的 SMILES。"""
    return "".join(tokenized_line.strip().split())


def _paired_lines(
    src_path: Path,
    tgt_path: Path,
    *,
    max_lines: int | None,
) -> Iterator[tuple[int, str, str]]:
    """作用：按行读取成对文件并检查长度。输入：源/目标路径和可选行数上限。输出：行号及配对文本迭代器。"""
    with src_path.open() as src_handle, tgt_path.open() as tgt_handle:
        line_no = 0
        while True:
            src_line = src_handle.readline()
            tgt_line = tgt_handle.readline()
            if not src_line and not tgt_line:
                break
            line_no += 1
            if not src_line or not tgt_line:
                raise ValueError(
                    "source/target line-count mismatch at line "
                    f"{line_no}: {src_path} vs {tgt_path}"
                )
            yield line_no, src_line, tgt_line
            if max_lines is not None and line_no >= max_lines:
                break


def _load_tokenizer(codes_path: Path, *, merges: int = -1):
    """作用：载入 SmilesPE 分词器。输入：SPE 规则文件和合并次数。输出：配置好的 tokenizer。"""
    try:
        from SmilesPE.tokenizer import SPE_Tokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "SmilesPE is required; install it with "
            "python -m pip install 'SmilesPE==0.0.3'"
        ) from exc
    codes = codes_path.open()
    try:
        tokenizer = SPE_Tokenizer(codes, merges=merges)
    finally:
        codes.close()
    return tokenizer


def tokenize_smiles(tokenizer, smiles: str) -> list[str]:
    """作用：确定性地切分一条 SMILES 并检查可逆性。输入：分词器和完整 SMILES。输出：SPE token 列表。"""
    tokens = tokenizer.tokenize(smiles, dropout=0).split()
    restored = "".join(tokens)
    if restored != smiles:
        raise ValueError(
            "SPE tokenization is not lossless: "
            f"original={smiles!r}, restored={restored!r}"
        )
    return tokens


def _tokenize_pair(
    pair: tuple[int, str, str],
    tokenizer,
) -> tuple[list[str], list[str]]:
    """作用：重建并切分一对反应 SMILES。输入：行号及源/目标文本行、分词器。输出：源和目标 SPE token 列表。"""
    line_no, src_line, tgt_line = pair
    src_smiles = restore_smiles(src_line)
    tgt_smiles = restore_smiles(tgt_line)
    if not src_smiles or not tgt_smiles:
        raise ValueError(f"empty reconstructed SMILES at line {line_no}")
    src_tokens = tokenize_smiles(tokenizer, src_smiles)
    tgt_tokens = tokenize_smiles(tokenizer, tgt_smiles)
    if "<GAP>" in src_tokens or "<GAP>" in tgt_tokens:
        raise ValueError(f"<GAP> appeared in unaligned SPE tokens at line {line_no}")
    return src_tokens, tgt_tokens


def _init_tokenizer_worker(
    codes_path: str,
    merges: int,
    cache_reset_interval: int,
) -> None:
    """作用：初始化一个并行 worker 的分词状态。输入：规则文件、合并次数和缓存清理间隔。输出：设置进程内 tokenizer。"""
    global _WORKER_TOKENIZER
    global _WORKER_CACHE_RESET_INTERVAL
    global _WORKER_PAIR_COUNT
    _WORKER_TOKENIZER = _load_tokenizer(Path(codes_path), merges=merges)
    _WORKER_CACHE_RESET_INTERVAL = cache_reset_interval
    _WORKER_PAIR_COUNT = 0


def _tokenize_pair_worker(
    pair: tuple[int, str, str],
) -> tuple[list[str], list[str]]:
    """作用：在 worker 中切分一对反应。输入：带行号的源/目标文本。输出：源和目标 token 列表。"""
    global _WORKER_PAIR_COUNT
    if _WORKER_TOKENIZER is None:  # pragma: no cover - defensive guard
        raise RuntimeError("SPE tokenizer worker was not initialized")
    result = _tokenize_pair(pair, _WORKER_TOKENIZER)
    _WORKER_PAIR_COUNT += 1
    if (
        _WORKER_CACHE_RESET_INTERVAL > 0
        and _WORKER_PAIR_COUNT % _WORKER_CACHE_RESET_INTERVAL == 0
    ):
        _WORKER_TOKENIZER.cache.clear()
    return result


def _tokenize_split(
    source_dir: Path,
    output_dir: Path,
    split: str,
    tokenizer,
    *,
    codes_path: Path,
    merges: int,
    num_workers: int,
    max_lines: int | None,
    cache_reset_interval: int,
) -> dict:
    """作用：处理一个数据分割并保存 SPE 文件。输入：源/输出目录、分割名和 tokenizer 配置。输出：该分割统计与校验和。"""
    split_source_dir = source_dir / split
    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)
    src_path = split_source_dir / f"src-{split}.txt"
    tgt_path = split_source_dir / f"tgt-{split}.txt"
    out_src_path = split_output_dir / f"src-{split}.txt"
    out_tgt_path = split_output_dir / f"tgt-{split}.txt"
    for path in (src_path, tgt_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    pair_count = 0
    src_token_count = 0
    tgt_token_count = 0
    src_max_tokens = 0
    tgt_max_tokens = 0
    pairs = _paired_lines(src_path, tgt_path, max_lines=max_lines)
    pool = None
    if num_workers == 1:
        tokenized_pairs = (_tokenize_pair(pair, tokenizer) for pair in pairs)
    else:
        pool = Pool(
            processes=num_workers,
            initializer=_init_tokenizer_worker,
            initargs=(str(codes_path), merges, cache_reset_interval),
        )
        tokenized_pairs = pool.imap(_tokenize_pair_worker, pairs, chunksize=200)

    try:
        with (
            out_src_path.open("w") as out_src,
            out_tgt_path.open("w") as out_tgt,
        ):
            for src_tokens, tgt_tokens in tokenized_pairs:
                out_src.write(" ".join(src_tokens) + "\n")
                out_tgt.write(" ".join(tgt_tokens) + "\n")
                pair_count += 1
                src_token_count += len(src_tokens)
                tgt_token_count += len(tgt_tokens)
                src_max_tokens = max(src_max_tokens, len(src_tokens))
                tgt_max_tokens = max(tgt_max_tokens, len(tgt_tokens))
                if (
                    num_workers == 1
                    and cache_reset_interval > 0
                    and pair_count % cache_reset_interval == 0
                ):
                    # SmilesPE caches every unique complete SMILES. Bounding
                    # it keeps the full run memory-stable without changing
                    # deterministic tokenization.
                    tokenizer.cache.clear()
    except BaseException:
        if pool is not None:
            pool.terminate()
        raise
    else:
        if pool is not None:
            pool.close()
    finally:
        if pool is not None:
            pool.join()

    return {
        "pair_count": pair_count,
        "src_token_count": src_token_count,
        "tgt_token_count": tgt_token_count,
        "src_mean_tokens": src_token_count / pair_count if pair_count else 0.0,
        "tgt_mean_tokens": tgt_token_count / pair_count if pair_count else 0.0,
        "src_max_tokens": src_max_tokens,
        "tgt_max_tokens": tgt_max_tokens,
        "source_sha256": {
            "src": _sha256(src_path),
            "tgt": _sha256(tgt_path),
        },
        "output_sha256": {
            "src": _sha256(out_src_path),
            "tgt": _sha256(out_tgt_path),
        },
    }


def _summarize_existing_split(
    source_dir: Path,
    output_dir: Path,
    split: str,
    *,
    max_lines: int | None,
) -> dict:
    """作用：校验已生成的 SPE 文件并汇总统计。输入：源/输出目录、分割名和可选行数上限。输出：样本数、token 统计及文件校验和。"""
    split_source_dir = source_dir / split
    split_output_dir = output_dir / split
    src_path = split_source_dir / f"src-{split}.txt"
    tgt_path = split_source_dir / f"tgt-{split}.txt"
    out_src_path = split_output_dir / f"src-{split}.txt"
    out_tgt_path = split_output_dir / f"tgt-{split}.txt"
    for path in (src_path, tgt_path, out_src_path, out_tgt_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    pair_count = 0
    src_token_count = 0
    tgt_token_count = 0
    src_max_tokens = 0
    tgt_max_tokens = 0
    with (
        src_path.open() as src_handle,
        tgt_path.open() as tgt_handle,
        out_src_path.open() as out_src_handle,
        out_tgt_path.open() as out_tgt_handle,
    ):
        while max_lines is None or pair_count < max_lines:
            src_line = src_handle.readline()
            tgt_line = tgt_handle.readline()
            out_src_line = out_src_handle.readline()
            out_tgt_line = out_tgt_handle.readline()
            if not any((src_line, tgt_line, out_src_line, out_tgt_line)):
                break
            if not all((src_line, tgt_line, out_src_line, out_tgt_line)):
                raise ValueError(
                    "source/target/output line-count mismatch at "
                    f"{split}:{pair_count + 1}"
                )

            pair_count += 1
            src_tokens = out_src_line.split()
            tgt_tokens = out_tgt_line.split()
            if "<GAP>" in src_tokens or "<GAP>" in tgt_tokens:
                raise ValueError(
                    f"<GAP> appeared in unaligned SPE tokens at "
                    f"{split}:{pair_count}"
                )
            if "".join(src_tokens) != restore_smiles(src_line):
                raise ValueError(
                    f"saved source SPE tokens do not reconstruct input at "
                    f"{split}:{pair_count}"
                )
            if "".join(tgt_tokens) != restore_smiles(tgt_line):
                raise ValueError(
                    f"saved target SPE tokens do not reconstruct input at "
                    f"{split}:{pair_count}"
                )
            src_token_count += len(src_tokens)
            tgt_token_count += len(tgt_tokens)
            src_max_tokens = max(src_max_tokens, len(src_tokens))
            tgt_max_tokens = max(tgt_max_tokens, len(tgt_tokens))

        # A partial --max-lines build deliberately ignores trailing source
        # records, but its saved output must contain exactly that many lines.
        if out_src_handle.readline() or out_tgt_handle.readline():
            raise ValueError(
                "existing tokenized output has more lines than requested at " f"{split}"
            )

    return {
        "pair_count": pair_count,
        "src_token_count": src_token_count,
        "tgt_token_count": tgt_token_count,
        "src_mean_tokens": src_token_count / pair_count if pair_count else 0.0,
        "tgt_mean_tokens": tgt_token_count / pair_count if pair_count else 0.0,
        "src_max_tokens": src_max_tokens,
        "tgt_max_tokens": tgt_max_tokens,
        "source_sha256": {
            "src": _sha256(src_path),
            "tgt": _sha256(tgt_path),
        },
        "output_sha256": {
            "src": _sha256(out_src_path),
            "tgt": _sha256(out_tgt_path),
        },
    }


def build_vocab(output_dir: Path, *, train_split: str = "train") -> dict:
    """作用：按训练源、目标 token 频次生成词表。输入：数据输出目录和训练分割名。输出：词表统计字典，并写入词表文件。"""
    counter: Counter[str] = Counter()
    for side in ("src", "tgt"):
        path = output_dir / train_split / f"{side}-{train_split}.txt"
        with path.open() as handle:
            for line in handle:
                counter.update(line.split())
    vocab_path = output_dir / "example.vocab.src"
    with vocab_path.open("w") as handle:
        for token, count in counter.most_common():
            handle.write(f"{token}\t{count}\n")
    return {
        "path": str(vocab_path),
        "token_count": len(counter),
        "model_vocab_size": len(counter) + 4,
        "sha256": _sha256(vocab_path),
        "total_training_token_count": sum(counter.values()),
    }


def _validate_paths(source_dir: Path, output_dir: Path, codes_path: Path) -> None:
    """作用：检查输入、输出和规则文件路径。输入：三个路径。输出：校验通过；无效时抛出异常。"""
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("output_dir must differ from the source #global# directory")
    if not codes_path.is_file():
        raise FileNotFoundError(codes_path)
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)


def _validate_options(
    *,
    splits: Sequence[str],
    merges: int,
    num_workers: int,
    max_lines: int | None,
    cache_reset_interval: int,
) -> None:
    """作用：检查预处理参数是否合法。输入：分割、合并数、worker 及缓存选项。输出：校验通过；无效时抛出异常。"""
    if max_lines is not None and max_lines < 1:
        raise ValueError("max_lines must be positive when provided")
    if merges < -1:
        raise ValueError("merges must be -1 (all rules) or non-negative")
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    if cache_reset_interval < 0:
        raise ValueError("cache_reset_interval must be non-negative")
    unknown_splits = sorted(set(splits) - set(SPLITS))
    if unknown_splits:
        raise ValueError(f"unknown split(s): {unknown_splits}")


def _make_metadata(
    *,
    source_dir: Path,
    output_dir: Path,
    codes_path: Path,
    merges: int,
    num_workers: int,
    max_lines: int | None,
    cache_reset_interval: int,
    split_stats: dict,
    vocab_stats: dict | None,
    finalized_existing_outputs: bool = False,
) -> dict:
    """作用：组装预处理记录。输入：路径、参数、分割统计和词表统计。输出：可写入 JSON 的元数据字典。"""
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "codes_path": str(codes_path.resolve()),
        "codes_sha256": _sha256(codes_path),
        "smilespe_version": "0.0.3",
        "merges": merges,
        "num_workers": num_workers,
        "dropout": 0,
        "max_lines": max_lines,
        "cache_reset_interval": cache_reset_interval,
        "source_kind": "unaligned src/tgt only; old aligned files ignored",
        "splits": split_stats,
        "vocab": vocab_stats,
    }
    if finalized_existing_outputs:
        metadata["finalized_existing_outputs"] = True
    return metadata


def preprocess(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    codes_path: Path = DEFAULT_CODES_PATH,
    *,
    splits: Sequence[str] = SPLITS,
    merges: int = -1,
    num_workers: int = 1,
    max_lines: int | None = None,
    cache_reset_interval: int = 50_000,
) -> dict:
    """作用：执行 SPE 预处理并生成训练词表。输入：源目录、输出目录及预处理参数。输出：元数据字典，并写入数据和元数据文件。"""
    _validate_paths(source_dir, output_dir, codes_path)
    _validate_options(
        splits=splits,
        merges=merges,
        num_workers=num_workers,
        max_lines=max_lines,
        cache_reset_interval=cache_reset_interval,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = _load_tokenizer(codes_path, merges=merges)
    split_stats = {}
    for split in splits:
        split_stats[split] = _tokenize_split(
            source_dir,
            output_dir,
            split,
            tokenizer,
            codes_path=codes_path,
            merges=merges,
            num_workers=num_workers,
            max_lines=max_lines,
            cache_reset_interval=cache_reset_interval,
        )
    vocab_stats = None
    if "train" in splits:
        vocab_stats = build_vocab(output_dir)

    metadata = _make_metadata(
        source_dir=source_dir,
        output_dir=output_dir,
        codes_path=codes_path,
        merges=merges,
        num_workers=num_workers,
        max_lines=max_lines,
        cache_reset_interval=cache_reset_interval,
        split_stats=split_stats,
        vocab_stats=vocab_stats,
    )
    metadata_path = output_dir / "spe_preprocessing_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def finalize_existing(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    codes_path: Path = DEFAULT_CODES_PATH,
    *,
    splits: Sequence[str] = SPLITS,
    merges: int = -1,
    num_workers: int = 1,
    max_lines: int | None = None,
    cache_reset_interval: int = 50_000,
) -> dict:
    """作用：校验已有 SPE 输出并补写词表与元数据。输入：源/输出目录及处理参数。输出：最终元数据字典，不重新切分数据。"""
    _validate_paths(source_dir, output_dir, codes_path)
    _validate_options(
        splits=splits,
        merges=merges,
        num_workers=num_workers,
        max_lines=max_lines,
        cache_reset_interval=cache_reset_interval,
    )

    split_stats = {
        split: _summarize_existing_split(
            source_dir,
            output_dir,
            split,
            max_lines=max_lines,
        )
        for split in splits
    }
    vocab_stats = build_vocab(output_dir) if "train" in splits else None
    metadata = _make_metadata(
        source_dir=source_dir,
        output_dir=output_dir,
        codes_path=codes_path,
        merges=merges,
        num_workers=num_workers,
        max_lines=max_lines,
        cache_reset_interval=cache_reset_interval,
        split_stats=split_stats,
        vocab_stats=vocab_stats,
        finalized_existing_outputs=True,
    )
    metadata_path = output_dir / "spe_preprocessing_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    """作用：解析命令行并启动预处理或恢复收尾。输入：可选命令行参数列表。输出：退出码，并写出预处理结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--codes", type=Path, default=DEFAULT_CODES_PATH)
    parser.add_argument(
        "--merges",
        type=int,
        default=500,
        help="Use only the first K merge rules; -1 uses the complete codes file",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel SPE tokenization workers; 1 preserves historical serial behavior",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Process only the first N paired lines per selected split",
    )
    parser.add_argument(
        "--cache-reset-interval",
        type=int,
        default=50_000,
        help="Clear SmilesPE's complete-SMILES cache periodically; 0 disables",
    )
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help=(
            "Validate existing unaligned output and write its vocabulary/metadata "
            "without re-tokenizing (interrupted-build recovery)"
        ),
    )
    args = parser.parse_args(argv)
    builder = finalize_existing if args.finalize_existing else preprocess
    metadata = builder(
        args.source_dir,
        args.output_dir,
        args.codes,
        splits=args.splits,
        merges=args.merges,
        num_workers=args.num_workers,
        max_lines=args.max_lines,
        cache_reset_interval=args.cache_reset_interval,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
