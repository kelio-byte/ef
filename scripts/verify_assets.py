"""用途：校验仓库冻结数据和 checkpoint 的完整性。
输入：assets.json 中列出的文件及 SHA-256 校验值。
输出：校验结果；文件缺失或内容不符时报告错误。
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    """作用：计算文件的 SHA-256。输入：文件路径。输出：十六进制校验和。"""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    """作用：核对资产清单中的文件和 dev1000 划分。输入：仓库资产与清单。输出：校验信息；不匹配时抛出异常。"""
    manifest = json.loads((ROOT / "assets.json").read_text())
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing file or forbidden symlink: {relative}")
        if (
            path.stat().st_size != expected["bytes"]
            or digest(path) != expected["sha256"]
        ):
            raise ValueError(f"Asset checksum mismatch: {relative}")
    split = json.loads((ROOT / "data/uspto50k_m500/dev1000/manifest.json").read_text())
    indices = split["original_reaction_indices"]
    if len(indices) != 1000 or len(set(indices)) != 1000:
        raise ValueError("dev1000 must contain 1000 distinct reaction indices")
    for side in ("src", "tgt"):
        source = (ROOT / split["source"][f"{side}_path"]).read_text().splitlines()
        expected = [line for i in indices for line in source[20 * i : 20 * (i + 1)]]
        actual = (ROOT / split["output"][f"{side}_path"]).read_text().splitlines()
        if len(actual) != 20000 or actual != expected:
            raise ValueError(
                f"dev1000 {side} differs from the frozen validation projection"
            )
    print(
        f"OK: {len(manifest['files'])} assets verified; dev1000 = 1000 reactions × 20 views"
    )


if __name__ == "__main__":
    main()
