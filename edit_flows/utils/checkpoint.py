"""用途：将训练状态安全地写入 checkpoint 文件。
输入：待保存的状态对象和目标文件路径。
输出：原子替换完成的 checkpoint 文件。
"""

import os
import tempfile
from pathlib import Path
from typing import Any
import torch


def atomic_torch_save(state, path: str | os.PathLike) -> None:
    """Serialize beside the destination, then atomically replace it.

    A same-directory temporary file keeps the rename on one filesystem.
    This protects readers and an existing best checkpoint from partial writes;
    it does not promise recovery from disk failure or concurrent writers.
    """
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
