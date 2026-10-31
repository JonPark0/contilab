"""Crash-safe file writes: write a temporary sibling, then atomically replace the target."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


def atomic_write(path: Path | str, write: Callable[[Path], None]) -> None:
    """`write(tmp)` fills a per-process temporary file that then replaces `path` (readers never see a partial file)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path | str, text: str) -> None:
    atomic_write(path, lambda p: p.write_text(text))
