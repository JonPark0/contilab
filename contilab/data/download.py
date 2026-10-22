"""Plain HTTPS downloads into the contilab cache (no hub client needed)."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from contilab.data.cache import cache_dir


def download_error(url: str, err: Exception) -> RuntimeError:
    return RuntimeError(
        f"failed to download '{url}': {err}. Cache directory: {cache_dir()}. "
        "Run `contilab data prepare <benchmark>` on a machine with network access and copy the cache directory here."
    )


def fetch(url: str) -> Path:
    """Download url once into <cache>/downloads/ and return the local path."""
    dest = cache_dir() / "downloads" / url.rsplit("/", 1)[-1]
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise download_error(url, e) from e
    tmp.replace(dest)
    return dest
