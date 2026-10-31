"""On-disk tensor cache for preprocessed datasets and experts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

from safetensors.torch import load_file, save_file
from torch import Tensor

from contilab.core.io import atomic_write, atomic_write_text


def cache_dir() -> Path:
    return Path(os.environ.get("CONTILAB_CACHE", Path.home() / ".cache" / "contilab"))


def key_hash(key: dict) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cached_tensors(namespace: str, key: dict, builder: Callable[[], dict[str, Tensor]]) -> dict[str, Tensor]:
    """Load tensors cached under (namespace, key) or build and store them."""
    d = cache_dir() / namespace
    path = d / f"{key_hash(key)}.safetensors"
    if path.exists():
        return load_file(str(path))
    tensors = builder()
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda p: save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(p)))
    atomic_write_text(d / f"{key_hash(key)}.json", json.dumps(key, sort_keys=True, default=str))
    return tensors
