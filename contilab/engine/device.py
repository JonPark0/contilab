"""Backend selection: device, mixed precision and peak-memory probing."""

from __future__ import annotations

import resource
import sys
from pathlib import Path

import torch


def select_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cpu_has_native_bf16() -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return "amx_bf16" in flags or "avx512_bf16" in flags


def select_precision(pref: str, device: torch.device) -> tuple[torch.dtype | None, str]:
    """(autocast dtype or None for fp32, reason)."""
    if pref == "fp32":
        return None, "requested fp32"
    if pref == "bf16":
        return torch.bfloat16, "requested bf16"
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16, "auto: CUDA bf16 supported"
    if device.type == "cpu" and _cpu_has_native_bf16():
        return torch.bfloat16, "auto: CPU AMX/AVX512-BF16 detected"
    return None, f"auto: no native bf16 on {device.type} ({torch.backends.cpu.get_cpu_capability()})"


def peak_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)
