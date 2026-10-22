"""Task-vector merging (§5.5): Task Arithmetic and TIES-Merging, all ops keep the leading member axis."""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import Tensor


def task_vector(base: dict[str, Tensor], tuned: dict[str, Tensor], names: list[str]) -> dict[str, Tensor]:
    return {k: tuned[k] - base[k] for k in names}


def merge_ta(tvs: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {k: torch.stack([tv[k] for tv in tvs]).sum(0) for k in tvs[0]}


def _trim(tv: dict[str, Tensor], k: float) -> dict[str, Tensor]:
    """Keep the top-k fraction of entries by magnitude, per member, across all tensors of the vector."""
    if k >= 1.0:
        return tv
    keys = list(tv)
    E = tv[keys[0]].shape[0]
    flat = torch.cat([tv[n].reshape(E, -1) for n in keys], 1)
    keep_n = max(1, int(round(k * flat.shape[1])))
    idx = flat.abs().topk(keep_n, dim=1).indices
    mask = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, idx, True)
    out, off = {}, 0
    for n in keys:
        size = tv[n][0].numel()
        out[n] = tv[n] * mask[:, off : off + size].view_as(tv[n])
        off += size
    return out


def merge_ties(tvs: list[dict[str, Tensor]], k: float = 0.2) -> dict[str, Tensor]:
    trimmed = [_trim(tv, k) for tv in tvs]
    out = {}
    for n in tvs[0]:
        stack = torch.stack([t[n] for t in trimmed])  # [K, E, ...]
        sign = torch.sign(stack.sum(0))
        agree = (torch.sign(stack) == sign) & (stack != 0)
        cnt = agree.sum(0).clamp(min=1)
        out[n] = (stack * agree).sum(0) / cnt
    return out


def apply_tv(base: dict[str, Tensor], tv: dict[str, Tensor], lam: Tensor | float) -> dict[str, Tensor]:
    out = dict(base)
    for k, v in tv.items():
        lam_k = lam.to(v.device, v.dtype).view(-1, *([1] * (v.dim() - 1))) if isinstance(lam, Tensor) else lam
        out[k] = base[k] + lam_k * v
    return out


@torch.no_grad()
def select_lambda(build: Callable[[float], dict[str, Tensor]], score: Callable[[dict[str, Tensor]], Tensor],
                  grid: Sequence[float]) -> tuple[Tensor, dict[str, Tensor]]:
    """Per-member best λ on the grid (ties → smaller λ); returns (λ [E], assembled params)."""
    best_s, best_lam, best_p = None, None, None
    for lam in sorted(grid):
        p = build(lam)
        s = score(p).float().cpu()
        if best_s is None:
            best_s, best_lam = s, torch.full_like(s, lam)
            best_p = {k: v.clone() for k, v in p.items()}
            continue
        better = s > best_s + 1e-12
        if better.any():
            best_s = torch.where(better, s, best_s)
            best_lam = torch.where(better, torch.full_like(s, lam), best_lam)
            for k, v in p.items():
                best_p[k][better.to(v.device)] = v[better.to(v.device)]
    return best_lam, best_p
