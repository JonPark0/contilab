from __future__ import annotations

import torch
from torch import Tensor

from contilab.core.config import OptimConfig


def build_optimizer(params: list[Tensor], cfg: OptimConfig) -> torch.optim.Optimizer:
    """Element-wise optimizers only, so stacked members never interact."""
    if cfg.name == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, foreach=True)
    if cfg.name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, foreach=True)
    raise ValueError(f"unknown optimizer: {cfg.name}")


@torch.no_grad()
def clip_per_member_(grads: dict[str, Tensor], max_norm: float) -> None:
    """Clip each member's global gradient norm independently."""
    gs = list(grads.values())
    E = gs[0].shape[0]
    sq = torch.zeros(E, device=gs[0].device)
    for g in gs:
        sq += g.reshape(E, -1).float().pow(2).sum(1)
    scale = (max_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)
    for g in gs:
        g.mul_(scale.view(E, *([1] * (g.dim() - 1))).to(g.dtype))
