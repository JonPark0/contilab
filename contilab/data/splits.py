from __future__ import annotations

import torch

from contilab.core.types import TensorSplit


def train_val_split(split: TensorSplit, frac: float = 0.1, seed: int = 0) -> tuple[TensorSplit, TensorSplit]:
    n = len(split)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = int(round(n * frac))
    return split.subset(perm[n_val:].sort().values), split.subset(perm[:n_val].sort().values)


def subsample(split: TensorSplit, n: int | None, seed: int = 0) -> TensorSplit:
    if n is None or n >= len(split):
        return split
    perm = torch.randperm(len(split), generator=torch.Generator().manual_seed(seed))
    return split.subset(perm[:n].sort().values)
