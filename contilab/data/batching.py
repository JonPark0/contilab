"""DataLoader-free batching over device-resident tensors with a leading member axis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor

from contilab.core.types import Batch, Stream, TensorSplit


@dataclass
class TrainPool:
    """All tasks' training data concatenated; samples are addressed by global index."""

    x: Tensor
    y: Tensor
    task_id: Tensor
    attn_mask: Tensor | None
    offsets: Tensor  # Long[T+1]

    @classmethod
    def from_stream(cls, stream: Stream) -> TrainPool:
        tr = [t.train for t in stream.tasks]
        sizes = torch.tensor([len(s) for s in tr])
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)])
        device = tr[0].x.device
        am = torch.cat([s.attn_mask for s in tr]) if tr[0].attn_mask is not None else None
        task_id = torch.repeat_interleave(torch.arange(len(tr)), sizes).to(device)
        return cls(torch.cat([s.x for s in tr]), torch.cat([s.y for s in tr]), task_id, am, offsets)

    def task_size(self, t: int) -> int:
        return int(self.offsets[t + 1] - self.offsets[t])

    def gather(self, gidx: Tensor) -> Batch:
        am = self.attn_mask[gidx] if self.attn_mask is not None else None
        return Batch(x=self.x[gidx], y=self.y[gidx], task_id=self.task_id[gidx], attn_mask=am, index=gidx)


def epoch_perms(n: int, gens: list[torch.Generator]) -> Tensor:
    """Long[E, n]: one permutation per member, each drawn from that member's generator."""
    return torch.stack([torch.randperm(n, generator=g) for g in gens])


def task_batches(pool: TrainPool, task_idx: int, batch_size: int, gens: list[torch.Generator]) -> Iterator[Batch]:
    n = pool.task_size(task_idx)
    perms = epoch_perms(n, gens).to(pool.x.device) + pool.offsets[task_idx].to(pool.x.device)
    for i in range(0, n, batch_size):
        yield pool.gather(perms[:, i : i + batch_size])


def eval_batches(split: TensorSplit, batch_size: int) -> Iterator[tuple[Tensor, Tensor, Tensor | None]]:
    for i in range(0, len(split), batch_size):
        am = split.attn_mask[i : i + batch_size] if split.attn_mask is not None else None
        yield split.x[i : i + batch_size], split.y[i : i + batch_size], am
