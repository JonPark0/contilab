"""Transfer data for distillation at three data-availability levels: full / buffer / proxy (§5.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from contilab.core.types import Stream, TensorSplit
from contilab.data.batching import TrainPool


@dataclass
class TransferSet:
    gidx: Tensor | None  # [S, N] global pool indices (full / buffer)
    proxy: TensorSplit | None  # shared unlabelled data (proxy)

    def __len__(self) -> int:
        return self.gidx.shape[1] if self.gidx is not None else len(self.proxy)


def build_transfer(stream: Stream, pool: TrainPool, upto: int, kind: str, M: int, gens: list[torch.Generator],
                   tokenizer: str | None = None, max_len: int = 128) -> TransferSet:
    dev = pool.x.device
    if kind == "full":
        n = int(pool.offsets[upto + 1])
        return TransferSet(torch.arange(n, device=dev).expand(len(gens), n).clone(), None)
    if kind == "buffer":
        per = M // (upto + 1)
        rows = []
        for g in gens:
            parts = [torch.randperm(pool.task_size(t), generator=g)[:per] + int(pool.offsets[t]) for t in range(upto + 1)]
            rows.append(torch.cat(parts))
        return TransferSet(torch.stack(rows).to(dev), None)
    if kind == "proxy":
        from contilab.data.proxy import build_proxy

        return TransferSet(None, build_proxy(stream.name, M, tokenizer, max_len).to(dev))
    raise ValueError(f"unknown transfer kind: {kind}")
