"""Experience Replay with an index reservoir; replay samples keep their own task masks."""

from __future__ import annotations

import torch

from contilab.core.scenario import sample_mask
from contilab.core.types import Batch
from contilab.strategies.base import Strategy, cross_entropy, derived_gens, register_strategy
from contilab.strategies.buffer import ReservoirBuffer

BUFFER_SALT = 7919


def cat_batches(a: Batch, b: Batch) -> Batch:
    """Concatenate along the batch axis (dim 1 when a member axis is present, else dim 0)."""
    d = 1 if a.y.dim() == 2 else 0

    def c(u, v):
        return None if u is None else torch.cat([u, v], d)

    return Batch(c(a.x, b.x), c(a.y, b.y), c(a.task_id, b.task_id), c(a.attn_mask, b.attn_mask),
                 c(a.index, b.index), c(a.mask, b.mask))


class ReplayStrategy(Strategy):
    """Shared buffer plumbing for ER / DER++ / A-GEM."""

    store_logits = False

    def __init__(self, M: int = 500, replay_batch_size: int = 32):
        super().__init__()
        self.M, self.replay_batch_size = M, replay_batch_size
        self.buffer: ReservoirBuffer | None = None
        self._gens: list[torch.Generator] | None = None

    def before_task(self, ctx, task):
        if self.buffer is None:
            C = ctx.stream.num_classes if self.store_logits else None
            self.buffer = ReservoirBuffer(ctx.E, self.M, C, ctx.pool.x.device)
            self._gens = derived_gens(ctx, BUFFER_SALT)

    def _replay(self, ctx, n: int) -> tuple[Batch, torch.Tensor] | None:
        if self.M == 0 or self.buffer is None:
            return None
        slots = self.buffer.sample(n, self._gens)
        if slots is None:
            return None
        b = ctx.pool.gather(self.buffer.gather(slots))
        b.mask = sample_mask(ctx.stream, b.task_id, ctx.task_idx)
        return b, slots

    def replay_batch(self, ctx):
        r = self._replay(ctx, self.replay_batch_size)
        return None if r is None else r[0]

    def after_step(self, ctx, batch, out):
        if self.M > 0:
            self.buffer.add(batch.index, out.logits if self.store_logits else None, self._gens)

    def state_dict(self):
        d = super().state_dict()
        d["buffer"] = self.buffer.state_dict() if self.buffer is not None else None
        d["gens"] = [g.get_state() for g in self._gens] if self._gens is not None else None
        return d

    def load_state_dict(self, d):
        super().load_state_dict(d)
        if d.get("buffer") is not None:
            t = d["buffer"]["tensors"]
            C = t["logits"].shape[-1] if "logits" in t else None
            self.buffer = ReservoirBuffer(t["gidx"].shape[0], self.M, C, t["gidx"].device)
            self.buffer.load_state_dict(d["buffer"])
        if d.get("gens") is not None:
            self._gens = [torch.Generator() for _ in d["gens"]]
            for g, s in zip(self._gens, d["gens"]):
                g.set_state(s)


@register_strategy("er")
class ER(ReplayStrategy):
    DEFAULTS = {"vision": {"M": 500, "replay_batch_size": 32}, "text": {"M": 200, "replay_batch_size": 16}}

    def loss(self, p, s, batch, replay, f):
        B = batch.y.shape[0]
        both = batch if replay is None else cat_batches(batch, replay)
        z = f.logits(p, both.x, both.attn_mask, both.mask)
        return cross_entropy(z, both.y), z[:B]
