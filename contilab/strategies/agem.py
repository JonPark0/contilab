"""A-GEM: project the current gradient when it conflicts with the replay reference gradient (per member)."""

from __future__ import annotations

import torch
from torch import Tensor

from contilab.core.scenario import sample_mask
from contilab.strategies.base import cross_entropy, register_strategy
from contilab.strategies.er import ReplayStrategy


def project_agem(g: dict[str, Tensor], g_ref: dict[str, Tensor]) -> dict[str, Tensor]:
    keys = list(g)
    E = g[keys[0]].shape[0]
    dot = sum((g[k] * g_ref[k]).reshape(E, -1).sum(1) for k in keys)
    rr = sum(g_ref[k].pow(2).reshape(E, -1).sum(1) for k in keys)
    coef = torch.where(dot < 0, dot / (rr + 1e-12), torch.zeros_like(dot))
    return {k: g[k] - coef.view(E, *([1] * (g[k].dim() - 1))) * g_ref[k] for k in keys}


@register_strategy("agem")
class AGEM(ReplayStrategy):
    uses_ref_grad = True
    # text: a 256-sample reference gradient per step is ~17× a training step on a 70M LM (CPU budget)
    DEFAULTS = {"vision": {"M": 500, "ref_batch": 256}, "text": {"M": 200, "ref_batch": 64}}

    def __init__(self, M: int = 500, ref_batch: int = 256):
        super().__init__(M, replay_batch_size=ref_batch)

    def replay_batch(self, ctx):
        """Reference batch from the episodic memory of *previous* tasks only (Chaudhry et al. 2019)."""
        if self.M == 0 or self.buffer is None:
            return None
        gidx = self.buffer.tensors["gidx"]
        valid = ctx.pool.task_id[gidx.clamp(min=0)] < ctx.task_idx
        slots = self.buffer.sample(self.replay_batch_size, self._gens, valid)
        if slots is None:
            return None
        b = ctx.pool.gather(self.buffer.gather(slots))
        b.mask = sample_mask(ctx.stream, b.task_id, ctx.task_idx)
        return b

    def loss(self, p, s, batch, replay, f):
        z = f.logits(p, batch.x, batch.attn_mask, batch.mask)
        return cross_entropy(z, batch.y), z

    def ref_loss(self, p, s, replay, f):
        z = f.logits(p, replay.x, replay.attn_mask, replay.mask)
        return cross_entropy(z, replay.y)

    def transform_grad(self, grads, ref_grads):
        return grads if ref_grads is None else project_agem(grads, ref_grads)
