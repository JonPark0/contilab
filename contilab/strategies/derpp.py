"""DER++: logit (MSE) + label replay from a reservoir that also stores logits at insertion time."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from contilab.strategies.base import MASK_FILL, cross_entropy, register_strategy
from contilab.strategies.er import ReplayStrategy, cat_batches


@register_strategy("derpp")
class DERpp(ReplayStrategy):
    store_logits = True
    DEFAULTS = {"vision": {"M": 500, "alpha": 0.1, "beta": 0.5, "replay_batch_size": 32},
                "text": {"M": 200, "alpha": 0.1, "beta": 0.5, "replay_batch_size": 16}}

    def __init__(self, M: int = 500, alpha: float = 0.1, beta: float = 0.5, replay_batch_size: int = 32):
        super().__init__(M, replay_batch_size)
        self.alpha, self.beta = alpha, beta

    def replay_batch(self, ctx):
        if self.alpha == 0 and self.beta == 0:
            return None
        r1, r2 = self._replay(ctx, self.replay_batch_size), self._replay(ctx, self.replay_batch_size)
        if r1 is None:
            return None
        (b1, s1), (b2, _) = r1, r2
        out = cat_batches(b1, b2)
        out.extra = {"buf_logits": self.buffer.gather(s1, "logits")}
        return out

    def loss(self, p, s, batch, replay, f):
        B = batch.y.shape[0]
        both = batch if replay is None else cat_batches(batch, replay)
        z_raw = f.logits(p, both.x, both.attn_mask, None)
        z = z_raw.masked_fill(~both.mask, MASK_FILL)
        loss = cross_entropy(z[:B], batch.y)
        if replay is not None:
            buf = replay.extra["buf_logits"]
            n1 = buf.shape[0]
            loss = loss + self.alpha * F.mse_loss(z_raw[B : B + n1], buf.to(z_raw.dtype))
            loss = loss + self.beta * cross_entropy(z[B + n1 :], both.y[B + n1 :])
        return loss, z_raw[:B]
