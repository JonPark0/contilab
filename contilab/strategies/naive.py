from __future__ import annotations

from contilab.strategies.base import Strategy, cross_entropy, register_strategy


@register_strategy("naive")
class Naive(Strategy):
    """Plain fine-tuning (lower bound); replay is ignored."""

    def loss(self, p, s, batch, replay, f):
        z = f.logits(p, batch.x, batch.attn_mask, batch.mask)
        return cross_entropy(z, batch.y), z
