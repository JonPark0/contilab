"""Learning without Forgetting: KD from a frozen snapshot of the previous model on the current batch."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from contilab.strategies.base import MASK_FILL, Strategy, cross_entropy, register_strategy


def kd_loss(z_student, z_teacher, mask, T: float):
    """T²·KL(softmax(z_t/T) ‖ softmax(z_s/T)) restricted to `mask` classes."""
    zs = z_student.masked_fill(~mask, MASK_FILL) / T
    zt = z_teacher.masked_fill(~mask, MASK_FILL) / T
    pt = zt.softmax(-1)
    return (T * T) * (pt * (pt.clamp_min(1e-12).log() - zs.log_softmax(-1))).sum(-1).mean()


@register_strategy("lwf")
class LwF(Strategy):
    DEFAULTS = {"vision": {"T": 2.0, "alpha": 1.0}, "text": {"T": 2.0, "alpha": 1.0}}

    def __init__(self, T: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.T, self.alpha = T, alpha

    def state(self):
        return self.tensors

    def loss(self, p, s, batch, replay, f):
        z_raw = f.logits(p, batch.x, batch.attn_mask, None)  # one student forward serves both terms
        z = z_raw.masked_fill(~batch.mask, MASK_FILL)
        loss = cross_entropy(z, batch.y)
        if "old_mask" in s and self.alpha > 0:
            old = {k[4:]: v for k, v in s.items() if k.startswith("old.")}
            z_old = f.logits(old, batch.x, batch.attn_mask, None).detach()
            mask = s["old_mask"].expand_as(z_raw)
            loss = loss + self.alpha * kd_loss(z_raw, z_old, mask, self.T)
        return loss, z

    def after_task(self, ctx, task):
        ens = ctx.ens
        for k, v in ens.params.items():
            self.tensors[f"old.{k}"] = v.detach().clone()
        seen = ctx.stream.class_table()[: ctx.task_idx + 1].any(0).to(ctx.device)
        self.tensors["old_mask"] = seen.expand(ens.E, -1).clone()
