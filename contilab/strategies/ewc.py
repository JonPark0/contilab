"""Online EWC: diagonal empirical Fisher from per-sample gradients computed with nested vmap.

Under plain SGD the penalty ½·λ·F·(θ − θ*)² multiplies a parameter's distance from its anchor by (1 − lr·λ·F) per
step, so a parameter whose stiffness lr·λ·F exceeds 2 oscillates with growing amplitude and the run diverges. The
online Fisher only grows from task to task (γ = 1 sums it), so a long stream crosses that bound sooner or later.
The stored Fisher is therefore clamped at max_stiffness / (lr·λ) when the optimiser is SGD: with the default 1, the
stiffest parameters return to their anchor in one step, the most rigid an SGD step can be. Adam-type optimisers
normalise the step, so the bound does not apply to them and nothing is clamped.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import grad, vmap

from contilab.core.scenario import sample_mask
from contilab.strategies.base import MemberFns, Strategy, cross_entropy, derived_gens, register_strategy

FISHER_CHUNK = 128
FISHER_BUDGET_BYTES = 2**30  # per-sample gradients held at once (fp32)


def fisher_chunk(n_params: int, budget_bytes: int = FISHER_BUDGET_BYTES) -> int:
    return max(1, min(FISHER_CHUNK, budget_bytes // (4 * n_params)))


@register_strategy("ewc")
class EWC(Strategy):
    DEFAULTS = {"vision": {"lam": 30.0, "gamma": 1.0, "fisher_n": 1024, "max_stiffness": 1.0},
                "text": {"lam": 10.0, "gamma": 1.0, "fisher_n": 1024, "max_stiffness": 1.0}}

    def __init__(self, lam: float = 100.0, gamma: float = 1.0, fisher_n: int = 1024,
                 max_stiffness: float | None = 1.0):
        super().__init__()
        self.lam, self.gamma, self.fisher_n, self.max_stiffness = lam, gamma, fisher_n, max_stiffness

    def fisher_cap(self, ctx) -> float | None:
        """The largest Fisher value that keeps lr·λ·F ≤ max_stiffness under SGD (None: no clamp)."""
        opt = getattr(ctx, "optim", None)
        if self.max_stiffness is None or opt is None or opt.name != "sgd" or self.lam <= 0 or opt.lr <= 0:
            return None
        return self.max_stiffness / (opt.lr * self.lam)

    def state(self):
        return self.tensors

    def loss(self, p, s, batch, replay, f):
        z = f.logits(p, batch.x, batch.attn_mask, batch.mask)
        loss = cross_entropy(z, batch.y)
        pen = [(s[f"F.{k}"] * (p[k] - s[f"star.{k}"]).pow(2)).sum() for k in p if f"F.{k}" in s]
        if pen:
            loss = loss + 0.5 * self.lam * torch.stack(pen).sum()
        return loss, z

    def after_task(self, ctx, task):
        ens = ctx.ens
        names = [k for k in ens.params if k not in ens.template.frozen_names()]
        n = min(self.fisher_n, ctx.pool.task_size(ctx.task_idx))
        off = ctx.pool.offsets[ctx.task_idx]
        gidx = torch.stack([torch.randperm(ctx.pool.task_size(ctx.task_idx), generator=g)[:n] for g in derived_gens(ctx, ctx.task_idx)]) + off
        batch = ctx.pool.gather(gidx.to(ctx.pool.x.device))
        mask = sample_mask(ctx.stream, batch.task_id, ctx.task_idx)
        fisher = self._fisher(ens, names, batch.x, batch.attn_mask, batch.y, mask)
        cap = self.fisher_cap(ctx)
        for k in names:
            prev = self.tensors.get(f"F.{k}")
            acc = fisher[k] if prev is None else self.gamma * prev + fisher[k]
            self.tensors[f"F.{k}"] = acc if cap is None else acc.clamp(max=cap)
            self.tensors[f"star.{k}"] = ens.params[k].detach().clone()

    @staticmethod
    def _fisher(ens, names, x, am, y, mask):
        tpl = ens.template

        def sample_sq_grad(p_train, p_rest, b, xi, ami, yi, mi):
            fns = MemberFns(tpl, b)

            def l(pt):
                z = fns.logits({**p_rest, **pt}, xi[None], None if ami is None else ami[None], mi[None])
                return F.cross_entropy(z, yi[None])

            return {k: g.pow(2) for k, g in grad(l)(p_train).items()}

        am_dim = None if am is None else 0
        per_sample = vmap(sample_sq_grad, in_dims=(None, None, None, 0, am_dim, 0, 0))

        chunk = fisher_chunk(sum(ens.params[k][0].numel() for k in names))

        def member_fisher(p_train, p_rest, b, xs, ams, ys, ms):
            tot = {k: torch.zeros_like(v) for k, v in p_train.items()}
            for i in range(0, xs.shape[0], chunk):
                sl = slice(i, i + chunk)
                g = per_sample(p_train, p_rest, b, xs[sl], None if ams is None else ams[sl], ys[sl], ms[sl])
                for k in tot:
                    tot[k] = tot[k] + g[k].sum(0)
            return {k: v / xs.shape[0] for k, v in tot.items()}

        p = {k: v.detach() for k, v in ens.params.items()}
        p_train = {k: p[k] for k in names}
        p_rest = {k: v for k, v in p.items() if k not in p_train}
        out = []
        for e in range(ens.E):  # members loop on the host; samples are vmapped
            pick = lambda d: {k: v[e] for k, v in d.items()}
            out.append(member_fisher(pick(p_train), pick(p_rest), pick(ens.buffers), x[e],
                                     None if am is None else am[e], y[e], mask[e]))
        return {k: torch.stack([o[k] for o in out]) for k in names}
