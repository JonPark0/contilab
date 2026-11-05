"""GEM (Lopez-Paz & Ranzato, 2017): one reference gradient per previous task; when the current gradient conflicts
with any of them, it is projected by the dual QP of the paper, with margin and ridge as in the reference code."""

from __future__ import annotations

import itertools

import torch
from torch import Tensor

from contilab.core.scenario import sample_mask
from contilab.strategies.agem import AGEM
from contilab.strategies.base import register_strategy

EXACT_MAX_K = 10  # 2^K candidate active sets; beyond this the QP falls back to projected coordinate descent


def _free_sets(K: int, device) -> Tensor:
    return torch.tensor(list(itertools.product([False, True], repeat=K)), dtype=torch.bool, device=device)


def solve_gem_dual(P: Tensor, d: Tensor, margin: float, eps: float = 1e-3) -> Tensor:
    """argmin_v ½ vᵀ(P + eps·I)v + dᵀv  s.t.  v ≥ margin, per member. P [E, K, K], d [E, K] → v [E, K].

    Exact for K ≤ EXACT_MAX_K: the optimum solves the reduced linear system of its free set (v_i > margin), so every
    free set is solved, its solution clipped to the feasible region, and the lowest objective wins."""
    E, K = d.shape
    P = 0.5 * (P + P.transpose(1, 2)) + eps * torch.eye(K, dtype=P.dtype, device=P.device)
    if K > EXACT_MAX_K:
        v = torch.full_like(d, margin)
        for _ in range(500):
            for i in range(K):
                g_i = (P[:, i] * v).sum(1) + d[:, i]
                v[:, i] = (v[:, i] - g_i / P[:, i, i]).clamp(min=margin)
        return v
    S = _free_sets(K, P.device)  # [N, K]
    both = (S[:, :, None] & S[:, None, :]).to(P.dtype)  # [N, K, K]
    eye = torch.eye(K, dtype=P.dtype, device=P.device)
    A = P[:, None] * both[None] + eye * (~S).to(P.dtype)[None, :, :, None]  # fixed rows → identity
    fixed = margin * (~S).to(P.dtype)  # [N, K]
    rhs = -(d[:, None, :] + torch.einsum("eij,nj->eni", P, fixed))
    b = torch.where(S[None], rhs, torch.full_like(rhs, margin))
    v = torch.linalg.solve(A, b).clamp(min=margin)  # [E, N, K]
    obj = 0.5 * torch.einsum("eni,eij,enj->en", v, P, v) + (v * d[:, None, :]).sum(-1)
    return v[torch.arange(E, device=v.device), obj.argmin(1)]


def project_gem(g: dict[str, Tensor], G: dict[str, Tensor], margin: float = 0.5, eps: float = 1e-3) -> dict[str, Tensor]:
    """g: {name: [E, ...]}, G: {name: [E, K, ...]} reference gradients. Members whose gradient conflicts with no
    reference gradient keep it unchanged; the others get g + Σ_k v_k G_k."""
    keys = list(g)
    E, K = G[keys[0]].shape[:2]
    flat = {k: G[k].reshape(E, K, -1) for k in keys}
    dot = sum(torch.bmm(flat[k], g[k].reshape(E, -1, 1)).squeeze(-1) for k in keys)  # [E, K]
    conflict = (dot < 0).any(1)
    if not bool(conflict.any()):
        return g
    P = sum(torch.bmm(flat[k], flat[k].transpose(1, 2)) for k in keys)
    v = solve_gem_dual(P.double(), dot.double(), margin, eps)
    v = torch.where(conflict[:, None], v, torch.zeros_like(v))
    out = {}
    for k in keys:
        coef = v.to(G[k].dtype)
        out[k] = g[k] + torch.bmm(coef[:, None, :], flat[k]).reshape_as(g[k])
    return out


@register_strategy("gem")
class GEM(AGEM):
    """Same episodic memory and reference-batch size as A-GEM; the reference batch is split evenly over the previous
    tasks, and each task's part gives its own gradient constraint."""

    DEFAULTS = {"vision": {"M": 500, "ref_batch": 256}, "text": {"M": 200, "ref_batch": 64}}

    def __init__(self, M: int = 500, ref_batch: int = 256, margin: float = 0.5, eps: float = 1e-3):
        super().__init__(M, ref_batch)
        self.margin, self.eps = margin, eps
        self._groups: list[int] | None = None

    def replay_batch(self, ctx):
        self._groups = None
        if self.M == 0 or self.buffer is None or ctx.task_idx == 0:
            return None
        gidx = self.buffer.tensors["gidx"]
        tid = ctx.pool.task_id[gidx.clamp(min=0)]
        per = max(1, self.replay_batch_size // ctx.task_idx)
        parts, sizes = [], []
        for k in range(ctx.task_idx):
            slots = self.buffer.sample(per, self._gens, tid == k)
            if slots is not None:
                parts.append(slots)
                sizes.append(slots.shape[1])
        if not parts:
            return None
        b = ctx.pool.gather(self.buffer.gather(torch.cat(parts, 1)))
        b.mask = sample_mask(ctx.stream, b.task_id, ctx.task_idx)
        self._groups = sizes
        return b

    def ref_groups(self):
        return self._groups

    def transform_grad(self, grads, ref_grads):
        return grads if ref_grads is None else project_gem(grads, ref_grads, self.margin, self.eps)
