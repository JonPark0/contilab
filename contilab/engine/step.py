"""One optimisation step for all E members: vmapped (and optionally compiled) loss, autograd, element-wise update."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor
from torch.func import vmap

from contilab.core.config import EngineConfig, OptimConfig
from contilab.core.types import Batch, RunContext, Stream
from contilab.engine.device import select_precision
from contilab.engine.ensemble import Ensemble
from contilab.engine.optim import build_optimizer, clip_per_member_
from contilab.strategies.base import MemberFns, Strategy

_FIELDS = ("x", "y", "task_id", "attn_mask", "index", "mask")


@dataclass
class StepOutput:
    loss: Tensor  # [E]
    logits: Tensor  # [E, B, C]


def batch_tree(b: Batch | None) -> dict[str, Any] | None:
    if b is None:
        return None
    d = {k: getattr(b, k) for k in _FIELDS if getattr(b, k) is not None}
    if b.extra:
        d["extra"] = b.extra
    return d


def tree_batch(d: dict[str, Any] | None) -> Batch | None:
    if d is None:
        return None
    return Batch(**{k: d.get(k) for k in _FIELDS}, extra=d.get("extra"))


def _index0(tree):
    return torch.utils._pytree.tree_map(lambda t: t[0], tree)


def maybe_compile(fn: Callable, mode: str) -> tuple[Callable, dict[str, Any]]:
    if mode == "off":
        return fn, {"requested": mode, "used": False, "reason": "compile disabled"}
    return torch.compile(fn, dynamic=False), {"requested": mode, "used": True, "reason": "pending first call"}


class TrainStep:
    def __init__(self, ens: Ensemble, strategy: Strategy, stream: Stream, engine_cfg: EngineConfig, optim_cfg: OptimConfig):
        self.ens, self.strategy, self.stream = ens, strategy, stream
        self.optim_cfg = optim_cfg
        self.frozen = ens.template.frozen_names()
        self.autocast_dtype, self.precision_reason = select_precision(engine_cfg.precision, ens.device)
        self._eager = (self._batched_loss, self._batched_ref)
        fwd, self.compile_decision = maybe_compile(self._batched_loss, engine_cfg.compile)
        ref, _ = maybe_compile(self._batched_ref, engine_cfg.compile)
        self._fwd, self._ref = fwd, ref
        self._first_call = self.compile_decision["used"]
        self.trainable: list[str] = []
        self.opt: torch.optim.Optimizer | None = None
        self.count_flops = False
        self.flops_per_step: int | None = None  # per member; max over the step structures seen so far
        self._flops_seen: set = set()

    # --- hot path ----------------------------------------------------------
    def _member_loss(self, p, b, s, bt, rt):
        return self.strategy.loss(p, s, tree_batch(bt), tree_batch(rt), MemberFns(self.ens.template, b))

    def _member_ref(self, p, b, s, rt):
        return self.strategy.ref_loss(p, s, tree_batch(rt), MemberFns(self.ens.template, b))

    def _batched_loss(self, params, buffers, state, bt, rt):
        if self.ens.E == 1:
            rt0 = _index0(rt) if rt is not None else None
            loss, z = self._member_loss(_index0(params), _index0(buffers), _index0(state), _index0(bt), rt0)
            return loss.unsqueeze(0), z.unsqueeze(0)
        r_dim = None if rt is None else 0
        return vmap(self._member_loss, in_dims=(0, 0, 0, 0, r_dim), randomness="different")(params, buffers, state, bt, rt)

    def _batched_ref(self, params, buffers, state, rt):
        if self.ens.E == 1:
            return self._member_ref(_index0(params), _index0(buffers), _index0(state), _index0(rt)).unsqueeze(0)
        return vmap(self._member_ref, in_dims=(0, 0, 0, 0), randomness="different")(params, buffers, state, rt)

    # --- host side -----------------------------------------------------------
    def set_task(self, task_idx: int) -> None:
        names = [n for n in self.strategy.trainable_names(list(self.ens.params), task_idx) if n not in self.frozen]
        self.trainable = names
        self.opt = build_optimizer([self.ens.params[n] for n in names], self.optim_cfg)

    def _autocast(self):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.ens.device.type, dtype=self.autocast_dtype)

    def _grads(self, vec: Tensor) -> dict[str, Tensor]:
        ts = [self.ens.params[n] for n in self.trainable]
        gs = torch.autograd.grad(vec.sum(), ts, allow_unused=True)
        return {n: (g if g is not None else torch.zeros_like(t)) for n, g, t in zip(self.trainable, gs, ts)}

    def _compute(self, bt, rt, state):
        params, buffers = self.ens.params, self.ens.buffers
        with self._autocast():
            loss_vec, logits = self._fwd(params, buffers, state, bt, rt)
        grads = self._grads(loss_vec)
        ref = None
        if self.strategy.uses_ref_grad and rt is not None:
            with self._autocast():
                ref_vec = self._ref(params, buffers, state, rt)
            ref = self._grads(ref_vec)
        return loss_vec, logits, grads, ref

    def measure_flops(self, bt, rt, state) -> int:
        """FLOPs of one eager forward+backward (all members, incl. reference gradients); no update."""
        from torch.utils.flop_counter import FlopCounterMode

        fwd, ref = self._fwd, self._ref
        self._fwd, self._ref = self._eager
        try:
            with FlopCounterMode(display=False) as fc:
                self._compute(bt, rt, state)
        finally:
            self._fwd, self._ref = fwd, ref
        return int(fc.get_total_flops())

    def __call__(self, batch: Batch, replay: Batch | None, ctx: RunContext) -> StepOutput:
        bt, rt, state = batch_tree(batch), batch_tree(replay), self.strategy.state()
        if self.count_flops:
            sig = (tuple(sorted(bt)), None if rt is None else tuple(sorted(rt)), tuple(sorted(state)))
            if sig not in self._flops_seen:  # e.g. replay appears at step 1, LwF/EWC state at task 1
                self._flops_seen.add(sig)
                per_member = self.measure_flops(bt, rt, state) // self.ens.E
                self.flops_per_step = max(self.flops_per_step or 0, per_member)
        if self.compile_decision["used"]:
            # compile failures fall back to eager (§7.5) — at the first call or at any later recompilation
            # (new replay / state structure); a genuine bug re-raises from the eager retry
            try:
                out = self._compute(bt, rt, state)
                if self._first_call:
                    self.compile_decision["reason"] = "compiled"
            except Exception as e:
                self._fwd, self._ref = self._eager
                self.compile_decision.update(used=False, reason=f"fallback to eager at step {ctx.step}: "
                                                                f"{type(e).__name__}: {str(e)[:300]}")
                out = self._compute(bt, rt, state)
            self._first_call = False
        else:
            out = self._compute(bt, rt, state)
        loss_vec, logits, grads, ref = out
        grads = self.strategy.transform_grad(grads, ref)
        self._apply(grads, ctx)
        return StepOutput(loss_vec.detach().float(), logits.detach())

    def _apply(self, grads: dict[str, Tensor], ctx: RunContext) -> None:
        params, E = self.ens.params, self.ens.E
        all_active = bool(ctx.active.all())
        saved = {}
        if not all_active:
            keep = ctx.active.to(next(iter(grads.values())).device)
            for n, g in grads.items():
                grads[n] = torch.nan_to_num(g) * keep.view(E, *([1] * (g.dim() - 1))).to(g.dtype)
                saved[n] = params[n].detach()[~keep].clone()
        if self.optim_cfg.clip is not None:
            clip_per_member_(grads, self.optim_cfg.clip)
        for n, g in grads.items():
            params[n].grad = g
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        if saved:
            with torch.no_grad():
                inactive = ~ctx.active.to(params[self.trainable[0]].device)
                for n, v in saved.items():
                    params[n][inactive] = v
