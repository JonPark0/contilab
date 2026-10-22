"""Strategy contract (§5.1): host-side hooks plus a per-member hot path that the engine vmaps and compiles."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor
from torch.func import functional_call

from contilab.core.scenario import MASK_FILL
from contilab.core.types import Batch, RunContext, Stream, Task
from contilab.models.base import CLModel



class MemberFns:
    """Model calls for ONE member inside the vmapped hot path."""

    def __init__(self, template: CLModel, buffers: dict[str, Tensor]):
        self.template = template
        self.buffers = buffers

    def _call(self, p, x, attn_mask, mode):
        return functional_call(self.template, (p, self.buffers), (x,), {"attn_mask": attn_mask, "mode": mode})

    def logits(self, p: dict[str, Tensor], x: Tensor, attn_mask: Tensor | None, mask: Tensor | None) -> Tensor:
        z = self._call(p, x, attn_mask, "logits")
        return z if mask is None else z.masked_fill(~mask, MASK_FILL)

    def features(self, p: dict[str, Tensor], x: Tensor, attn_mask: Tensor | None) -> Tensor:
        return self._call(p, x, attn_mask, "features")


class Strategy:
    name = "base"
    vmappable = True
    uses_ref_grad = False
    DEFAULTS: dict[str, dict[str, Any]] = {"vision": {}, "text": {}}

    def __init__(self, **params: Any):
        if params:
            raise TypeError(f"{type(self).__name__} got unexpected params: {sorted(params)}")
        self.tensors: dict[str, Tensor] = {}
        self.meta: dict[str, Any] = {}

    # --- host side -------------------------------------------------------
    def prepare_model(self, model: CLModel, stream: Stream) -> CLModel:
        return model

    def trainable_names(self, names: list[str], task_idx: int) -> list[str]:
        return names

    def before_task(self, ctx: RunContext, task: Task) -> None:
        pass

    def replay_batch(self, ctx: RunContext) -> Batch | None:
        return None

    def after_step(self, ctx: RunContext, batch: Batch, out) -> None:
        pass

    def after_task(self, ctx: RunContext, task: Task) -> None:
        pass

    def eval_setup(self, ctx: RunContext, task_eval: int | None) -> None:
        """Called before evaluating task `task_eval`; None restores the training configuration."""

    def transform_grad(self, grads: dict[str, Tensor], ref_grads: dict[str, Tensor] | None) -> dict[str, Tensor]:
        return grads

    # --- hot path (one member; the engine vmaps over members) -------------
    def loss(self, p, s, batch: Batch, replay: Batch | None, f: MemberFns) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def ref_loss(self, p, s, replay: Batch, f: MemberFns) -> Tensor | None:
        return None

    # --- state -------------------------------------------------------------
    def state(self) -> dict[str, Tensor]:
        """Stacked [E, ...] tensors passed into the hot path."""
        return {}

    def state_dict(self) -> dict[str, Any]:
        return {"tensors": dict(self.tensors), "meta": copy.deepcopy(self.meta)}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.tensors = dict(d["tensors"])
        self.meta = copy.deepcopy(d["meta"])


STRATEGIES: dict[str, type[Strategy]] = {}


def register_strategy(name: str):
    def deco(cls: type[Strategy]) -> type[Strategy]:
        cls.name = name
        STRATEGIES[name] = cls
        return cls

    return deco


def build_strategy(name: str, params: dict[str, Any], modality: str) -> Strategy:
    import contilab.strategies  # noqa: F401  (registers built-ins)

    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy: {name}")
    cls = STRATEGIES[name]
    return cls(**{**cls.DEFAULTS.get(modality, {}), **params})


def derived_gens(ctx: RunContext, salt: int) -> list[torch.Generator]:
    """Per-member generators independent of the data-order generators (so strategies never perturb data order)."""
    return [torch.Generator().manual_seed((g.initial_seed() * 1_000_003 + salt) % (2**63)) for g in ctx.gens]


def cross_entropy(logits: Tensor, y: Tensor) -> Tensor:
    return torch.nn.functional.cross_entropy(logits, y)
