"""Stacked-parameter ensemble: E independent models evaluated by one vmapped functional call (§4.4)."""

from __future__ import annotations

import copy
from typing import Callable

import torch
from torch import Tensor
from torch.func import functional_call, stack_module_state, vmap
from torch.utils._pytree import tree_map

from contilab.models.base import CLModel


def _set_tensor(module: torch.nn.Module, name: str, value: Tensor) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    if leaf in parent._parameters:
        parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=value.requires_grad)
    else:
        parent._buffers[leaf] = value


class Ensemble:
    def __init__(self, factory: Callable[[int], CLModel], init_seeds: list[int], device: torch.device):
        models = [factory(s).to(device) for s in init_seeds]
        self.E = len(models)
        self.device = device
        params, buffers = stack_module_state(models)
        self.params: dict[str, Tensor] = {k: v.detach().requires_grad_() for k, v in params.items()}
        self.buffers: dict[str, Tensor] = {k: v.detach() for k, v in buffers.items()}
        # Weights live only in the stacked tensors; the template is a weightless skeleton.
        self.template = copy.deepcopy(models[0]).to("meta")
        del models

    def apply(self, params: dict[str, Tensor], x: Tensor, attn_mask: Tensor | None = None,
              mode: str = "logits", shared_input: bool = False):
        """Run every member. x is [E, B, ...] (or [B, ...] when shared_input); outputs gain a leading E axis."""
        tpl = self.template

        def f(p, b, xi, ami):
            return functional_call(tpl, (p, b), (xi,), {"attn_mask": ami, "mode": mode})

        if self.E == 1:
            p = {k: v[0] for k, v in params.items()}
            b = {k: v[0] for k, v in self.buffers.items()}
            xi = x if shared_input else x[0]
            ami = attn_mask if (attn_mask is None or shared_input) else attn_mask[0]
            return tree_map(lambda t: t.unsqueeze(0), f(p, b, xi, ami))
        x_dim = None if shared_input else 0
        am_dim = None if (attn_mask is None or shared_input) else 0
        return vmap(f, in_dims=(0, 0, x_dim, am_dim), randomness="different")(params, self.buffers, x, attn_mask)

    def member_state(self, i: int) -> dict[str, Tensor]:
        out = {k: v[i].detach().clone() for k, v in self.params.items()}
        out.update({k: v[i].detach().clone() for k, v in self.buffers.items()})
        return out

    @torch.no_grad()
    def load_member_state(self, i: int, sd: dict[str, Tensor]) -> None:
        for k, v in sd.items():
            target = self.params.get(k, self.buffers.get(k))
            target[i].copy_(v)

    def member_module(self, i: int) -> CLModel:
        """A standalone (non-functional) copy of member i."""
        m = copy.deepcopy(self.template)
        for k, v in self.member_state(i).items():
            _set_tensor(m, k, v)
        return m

    def snapshot(self) -> dict[str, Tensor]:
        return {k: v.detach().clone() for k, v in self.params.items()}

    @torch.no_grad()
    def load_params(self, p: dict[str, Tensor]) -> None:
        for k, v in p.items():
            self.params[k].copy_(v)
