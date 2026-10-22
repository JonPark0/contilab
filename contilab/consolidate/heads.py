"""Classifier-head assembly for merged models: each class row comes from the expert of the task that owns it."""

from __future__ import annotations

import torch
from torch import Tensor


def body_names(params: dict, prefix: str = "head.") -> list[str]:
    return [k for k in params if not k.startswith(prefix)]


def assemble_head(experts: list[dict[str, Tensor]], class_table: Tensor, upto: int, scenario: str,
                  prefix: str = "head.") -> dict[str, Tensor]:
    keys = [k for k in experts[0] if k.startswith(prefix)]
    if scenario == "domain":
        return {k: torch.stack([e[k] for e in experts[: upto + 1]]).mean(0) for k in keys}
    out = {k: experts[upto][k].clone() for k in keys}
    for t in range(upto + 1):
        rows = class_table[t].nonzero().flatten()
        for k in keys:
            out[k][:, rows.to(out[k].device)] = experts[t][k][:, rows.to(out[k].device)]
    return out
