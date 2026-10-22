"""Reduced ResNet-18 (Lopez-Paz & Ranzato 2017, nf=20) with GroupNorm instead of BatchNorm (vmap-safe)."""

from __future__ import annotations

import math

import torch.nn.functional as F
from torch import Tensor, nn

from contilab.models.base import CLModel


def norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(32, ch), ch)


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = norm(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = norm(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), norm(cout))

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        return F.relu(out + (self.short(x) if self.short is not None else x))


class ReducedResNet18(CLModel):
    def __init__(self, num_classes: int, nf: int = 20, in_ch: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, nf, 3, 1, 1, bias=False)
        self.n1 = norm(nf)
        widths = [nf, 2 * nf, 4 * nf, 8 * nf]
        stages, cin = [], nf
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            stages.append(nn.Sequential(BasicBlock(cin, w, stride), BasicBlock(w, w, 1)))
            cin = w
        self.stages = nn.ModuleList(stages)
        self.head = nn.Linear(8 * nf, num_classes)

    def _stage_outputs(self, x: Tensor) -> list[Tensor]:
        h = F.relu(self.n1(self.conv1(x)))
        outs = []
        for s in self.stages:
            h = s(h)
            outs.append(h)
        return outs

    def activations(self, x: Tensor, attn_mask: Tensor | None = None) -> list[Tensor]:
        return [o.mean((2, 3)) for o in self._stage_outputs(x)]

    def features(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        return self._stage_outputs(x)[-1].mean((2, 3))
