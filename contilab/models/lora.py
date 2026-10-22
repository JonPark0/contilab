"""In-house LoRA (no peft): one adapter slot per task, mergeable as ΔW = (alpha/r)·B@A."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

TEXT_TARGETS = ("query_key_value", "q_proj", "v_proj")


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


class _LoRABase(nn.Module):
    def _init_adapters(self, r: int, alpha: int, n_adapters: int, fan_in: int, fan_out: int) -> None:
        self.r, self.alpha = r, alpha
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for i in range(n_adapters):
            a = torch.empty(r, fan_in)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.lora_A[f"t{i}"] = nn.Parameter(a)
            self.lora_B[f"t{i}"] = nn.Parameter(torch.zeros(fan_out, r))
        self.active: list[int] = []
        self.scale = 1.0
        # per-member multiplier (stacked by the Ensemble), lets each member merge adapters with its own λ
        self.register_buffer("member_scale", torch.ones(()))

    def delta(self, t: int) -> Tensor:
        return (self.alpha / self.r) * self.lora_B[f"t{t}"] @ self.lora_A[f"t{t}"]


class LoRALinear(_LoRABase):
    def __init__(self, base: nn.Linear, r: int, alpha: int, n_adapters: int):
        super().__init__()
        self.base = base
        self._init_adapters(r, alpha, n_adapters, base.in_features, base.out_features)

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        for t in self.active:
            a, b = self.lora_A[f"t{t}"], self.lora_B[f"t{t}"]
            y = y + (self.scale * self.member_scale * self.alpha / self.r) * F.linear(F.linear(x, a), b)
        return y


class LoRAConv2d(_LoRABase):
    def __init__(self, base: nn.Conv2d, r: int, alpha: int, n_adapters: int):
        super().__init__()
        if base.groups != 1:
            raise ValueError("LoRAConv2d supports groups=1 only")
        self.base = base
        k = base.kernel_size[0] * base.kernel_size[1]
        self._init_adapters(r, alpha, n_adapters, base.in_channels * k, base.out_channels)

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        if self.active:
            dw = sum(self.delta(t) for t in self.active).view_as(self.base.weight)
            c = self.base
            y = y + (self.scale * self.member_scale) * F.conv2d(x, dw, None, c.stride, c.padding, c.dilation)
        return y


def _is_target(name: str, mod: nn.Module, targets: list[str] | None, modality: str) -> bool:
    if isinstance(mod, _LoRABase):
        return False
    if targets is not None:
        return isinstance(mod, (nn.Linear, nn.Conv2d)) and any(t in name for t in targets)
    if modality == "text":
        return isinstance(mod, nn.Linear) and any(t in name for t in TEXT_TARGETS)
    return isinstance(mod, (nn.Linear, nn.Conv2d)) and not (name == "head" or name.startswith("head."))


def inject_lora(model: nn.Module, targets: list[str] | None, r: int, alpha: int, n_adapters: int, modality: str) -> list[str]:
    """Wrap target modules in place; returns the wrapped module names."""
    chosen = [n for n, m in model.named_modules() if n and _is_target(n, m, targets, modality)]
    for name in chosen:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        mod = getattr(parent, child)
        cls = LoRALinear if isinstance(mod, nn.Linear) else LoRAConv2d
        setattr(parent, child, cls(mod, r, alpha, n_adapters))
    return chosen


def set_active(model: nn.Module, adapters: list[int], scale: float = 1.0) -> None:
    for m in model.modules():
        if isinstance(m, _LoRABase):
            m.active = list(adapters)
            m.scale = scale


def lora_param_names(model: nn.Module, adapter: int) -> list[str]:
    tag = f".t{adapter}"
    return [n for n, _ in model.named_parameters() if ("lora_A" in n or "lora_B" in n) and n.endswith(tag)]


def lora_task_vector(params: dict[str, Tensor], adapter: int, lora_modules: list[str], r: int, alpha: int) -> dict[str, Tensor]:
    """{<module>.base.weight: ΔW} with the leading member axis kept."""
    out = {}
    for m in lora_modules:
        a = params[_join(m, f"lora_A.t{adapter}")]
        b = params[_join(m, f"lora_B.t{adapter}")]
        w_key = _join(m, "base.weight")
        out[w_key] = ((alpha / r) * (b @ a)).reshape(params[w_key].shape)
    return out
