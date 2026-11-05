"""O-LoRA (Wang et al., 2023): a new LoRA adapter per task on a frozen backbone, trained while the previous adapters
stay active and frozen, with a penalty that keeps its subspace orthogonal to theirs. Inference uses the sum of all
adapters learned so far, so it needs no task identity. Penalty as in the reference code: λ₁·Σ|A_i A_tᵀ| (L1) over
previous adapters i, plus an optional λ₂·(‖A_t‖ + ‖B_t‖); defaults λ₁ = 0.5, λ₂ = 0 from its training scripts."""

from __future__ import annotations

import torch

from contilab.models.lora import _join, inject_lora, set_active
from contilab.strategies.base import Strategy, cross_entropy, register_strategy


@register_strategy("olora")
class OLoRA(Strategy):
    DEFAULTS = {"vision": {"r": 8, "alpha": 16}, "text": {"r": 8, "alpha": 16}}

    def __init__(self, r: int = 8, alpha: int = 16, targets: list[str] | None = None, lam_orth: float = 0.5,
                 lam_l2: float = 0.0):
        super().__init__()
        self.r, self.alpha, self.targets = r, alpha, targets
        self.lam_orth, self.lam_l2 = lam_orth, lam_l2
        self.lora_modules: list[str] = []
        self._t = 0

    def prepare_model(self, model, stream):
        self.lora_modules = inject_lora(model, self.targets, self.r, self.alpha, len(stream.tasks), stream.modality)
        return model

    def trainable_names(self, names, task_idx):
        tag = f".t{task_idx}"
        return [n for n in names if (("lora_A" in n or "lora_B" in n) and n.endswith(tag)) or n.startswith("head.")]

    def before_task(self, ctx, task):
        self._t = ctx.task_idx
        set_active(ctx.ens.template, list(range(ctx.task_idx + 1)))

    def eval_setup(self, ctx, task_eval):
        set_active(ctx.ens.template, list(range(ctx.task_idx + 1)))

    def penalty(self, p) -> torch.Tensor:
        t = self._t
        out = torch.zeros((), device=next(iter(p.values())).device)
        for m in self.lora_modules:
            a_t = p[_join(m, f"lora_A.t{t}")]
            if self.lam_orth:
                for i in range(t):
                    out = out + self.lam_orth * (p[_join(m, f"lora_A.t{i}")] @ a_t.T).abs().sum()
            if self.lam_l2:
                out = out + self.lam_l2 * (a_t.norm() + p[_join(m, f"lora_B.t{t}")].norm())
        return out

    def loss(self, p, s, batch, replay, f):
        z = f.logits(p, batch.x, batch.attn_mask, batch.mask)
        return cross_entropy(z, batch.y) + self.penalty(p), z

    def state_dict(self):
        d = super().state_dict()
        d["t"] = self._t
        return d

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self._t = d.get("t", 0)
