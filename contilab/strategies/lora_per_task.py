"""Parameter isolation: frozen backbone, one LoRA adapter per task, shared head.
Task-IL evaluation selects the task's adapter; class/domain-IL evaluation merges all seen adapters (scaled by λ*)."""

from __future__ import annotations

from typing import Sequence

import torch

from contilab.core.scenario import sample_mask
from contilab.engine.evaluate import evaluate_split
from contilab.models.lora import inject_lora, set_active
from contilab.strategies.base import Strategy, cross_entropy, register_strategy

LAM_GRID = tuple(round(0.1 * i, 1) for i in range(1, 11))


@register_strategy("lora_per_task")
class LoRAPerTask(Strategy):
    DEFAULTS = {"vision": {"r": 8, "alpha": 16}, "text": {"r": 8, "alpha": 16}}

    def __init__(self, r: int = 8, alpha: int = 16, targets: list[str] | None = None, lam_grid: Sequence[float] = LAM_GRID):
        super().__init__()
        self.r, self.alpha, self.targets, self.lam_grid = r, alpha, targets, tuple(lam_grid)
        self.meta["lam"] = 1.0
        self.lora_modules: list[str] = []

    def prepare_model(self, model, stream):
        self.lora_modules = inject_lora(model, self.targets, self.r, self.alpha, len(stream.tasks), stream.modality)
        return model

    def trainable_names(self, names, task_idx):
        tag = f".t{task_idx}"
        return [n for n in names if (("lora_A" in n or "lora_B" in n) and n.endswith(tag)) or n.startswith("head.")]

    def before_task(self, ctx, task):
        set_active(ctx.ens.template, [ctx.task_idx])

    def loss(self, p, s, batch, replay, f):
        z = f.logits(p, batch.x, batch.attn_mask, batch.mask)
        return cross_entropy(z, batch.y), z

    @torch.no_grad()
    def after_task(self, ctx, task):
        if ctx.stream.scenario == "task":
            return
        t, ens = ctx.task_idx, ctx.ens
        best, best_lam = -1.0, 1.0
        for lam in self.lam_grid:
            set_active(ens.template, list(range(t + 1)), scale=lam)
            accs = []
            for j in range(t + 1):
                val = ctx.stream.tasks[j].val
                mask = sample_mask(ctx.stream, torch.tensor(j), t)
                accs.append(evaluate_split(ens, ens.params, val, mask, 256).correct[0].float().mean().item())
            score = sum(accs) / len(accs)
            if score > best + 1e-12:
                best, best_lam = score, lam
        self.meta["lam"] = best_lam
        set_active(ens.template, [t])

    def eval_setup(self, ctx, task_eval):
        tpl = ctx.ens.template
        if task_eval is None:  # back to training configuration
            set_active(tpl, [ctx.task_idx])
        elif ctx.stream.scenario == "task":
            set_active(tpl, [task_eval])
        else:
            set_active(tpl, list(range(ctx.task_idx + 1)), scale=self.meta["lam"])
