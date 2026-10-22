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

    def _set_member_scales(self, ens, lams) -> None:
        lam = torch.as_tensor(lams, dtype=torch.float32)
        for m in self.lora_modules:
            buf = ens.buffers[f"{m}.member_scale" if m else "member_scale"]
            buf.copy_(lam.to(buf.device, buf.dtype))

    @torch.no_grad()
    def after_task(self, ctx, task):
        """Pick λ* per member on the seen tasks' validation sets (ties → smaller λ)."""
        if ctx.stream.scenario == "task":
            return
        t, ens = ctx.task_idx, ctx.ens
        set_active(ens.template, list(range(t + 1)))
        best = torch.full((ens.E,), -1.0)
        best_lam = torch.ones(ens.E)
        for lam in sorted(self.lam_grid):
            self._set_member_scales(ens, [lam] * ens.E)
            accs = []
            for j in range(t + 1):
                mask = sample_mask(ctx.stream, torch.tensor(j), t)
                accs.append(evaluate_split(ens, ens.params, ctx.stream.tasks[j].val, mask, 256).correct.float().mean(1).cpu())
            score = torch.stack(accs, 1).mean(1)
            better = score > best + 1e-12
            best = torch.where(better, score, best)
            best_lam = torch.where(better, torch.full_like(best_lam, lam), best_lam)
        self.meta["lam"] = best_lam.tolist()
        self._set_member_scales(ens, [1.0] * ens.E)
        set_active(ens.template, [t])

    def eval_setup(self, ctx, task_eval):
        ens = ctx.ens
        tpl = ens.template
        lams = self.meta["lam"] if isinstance(self.meta["lam"], list) else [self.meta["lam"]] * ens.E
        if task_eval is None:  # back to training configuration
            set_active(tpl, [ctx.task_idx])
            self._set_member_scales(ens, [1.0] * ens.E)
        elif ctx.stream.scenario == "task":
            set_active(tpl, [task_eval])
            self._set_member_scales(ens, [1.0] * ens.E)
        else:
            set_active(tpl, list(range(ctx.task_idx + 1)))
            self._set_member_scales(ens, lams)
