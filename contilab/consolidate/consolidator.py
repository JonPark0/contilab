"""Train-then-consolidate pipeline (§5.5): merge experts after every task so all CL metrics stay defined."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from contilab.consolidate.experts import Experts, expert_factory, train_experts
from contilab.consolidate.heads import assemble_head, body_names
from contilab.consolidate.merge import apply_tv, merge_ta, merge_ties, select_lambda, task_vector
from contilab.core.config import ConsolidateConfig, RunConfig
from contilab.core.scenario import sample_mask
from contilab.core.types import RunContext, Stream
from contilab.data.batching import TrainPool
from contilab.engine.ensemble import Ensemble
from contilab.engine.evaluate import evaluate_split
from contilab.models.lora import lora_task_vector, set_active


@dataclass
class ConsolidationContext:
    stream: Stream
    ens: Ensemble
    cfg: RunConfig
    val_score: Callable[[dict[str, Tensor], int], Tensor]  # (params, upto) → mean val acc [S]
    seeds: list[int]


class Consolidator:
    def __init__(self, cfg: ConsolidateConfig, student_factory=None):
        self.cfg = cfg
        self.student_factory = student_factory  # extension point: teacher→student continual distillation (§5.6)
        self.flops = 0  # per member, training done inside consolidate() (distillation)
        self.stored_samples = 0  # stream samples kept for consolidation (counted as buffer memory)
        self.diverged: set[int] = set()  # members whose consolidation training diverged

    def on_new_expert(self, expert: dict[str, Tensor], ctx: RunContext) -> None:
        pass

    def consolidate(self, experts: Experts, upto: int, cctx: ConsolidationContext) -> dict[str, Tensor]:
        raise NotImplementedError

    def _task_vectors(self, experts: Experts, upto: int, cctx: ConsolidationContext) -> list[dict[str, Tensor]]:
        if experts.lora_modules:
            lc = cctx.cfg.model.lora
            return [lora_task_vector(experts.tuned[t], 0, experts.lora_modules, lc.r, lc.alpha) for t in range(upto + 1)]
        names = body_names(experts.base)
        return [task_vector(experts.base, experts.tuned[t], names) for t in range(upto + 1)]

    def _merge(self, experts: Experts, upto: int, cctx: ConsolidationContext, tv: dict[str, Tensor]) -> dict[str, Tensor]:
        head = assemble_head(experts.tuned, cctx.stream.class_table(), upto, cctx.stream.scenario)
        dev = next(iter(cctx.ens.params.values())).device
        base = {k: v.to(dev) for k, v in experts.base.items()}
        tv = {k: v.to(dev) for k, v in tv.items()}
        head = {k: v.to(dev) for k, v in head.items()}

        def build(lam: float) -> dict[str, Tensor]:
            return {**apply_tv(base, tv, lam), **head}

        _, params = select_lambda(build, lambda p: cctx.val_score(p, upto), self.cfg.lam_grid)
        return params


class TAConsolidator(Consolidator):
    def consolidate(self, experts, upto, cctx):
        return self._merge(experts, upto, cctx, merge_ta(self._task_vectors(experts, upto, cctx)))


class TIESConsolidator(Consolidator):
    def consolidate(self, experts, upto, cctx):
        return self._merge(experts, upto, cctx, merge_ties(self._task_vectors(experts, upto, cctx), self.cfg.ties_k))


def build_consolidator(method_name: str, cfg: ConsolidateConfig) -> Consolidator:
    kind = method_name.split(":", 1)[1]
    if kind == "ta":
        return TAConsolidator(cfg)
    if kind == "ties":
        return TIESConsolidator(cfg)
    if kind == "distill":
        from contilab.consolidate.distill import DistillConsolidator

        return DistillConsolidator(cfg)
    raise ValueError(f"unknown consolidation method: {method_name}")


def make_val_score(stream: Stream, ens: Ensemble, batch_size: int):
    @torch.no_grad()
    def score(params: dict[str, Tensor], upto: int) -> Tensor:
        accs = []
        for j in range(upto + 1):
            mask = sample_mask(stream, torch.tensor(j), upto)
            accs.append(evaluate_split(ens, params, stream.tasks[j].val, mask, batch_size).correct.float().mean(1))
        return torch.stack(accs, 1).mean(1)

    return score


def run_consolidation(stream: Stream, cfg: RunConfig, seeds: list[int], hooks, ctx: RunContext) -> None:
    experts = train_experts(stream, cfg, seeds)
    fac, holder = expert_factory(stream, cfg)
    ens = Ensemble(fac, [cfg.consolidate.init_seed] * len(seeds), ctx.device)
    if holder.get("modules"):
        set_active(ens.template, [])  # merged ΔW lives in the base weights
    ctx.ens = ens
    if ctx.pool is None:
        ctx.pool = TrainPool.from_stream(stream)
    cctx = ConsolidationContext(stream, ens, cfg, make_val_score(stream, ens, cfg.train.eval_batch_size), list(seeds))
    consolidator = build_consolidator(cfg.method.name, cfg.consolidate)
    if hasattr(hooks, "consolidation"):
        hooks.consolidation = True
    hooks.on_run_start(ctx)
    for t in range(len(stream.tasks)):
        ctx.task_idx = t
        hooks.on_task_start(ctx)
        consolidator.on_new_expert(experts.tuned[t], ctx)
        ens.load_params(consolidator.consolidate(experts, t, cctx))
        hooks.on_task_end(ctx)
    if hasattr(hooks, "extra_cost"):
        hooks.extra_cost.update(expert_wall_s=experts.wall_s, expert_flops=experts.flops + consolidator.flops,
                                stored_experts_bytes=experts.nbytes(),
                                transfer_bytes=consolidator.stored_samples * ctx.pool.sample_bytes())
    ctx.diverged |= set(experts.diverged) | consolidator.diverged
    hooks.on_run_end(ctx)
