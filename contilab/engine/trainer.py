"""Stream training loop: tasks → epochs → batches, with strategy hooks, collectors and divergence isolation."""

from __future__ import annotations

from typing import Callable, Protocol

import torch

from contilab.core.config import EngineConfig, OptimConfig, TrainConfig
from contilab.core.scenario import sample_mask
from contilab.core.types import RunContext, Stream
from contilab.data.batching import TrainPool, task_batches
from contilab.engine.ensemble import Ensemble
from contilab.engine.step import StepOutput, TrainStep
from contilab.strategies.base import Strategy

DIVERGENCE_FACTOR = 1e3


class CollectorHooks(Protocol):
    def on_run_start(self, ctx: RunContext) -> None: ...
    def on_task_start(self, ctx: RunContext) -> None: ...
    def on_step(self, ctx: RunContext, out: StepOutput) -> None: ...
    def on_task_end(self, ctx: RunContext) -> None: ...
    def on_run_end(self, ctx: RunContext) -> None: ...


class NullHooks:
    def on_run_start(self, ctx): pass
    def on_task_start(self, ctx): pass
    def on_step(self, ctx, out): pass
    def on_task_end(self, ctx): pass
    def on_run_end(self, ctx): pass


class DivergenceGuard:
    """Marks members whose loss is non-finite or exceeds 1e3 × their first loss (§7.5)."""

    def __init__(self) -> None:
        self.first: torch.Tensor | None = None

    def check(self, ctx: RunContext, loss: torch.Tensor) -> None:
        loss = loss.detach().cpu()
        if self.first is None:
            self.first = torch.where(torch.isfinite(loss), loss.abs(), torch.full_like(loss, float("inf")))
        bad = ~torch.isfinite(loss) | (loss > DIVERGENCE_FACTOR * self.first.clamp(min=1e-8))
        for e in torch.nonzero(bad & ctx.active.cpu()).flatten().tolist():
            ctx.active[e] = False
            ctx.diverged.add(e)


def train_stream(ens: Ensemble, strategy: Strategy, stream: Stream, train_cfg: TrainConfig, engine_cfg: EngineConfig,
                 optim_cfg: OptimConfig, hooks: CollectorHooks, ctx: RunContext, start_task: int = 0,
                 on_task_done: Callable[[int], None] | None = None, end_task: int | None = None) -> TrainStep:
    ctx.ens = ens
    if ctx.pool is None:
        ctx.pool = TrainPool.from_stream(stream)
    step = TrainStep(ens, strategy, stream, engine_cfg, optim_cfg)
    if hasattr(hooks, "attach_step"):
        hooks.attach_step(step)
    guard = DivergenceGuard()
    end = len(stream.tasks) if end_task is None else end_task
    ctx.task_idx = start_task
    if start_task == 0:
        hooks.on_run_start(ctx)
    for t in range(start_task, end):
        ctx.task_idx = t
        task = stream.tasks[t]
        strategy.before_task(ctx, task)
        step.set_task(t)
        hooks.on_task_start(ctx)
        for _ in range(train_cfg.epochs):
            for batch in task_batches(ctx.pool, t, train_cfg.batch_size, ctx.gens):
                batch.mask = sample_mask(stream, batch.task_id, t)
                replay = strategy.replay_batch(ctx)
                out = step(batch, replay, ctx)
                strategy.after_step(ctx, batch, out)
                guard.check(ctx, out.loss)
                ctx.step += 1
                hooks.on_step(ctx, out)
        strategy.after_task(ctx, task)
        hooks.on_task_end(ctx)
        if on_task_done is not None:
            on_task_done(t)
    hooks.on_run_end(ctx)
    return step
