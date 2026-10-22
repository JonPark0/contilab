import torch

from contilab.core.scenario import sample_mask
from contilab.data.synthetic import build_synthetic
from contilab.engine.evaluate import evaluate_split
from contilab.strategies.naive import Naive
from helpers import train_on


class Recorder:
    def __init__(self):
        self.events = []

    def on_run_start(self, ctx): self.events.append(("run_start", ctx.task_idx))
    def on_task_start(self, ctx): self.events.append(("task_start", ctx.task_idx))
    def on_step(self, ctx, out): pass
    def on_task_end(self, ctx): self.events.append(("task_end", ctx.task_idx))
    def on_run_end(self, ctx): self.events.append(("run_end", ctx.task_idx))


def test_naive_learns_task_il():
    s = build_synthetic(scenario="task")
    ens, ctx = train_on(s, Naive(), seeds=[0], epochs=3)
    acc = [evaluate_split(ens, ens.params, t.test, s.class_table()[t.id], 256).correct.float().mean().item() for t in s.tasks]
    assert min(acc) > 0.9


def test_nan_member_marked_diverged():
    s = build_synthetic()
    ens, ctx = train_on(s, Naive(), seeds=[0, 1], epochs=1, poison_member=1)
    assert ctx.diverged == {1} and all(torch.isfinite(p[0]).all() for p in ens.params.values())


def test_hook_order_and_end_task():
    rec = Recorder()
    train_on(build_synthetic(n_tasks=3), Naive(), [0], tasks=2, hooks=rec)
    assert rec.events == [("run_start", 0), ("task_start", 0), ("task_end", 0), ("task_start", 1), ("task_end", 1), ("run_end", 1)]
