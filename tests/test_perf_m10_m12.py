"""Deferred review minors M10 (bench precision), M11 (redundant work), M12 (non-vmap and memory fallbacks)."""

import json

import pytest
import torch

from contilab.data.synthetic import build_synthetic
from contilab.metrics.rawlog import RawLog
from contilab.metrics.registry import compute_metrics
from contilab.runner.scheduler import run_experiment
from contilab.strategies.base import register_strategy
from contilab.strategies.naive import Naive
from helpers import fake_log, mctx, run_consolidation_for_test, statuses, tiny_experiment, train_on


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path / "cache"))


# M10 --------------------------------------------------------------------------------------------
def test_bench_measures_each_precision(tmp_path):
    from contilab.runner.bench import run_bench

    rows = run_bench("mlp", [1, 2], steps=5, compile_modes=["off"], out=tmp_path, precisions=["fp32", "auto"])
    assert {r["precision"] for r in rows} == {"fp32", "auto"} and all("precision_used" in r for r in rows)
    for r in rows:  # vmap speed-up is relative to E=1 at the same compile mode and precision
        base = next(x for x in rows if x["E"] == 1 and x["precision"] == r["precision"])
        assert r["vmap_speedup"] == pytest.approx(r["samples_per_s"] / base["samples_per_s"])


# M11 --------------------------------------------------------------------------------------------
def test_class_table_is_built_once_per_stream(monkeypatch):
    from contilab.core.scenario import sample_mask

    s = build_synthetic(n_tasks=3)
    first = s.class_table()
    calls = {"n": 0}
    real = torch.zeros

    def counting_zeros(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(torch, "zeros", counting_zeros)
    for _ in range(5):
        sample_mask(s, torch.tensor([0, 1]), 1)
        assert torch.equal(s.class_table(), first)
    assert calls["n"] == 0


def test_lwf_runs_two_forwards_per_step(monkeypatch):
    from contilab.strategies.base import MemberFns
    from contilab.strategies.lwf import LwF

    s = build_synthetic(n_tasks=2)
    calls = {"n": 0}
    real = MemberFns._call

    def counting(self, *a, **k):
        calls["n"] += 1
        return real(self, *a, **k)

    lwf = LwF()
    ens, ctx = train_on(s, lwf, [0], tasks=1)  # task 0: no distillation yet
    monkeypatch.setattr(MemberFns, "_call", counting)
    out = lwf.loss({k: v[0] for k, v in ens.params.items()}, {k: v[0] for k, v in lwf.state().items()},
                   _first_batch(s, ctx), None, MemberFns(ens.template, {k: v[0] for k, v in ens.buffers.items()}))
    assert calls["n"] == 2 and torch.isfinite(out[0])


def _first_batch(s, ctx):
    from contilab.core.scenario import sample_mask
    from contilab.data.batching import task_batches

    b = next(task_batches(ctx.pool, 1, 8, ctx.gens))
    b.mask = sample_mask(s, b.task_id, 1)
    return type(b)(x=b.x[0], y=b.y[0], task_id=b.task_id[0], attn_mask=None, index=b.index[0], mask=b.mask[0])


def test_distillation_reuses_the_run_pool(tmp_path, monkeypatch):
    from contilab.data import batching

    calls = {"n": 0}
    real = batching.TrainPool.from_stream.__func__

    def counting(cls, stream):
        calls["n"] += 1
        return real(cls, stream)

    monkeypatch.setattr(batching.TrainPool, "from_stream", classmethod(counting))
    run_consolidation_for_test(build_synthetic(n_tasks=3), "consolidate:distill", [0], RawLog(tmp_path))
    assert calls["n"] <= 2  # the run's pool (+ the expert trainer's), not one per consolidation step


def test_probe_matrix_fitted_once_per_log(tmp_path, monkeypatch):
    import contilab.metrics.representation as rep

    log = fake_log(tmp_path, [[0.9, 0.1], [0.4, 0.9]], [0.1, 0.1])
    y = torch.arange(40) % 2
    for t in range(2):
        for j in range(2):
            log.put_tensor("features", f"t{t}.task{j}", torch.randn(40, 6) + y[:, None] * 3)
            log.put_tensor("features", f"t{t}.task{j}.y", y + 2 * j)
    log.flush()
    calls = {"n": 0}
    real = rep.probe_accuracy

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(rep, "probe_accuracy", counting)
    compute_metrics(log, mctx(2), "extended", only=["probe_acc", "repr_forgetting", "head_forgetting"])
    assert calls["n"] == 4  # one 2×2 probe matrix shared by the three metrics


# M12 --------------------------------------------------------------------------------------------
@register_strategy("not_vmappable")
class NotVmappable(Naive):
    vmappable = False


def test_non_vmappable_fallback_is_recorded(tmp_path):
    out = run_experiment(tiny_experiment(["not_vmappable"], seeds=[0, 1]), tmp_path)
    envs = [json.loads(p.read_text()) for p in out.glob("syn_class-not_vmappable-*/env.json")]
    assert len(envs) == 2 and all("not vmappable" in e.get("vmap_fallback", "") for e in envs)


@register_strategy("oom_when_wide")
class OomWhenWide(Naive):
    """Raises an out-of-memory error once training is under way, whenever more than one member trains together."""

    def after_step(self, ctx, batch, out):
        if ctx.E > 1 and ctx.step >= 3:
            raise torch.OutOfMemoryError("simulated: not enough memory")


def test_out_of_memory_splits_the_group(tmp_path):
    out = run_experiment(tiny_experiment(["oom_when_wide"], seeds=[0, 1, 2, 3]), tmp_path)
    assert statuses(out)["oom_when_wide"] == {"done"}
    envs = [json.loads(p.read_text()) for p in out.glob("syn_class-oom_when_wide-*/env.json")]
    assert len(envs) == 4 and all("memory_split" in e for e in envs) and all(e["E"] == 1 for e in envs)
