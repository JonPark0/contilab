"""Regression tests for the deferred review minors fixed in v0.2 (M4, M6, M8, M13)."""

import pytest
import torch

from contilab.core.config import MetricsConfig, OptimConfig, TrainConfig
from contilab.data.batching import TrainPool
from contilab.data.synthetic import build_synthetic
from contilab.data.transforms import make_joint_stream
from contilab.metrics.collectors import RUN_END, Collectors
from contilab.metrics.rawlog import RawLog
from contilab.metrics.registry import compute_metrics
from contilab.metrics.representation import linear_cka
from contilab.strategies.naive import Naive
from helpers import fake_log, mctx, run_consolidation_for_test, run_with_collectors, train_on, values


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path / "cache"))


# M4 ---------------------------------------------------------------------------------------------
def test_cka_final_excludes_trivial_last_task(tmp_path):
    g = torch.Generator().manual_seed(0)
    log = fake_log(tmp_path, [[0.9, 0.1], [0.4, 0.9]], [0.1, 0.1])
    f00, f10, f11 = (torch.randn(40, 6, generator=g) for _ in range(3))
    for key, f in {"t0.task0": f00, "t1.task0": f10, "t1.task1": f11, "t0.task1": f11}.items():
        log.put_tensor("features", key, f)
        log.put_tensor("features", key + ".y", torch.arange(40) % 2)
    log.flush()
    v = values(compute_metrics(log, mctx(2), "extended", only=["cka_final"]))
    assert v["cka_final"] == pytest.approx(linear_cka(f00.numpy(), f10.numpy()))


def test_cka_final_undefined_for_single_task(tmp_path):
    log = fake_log(tmp_path, [[0.9]], [0.1])
    log.put_tensor("features", "t0.task0", torch.randn(40, 6))
    log.put_tensor("features", "t0.task0.y", torch.arange(40) % 2)
    log.flush()
    assert values(compute_metrics(log, mctx(1), "extended", only=["cka_final"]))["cka_final"] is None


# M6 ---------------------------------------------------------------------------------------------
def _end_task_cost(log):
    return [r for r in log.records("cost") if r["task"] != RUN_END][-1]


def test_empty_derpp_buffer_costs_nothing(tmp_path):
    from contilab.strategies.derpp import DERpp

    log = RawLog(tmp_path)
    run_with_collectors(build_synthetic(n_tasks=2), DERpp(M=0), [0], tier="core", logs=[log])
    assert _end_task_cost(log)["buffer_bytes"] == 0


@pytest.mark.parametrize("name", ["er", "derpp"])
def test_buffer_bytes_count_stored_samples(tmp_path, name):
    from contilab.strategies.derpp import DERpp
    from contilab.strategies.er import ER

    s = build_synthetic(n_tasks=2)
    log = RawLog(tmp_path)
    strat = ER(M=50) if name == "er" else DERpp(M=50)
    run_with_collectors(s, strat, [0, 1], tier="core", logs=[log, RawLog(tmp_path / "m1")])
    per_item = TrainPool.from_stream(s).sample_bytes() + (4 * s.num_classes if name == "derpp" else 0)
    rec = _end_task_cost(log)
    assert rec["buffer_items"] == 50 and rec["buffer_bytes"] == 50 * per_item


def test_state_bytes_metric(tmp_path):
    cost = [{"task": 0, "wall_s": 1.0, "samples": 1, "samples_per_s": 1.0, "peak_mem": 1, "trainable_params": 1,
             "buffer_bytes": 0, "state_bytes": 123, "model_bytes": 1}]
    v = values(compute_metrics(fake_log(tmp_path, [[0.9]], [0.1], cost=cost), mctx(1), "core", only=["state_bytes"]))
    assert v["state_bytes"] == 123


def test_distill_buffer_transfer_counts_as_stored_samples(tmp_path):
    s = build_synthetic(n_tasks=2)
    log = RawLog(tmp_path)
    run_consolidation_for_test(s, "consolidate:distill", [0], log, transfer="buffer")
    v = values(compute_metrics(log, mctx(2), "core", only=["buffer_bytes"]))
    assert v["buffer_bytes"] == 60 * TrainPool.from_stream(s).sample_bytes()  # smoke_cfg M=60
    log2 = RawLog(tmp_path / "proxy-free")
    run_consolidation_for_test(s, "consolidate:ta", [0], log2)
    assert values(compute_metrics(log2, mctx(2), "core", only=["buffer_bytes"]))["buffer_bytes"] == 0


# M8 ---------------------------------------------------------------------------------------------
def test_recovery_uses_the_run_optimizer(tmp_path, monkeypatch):
    import contilab.engine.step as step_mod

    seen = []
    real = step_mod.TrainStep

    class Spy(real):
        def __init__(self, ens, strategy, stream, engine, optim, *a, **k):
            seen.append(optim)
            super().__init__(ens, strategy, stream, engine, optim, *a, **k)

    s = build_synthetic(n_tasks=2)
    adamw = OptimConfig(name="adamw", lr=1e-3, weight_decay=0.01)
    hooks = Collectors("full", [RawLog(tmp_path)], s, Naive(), MetricsConfig(tier="full", recovery_steps=2),
                       TrainConfig(epochs=1, batch_size=32), adamw)
    monkeypatch.setattr(step_mod, "TrainStep", Spy)
    train_on(s, hooks.strategy, [0], hooks=hooks)
    rec = [o for o in seen if o.lr == 1e-3]  # training itself runs at train_on's lr 0.1
    assert rec and all(o.name == "adamw" and o.weight_decay == 0.01 for o in rec)


# M13 --------------------------------------------------------------------------------------------
def test_joint_stream_acc_tracks_every_task(tmp_path):
    s = build_synthetic(n_tasks=3)
    joint = make_joint_stream(s)
    log = RawLog(tmp_path)
    hooks = Collectors("extended", [log], joint, Naive(), MetricsConfig(tier="extended", rho=5),
                       TrainConfig(epochs=1, batch_size=32), OptimConfig(lr=0.1), eval_stream=s)
    train_on(joint, hooks.strategy, [0], hooks=hooks)
    assert all(len(r["accs"]) == 3 for r in log.records("stream_acc"))
    v = values(compute_metrics(log, mctx(3), "extended", only=["aaa", "cka_final"]))
    assert v["aaa"] is not None and v["cka_final"] is None
