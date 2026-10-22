"""Regression tests for the whole-branch review findings (I1–I9, re-graded M3/M7/M9)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from contilab.core.config import LoRAConfig, ModelConfig, OptimConfig, RunConfig
from contilab.data.synthetic import build_synthetic
from contilab.metrics.rawlog import RawLog
from contilab.metrics.registry import compute_metrics
from contilab.runner.scheduler import run_experiment
from contilab.strategies.base import register_strategy
from contilab.strategies.naive import Naive
from helpers import fake_log, mctx, run_with_collectors, statuses, tiny_experiment, train_on, values


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path / "cache"))


@register_strategy("boom_after0")
class BoomAfter0(Naive):
    def after_task(self, ctx, task):
        raise RuntimeError("boom in after_task")


# I1 ---------------------------------------------------------------------------------------------
def test_rerun_keeps_diverged_terminal_and_logs_single(tmp_path):
    exp = tiny_experiment(["naive"], cli={"optim.lr": 1e4})
    out = run_experiment(exp, tmp_path)
    assert statuses(out)["naive"] == {"diverged"}
    run_experiment(exp, tmp_path)
    for d in out.glob("syn_class-naive-*"):
        assert len(RawLog(d).records("acc_matrix")) == 3


def test_fresh_restart_clears_partial_raw(tmp_path):
    exp = tiny_experiment(["boom_after0"])
    run_experiment(exp, tmp_path)
    out = run_experiment(exp, tmp_path)
    for d in out.glob("syn_class-boom_after0-*"):
        assert len(RawLog(d).records("init_acc")) == 1


# I2 ---------------------------------------------------------------------------------------------
def _flops(strategy, seeds, tmp_path):
    logs = [RawLog(tmp_path / f"{strategy.name}-{len(seeds)}-{e}") for e in range(len(seeds))]
    run_with_collectors(build_synthetic(n_tasks=2), strategy, seeds, tier="core", logs=logs)
    return [r for r in logs[0].records("cost") if r["task"] == 10**6][0]["flops_per_step"]


def test_flops_include_replay_and_are_per_member(tmp_path):
    from contilab.strategies.er import ER

    naive1, er1 = _flops(Naive(), [0], tmp_path), _flops(ER(M=50), [0], tmp_path)
    assert er1 > naive1
    assert _flops(Naive(), [0, 1], tmp_path) == naive1


# I3 ---------------------------------------------------------------------------------------------
def test_peak_memory_can_be_reset():
    from contilab.engine.device import peak_memory_bytes, reset_peak_memory

    big = torch.ones(50_000_000)  # ~200 MB
    big.add_(1)
    high = peak_memory_bytes(torch.device("cpu"))
    del big
    reset_peak_memory(torch.device("cpu"))
    assert peak_memory_bytes(torch.device("cpu")) < high - 100 * 2**20


# I4 ---------------------------------------------------------------------------------------------
def test_lora_lambda_selected_per_member():
    from contilab.strategies.lora_per_task import LoRAPerTask

    s = build_synthetic(scenario="class", n_tasks=3)
    both = LoRAPerTask(r=2, alpha=4)
    train_on(s, both, [0, 1])
    alone = LoRAPerTask(r=2, alpha=4)
    train_on(s, alone, [1])
    assert len(both.meta["lam"]) == 2 and both.meta["lam"][1] == alone.meta["lam"][0]


# I5 ---------------------------------------------------------------------------------------------
def test_distill_randomness_follows_seed():
    from contilab.consolidate.distill import transfer_gens

    a = [g.initial_seed() for g in transfer_gens([0, 1], upto=2)]
    b = [g.initial_seed() for g in transfer_gens([1], upto=2)]
    c = [g.initial_seed() for g in transfer_gens([0], upto=2)]
    assert a[1] == b[0] and b[0] != c[0]


# I6 ---------------------------------------------------------------------------------------------
def test_agem_reference_excludes_current_task():
    from contilab.strategies.agem import AGEM

    s = build_synthetic(scenario="class", n_tasks=3)
    ag = AGEM(M=200, ref_batch=64)
    ens, ctx = train_on(s, ag, [0], tasks=2)  # buffer holds tasks 0 and 1
    ctx.task_idx = 1
    r = ag.replay_batch(ctx)
    assert r is not None and (r.task_id == 0).all()
    ctx.task_idx = 0
    assert ag.replay_batch(ctx) is None


# I7 ---------------------------------------------------------------------------------------------
def test_fisher_chunk_respects_memory_budget():
    from contilab.strategies.ewc import fisher_chunk

    assert fisher_chunk(45_000_000, 2**30) == 5
    assert fisher_chunk(10**12, 2**30) == 1
    assert fisher_chunk(1000, 2**30) == 128


# I8 ---------------------------------------------------------------------------------------------
def test_windowed_forgetting_unavailable_when_window_below_rho(tmp_path):
    log = fake_log(tmp_path, [[0.9, 0.1], [0.4, 0.9]], [0.1, 0.1])
    for r in [{"task": 0, "step": 0, "task_now": 0, "accs": [0.0]}, {"task": 0, "step": 10, "task_now": 0, "accs": [0.9]},
              {"task": 1, "step": 15, "task_now": 1, "accs": [0.4, 0.5]}, {"task": 1, "step": 20, "task_now": 1, "accs": [0.7, 0.9]}]:
        log.append("stream_acc", r)
    out = compute_metrics(log, mctx(2, rho=50), "extended", only=["windowed_forgetting"])["windowed_forgetting"]
    assert out["value"] is None and "ρ" in out["error"]


# I9 ---------------------------------------------------------------------------------------------
def test_sweep_variants_stay_separate(tmp_path):
    from contilab.report.tables import main_table

    out = run_experiment(tiny_experiment(["ewc"], seeds=[0], sweep={"method.params.lam": [1.0, 10.0]}), tmp_path)
    df = pd.read_csv(out / "results.csv")
    assert sorted(df.variant) == ["method.params.lam=1.0", "method.params.lam=10.0"]
    t = main_table(df, "syn_class")
    assert "ewc [method.params.lam=1.0]" in t and "ewc [method.params.lam=10.0]" in t


# M3 ---------------------------------------------------------------------------------------------
def test_fwt_undefined_for_class_il(tmp_path):
    R, B = [[0.9, 0.0], [0.5, 0.8]], [0.0, 0.0]
    assert compute_metrics(fake_log(tmp_path / "a", R, B), mctx(2, scenario="class"), "core")["fwt"]["value"] is None
    assert compute_metrics(fake_log(tmp_path / "b", R, B), mctx(2, scenario="task"), "core")["fwt"]["value"] is not None


# M7 ---------------------------------------------------------------------------------------------
def test_diverged_runs_marked_in_tables():
    from contilab.report.tables import main_table

    df = pd.DataFrame([{"run_id": "p-ewc-s0", "preset": "p", "method": "ewc", "seed": 0, "status": "diverged", "acc": 0.1},
                       {"run_id": "p-ewc-s1", "preset": "p", "method": "ewc", "seed": 1, "status": "done", "acc": 0.5}])
    row = next(l for l in main_table(df, "p").splitlines() if l.startswith("| ewc"))
    assert "diverged 1/2" in row


# M9 ---------------------------------------------------------------------------------------------
def test_lora_lr_only_overrides_adamw():
    from contilab.runner.execute import _lora_optim

    sgd = RunConfig(optim=OptimConfig(name="sgd", lr=0.03), model=ModelConfig(lora=LoRAConfig(lr=1e-3)))
    adamw = RunConfig(optim=OptimConfig(name="adamw", lr=1e-4), model=ModelConfig(lora=LoRAConfig(lr=1e-3)))
    assert _lora_optim(sgd).lr == 0.03 and _lora_optim(adamw).lr == 1e-3
