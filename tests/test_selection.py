"""Validation-based hyperparameter selection for sweeps (spec §5.3)."""

import pandas as pd
import pytest

from contilab.data.synthetic import build_synthetic
from contilab.metrics.collectors import RUN_END
from contilab.metrics.rawlog import RawLog
from contilab.metrics.registry import compute_metrics
from contilab.runner.scheduler import run_experiment
from contilab.strategies.naive import Naive
from helpers import fake_log, mctx, run_with_collectors, tiny_experiment, values


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path / "cache"))


def test_final_validation_accuracy_recorded(tmp_path):
    log = RawLog(tmp_path)
    run_with_collectors(build_synthetic(n_tasks=3), Naive(), [0], tier="core", logs=[log])
    rec = log.records("val_final")
    assert len(rec) == 1 and rec[0]["task"] == RUN_END and len(rec[0]["accs"]) == 3


def test_val_acc_metric(tmp_path):
    log = fake_log(tmp_path, [[0.9]], [0.1])
    log.append("val_final", {"task": RUN_END, "accs": [0.5, 0.7]})
    assert values(compute_metrics(log, mctx(2), "core", only=["val_acc"]))["val_acc"] == pytest.approx(0.6)


def test_sweep_selects_best_validation_variant(tmp_path):
    from contilab.report.tables import main_table, rank_table

    exp = tiny_experiment(["naive", "ewc"], seeds=[0, 1], sweep={"optim.lr": [1e-4, 0.05]})
    df = pd.read_csv(run_experiment(exp, tmp_path) / "results.csv")
    for (_, method), g in df.groupby(["preset", "method"]):
        sel = g[g.selected == True]  # noqa: E712
        assert sel.variant.nunique() == 1 and len(sel) == 2
        scores = g.groupby("variant").val_acc.mean()
        assert sel.variant.iloc[0] == scores.idxmax() == "optim.lr=0.05"
    assert "naive [optim.lr=0.05] ✓" in main_table(df, "syn_class")
    _, st = rank_table(df)
    assert set(st["avg_ranks"]) == {"naive [optim.lr=0.05] ✓", "ewc [optim.lr=0.05] ✓"}


def test_diverged_seeds_count_against_a_variant(tmp_path):
    from contilab.runner.finalize import select_variants

    rows = [{"preset": "p", "method": "m", "variant": "a", "status": "done", "val_acc": 0.9},
            {"preset": "p", "method": "m", "variant": "a", "status": "diverged", "val_acc": 0.9},
            {"preset": "p", "method": "m", "variant": "b", "status": "done", "val_acc": 0.6},
            {"preset": "p", "method": "m", "variant": "b", "status": "done", "val_acc": 0.6}]
    select_variants(rows)
    assert [r["selected"] for r in rows] == [False, False, True, True]
