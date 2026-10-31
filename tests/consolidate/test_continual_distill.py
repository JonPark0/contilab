"""Teacher→student continual distillation (spec §5.6)."""

import json

import pandas as pd
import pytest
import torch

from contilab.data.synthetic import build_synthetic
from contilab.metrics.collectors import RUN_END
from contilab.metrics.rawlog import RawLog
from contilab.runner.scheduler import run_experiment
from helpers import run_consolidation_for_test, tiny_experiment


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path / "cache"))


def test_runs_and_discards_teachers(tmp_path):
    s = build_synthetic(n_tasks=3)
    log = RawLog(tmp_path)
    run_consolidation_for_test(s, "consolidate:continual_distill", [0, 1], [log, RawLog(tmp_path / "m1")])
    assert len(log.records("acc_matrix")) == 3
    end = [r for r in log.records("cost") if r["task"] == RUN_END][-1]
    assert end["stored_experts_bytes"] == 0  # a teacher is dropped once the student has learned from it
    assert end["transfer_bytes"] > 0  # replay buffer of earlier tasks (budget M)


def test_student_keeps_old_tasks_better_than_naive(tmp_path):
    # soft targets (T=2) need more than the smoke profile's ~6 steps per task; both methods get 3 epochs
    out = run_experiment(tiny_experiment(["naive", "consolidate:continual_distill"], seeds=[0, 1],
                                         cli={"train.epochs": 3}), tmp_path)
    df = pd.read_csv(out / "results.csv")
    acc = df.groupby("method").acc.mean()
    assert acc["consolidate:continual_distill"] > acc["naive"] + 0.05


@pytest.mark.parametrize("method", ["consolidate:continual_distill", "consolidate:distill"])
def test_smaller_student_architecture(tmp_path, method):
    # routed targets and 30 passes over the 60-sample transfer set: the multi-teacher smoke default trains 2 steps
    cli = {"consolidate.routing": "routed", "consolidate.epochs": 30, "train.epochs": 3}
    big = run_experiment(tiny_experiment([method], seeds=[0], cli=cli), tmp_path / "big")
    small = run_experiment(tiny_experiment([method], seeds=[0], cli={**cli, "method.params.student.hidden": 4}),
                           tmp_path / "small")
    p = {k: pd.read_csv(d / "results.csv").iloc[0] for k, d in [("big", big), ("small", small)]}
    assert p["small"].status == "done" and p["small"].trainable_params < p["big"].trainable_params
    assert p["small"].acc > 0.4  # learns despite the smaller student (3 tasks × 2 classes; chance ≈ 0.17)


def test_merging_rejects_a_different_student(tmp_path):
    out = run_experiment(tiny_experiment(["consolidate:ta"], seeds=[0], cli={"method.params.student.hidden": 4}), tmp_path)
    st = json.loads(next(out.glob("syn_class-consolidate_ta-*/status.json")).read_text())
    assert st["status"] == "failed" and "teacher architecture" in st["error"]
