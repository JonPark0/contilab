from dataclasses import replace

import pandas as pd
import pytest

from contilab.runner.config import expand
from contilab.runner.scheduler import group_runs, run_experiment
from contilab.strategies.base import register_strategy
from contilab.strategies.naive import Naive
from helpers import acc_matrix, clear_crash, crash_after_task, mtimes, read_status, statuses, tiny_experiment


@register_strategy("boom")
class Boom(Naive):
    def before_task(self, ctx, task):
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path / "cache"))


def test_group_runs_split():
    base = expand(tiny_experiment(["naive"], seeds=[0]))[0]
    cfgs = [replace(base, seed=s) for s in range(7)]
    assert [len(g) for g in group_runs(cfgs, max_E=4)] == [4, 3]
    assert [c.seed for c in group_runs(cfgs, max_E=4)[1]] == [4, 5, 6]


def test_failure_isolated(tmp_path):
    out = run_experiment(tiny_experiment(["naive", "boom"]), tmp_path)
    st = statuses(out)
    assert st["naive"] == {"done"} and st["boom"] == {"failed"}
    assert "Traceback" in read_status(out, "boom")["error"]


def test_resume_matches_uninterrupted(tmp_path, monkeypatch):
    full = run_experiment(tiny_experiment(["er"]), tmp_path / "a")
    crash_after_task(1, monkeypatch)
    part = run_experiment(tiny_experiment(["er"]), tmp_path / "b")
    assert statuses(part)["er"] == {"failed"}
    clear_crash(monkeypatch)
    resumed = run_experiment(tiny_experiment(["er"]), tmp_path / "b")
    assert statuses(resumed)["er"] == {"done"}
    assert list(acc_matrix(full).values()) == list(acc_matrix(resumed).values())


def test_done_runs_skipped(tmp_path):
    out = run_experiment(tiny_experiment(["naive"]), tmp_path)
    m = mtimes(out)
    run_experiment(tiny_experiment(["naive"]), tmp_path)
    assert mtimes(out) == m


def test_results_csv_has_intransigence(tmp_path):
    out = run_experiment(tiny_experiment(["naive", "joint"]), tmp_path)
    df = pd.read_csv(out / "results.csv")
    assert len(df) == 4 and df.loc[df.method == "naive", "intransigence"].notna().all()
    assert (out / "experiment.yaml").exists()


def test_consolidation_and_env_recorded(tmp_path):
    import json

    out = run_experiment(tiny_experiment(["consolidate:ta"]), tmp_path)
    assert statuses(out)["consolidate_ta"] == {"done"}
    env = json.loads(next(out.glob("*/env.json")).read_text())
    assert {"torch", "device", "precision"} <= set(env)
