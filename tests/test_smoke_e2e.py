import os
import time

import pandas as pd

from contilab.runner.bench import run_bench
from contilab.runner.cli import main


def test_smoke_all_under_60s(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path / "cache"))
    t0 = time.perf_counter()
    assert main(["run", "configs/experiments/smoke_all.yaml", "--root", str(tmp_path)]) == 0
    elapsed = time.perf_counter() - t0
    df = pd.read_csv(tmp_path / "smoke_all" / "results.csv")
    assert len(df) == 66 and set(df.status) == {"done"}, df[df.status != "done"][["run_id", "status"]]
    # 60 s on a 4-core workstation (spec §1.5); shared CI runners set a looser budget
    assert elapsed < float(os.environ.get("CONTILAB_SMOKE_BUDGET_S", 60)), elapsed
    assert main(["report", str(tmp_path / "smoke_all")]) == 0
    assert (tmp_path / "smoke_all" / "report.html").exists()


def test_metrics_add_posthoc(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path / "cache"))
    assert main(["run", "configs/experiments/smoke_all.yaml", "--root", str(tmp_path), "matrix.method.name=[naive]",
                 "matrix.preset=[syn_class]", "metrics.tier=extended"]) == 0
    exp = tmp_path / "smoke_all"
    assert "cka_final" in pd.read_csv(exp / "results.csv").columns
    assert main(["metrics", str(exp), "--add", "cka_final", "loss_barrier"]) == 0
    df = pd.read_csv(exp / "results.csv")
    assert df.cka_final.notna().all() and "loss_barrier" not in df.columns  # barrier raw data needs the full tier


def test_bench_reports_speedup(tmp_path):
    rows = run_bench("mlp", [1, 4], steps=10, compile_modes=["off"], out=tmp_path)
    assert {r["E"] for r in rows} == {1, 4} and all(r["samples_per_s"] > 0 for r in rows)
    assert all("vmap_speedup" in r for r in rows) and (tmp_path / "bench.json").exists()


def test_cli_rejects_unknown_override(tmp_path, capsys):
    assert main(["run", "configs/experiments/smoke_all.yaml", "--root", str(tmp_path), "train.epoch=2"]) == 2
    assert "unknown config key" in capsys.readouterr().err
