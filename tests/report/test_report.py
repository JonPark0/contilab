import pandas as pd
import pytest

from contilab.report.render import build_report
from contilab.report.tables import axis_tables, main_table, rank_table
from contilab.runner.scheduler import run_experiment
from helpers import tiny_experiment


def fake_results(**methods):
    rows = []
    for m, vals in methods.items():
        for s, v in enumerate(vals):
            rows.append({"run_id": f"p-{m}-s{s}", "preset": "p", "method": m, "seed": s, "status": "done", "acc": v, "bwt": -v})
    return pd.DataFrame(rows)


def test_report_from_tiny_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path / "cache"))
    out = run_experiment(tiny_experiment(["naive", "er", "joint"], tier="full"), tmp_path)
    md, html = build_report(out)
    text = md.read_text()
    assert "| naive |" in text and "±" in text and "acc_heatmap.png" in text
    assert (out / "figs" / "syn_class" / "acc_heatmap.png").exists()
    assert (out / "figs" / "syn_class" / "stream_curves.png").exists()
    assert "data:image/png;base64," in html.read_text()
    assert "통계 검정력" in text


def test_main_table_marks_significance():
    df = fake_results(naive=[0.1] * 6, er=[0.5] * 6)
    t = main_table(df, "p")
    er_row = next(l for l in t.splitlines() if l.startswith("| er "))
    assert "*" in er_row and "—" in er_row  # missing metrics render as an em dash


def test_axis_tables_skip_empty_axes():
    tabs = axis_tables(fake_results(naive=[0.1, 0.2], er=[0.5, 0.6]), "p")
    assert "stability" in tabs and "representation" not in tabs


def test_rank_table_across_presets():
    df = pd.concat([fake_results(naive=[0.1], er=[0.5], ewc=[0.3]).assign(preset=p) for p in ("a", "b", "c")])
    md, st = rank_table(df, "acc")
    assert st["avg_ranks"]["er"] == pytest.approx(1.0) and "CD" in md
