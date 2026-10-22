from dataclasses import replace

import pytest

from contilab.metrics.registry import compute_metrics
from helpers import fake_log, mctx, values

R = [[0.9, 0.1, 0.0], [0.6, 0.8, 0.2], [0.5, 0.7, 0.9]]
B = [0.1, 0.1, 0.1]
MCTX = mctx(3, scenario="task")  # FWT is only defined outside Class-IL
MCTX1 = mctx(1)


def test_matrix_metrics(tmp_path):
    v = values(compute_metrics(fake_log(tmp_path, R, B), MCTX, "core"))
    assert v["acc"] == pytest.approx(0.7)
    assert v["aia"] == pytest.approx((0.9 + 0.7 + 0.7) / 3)
    assert v["bwt"] == pytest.approx(((0.5 - 0.9) + (0.7 - 0.8)) / 2)
    assert v["fwt"] == pytest.approx(((0.1 - 0.1) + (0.2 - 0.1)) / 2)
    assert v["forgetting"] == pytest.approx(((0.9 - 0.5) + (0.8 - 0.7)) / 2)
    assert v["relative_forgetting"] == pytest.approx((0.4 / 0.9 + 0.1 / 0.8) / 2)
    assert v["learning_acc"] == pytest.approx((0.9 + 0.8 + 0.9) / 3)


def test_intransigence_needs_joint(tmp_path):
    log = fake_log(tmp_path, R, B)
    assert compute_metrics(log, MCTX, "core")["intransigence"]["error"] is not None
    v = values(compute_metrics(log, replace(MCTX, joint_acc=[0.95, 0.9, 0.92]), "core"))
    assert v["intransigence"] == pytest.approx(((0.95 - 0.9) + (0.9 - 0.8) + (0.92 - 0.9)) / 3)


def test_single_task_metrics_none(tmp_path):
    out = compute_metrics(fake_log(tmp_path, [[0.8]], [0.1]), MCTX1, "core")
    assert out["acc"]["value"] == pytest.approx(0.8)
    for k in ("bwt", "fwt", "forgetting"):
        assert out[k]["value"] is None and out[k]["error"]


def test_missing_raw_reports_tier(tmp_path):
    from contilab.metrics.registry import METRICS, metric

    metric("_needs_stream", ("stream_acc",), "extended")(lambda log, m: 1.0)
    try:
        out = compute_metrics(fake_log(tmp_path, R, B), MCTX, "extended", only=["_needs_stream"])
    finally:
        METRICS.pop("_needs_stream")
    err = out["_needs_stream"]["error"]
    assert out["_needs_stream"]["value"] is None and "stream_acc" in err and "extended" in err


def test_efficiency_metrics(tmp_path):
    cost = [{"task": 0, "wall_s": 2.0, "samples": 10, "samples_per_s": 10.0, "peak_mem": 5, "trainable_params": 7,
             "buffer_bytes": 0, "model_bytes": 100},
            {"task": 1, "wall_s": 6.0, "samples": 10, "samples_per_s": 20.0, "peak_mem": 9, "trainable_params": 7,
             "buffer_bytes": 40, "model_bytes": 100},
            {"task": 10**6, "flops_per_step": 1000, "steps": 4}]
    v = values(compute_metrics(fake_log(tmp_path, R[:2], B[:2], cost), mctx(2), "core"))
    assert v["wall_time"] == 8.0 and v["samples_per_s"] == pytest.approx((10 * 2 + 20 * 6) / 8)
    assert v["peak_mem"] == 9 and v["total_flops"] == 4000 and v["buffer_bytes"] == 40
