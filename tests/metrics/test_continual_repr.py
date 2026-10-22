import numpy as np
import pytest
import torch

from contilab.metrics.plasticity import dormant_fraction, effective_rank, srank
from contilab.metrics.rawlog import RawLog
from contilab.metrics.registry import compute_metrics
from contilab.metrics.representation import linear_cka, probe_accuracy
from helpers import fake_log, mctx, values

R2 = [[0.9, 0.1], [0.4, 0.9]]
ROWS = [{"task": 0, "step": 0, "task_now": 0, "accs": [0.0]}, {"task": 0, "step": 5, "task_now": 0, "accs": [0.6]},
        {"task": 0, "step": 10, "task_now": 0, "accs": [0.9]}, {"task": 1, "step": 15, "task_now": 1, "accs": [0.4, 0.5]},
        {"task": 1, "step": 20, "task_now": 1, "accs": [0.7, 0.9]}]


def stream_log(tmp_path, rows=ROWS):
    log = fake_log(tmp_path, R2, [0.1, 0.1])
    for r in rows:
        log.append("stream_acc", r)
    return log


def test_stream_metrics(tmp_path):
    v = values(compute_metrics(stream_log(tmp_path), mctx(2), "extended"))
    assert v["min_acc"] == pytest.approx(0.4)
    assert v["wc_acc"] == pytest.approx(0.5 * 0.9 + 0.5 * 0.4)
    assert v["stability_gap"] == pytest.approx(0.9 - 0.4)
    assert v["aaa"] == pytest.approx(np.mean([0.0, 0.6, 0.9, 0.45, 0.8]))
    # window = 10% of mean task length (10 steps) → 1 step: no pair of records within 1 step → 0
    assert v["windowed_forgetting"] == pytest.approx(0.0)


def test_windowed_forgetting_wide_window(tmp_path):
    v = values(compute_metrics(stream_log(tmp_path), mctx(2, window_frac=1.0), "extended", only=["windowed_forgetting"]))
    assert v["windowed_forgetting"] == pytest.approx(0.9 - 0.4)


def test_lca(tmp_path):
    log = fake_log(tmp_path, R2, [0.1, 0.1])
    for t, accs in enumerate([[0.1, 0.5, 0.9], [0.3, 0.5, 0.7]]):
        for b, a in enumerate(accs):
            log.append("lca_curve", {"task": t, "batch": b, "acc": a})
    v = values(compute_metrics(log, mctx(2), "extended", only=["lca"]))
    assert v["lca"] == pytest.approx(np.mean([0.2, 0.5, 0.8]))


def test_cka_identity_and_rotation():
    X = np.random.default_rng(0).normal(size=(50, 8))
    Q = np.linalg.qr(np.random.default_rng(1).normal(size=(8, 8)))[0]
    assert linear_cka(X, X) == pytest.approx(1.0) and linear_cka(X, X @ Q) == pytest.approx(1.0, abs=1e-6)


def test_rank_measures():
    F = np.diag([1.0, 1.0, 0.0, 0.0])
    assert effective_rank(F) == pytest.approx(2.0) and srank(F, 0.01) == 2


def test_dormant_ratio():
    assert dormant_fraction(np.array([1, 1, 0, 0.02]), 0.025) == pytest.approx(0.25)


def test_probe_accuracy_separable():
    rng = np.random.default_rng(0)
    y = np.arange(100) % 2
    F = rng.normal(size=(100, 4)) + y[:, None] * 5
    assert probe_accuracy(F, y) == pytest.approx(1.0)


def test_representation_metrics_from_features(tmp_path):
    rng = torch.Generator().manual_seed(0)
    log = fake_log(tmp_path, R2, [0.1, 0.1])
    y = torch.arange(40) % 2
    base = [torch.randn(40, 6, generator=rng) + y[:, None] * 4 for _ in range(2)]
    for t in range(2):
        for j in range(2):
            f = base[j] if t == 0 or j == 1 else torch.randn(40, 6, generator=rng)  # task-0 features destroyed at t=1
            log.put_tensor("features", f"t{t}.task{j}", f)
            log.put_tensor("features", f"t{t}.task{j}.y", y + 2 * j)
    log.flush()
    v = values(compute_metrics(log, mctx(2), "extended"))
    assert v["repr_forgetting"] > 0.3 and v["cka_final"] < 1.0 and v["feature_drift"] > 0.5
    assert len(v["probe_acc"]["matrix"]) == 2
    assert v["head_forgetting"] == pytest.approx((0.9 - 0.4) - v["repr_forgetting"])
