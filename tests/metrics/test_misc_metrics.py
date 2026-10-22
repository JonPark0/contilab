import numpy as np
import pytest
import torch

from contilab.metrics.bias import expected_calibration_error
from contilab.metrics.collectors import RUN_END
from contilab.metrics.registry import compute_metrics
from helpers import fake_log, mctx, values

R2 = [[0.9, 0.1], [0.4, 0.9]]


def test_ece_perfect_and_overconfident():
    assert expected_calibration_error(np.array([1.0, 1.0]), np.array([1, 1])) == pytest.approx(0.0)
    assert expected_calibration_error(np.array([0.9] * 10), np.array([1] * 5 + [0] * 5)) == pytest.approx(0.4)


def fake_pred_log(tmp_path):
    log = fake_log(tmp_path, R2, [0.1, 0.1])
    # task-0 samples (classes 0/1): 3 of 4 predicted as task-1 classes; task-1 samples all correct
    log.put_tensor("pred_hist", "t1.pred", torch.tensor([2, 3, 2, 0, 2, 3, 2, 3]))
    log.put_tensor("pred_hist", "t1.y", torch.tensor([0, 1, 0, 1, 2, 3, 2, 3]))
    log.put_tensor("pred_hist", "t1.conf", torch.full((8,), 0.75))
    log.put_tensor("pred_hist", "t1.task", torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
    log.flush()
    return log


def test_recency_bias_and_task_confusion(tmp_path):
    v = values(compute_metrics(fake_pred_log(tmp_path), mctx(2), "core"))
    assert v["recency_bias"] == pytest.approx(0.75)
    assert v["task_confusion"]["matrix"] == [[0.25, 0.75], [0.0, 1.0]]
    # 4 correct of 8 at confidence 0.75 → |0.5 - 0.75|
    assert v["ece"] == pytest.approx(0.25)


def test_recency_bias_class_il_only(tmp_path):
    out = compute_metrics(fake_pred_log(tmp_path), mctx(2, scenario="task"), "core")
    assert out["recency_bias"]["value"] is None


def test_loss_barrier(tmp_path):
    log = fake_log(tmp_path, R2)
    log.append("barrier_curve", {"task": 1, "k": 0, "alphas": [0, 0.5, 1], "losses": [1.0, 3.0, 2.0]})
    v = values(compute_metrics(log, mctx(2), "full", only=["loss_barrier"]))
    assert v["loss_barrier"] == pytest.approx(3.0 - 1.5)


def test_recovery_formula(tmp_path):
    log = fake_log(tmp_path, R2)
    log.append("recovery", {"task": RUN_END, "k": 0, "acc_before": 0.4, "acc_after": 0.8, "acc_at_learn": 0.9})
    log.append("recovery", {"task": RUN_END, "k": 1, "acc_before": 0.9, "acc_after": 0.9, "acc_at_learn": 0.9})
    assert values(compute_metrics(log, mctx(2), "full", only=["recovery"]))["recovery"] == pytest.approx(0.8)


def test_grad_interference_offdiag(tmp_path):
    log = fake_log(tmp_path, R2)
    log.append("grad_cosine", {"task": 1, "matrix": [[1.0, -0.2], [-0.2, 1.0]]})
    assert values(compute_metrics(log, mctx(2), "full", only=["grad_interference"]))["grad_interference"] == pytest.approx(-0.2)


def test_general_ppl_delta(tmp_path):
    log = fake_log(tmp_path, R2)
    for t, p in [(-1, 20.0), (0, 22.0), (1, 25.0)]:
        log.append("general_ppl", {"task": t, "ppl": p})
    assert values(compute_metrics(log, mctx(2), "extended", only=["general_ppl_delta"]))["general_ppl_delta"] == pytest.approx(5.0)


def test_all_metrics_compute_on_real_full_run(tmp_path):
    from contilab.data.synthetic import build_synthetic
    from contilab.metrics.rawlog import RawLog
    from contilab.strategies.er import ER
    from helpers import run_with_collectors

    s = build_synthetic(n_tasks=3)
    log = RawLog(tmp_path)
    run_with_collectors(s, ER(M=50), [0], tier="full", logs=[log], rho=3, beta=3)
    out = compute_metrics(log, mctx(3), "full")
    crashed = {k: v["error"] for k, v in out.items() if v["error"] and "Error" in v["error"].split(":")[0]}
    assert not crashed, crashed
    assert out["acc"]["value"] is not None and out["loss_barrier"]["value"] is not None and out["min_acc"]["value"] is not None
