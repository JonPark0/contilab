from contilab.data.synthetic import build_synthetic
from contilab.metrics.rawlog import RawLog
from contilab.strategies.naive import Naive
from helpers import run_with_collectors


def test_core_collection(tmp_path):
    logs = [RawLog(tmp_path / f"m{e}") for e in range(2)]
    run_with_collectors(build_synthetic(), Naive(), [0, 1], tier="core", logs=logs)
    rows = logs[0].records("acc_matrix")
    assert [r["task"] for r in rows] == [0, 1, 2, 3, 4] and all(len(r["accs"]) == 5 for r in rows)
    assert len(logs[1].records("init_acc")[0]["accs"]) == 5
    assert logs[0].tensors("pred_hist")["t4.pred"].shape == (100 * 5,)
    c = logs[0].records("cost")
    assert c[-1]["flops_per_step"] > 0 and c[0]["samples_per_s"] > 0 and c[0]["trainable_params"] > 0
    assert not logs[0].has("stream_acc")


def test_members_logged_separately(tmp_path):
    logs = [RawLog(tmp_path / f"m{e}") for e in range(2)]
    run_with_collectors(build_synthetic(), Naive(), [0, 1], tier="core", logs=logs)
    assert logs[0].records("acc_matrix") != logs[1].records("acc_matrix")


def test_rawlog_truncate(tmp_path):
    log = RawLog(tmp_path)
    for i in range(3):
        log.append("acc_matrix", {"task": i, "accs": [i]})
    log.truncate_after_task(0)
    assert [r["task"] for r in log.records("acc_matrix")] == [0]


def test_rawlog_tensors_roundtrip_and_truncate(tmp_path):
    import torch

    log = RawLog(tmp_path)
    log.put_tensor("features", "t0.task0", torch.ones(2))
    log.put_tensor("features", "t1.task0", torch.zeros(2))
    log.flush()
    log2 = RawLog(tmp_path)
    assert set(log2.tensors("features")) == {"t0.task0", "t1.task0"}
    log2.truncate_after_task(0)
    assert set(RawLog(tmp_path).tensors("features")) == {"t0.task0"}
