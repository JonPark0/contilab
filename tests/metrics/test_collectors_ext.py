from contilab.data.synthetic import build_synthetic, build_synthetic_text
from contilab.metrics.rawlog import RawLog
from contilab.strategies.lora_per_task import LoRAPerTask
from contilab.strategies.naive import Naive
from helpers import run_with_collectors


def test_extended_keys(tmp_path):
    log = RawLog(tmp_path)
    run_with_collectors(build_synthetic(), Naive(), [0], tier="extended", logs=[log], rho=5, beta=3)
    sa = log.records("stream_acc")
    assert sa[0]["step"] == 0 and len(sa[0]["accs"]) == 1 and len(sa[-1]["accs"]) == 5
    assert len(log.records("lca_curve")) == 5 * 4
    feats = log.tensors("features")
    assert feats["t4.task0"].shape == (100, 16) and feats["t4.task0.y"].shape == (100,)
    assert "t0.layer0" in log.tensors("activations")
    assert not log.has("barrier_curve")


def test_full_keys(tmp_path):
    log = RawLog(tmp_path)
    run_with_collectors(build_synthetic(n_tasks=3), Naive(), [0], tier="full", logs=[log])
    bc = log.records("barrier_curve")
    assert len(bc) == 2 and len(bc[0]["losses"]) == 11 and bc[0]["k"] == 0
    m = log.records("grad_cosine")[-1]["matrix"]
    assert len(m) == 3 and abs(m[0][0] - 1) < 1e-5
    rec = log.records("recovery")
    assert len(rec) == 2 and {"acc_before", "acc_after", "acc_at_learn"} <= set(rec[0])
    assert "t2.layers.0.weight" in log.tensors("param_snapshots")


def test_general_ppl_for_text(tmp_path):
    log = RawLog(tmp_path)
    run_with_collectors(build_synthetic_text(), Naive(), [0], tier="extended", logs=[log], model="tiny")
    rows = log.records("general_ppl")
    assert [r["task"] for r in rows] == [-1, 0, 1, 2] and all(r["ppl"] > 1 for r in rows)


def test_lora_collection_restores_training_adapter(tmp_path):
    log = RawLog(tmp_path)
    run_with_collectors(build_synthetic(scenario="class", n_tasks=3), LoRAPerTask(r=2, alpha=4), [0],
                        tier="extended", logs=[log], rho=2)
    assert len(log.records("acc_matrix")) == 3
