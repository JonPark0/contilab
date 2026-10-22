import numpy as np
import torch

from contilab.consolidate.distill import mix_teachers
from contilab.consolidate.transfer import build_transfer
from contilab.data.batching import TrainPool
from contilab.data.synthetic import build_synthetic
from contilab.metrics.rawlog import RawLog
from helpers import run_consolidation_for_test

TABLE_2x2 = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.bool)


def G(s):
    return torch.Generator().manual_seed(s)


def test_buffer_transfer_budget():
    s = build_synthetic()
    pool = TrainPool.from_stream(s)
    t = build_transfer(s, pool, upto=3, kind="buffer", M=100, gens=[G(0), G(1)])
    assert t.gidx.shape == (2, 100) and set(pool.task_id[t.gidx[0]].tolist()) == {0, 1, 2, 3}
    assert not torch.equal(t.gidx[0], t.gidx[1])


def test_full_and_proxy_transfer():
    s = build_synthetic()
    pool = TrainPool.from_stream(s)
    full = build_transfer(s, pool, upto=1, kind="full", M=100, gens=[G(0)])
    assert full.gidx.shape == (1, int(pool.offsets[2])) and full.proxy is None
    prox = build_transfer(s, pool, upto=1, kind="proxy", M=64, gens=[G(0)])
    assert prox.gidx is None and prox.proxy.x.shape == (64, 32)


def test_entropy_routing_prefers_confident_teacher():
    p = mix_teachers(torch.tensor([[[5.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.1, 0.0]]]), TABLE_2x2, routing="entropy", T=1.0)
    assert p[0].argmax() == 0 and torch.allclose(p.sum(-1), torch.ones(1))


def test_routed_uses_owner_teacher_only():
    logits = torch.tensor([[[5.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 3.0]]])
    p = mix_teachers(logits, TABLE_2x2, routing="routed", T=1.0, task_of=torch.tensor([1]))
    assert p[0, :2].sum() == 0 and p[0].argmax() == 3


def test_distill_student_learns_from_teachers(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path))
    s = build_synthetic(n_tasks=2, scenario="task")
    log = RawLog(tmp_path / "r")
    run_consolidation_for_test(s, "consolidate:distill", [0], log, transfer="full")
    assert np.mean(log.records("acc_matrix")[-1]["accs"]) > 0.8
