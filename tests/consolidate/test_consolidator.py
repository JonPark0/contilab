import time

import numpy as np
import torch

from contilab.consolidate.experts import train_experts
from contilab.data.synthetic import build_synthetic
from contilab.metrics.rawlog import RawLog
from helpers import run_consolidation_for_test, smoke_cfg


def test_experts_share_init_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path))
    s = build_synthetic(n_tasks=3)
    ex = train_experts(s, smoke_cfg(), [0, 1])
    assert torch.equal(ex.base["head.weight"][0], ex.base["head.weight"][1]) and len(ex.tuned) == 3
    assert not torch.equal(ex.tuned[0]["head.weight"][0], ex.tuned[0]["head.weight"][1])  # different data order
    t0 = time.perf_counter()
    ex2 = train_experts(s, smoke_cfg(), [0, 1])
    assert time.perf_counter() - t0 < 1.0 and torch.equal(ex2.tuned[2]["layers.0.weight"], ex.tuned[2]["layers.0.weight"])


def test_grouped_expert_equals_single_task_training(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path))
    s = build_synthetic(n_tasks=2)
    both = train_experts(s, smoke_cfg(), [0])
    from contilab.core.types import Stream
    only1 = train_experts(Stream("synthetic-t1", [s.tasks[1]], s.scenario, s.num_classes, s.modality), smoke_cfg(), [0])
    assert torch.allclose(both.tuned[1]["layers.0.weight"], only1.tuned[0]["layers.0.weight"], atol=1e-5)


def test_ta_consolidation_task_il_reasonable(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path))
    s = build_synthetic(n_tasks=3, scenario="task")
    log = RawLog(tmp_path / "r")
    run_consolidation_for_test(s, "consolidate:ta", [0], log)
    rows = log.records("acc_matrix")
    assert len(rows) == 3 and np.mean(rows[-1]["accs"]) > 0.6
    end = [r for r in log.records("cost") if r["task"] == 10**6][0]
    assert end["expert_wall_s"] > 0 and end["stored_experts_bytes"] > 0


def test_lora_experts_merge_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path))
    s = build_synthetic(n_tasks=2, scenario="task")
    log = RawLog(tmp_path / "r")
    run_consolidation_for_test(s, "consolidate:ties", [0], log, experts_lora=True)
    assert len(log.records("acc_matrix")) == 2


def test_consolidation_marks_stream_metrics_not_applicable(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path))
    s = build_synthetic(n_tasks=2)
    log = RawLog(tmp_path / "r")
    run_consolidation_for_test(s, "consolidate:ta", [0], log, tier="extended")
    assert log.records("stream_acc") == [{"task": 0, "not_applicable": True}]


def test_expert_groups_respect_max_E_and_equal_sizes():
    from contilab.consolidate.experts import plan_expert_groups

    groups = plan_expert_groups([10, 10, 10, 5], [0, 1], max_E=3)
    assert all(len(g) <= 3 for g in groups)
    assert sorted(m for g in groups for m in g) == sorted((t, s) for t in range(4) for s in (0, 1))
    sizes = [10, 10, 10, 5]
    assert all(len({sizes[t] for t, _ in g}) == 1 for g in groups)


def test_experts_split_by_max_E_match_grouped(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path))
    s = build_synthetic(n_tasks=3)
    grouped = train_experts(s, smoke_cfg(), [0, 1])
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path / "other"))
    split = train_experts(s, smoke_cfg(**{"engine.max_E": 1}), [0, 1])
    for t in range(3):
        assert torch.allclose(grouped.tuned[t]["layers.0.weight"], split.tuned[t]["layers.0.weight"], atol=1e-5)
