import torch

from contilab.data.batching import TrainPool, epoch_perms, eval_batches, task_batches
from contilab.data.cache import cached_tensors
from contilab.data.synthetic import build_synthetic


def g(s):
    return torch.Generator().manual_seed(s)


def test_member_perms_follow_own_generator():
    p2 = epoch_perms(50, [g(1), g(2)])
    p1 = epoch_perms(50, [g(2)])
    assert torch.equal(p2[1], p1[0]) and not torch.equal(p2[0], p2[1])


def test_task_batches_cover_task_once():
    s = build_synthetic(n_train=70)
    pool = TrainPool.from_stream(s)
    idx = torch.cat([b.index for b in task_batches(pool, 2, 32, [g(0)])], 1)
    assert idx.shape == (1, 63)  # 70 - 10% val
    assert set(idx[0].tolist()) == set(range(pool.offsets[2].item(), pool.offsets[3].item()))
    assert (pool.task_id[idx] == 2).all()


def test_gather_returns_member_batches():
    s = build_synthetic()
    pool = TrainPool.from_stream(s)
    b = pool.gather(torch.tensor([[0, 1], [500, 501]]))
    assert b.x.shape == (2, 2, 32) and b.task_id[1].tolist() == [2, 2] and torch.equal(b.index, torch.tensor([[0, 1], [500, 501]]))


def test_eval_batches_cover_split():
    s = build_synthetic()
    ys = torch.cat([y for _, y, _ in eval_batches(s.tasks[0].test, 30)])
    assert torch.equal(ys, s.tasks[0].test.y)


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path))
    calls = []

    def b():
        calls.append(1)
        return {"a": torch.ones(2)}

    cached_tensors("t", {"k": 1}, b)
    out = cached_tensors("t", {"k": 1}, b)
    assert calls == [1] and torch.equal(out["a"], torch.ones(2))
