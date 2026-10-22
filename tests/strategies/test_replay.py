import torch

from contilab.data.synthetic import build_synthetic
from contilab.strategies.buffer import ReservoirBuffer
from contilab.strategies.derpp import DERpp
from contilab.strategies.er import ER
from contilab.strategies.naive import Naive
from helpers import CPU, default_stream, final_acc, params_equal, train_on


def G(s):
    return torch.Generator().manual_seed(s)


def test_reservoir_fills_then_keeps_size():
    b = ReservoirBuffer(E=2, M=10, C=None, device=CPU)
    gens = [G(0), G(1)]
    for i in range(10):
        b.add(torch.arange(i * 5, i * 5 + 5).expand(2, 5), None, gens)
    assert b.size().tolist() == [10, 10] and not torch.equal(b.tensors["gidx"][0], b.tensors["gidx"][1])


def test_reservoir_is_roughly_uniform():
    first = 0.0
    for rep in range(200):
        b = ReservoirBuffer(E=1, M=100, C=None, device=CPU)
        gens = [G(rep)]
        for i in range(100):
            b.add(torch.arange(i * 10, i * 10 + 10).view(1, 10), None, gens)
        first += (b.tensors["gidx"][0] < 500).float().mean().item()
    assert abs(first / 200 - 0.5) < 0.05


def test_sample_returns_none_when_empty_and_valid_indices():
    b = ReservoirBuffer(E=2, M=10, C=None, device=CPU)
    assert b.sample(4, [G(0), G(1)]) is None
    b.add(torch.arange(3).expand(2, 3), None, [G(0), G(1)])
    s = b.sample(4, [G(0), G(1)])
    assert s.shape == (2, 3) and (s >= 0).all()


def test_er_m0_equals_naive():
    s = default_stream()
    assert params_equal(train_on(s, ER(M=0), [0])[0], train_on(s, Naive(), [0])[0])


def test_derpp_zero_coeffs_equals_naive():
    s = default_stream()
    assert params_equal(train_on(s, DERpp(M=50, alpha=0, beta=0), [0])[0], train_on(s, Naive(), [0])[0], atol=1e-5)


def test_replay_uses_own_task_mask():
    s = build_synthetic(scenario="task")
    er = ER(M=50)
    ens, ctx = train_on(s, er, [0], tasks=1)
    ctx.task_idx = 1
    r = er.replay_batch(ctx)
    assert r.mask[0, :, :2].all() and not r.mask[0, :, 2:].any()


def test_er_beats_naive_class_il():
    s = build_synthetic(scenario="class")
    assert final_acc(s, ER(M=100), [0, 1]) > final_acc(s, Naive(), [0, 1]) + 0.1


def test_derpp_beats_naive_class_il():
    s = build_synthetic(scenario="class")
    assert final_acc(s, DERpp(M=100), [0, 1]) > final_acc(s, Naive(), [0, 1]) + 0.1
