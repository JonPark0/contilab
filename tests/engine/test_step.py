import pytest
import torch

from contilab.core.config import EngineConfig, OptimConfig
from contilab.core.scenario import sample_mask
from contilab.data.batching import TrainPool, task_batches
from contilab.data.synthetic import build_synthetic
from contilab.engine.ensemble import Ensemble
from contilab.engine.evaluate import evaluate_split
from contilab.engine.step import TrainStep
from contilab.strategies.base import build_strategy
from contilab.strategies.naive import Naive
from helpers import CPU, make_ctx, mlp_factory


def run_naive(seeds, steps=3, compile="off", active=None):
    s = build_synthetic()
    ens = Ensemble(mlp_factory(), seeds, CPU)
    pool = TrainPool.from_stream(s)
    ctx = make_ctx(s, seeds, ens=ens, pool=pool)
    if active is not None:
        ctx.active = torch.tensor(active)
    step = TrainStep(ens, Naive(), s, EngineConfig(compile=compile, precision="fp32"), OptimConfig(lr=0.1))
    step.set_task(0)
    for b, _ in zip(task_batches(pool, 0, 32, ctx.gens), range(steps)):
        b.mask = sample_mask(s, b.task_id, 0)
        step(b, None, ctx)
    return ens, step


def test_vmap_members_equal_individual_runs():
    e3, _ = run_naive([1, 2, 3])
    for i, sd in enumerate([1, 2, 3]):
        e1, _ = run_naive([sd])
        for k in e3.params:
            assert torch.allclose(e3.params[k][i], e1.params[k][0], atol=1e-5)


def test_compile_equals_eager():
    a, _ = run_naive([1], compile="off")
    b, step = run_naive([1], compile="on")
    if not step.compile_decision["used"]:
        pytest.skip(f"compile unavailable: {step.compile_decision['reason']}")
    for k in a.params:
        assert torch.allclose(a.params[k], b.params[k], atol=1e-5)


def test_inactive_member_frozen_others_unaffected():
    init, _ = run_naive([1, 2], steps=0)
    e, _ = run_naive([1, 2], active=[True, False])
    e1, _ = run_naive([1])
    for k in e.params:
        assert torch.equal(e.params[k][1], init.params[k][1])
        assert torch.allclose(e.params[k][0], e1.params[k][0], atol=1e-5)


def test_training_reduces_loss():
    s = build_synthetic()
    ens = Ensemble(mlp_factory(), [0], CPU)
    pool = TrainPool.from_stream(s)
    ctx = make_ctx(s, [0], ens=ens, pool=pool)
    step = TrainStep(ens, Naive(), s, EngineConfig(compile="off", precision="fp32"), OptimConfig(lr=0.1))
    step.set_task(0)
    losses = []
    for _ in range(5):
        for b in task_batches(pool, 0, 32, ctx.gens):
            b.mask = sample_mask(s, b.task_id, 0)
            losses.append(step(b, None, ctx).loss[0].item())
    assert losses[-1] < losses[0] * 0.5


def test_evaluate_split_accuracy_shape():
    ens, _ = run_naive([1, 2])
    s = build_synthetic()
    r = evaluate_split(ens, ens.params, s.tasks[0].test, s.class_table()[0], 64)
    assert r.correct.shape == (2, 100) and r.pred.max() <= 1 and (r.conf <= 1).all()


def test_build_strategy_merges_modality_defaults():
    assert isinstance(build_strategy("naive", {}, "vision"), Naive)
