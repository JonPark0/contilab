import torch

from contilab.data.synthetic import build_synthetic
from contilab.strategies.ewc import EWC
from contilab.strategies.lwf import LwF
from contilab.strategies.naive import Naive
from helpers import default_stream, params_equal, per_task_final_acc, train_on


def test_ewc_lambda0_equals_naive():
    s = default_stream()
    assert params_equal(train_on(s, EWC(lam=0.0), [0])[0], train_on(s, Naive(), [0])[0])


def test_fisher_nonnegative_and_shaped():
    ewc = EWC(lam=1.0, fisher_n=64)
    train_on(default_stream(), ewc, [0, 1], tasks=1)
    F = {k: v for k, v in ewc.state().items() if k.startswith("F.")}
    assert F and all((v >= 0).all() and v.shape[0] == 2 for v in F.values())
    assert any(v.sum() > 0 for v in F.values())


def test_lwf_alpha0_equals_naive():
    s = default_stream()
    assert params_equal(train_on(s, LwF(alpha=0.0), [0])[0], train_on(s, Naive(), [0])[0])


def test_lwf_changes_training_after_first_task():
    s = default_stream()
    assert not params_equal(train_on(s, LwF(alpha=1.0), [0])[0], train_on(s, Naive(), [0])[0])


def test_ewc_reduces_forgetting_on_domain_synthetic():
    s = build_synthetic(scenario="domain", n_tasks=3, classes_per_task=10, n_train=400)
    ewc = per_task_final_acc(s, EWC(lam=10.0), [0, 1], epochs=3)[0]
    naive = per_task_final_acc(s, Naive(), [0, 1], epochs=3)[0]
    assert ewc > naive


def _fisher(ewc):
    return {k: v for k, v in ewc.state().items() if k.startswith("F.")}


def test_ewc_caps_the_stiffness_under_sgd():
    lam, lr = 1e4, 0.1
    capped, free = EWC(lam=lam, fisher_n=64), EWC(lam=lam, fisher_n=64, max_stiffness=None)
    for ewc in (capped, free):
        train_on(default_stream(), ewc, [0, 1], tasks=1, lr=lr)  # one task: no penalty yet, same Fisher
    cap = 1.0 / (lr * lam)
    assert any((v > cap).any() for v in _fisher(free).values())
    for k, v in _fisher(free).items():
        assert torch.equal(_fisher(capped)[k], v.clamp(max=cap))


def test_ewc_stiffness_cap_only_applies_to_sgd():
    from contilab.core.config import OptimConfig

    ewc = EWC(lam=10.0)
    ctx = type("Ctx", (), {})()
    assert ewc.fisher_cap(ctx) is None
    ctx.optim = OptimConfig(name="adamw", lr=1e-3)
    assert ewc.fisher_cap(ctx) is None
    ctx.optim = OptimConfig(name="sgd", lr=0.05)
    assert ewc.fisher_cap(ctx) == 1.0 / (0.05 * 10.0)
    assert EWC(lam=10.0, max_stiffness=None).fisher_cap(ctx) is None
    assert EWC(lam=0.0).fisher_cap(ctx) is None


def test_ewc_with_a_stiff_penalty_does_not_diverge():
    """lr·λ·F > 2 makes SGD on the penalty oscillate with growing amplitude; the cap keeps it stable."""
    s = default_stream()
    assert train_on(s, EWC(lam=1e4, max_stiffness=None), [0, 1], lr=0.1)[1].diverged
    assert not train_on(s, EWC(lam=1e4), [0, 1], lr=0.1)[1].diverged
