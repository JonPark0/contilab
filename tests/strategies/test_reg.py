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
