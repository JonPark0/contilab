import numpy as np
import pytest
import torch
from scipy.optimize import minimize

from contilab.strategies.agem import AGEM, project_agem
from contilab.strategies.gem import GEM, project_gem, solve_gem_dual
from contilab.strategies.naive import Naive
from helpers import default_stream, params_equal, train_on


def _reference_dual(P, d, margin, eps):
    """The GEM dual by a generic bounded solver, for comparison."""
    P = 0.5 * (P + P.T) + eps * np.eye(len(d))
    res = minimize(lambda v: 0.5 * v @ P @ v + d @ v, np.full(len(d), margin), jac=lambda v: P @ v + d,
                   bounds=[(margin, None)] * len(d), method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-12})
    return res.x


@pytest.mark.parametrize("K", [1, 3, 6])
def test_dual_matches_a_generic_qp_solver(K):
    g = torch.Generator().manual_seed(K)
    G = torch.randn(4, K, 20, generator=g, dtype=torch.float64)
    x = torch.randn(4, 20, generator=g, dtype=torch.float64)
    P, d = G @ G.transpose(1, 2), (G @ x[..., None]).squeeze(-1)
    v = solve_gem_dual(P, d, margin=0.5, eps=1e-3)
    for e in range(4):
        ref = _reference_dual(P[e].numpy(), d[e].numpy(), 0.5, 1e-3)
        assert np.allclose(v[e].numpy(), ref, atol=1e-5), (e, v[e], ref)


def test_coordinate_descent_fallback_agrees_with_the_exact_solver(monkeypatch):
    import contilab.strategies.gem as gem

    g = torch.Generator().manual_seed(0)
    G = torch.randn(2, 5, 12, generator=g, dtype=torch.float64)
    x = torch.randn(2, 12, generator=g, dtype=torch.float64)
    P, d = G @ G.transpose(1, 2), (G @ x[..., None]).squeeze(-1)
    exact = solve_gem_dual(P, d, 0.5)
    monkeypatch.setattr(gem, "EXACT_MAX_K", 2)
    assert torch.allclose(gem.solve_gem_dual(P, d, 0.5), exact, atol=1e-6)


def test_projection_satisfies_every_constraint_and_leaves_agreeing_members_alone():
    g = {"w": torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])}
    G = {"w": torch.tensor([[[-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # member 0 conflicts with task 0
                            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])}  # member 1 agrees with both
    out = project_gem(g, G)["w"]
    assert torch.equal(out[1], g["w"][1])
    assert ((G["w"][0] @ out[0]) >= -1e-6).all()


def test_one_reference_without_margin_is_the_agem_projection():
    g = {"a": torch.randn(3, 4), "b": torch.randn(3, 2)}
    r = {"a": -g["a"] + 0.1 * torch.randn(3, 4), "b": torch.randn(3, 2)}
    gem = project_gem(g, {k: v[:, None] for k, v in r.items()}, margin=0.0, eps=0.0)
    agem = project_agem(g, r)
    assert all(torch.allclose(gem[k], agem[k], atol=1e-5) for k in g)


def test_reference_batch_is_split_by_previous_task():
    s = default_stream()
    strat = GEM(M=60, ref_batch=30)
    _, ctx = train_on(s, strat, [0, 1], tasks=3)
    ctx.task_idx = 3
    b = strat.replay_batch(ctx)
    groups = strat.ref_groups()
    assert len(groups) == 3 and sum(groups) == b.y.shape[1]
    off = 0
    for k, n in enumerate(groups):
        assert (b.task_id[:, off:off + n] == k).all()
        off += n


def test_gem_runs_and_differs_from_naive_and_agem():
    s = default_stream()
    gem, _ = train_on(s, GEM(M=50), [0, 1])
    assert not params_equal(gem, train_on(s, Naive(), [0, 1])[0])
    assert not params_equal(gem, train_on(s, AGEM(M=50), [0, 1])[0])
    assert all(torch.isfinite(v).all() for v in gem.params.values())


def test_gem_compiled_matches_eager():
    s = default_stream()
    eager, _ = train_on(s, GEM(M=40, ref_batch=24), [0], tasks=3)
    comp, _ = train_on(s, GEM(M=40, ref_batch=24), [0], tasks=3, compile="on")
    assert params_equal(eager, comp, atol=1e-4)
