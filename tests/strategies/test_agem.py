import torch

from contilab.strategies.agem import AGEM, project_agem
from contilab.strategies.naive import Naive
from helpers import default_stream, params_equal, train_on


def test_projection_only_when_conflicting():
    g = {"w": torch.tensor([[1.0, 0.0], [1.0, 0.0]])}
    r = {"w": torch.tensor([[1.0, 1.0], [-1.0, 0.0]])}
    out = project_agem(g, r)["w"]
    assert torch.equal(out[0], g["w"][0])
    assert torch.allclose(out[1], torch.tensor([0.0, 0.0]))
    assert (out[1] * r["w"][1]).sum() >= -1e-6


def test_projection_spans_multiple_tensors():
    g = {"a": torch.tensor([[1.0]]), "b": torch.tensor([[1.0]])}
    r = {"a": torch.tensor([[-1.0]]), "b": torch.tensor([[0.0]])}
    out = project_agem(g, r)
    assert torch.allclose(out["a"], torch.tensor([[0.0]])) and torch.allclose(out["b"], torch.tensor([[1.0]]))


def test_agem_runs_and_differs_from_naive():
    s = default_stream()
    assert not params_equal(train_on(s, AGEM(M=50), [0])[0], train_on(s, Naive(), [0])[0])
