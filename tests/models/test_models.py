import torch
from torch import nn

from contilab.core.config import ModelConfig
from contilab.data.synthetic import build_synthetic, build_synthetic_text
from contilab.models.hf import HFClassifier
from contilab.models.mlp import MLP
from contilab.models.registry import build_model
from contilab.models.resnet import ReducedResNet18


def test_mlp_modes():
    m = MLP(32, 10)
    x = torch.randn(4, 32)
    assert m(x).shape == (4, 10) and m(x, mode="features").shape == (4, 400) and len(m(x, mode="acts")) == 2


def test_resnet_no_batchnorm():
    m = ReducedResNet18(10)
    assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert not any(isinstance(l, nn.modules.batchnorm._BatchNorm) for l in m.modules())
    assert [a.shape[1] for a in m(torch.randn(2, 3, 32, 32), mode="acts")] == [20, 40, 80, 160]


def test_last_token_pooling_ignores_padding():
    m = HFClassifier("tiny-gptneox", 6).eval()
    ids = torch.tensor([[5, 6, 7, 0, 0], [5, 6, 7, 9, 9]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    f_pad = m(ids, mask, mode="features")[0]
    f_short = m(ids[:1, :3], mask[:1, :3], mode="features")[0]
    assert torch.allclose(f_pad, f_short, atol=1e-5)
    assert m(ids, mask, mode="lm").shape == (2, 5, 128)


def test_build_model_from_stream():
    assert isinstance(build_model(ModelConfig(name="mlp", hidden=16), build_synthetic()), MLP)
    m = build_model(ModelConfig(name="hf", hf_name="tiny-gptneox"), build_synthetic_text())
    assert m.head.out_features == 6
