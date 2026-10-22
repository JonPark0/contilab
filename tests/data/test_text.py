import pytest
import torch

from contilab.data.proxy import build_proxy
from contilab.data.text import TEXT5_SOURCES, build_text5


def test_text5_label_layout():
    offs = [(o, n) for _, _, o, n in TEXT5_SOURCES]
    assert offs == [(0, 4), (4, 5), (4, 5), (9, 14), (23, 10)]


def test_proxy_synthetic_shape():
    assert build_proxy("synthetic", 16).x.shape == (16, 32)
    p = build_proxy("synthetic_text", 8)
    assert p.x.shape == (8, 16) and p.attn_mask.all()


@pytest.mark.network
def test_build_text5_small():
    s = build_text5("EleutherAI/pythia-70m", n_train=20, n_test=10, max_len=32)
    assert s.num_classes == 33 and s.tasks[1].classes.tolist() == s.tasks[2].classes.tolist()
    x, m = s.tasks[0].train.x, s.tasks[0].train.attn_mask
    assert x.shape[1] == 32 and m[:, 0].all() and s.lm_eval is not None
    assert s.tasks[3].train.y.min() >= 9 and s.tasks[4].test.y.max() <= 32


@pytest.mark.network
def test_proxy_for_mnist_is_cifar_gray():
    p = build_proxy("split_mnist", 10)
    assert p.x.shape == (10, 784)
