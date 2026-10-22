import pytest
import torch

from contilab.core.config import BenchmarkConfig
from contilab.data.registry import build_stream
from contilab.data.vision import build_permuted_mnist, build_split_mnist, split_by_classes


def test_split_by_classes():
    y = torch.tensor([0, 1, 2, 3, 0, 2])
    x = torch.arange(6.0).view(6, 1)
    parts = split_by_classes(x, y, [[0, 1], [2, 3]])
    assert parts[0][1].tolist() == [0, 1, 0] and parts[1][0].view(-1).tolist() == [2.0, 3.0, 5.0]


def test_registry_builds_synthetic_with_task_order():
    s = build_stream(BenchmarkConfig(name="synthetic", scenario="task", n_tasks=3, task_order_seed=1))
    assert [t.id for t in s.tasks] == [0, 1, 2] and s.scenario == "task"
    assert sorted(t.classes.tolist()[0] for t in s.tasks) == [0, 2, 4]


def test_registry_unknown_benchmark():
    with pytest.raises(ValueError, match="unknown benchmark"):
        build_stream(BenchmarkConfig(name="nope"))


@pytest.mark.network
def test_split_mnist_shapes():
    s = build_split_mnist("class", n_train=100, n_test=50)
    assert len(s.tasks) == 5 and s.tasks[0].train.x.shape[1] == 784 and s.num_classes == 10
    assert s.tasks[4].classes.tolist() == [8, 9] and len(s.tasks[0].test) == 50


@pytest.mark.network
def test_permuted_mnist_first_task_identity():
    s = build_permuted_mnist(n_tasks=3, n_train=50, n_test=20)
    assert s.scenario == "domain" and s.tasks[0].classes.tolist() == list(range(10))
    assert not torch.equal(s.tasks[1].test.x, s.tasks[0].test.x)


@pytest.mark.network
def test_split_cifar10_labels_canonical():
    from contilab.data.vision import build_split_cifar10, load_raw

    s = build_split_cifar10("class", n_train=50, n_test=20)
    assert s.tasks[0].train.x.shape[1:] == (3, 32, 32) and s.tasks[2].classes.tolist() == [4, 5]
    raw = load_raw("cifar10", "test")
    assert raw["x"].shape == (10000, 32, 32, 3) and torch.bincount(raw["y"]).tolist() == [1000] * 10
