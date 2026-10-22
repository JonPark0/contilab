"""Vision CL benchmarks (§4.1), downloaded from the original dataset mirrors."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from contilab.core.types import Stream, Task, TensorSplit
from contilab.data.cache import cached_tensors
from contilab.data.download import fetch
from contilab.data.splits import subsample, train_val_split

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
CIFAR_URLS = {
    "cifar10": "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz",
    "cifar100": "https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz",
}


def _read_png_tgz(path) -> dict[str, dict[str, Tensor]]:
    """fast.ai CIFAR archives: <root>/<split>/[<coarse>/]<class>/<file>.png; label = sorted class-name index
    (alphabetical order equals the canonical CIFAR label order)."""
    import io
    import tarfile

    from PIL import Image

    items: dict[str, list[tuple[str, np.ndarray]]] = {"train": [], "test": []}
    with tarfile.open(path, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".png"):
                continue
            parts = m.name.split("/")
            img = np.asarray(Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB"), dtype=np.uint8)
            items[parts[1]].append((parts[-2], img))
    names = sorted({c for v in items.values() for c, _ in v})
    index = {c: i for i, c in enumerate(names)}
    out = {}
    for split, v in items.items():
        v.sort(key=lambda t: t[0])
        out[split] = {"x": torch.from_numpy(np.stack([im for _, im in v])),
                      "y": torch.tensor([index[c] for c, _ in v], dtype=torch.long)}
    return out


def load_raw(name: str, split: str) -> dict[str, Tensor]:
    """uint8 images + labels for one split, cached on disk."""
    url = MNIST_URL if name == "mnist" else CIFAR_URLS[name]

    def build() -> dict[str, Tensor]:
        path = fetch(url)
        if name == "mnist":
            with np.load(path) as d:
                key = "x_train" if split == "train" else "x_test"
                return {"x": torch.from_numpy(d[key]), "y": torch.from_numpy(d[key.replace("x", "y")]).long()}
        return _read_png_tgz(path)[split]

    return cached_tensors("vision", {"url": url, "split": split}, build)


def mnist_tensor(raw: Tensor) -> Tensor:
    return ((raw.float() / 255.0 - MNIST_MEAN) / MNIST_STD).reshape(raw.shape[0], -1)


def cifar_tensor(raw: Tensor) -> Tensor:
    x = raw.float().permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def split_by_classes(x: Tensor, y: Tensor, groups: list[list[int]]) -> list[tuple[Tensor, Tensor]]:
    out = []
    for grp in groups:
        sel = torch.isin(y, torch.tensor(grp))
        out.append((x[sel], y[sel]))
    return out


def _make_tasks(train: tuple[Tensor, Tensor], test: tuple[Tensor, Tensor], groups: list[list[int]],
                n_train: int | None, n_test: int | None) -> list[Task]:
    tasks = []
    tr_parts = split_by_classes(*train, groups)
    te_parts = split_by_classes(*test, groups)
    for t, (grp, tr, te) in enumerate(zip(groups, tr_parts, te_parts)):
        tr_split, va_split = train_val_split(subsample(TensorSplit(*tr), n_train), 0.1, seed=t)
        tasks.append(Task(t, tr_split, va_split, subsample(TensorSplit(*te), n_test), torch.tensor(grp)))
    return tasks


def _split_stream(name: str, prep, n_classes: int, n_tasks: int, scenario: str,
                  n_train: int | None, n_test: int | None) -> Stream:
    tr, te = load_raw(name, "train"), load_raw(name, "test")
    per = n_classes // n_tasks
    groups = [list(range(t * per, (t + 1) * per)) for t in range(n_tasks)]
    tasks = _make_tasks((prep(tr["x"]), tr["y"]), (prep(te["x"]), te["y"]), groups, n_train, n_test)
    return Stream(f"split_{name}", tasks, scenario, n_classes, "vision")


def build_split_mnist(scenario: str = "class", n_train: int | None = None, n_test: int | None = None) -> Stream:
    return _split_stream("mnist", mnist_tensor, 10, 5, scenario, n_train, n_test)


def build_split_cifar10(scenario: str = "class", n_train: int | None = None, n_test: int | None = None) -> Stream:
    return _split_stream("cifar10", cifar_tensor, 10, 5, scenario, n_train, n_test)


def build_split_cifar100(scenario: str = "class", n_train: int | None = None, n_test: int | None = None) -> Stream:
    return _split_stream("cifar100", cifar_tensor, 100, 10, scenario, n_train, n_test)


def build_permuted_mnist(n_tasks: int = 10, n_train: int | None = None, n_test: int | None = None, seed: int = 0) -> Stream:
    tr, te = load_raw("mnist", "train"), load_raw("mnist", "test")
    xtr, xte = mnist_tensor(tr["x"]), mnist_tensor(te["x"])
    g = torch.Generator().manual_seed(seed)
    classes = torch.arange(10)
    tasks = []
    for t in range(n_tasks):
        perm = torch.arange(784) if t == 0 else torch.randperm(784, generator=g)
        train = subsample(TensorSplit(xtr[:, perm], tr["y"]), n_train, seed=t)
        train, val = train_val_split(train, 0.1, seed=t)
        test = subsample(TensorSplit(xte[:, perm], te["y"]), n_test, seed=t)
        tasks.append(Task(t, train, val, test, classes.clone()))
    return Stream("permuted_mnist", tasks, "domain", 10, "vision")
