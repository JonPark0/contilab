"""Small deterministic synthetic streams for smoke runs and tests."""

from __future__ import annotations

import torch

from contilab.core.types import Stream, Task, TensorSplit
from contilab.data.splits import train_val_split


def build_synthetic(
    n_tasks: int = 5,
    classes_per_task: int = 2,
    dim: int = 32,
    n_train: int = 200,
    n_test: int = 100,
    seed: int = 0,
    scenario: str = "class",
) -> Stream:
    """Gaussian clusters. Domain scenario: classes shared, cluster centres differ per task."""
    g = torch.Generator().manual_seed(seed)
    domain = scenario == "domain"
    C = classes_per_task if domain else n_tasks * classes_per_task
    tasks = []
    for t in range(n_tasks):
        local = list(range(classes_per_task))
        glob = local if domain else [t * classes_per_task + c for c in local]
        centers = torch.randn(classes_per_task, dim, generator=g) * 3.0

        def draw(n: int) -> TensorSplit:
            lab = torch.arange(n) % classes_per_task
            x = centers[lab] + torch.randn(n, dim, generator=g)
            return TensorSplit(x, torch.tensor(glob)[lab])

        train, val = train_val_split(draw(n_train), 0.1, seed=t)
        tasks.append(Task(t, train, val, draw(n_test), torch.tensor(glob)))
    return Stream("synthetic", tasks, scenario, C, "vision")


def build_synthetic_text(
    n_tasks: int = 3,
    classes_per_task: int = 2,
    vocab: int = 128,
    seq_len: int = 16,
    n_train: int = 200,
    n_test: int = 100,
    seed: int = 0,
    scenario: str = "class",
) -> Stream:
    """Token sequences; each class has a unique marker token. Pad id 0, right padding."""
    g = torch.Generator().manual_seed(seed)
    C = n_tasks * classes_per_task
    marker0 = vocab - C  # marker tokens occupy the top of the vocabulary

    def draw(n: int, glob: list[int]) -> TensorSplit:
        lab = torch.arange(n) % len(glob)
        lengths = torch.randint(6, seq_len + 1, (n,), generator=g)
        x = torch.randint(1, marker0, (n, seq_len), generator=g)
        pos = torch.arange(seq_len).expand(n, seq_len)
        mask = (pos < lengths[:, None]).long()
        where = (torch.rand(n, generator=g) * lengths).long()
        y = torch.tensor(glob)[lab]
        x[torch.arange(n), where] = marker0 + y
        x = x * mask
        return TensorSplit(x, y, mask)

    tasks = []
    for t in range(n_tasks):
        glob = [t * classes_per_task + c for c in range(classes_per_task)]
        train, val = train_val_split(draw(n_train, glob), 0.1, seed=t)
        tasks.append(Task(t, train, val, draw(n_test, glob), torch.tensor(glob)))
    lm_x = torch.randint(1, marker0, (64, seq_len), generator=g)
    lm = TensorSplit(lm_x, torch.zeros(64, dtype=torch.long), torch.ones(64, seq_len, dtype=torch.long))
    return Stream("synthetic_text", tasks, scenario, C, "text", lm_eval=lm)
