"""Shared test helpers (imported by test modules; conftest only handles options/markers)."""

from __future__ import annotations

import torch

from contilab.core.types import Stream, Task, TensorSplit


def make_stream(classes: list[list[int]], scenario: str = "class", C: int | None = None, n: int = 4, dim: int = 3) -> Stream:
    """Dummy stream whose tasks own the given global class ids."""
    C = C if C is not None else max(max(c) for c in classes) + 1
    tasks = []
    for i, cls in enumerate(classes):
        y = torch.tensor([cls[j % len(cls)] for j in range(n)])
        split = TensorSplit(torch.randn(n, dim), y)
        tasks.append(Task(id=i, train=split, val=split, test=split, classes=torch.tensor(cls)))
    return Stream(name="dummy", tasks=tasks, scenario=scenario, num_classes=C, modality="vision")
