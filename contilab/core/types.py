"""Core data types shared by every layer of contilab."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor

if TYPE_CHECKING:
    from contilab.data.batching import TrainPool
    from contilab.engine.ensemble import Ensemble

Scenario = Literal["task", "class", "domain"]
Modality = Literal["vision", "text"]


@dataclass
class TensorSplit:
    """Device-resident tensors of one data split."""

    x: Tensor
    y: Tensor
    attn_mask: Tensor | None = None

    def __len__(self) -> int:
        return self.y.shape[0]

    def subset(self, idx: Tensor) -> TensorSplit:
        am = self.attn_mask[idx] if self.attn_mask is not None else None
        return TensorSplit(self.x[idx], self.y[idx], am)

    def to(self, device: torch.device | str) -> TensorSplit:
        am = self.attn_mask.to(device) if self.attn_mask is not None else None
        return TensorSplit(self.x.to(device), self.y.to(device), am)


@dataclass
class Task:
    id: int
    train: TensorSplit
    val: TensorSplit
    test: TensorSplit
    classes: Tensor  # global class ids (long)


@dataclass
class Stream:
    name: str
    tasks: list[Task]
    scenario: Scenario
    num_classes: int
    modality: Modality
    lm_eval: TensorSplit | None = None

    def class_table(self, device: torch.device | str | None = None) -> Tensor:
        """Bool[T, C]: which global classes each task owns. Built once per stream and device, then cached
        (the hot path asks for it every batch); callers must not modify it in place."""
        cache = self.__dict__.setdefault("_class_tables", {})
        key = str(torch.device(device)) if device is not None else "cpu"
        if key not in cache:
            if "cpu" not in cache:
                table = torch.zeros(len(self.tasks), self.num_classes, dtype=torch.bool)
                for i, t in enumerate(self.tasks):
                    table[i, t.classes.cpu()] = True
                cache["cpu"] = table
            cache[key] = cache["cpu"].to(key)
        return cache[key]

    def to(self, device: torch.device | str) -> Stream:
        tasks = [replace(t, train=t.train.to(device), val=t.val.to(device), test=t.test.to(device)) for t in self.tasks]
        lm = self.lm_eval.to(device) if self.lm_eval is not None else None
        return replace(self, tasks=tasks, lm_eval=lm)


@dataclass
class Batch:
    """A training batch; during training every field carries a leading member axis [E, B, ...]."""

    x: Tensor
    y: Tensor
    task_id: Tensor
    attn_mask: Tensor | None = None
    index: Tensor | None = None
    mask: Tensor | None = None
    extra: dict[str, Any] | None = None


@dataclass
class RunContext:
    task_idx: int
    step: int
    device: torch.device
    E: int
    gens: list[torch.Generator]
    stream: Stream
    active: Tensor  # Bool[E]
    diverged: set[int] = field(default_factory=set)
    pool: TrainPool | None = None
    ens: Ensemble | None = None
    loss_baseline: Tensor | None = None  # first loss per member (divergence reference; checkpointed)
