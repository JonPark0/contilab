"""Scenario-specific logit masks (§4.5): one shared head, masked per sample."""

from __future__ import annotations

import torch
from torch import Tensor

from contilab.core.types import Stream

MASK_FILL = -1e9  # finite stand-in for -inf on disallowed logits (stable CE/KL)


def sample_mask(stream: Stream, task_ids: Tensor, current_task: int) -> Tensor:
    """Bool[*task_ids.shape, C] of classes each sample may predict."""
    table = stream.class_table(task_ids.device)
    C = stream.num_classes
    if stream.scenario == "task":
        return table[task_ids]
    if stream.scenario == "class":
        seen = table[: current_task + 1].any(0)
        return seen.expand(*task_ids.shape, C)
    if stream.scenario == "domain":
        return torch.ones(*task_ids.shape, C, dtype=torch.bool, device=task_ids.device)
    raise ValueError(f"unknown scenario: {stream.scenario}")
