from __future__ import annotations

import torch

from contilab.core.types import Stream, Task, TensorSplit


def _cat(splits: list[TensorSplit]) -> TensorSplit:
    am = torch.cat([s.attn_mask for s in splits]) if splits[0].attn_mask is not None else None
    return TensorSplit(torch.cat([s.x for s in splits]), torch.cat([s.y for s in splits]), am)


def make_joint_stream(stream: Stream) -> Stream:
    """All tasks merged into one task (the Joint upper bound)."""
    ts = stream.tasks
    classes = torch.unique(torch.cat([t.classes for t in ts]))
    joint = Task(0, _cat([t.train for t in ts]), _cat([t.val for t in ts]), _cat([t.test for t in ts]), classes)
    return Stream(f"{stream.name}-joint", [joint], stream.scenario, stream.num_classes, stream.modality, stream.lm_eval)
