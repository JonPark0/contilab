from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from contilab.core.types import TensorSplit
from contilab.data.batching import eval_batches
from contilab.engine.ensemble import Ensemble
from contilab.strategies.base import MASK_FILL


@dataclass
class EvalResult:
    correct: Tensor  # Bool[E, N]
    pred: Tensor  # Long[E, N]
    conf: Tensor  # [E, N] max softmax probability over allowed classes
    logits: Tensor | None = None  # [E, N, C]


@torch.inference_mode()
def evaluate_split(ens: Ensemble, params: dict[str, Tensor], split: TensorSplit, mask_row: Tensor, batch_size: int,
                   keep_logits: bool = False) -> EvalResult:
    mask_row = mask_row.to(split.x.device)
    correct, pred, conf, logits = [], [], [], []
    for x, y, am in eval_batches(split, batch_size):
        z = ens.apply(params, x, am, shared_input=True).float().masked_fill(~mask_row, MASK_FILL)
        prob = z.softmax(-1)
        c, p = prob.max(-1)
        correct.append(p == y)
        pred.append(p)
        conf.append(c)
        if keep_logits:
            logits.append(z)
    return EvalResult(torch.cat(correct, 1), torch.cat(pred, 1), torch.cat(conf, 1),
                      torch.cat(logits, 1) if keep_logits else None)
