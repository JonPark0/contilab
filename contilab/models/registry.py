from __future__ import annotations

import math

from contilab.core.config import ModelConfig
from contilab.core.types import Stream
from contilab.models.base import CLModel


def build_model(cfg: ModelConfig, stream: Stream) -> CLModel:
    x = stream.tasks[0].train.x
    C = stream.num_classes
    if cfg.name == "mlp":
        from contilab.models.mlp import MLP

        return MLP(math.prod(x.shape[1:]), C, cfg.hidden, cfg.depth)
    if cfg.name == "resnet18r":
        from contilab.models.resnet import ReducedResNet18

        return ReducedResNet18(C, cfg.nf, x.shape[1])
    if cfg.name == "hf":
        from contilab.models.hf import HFClassifier

        if cfg.hf_name is None:
            raise ValueError("model.hf_name is required for name='hf'")
        return HFClassifier(cfg.hf_name, C)
    raise ValueError(f"unknown model: {cfg.name}")
