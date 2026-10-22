from __future__ import annotations

from torch import Tensor, nn

from contilab.models.base import CLModel


class MLP(CLModel):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 400, depth: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden] * depth
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = nn.Linear(hidden, num_classes)

    def activations(self, x: Tensor, attn_mask: Tensor | None = None) -> list[Tensor]:
        acts = []
        h = x.flatten(1)
        for layer in self.layers:
            h = layer(h).relu()
            acts.append(h)
        return acts

    def features(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        return self.activations(x)[-1]
