"""Common model contract: every path is reachable through forward(mode=...) so functional_call can use it."""

from __future__ import annotations

from torch import Tensor, nn


class CLModel(nn.Module):
    head: nn.Linear

    def features(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        raise NotImplementedError

    def activations(self, x: Tensor, attn_mask: Tensor | None = None) -> list[Tensor]:
        raise NotImplementedError

    def lm_logits(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__} has no language-model head")

    def frozen_names(self) -> set[str]:
        """Parameters never trained by classification (e.g. the original LM head)."""
        return set()

    def forward(self, x: Tensor, attn_mask: Tensor | None = None, mode: str = "logits"):
        if mode == "logits":
            return self.head(self.features(x, attn_mask))
        if mode == "features":
            return self.features(x, attn_mask)
        if mode == "acts":
            return self.activations(x, attn_mask)
        if mode == "lm":
            return self.lm_logits(x, attn_mask)
        raise ValueError(f"unknown mode: {mode}")
