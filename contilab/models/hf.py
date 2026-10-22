"""Hugging Face causal LM with a classification head on the last non-padding token."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from contilab.models.base import CLModel

TINY_CONFIG = dict(vocab_size=128, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                   intermediate_size=64, max_position_embeddings=64)


def load_lm(hf_name: str) -> nn.Module:
    from transformers import AutoModelForCausalLM, GPTNeoXConfig

    if hf_name.startswith("tiny-"):
        return AutoModelForCausalLM.from_config(GPTNeoXConfig(**TINY_CONFIG), attn_implementation="eager")
    return AutoModelForCausalLM.from_pretrained(hf_name, torch_dtype=torch.float32, attn_implementation="eager")


class HFClassifier(CLModel):
    def __init__(self, hf_name: str, num_classes: int, pooling: str = "last_token"):
        super().__init__()
        if pooling != "last_token":
            raise ValueError(f"unsupported pooling: {pooling}")
        self.lm = load_lm(hf_name)
        self.head = nn.Linear(self.lm.config.hidden_size, num_classes)

    def _hidden(self, x: Tensor, attn_mask: Tensor | None, all_layers: bool = False):
        out = self.lm.base_model(input_ids=x, attention_mask=attn_mask, output_hidden_states=all_layers)
        return out.hidden_states if all_layers else out.last_hidden_state

    @staticmethod
    def _last_index(x: Tensor, attn_mask: Tensor | None) -> Tensor:
        if attn_mask is None:
            return torch.full(x.shape[:1], x.shape[1] - 1, device=x.device)
        return (attn_mask.sum(-1) - 1).clamp(min=0)

    def features(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        h = self._hidden(x, attn_mask)
        idx = self._last_index(x, attn_mask)
        return h[torch.arange(h.shape[0], device=h.device), idx]

    def activations(self, x: Tensor, attn_mask: Tensor | None = None) -> list[Tensor]:
        hs = self._hidden(x, attn_mask, all_layers=True)[1:]
        m = attn_mask.unsqueeze(-1).float() if attn_mask is not None else torch.ones(*x.shape, 1, device=x.device)
        return [(h * m).sum(1) / m.sum(1).clamp(min=1) for h in hs]

    def lm_logits(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        return self.lm(input_ids=x, attention_mask=attn_mask).logits

    def frozen_names(self) -> set[str]:
        out_emb = self.lm.get_output_embeddings()
        ids = {id(p) for p in out_emb.parameters()}
        return {n for n, p in self.named_parameters() if id(p) in ids}
