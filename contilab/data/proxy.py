"""Unlabelled proxy transfer data for data-free-ish distillation (§5.5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from contilab.core.types import TensorSplit


def _unlabelled(x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> TensorSplit:
    return TensorSplit(x, torch.zeros(x.shape[0], dtype=torch.long), attn_mask)


def _pick(n_total: int, n: int) -> torch.Tensor:
    return torch.randperm(n_total, generator=torch.Generator().manual_seed(0))[:n]


def build_proxy(stream_name: str, n: int, tokenizer_name: str | None = None, max_len: int = 128) -> TensorSplit:
    from contilab.data import vision

    if stream_name in ("split_mnist", "permuted_mnist"):
        raw = vision.load_raw("cifar10", "train")["x"]
        x = raw[_pick(raw.shape[0], n)].float().mean(-1, keepdim=True).permute(0, 3, 1, 2) / 255.0
        x = F.interpolate(x, size=(28, 28), mode="bilinear", align_corners=False)
        return _unlabelled(((x - vision.MNIST_MEAN) / vision.MNIST_STD).reshape(n, -1))
    if stream_name in ("split_cifar10", "split_cifar100"):
        other = "cifar100" if stream_name == "split_cifar10" else "cifar10"
        raw = vision.load_raw(other, "train")["x"]
        return _unlabelled(vision.cifar_tensor(raw[_pick(raw.shape[0], n)]))
    if stream_name == "text5":
        from contilab.data.text import wikitext_split

        return wikitext_split("train", n, tokenizer_name or "EleutherAI/pythia-70m", max_len)
    if stream_name == "synthetic":
        return _unlabelled(torch.randn(n, 32, generator=torch.Generator().manual_seed(0)))
    if stream_name == "synthetic_text":
        x = torch.randint(1, 122, (n, 16), generator=torch.Generator().manual_seed(0))
        return _unlabelled(x, torch.ones_like(x))
    raise ValueError(f"no proxy data for stream '{stream_name}'")
