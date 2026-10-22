"""Per-member reservoir buffer that stores global dataset indices (and optionally logits), never sample copies."""

from __future__ import annotations

import torch
from torch import Tensor


class ReservoirBuffer:
    def __init__(self, E: int, M: int, C: int | None, device: torch.device):
        self.E, self.M = E, M
        self.tensors: dict[str, Tensor] = {"gidx": torch.full((E, M), -1, dtype=torch.long, device=device)}
        if C is not None:
            self.tensors["logits"] = torch.zeros(E, M, C, device=device)
        self.seen = [0] * E

    def size(self) -> Tensor:
        return torch.tensor([min(s, self.M) for s in self.seen])

    @torch.no_grad()
    def add(self, batch_index: Tensor, logits: Tensor | None, gens: list[torch.Generator]) -> None:
        if self.M == 0:
            return
        B = batch_index.shape[1]
        dev = self.tensors["gidx"].device
        for e, g in enumerate(gens):
            j = torch.arange(self.seen[e], self.seen[e] + B)
            r = (torch.rand(B, generator=g) * (j + 1).float()).long()
            slot = torch.where(j < self.M, j, r)
            keep = slot < self.M
            slot, src = slot[keep].to(dev), torch.nonzero(keep).flatten().to(dev)
            self.tensors["gidx"][e, slot] = batch_index[e, src]
            if logits is not None:
                self.tensors["logits"][e, slot] = logits[e, src].float()
            self.seen[e] += B

    def sample(self, n: int, gens: list[torch.Generator]) -> Tensor | None:
        """Slot indices [E, n'] (n' = min(n, filled)); filled count is equal across members."""
        k = min(self.seen[0], self.M)
        if k == 0:
            return None
        n = min(n, k)
        dev = self.tensors["gidx"].device
        return torch.stack([torch.randperm(k, generator=g)[:n] for g in gens]).to(dev)

    def gather(self, slots: Tensor, key: str = "gidx") -> Tensor:
        return torch.gather(self.tensors[key], 1, slots) if key == "gidx" else \
            torch.gather(self.tensors[key], 1, slots[..., None].expand(-1, -1, self.tensors[key].shape[-1]))

    def state_dict(self) -> dict:
        return {"tensors": dict(self.tensors), "seen": list(self.seen)}

    def load_state_dict(self, d: dict) -> None:
        self.tensors = dict(d["tensors"])
        self.seen = list(d["seen"])
