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

    def sample(self, n: int, gens: list[torch.Generator], valid: Tensor | None = None) -> Tensor | None:
        """Slot indices [E, n'] drawn without replacement from filled (and `valid`) slots;
        n' = min(n, smallest eligible count over members)."""
        ok = self.tensors["gidx"] >= 0
        if valid is not None:
            ok = ok & valid
        counts = ok.sum(1)
        n = min(n, int(counts.min()))
        if n == 0:
            return None
        dev = self.tensors["gidx"].device
        rows = []
        for e, g in enumerate(gens):
            slots = torch.nonzero(ok[e]).flatten().cpu()
            rows.append(slots[torch.randperm(len(slots), generator=g)[:n]])
        return torch.stack(rows).to(dev)

    def gather(self, slots: Tensor, key: str = "gidx") -> Tensor:
        return torch.gather(self.tensors[key], 1, slots) if key == "gidx" else \
            torch.gather(self.tensors[key], 1, slots[..., None].expand(-1, -1, self.tensors[key].shape[-1]))

    def state_dict(self) -> dict:
        return {"tensors": dict(self.tensors), "seen": list(self.seen)}

    def load_state_dict(self, d: dict) -> None:
        self.tensors = dict(d["tensors"])
        self.seen = list(d["seen"])
