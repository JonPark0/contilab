"""Multi-teacher distillation of task experts into one student; teacher targets are computed once per call."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from contilab.consolidate.consolidator import ConsolidationContext, Consolidator, TAConsolidator, TIESConsolidator
from contilab.consolidate.experts import Experts
from contilab.consolidate.transfer import TransferSet, build_transfer
from contilab.core.scenario import MASK_FILL
from contilab.core.types import Batch, RunContext
from contilab.data.batching import TrainPool, epoch_perms
from contilab.engine.step import TrainStep
from contilab.models.lora import set_active
from contilab.strategies.base import Strategy


def mix_teachers(logits: Tensor, class_table: Tensor, routing: str, T: float, task_of: Tensor | None = None) -> Tensor:
    """logits [K, N, C] from K teachers → target distribution [N, C]. Teacher k only speaks for its own classes."""
    table = class_table[: logits.shape[0]].to(logits.device)  # [K, C]
    p = (logits / T).masked_fill(~table[:, None, :], MASK_FILL).softmax(-1) * table[:, None, :]
    if routing == "routed" and task_of is not None:
        return p[task_of.to(p.device), torch.arange(p.shape[1], device=p.device)]
    H = -(p * p.clamp_min(1e-12).log()).sum(-1)  # [K, N]
    w = (-H).softmax(0)
    return (w[..., None] * p).sum(0)


class DistillStrategy(Strategy):
    def __init__(self, T: float):
        super().__init__()
        self.T = T

    def loss(self, p, s, batch, replay, f):
        z = f.logits(p, batch.x, batch.attn_mask, batch.mask)
        tgt = batch.extra["target"]
        logq = (z / self.T).log_softmax(-1)
        kl = (tgt * (tgt.clamp_min(1e-12).log() - logq)).sum(-1).mean()
        return self.T * self.T * kl, z


class DistillConsolidator(Consolidator):
    @torch.no_grad()
    def _teacher_logits(self, experts: Experts, upto: int, x: Tensor, am: Tensor | None, shared: bool,
                        cctx: ConsolidationContext) -> Tensor:
        ens = cctx.ens
        dev = next(iter(ens.params.values())).device
        bs = cctx.cfg.train.eval_batch_size
        if experts.lora_modules:
            set_active(ens.template, [0])
        out = []
        for t in range(upto + 1):
            p = {k: v.to(dev) for k, v in experts.tuned[t].items()}
            chunks = []
            n = x.shape[0] if shared else x.shape[1]
            for i in range(0, n, bs):
                xi = x[i : i + bs] if shared else x[:, i : i + bs]
                ami = None if am is None else (am[i : i + bs] if shared else am[:, i : i + bs])
                chunks.append(ens.apply(p, xi, ami, shared_input=shared).float())
            out.append(torch.cat(chunks, 1))
        if experts.lora_modules:
            set_active(ens.template, [])
        return torch.stack(out)  # [K, S, N, C]

    def _init(self, experts: Experts, upto: int, cctx: ConsolidationContext) -> dict[str, Tensor]:
        if self.cfg.init == "ta":
            return TAConsolidator(self.cfg).consolidate(experts, upto, cctx)
        if self.cfg.init == "ties":
            return TIESConsolidator(self.cfg).consolidate(experts, upto, cctx)
        return {k: v.clone() for k, v in experts.base.items()}

    def consolidate(self, experts: Experts, upto: int, cctx: ConsolidationContext) -> dict[str, Tensor]:
        ens, stream, cfg = cctx.ens, cctx.stream, cctx.cfg
        dev = next(iter(ens.params.values())).device
        S = ens.E
        pool = TrainPool.from_stream(stream)
        gens = [torch.Generator().manual_seed(10_007 * (upto + 1) + e) for e in range(S)]
        tr: TransferSet = build_transfer(stream, pool, upto, self.cfg.transfer, self.cfg.M, gens,
                                         cfg.benchmark.tokenizer, cfg.benchmark.max_len)
        table = stream.class_table().to(dev)
        seen = table[: upto + 1].any(0)
        if tr.gidx is not None:
            b = pool.gather(tr.gidx)
            x, am, task_of, shared = b.x, b.attn_mask, b.task_id, False
        else:
            x, am, task_of, shared = tr.proxy.x, tr.proxy.attn_mask, None, True
        logits = self._teacher_logits(experts, upto, x, am, shared, cctx)  # cached for all epochs
        if task_of is None:
            routing = "entropy"
        elif stream.scenario == "task":
            routing = "routed"  # task identity is given in task-IL: the owner teacher speaks
        else:
            routing = self.cfg.routing
        targets = torch.stack([mix_teachers(logits[:, e], table, routing, self.cfg.temperature,
                                            None if task_of is None else task_of[e]) for e in range(S)])  # [S, N, C]
        if stream.scenario == "task" and task_of is not None:
            masks = table[task_of]
        else:
            masks = seen.expand(S, len(tr), -1)
        targets = targets * masks  # targets live on the classes the student may predict
        mass = targets.sum(-1, keepdim=True)
        targets = torch.where(mass > 0, targets / mass.clamp_min(1e-12), masks.float() / masks.sum(-1, keepdim=True))
        if shared:
            x = x.expand(S, *x.shape)
            am = None if am is None else am.expand(S, *am.shape)

        ens.load_params({k: v.to(dev) for k, v in self._init(experts, upto, cctx).items()})
        step = TrainStep(ens, DistillStrategy(self.cfg.temperature), stream, cfg.engine, cfg.optim)
        step.set_task(upto)
        ctx = RunContext(task_idx=upto, step=0, device=dev, E=S, gens=gens, stream=stream,
                         active=torch.ones(S, dtype=torch.bool), ens=ens)
        N = len(tr)
        rows = torch.arange(S, device=dev)[:, None]
        for _ in range(self.cfg.epochs):
            perm = epoch_perms(N, gens).to(dev)
            for i in range(0, N, cfg.train.batch_size):
                idx = perm[:, i : i + cfg.train.batch_size]
                batch = Batch(x=x[rows, idx], y=torch.zeros_like(idx), task_id=torch.zeros_like(idx),
                              attn_mask=None if am is None else am[rows, idx], mask=masks[rows, idx],
                              extra={"target": targets[rows, idx]})
                step(batch, None, ctx)
        return ens.snapshot()
