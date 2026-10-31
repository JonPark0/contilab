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
from contilab.engine.ensemble import Ensemble
from contilab.engine.step import TrainStep
from contilab.engine.trainer import DivergenceGuard
from contilab.models.lora import set_active
from contilab.strategies.base import Strategy


def transfer_gens(seeds: list[int], upto: int) -> list[torch.Generator]:
    """Per-run generators for transfer sampling and batch order, keyed by seed (never by ensemble slot)."""
    return [torch.Generator().manual_seed((s * 1_000_003 + 10_007 * (upto + 1)) % (2**63)) for s in seeds]


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


@torch.no_grad()
def _member_logits(model: Ensemble, params: dict[str, Tensor], x: Tensor, am: Tensor | None, shared: bool,
                   bs: int) -> Tensor:
    """Logits [S, N, C] of every member of `model` (with `params`) on x ([N, ...] if shared, else [S, N, ...])."""
    dev = next(iter(model.params.values())).device
    p = {k: v.to(dev) for k, v in params.items()}
    n = x.shape[0] if shared else x.shape[1]
    chunks = []
    for i in range(0, n, bs):
        xi = x[i : i + bs] if shared else x[:, i : i + bs]
        ami = None if am is None else (am[i : i + bs] if shared else am[:, i : i + bs])
        chunks.append(model.apply(p, xi, ami, shared_input=shared).float())
    return torch.cat(chunks, 1)


def _student_targets(mixed: Tensor, masks: Tensor) -> Tensor:
    """Restrict teacher targets to the classes the student may predict and renormalise (uniform if no mass)."""
    targets = mixed * masks
    mass = targets.sum(-1, keepdim=True)
    return torch.where(mass > 0, targets / mass.clamp_min(1e-12), masks.float() / masks.sum(-1, keepdim=True))


class DistillConsolidator(Consolidator):
    supports_student = True

    def __init__(self, cfg, params=None):
        super().__init__(cfg, params)
        self._student_init: dict[str, Tensor] | None = None

    def _teacher_logits(self, experts: Experts, tasks: list[int], x: Tensor, am: Tensor | None, shared: bool,
                        cctx: ConsolidationContext) -> Tensor:
        teachers = cctx.teachers
        if experts.lora_modules:
            set_active(teachers.template, [0])
        out = [_member_logits(teachers, experts.tuned[t], x, am, shared, cctx.cfg.train.eval_batch_size) for t in tasks]
        if experts.lora_modules:
            set_active(teachers.template, [])
        return torch.stack(out)  # [K, S, N, C]

    def _init(self, experts: Experts, upto: int, cctx: ConsolidationContext) -> dict[str, Tensor]:
        if cctx.teacher is not None:  # another student architecture: its own fixed initialisation
            if self.cfg.init != "base":
                raise ValueError(f"consolidate.init={self.cfg.init} merges expert weights: it needs the teacher architecture")
            if self._student_init is None:
                self._student_init = cctx.ens.snapshot()
            return self._student_init
        if self.cfg.init == "ta":
            return TAConsolidator(self.cfg).consolidate(experts, upto, cctx)
        if self.cfg.init == "ties":
            return TIESConsolidator(self.cfg).consolidate(experts, upto, cctx)
        return {k: v.clone() for k, v in experts.base.items()}

    def _routing(self, stream, known_task: bool) -> str:
        if not known_task:
            return "entropy"
        if stream.scenario == "task":
            return "routed"  # task identity is given in task-IL: the owner teacher speaks
        return self.cfg.routing

    def _masks(self, stream, table: Tensor, upto: int, task_of: Tensor | None, S: int, N: int) -> Tensor:
        if stream.scenario == "task" and task_of is not None:
            return table[task_of]
        return table[: upto + 1].any(0).expand(S, N, -1)

    def _fit(self, init: dict[str, Tensor], x: Tensor, am: Tensor | None, masks: Tensor, targets: Tensor,
             gens: list[torch.Generator], epochs: int, upto: int, cctx: ConsolidationContext) -> dict[str, Tensor]:
        """KL(T) training of the student on per-member inputs x [S, N, ...] towards targets [S, N, C]."""
        ens, stream, cfg = cctx.ens, cctx.stream, cctx.cfg
        dev = next(iter(ens.params.values())).device
        S, N = targets.shape[:2]
        ens.load_params({k: v.to(dev) for k, v in init.items()})
        step = TrainStep(ens, DistillStrategy(self.cfg.temperature), stream, cfg.engine, cfg.optim)
        step.count_flops = True
        step.set_task(upto)
        ctx = RunContext(task_idx=upto, step=0, device=dev, E=S, gens=gens, stream=stream,
                         active=torch.ones(S, dtype=torch.bool), ens=ens)
        rows = torch.arange(S, device=dev)[:, None]
        guard = DivergenceGuard(floor=1.0)  # KL from a merged init can start near 0
        for _ in range(epochs):
            perm = epoch_perms(N, gens).to(dev)
            for i in range(0, N, cfg.train.batch_size):
                idx = perm[:, i : i + cfg.train.batch_size]
                batch = Batch(x=x[rows, idx], y=torch.zeros_like(idx), task_id=torch.zeros_like(idx),
                              attn_mask=None if am is None else am[rows, idx], mask=masks[rows, idx],
                              extra={"target": targets[rows, idx]})
                out = step(batch, None, ctx)
                guard.check(ctx, out.loss)
                self.flops += step.flops_per_step or 0
        self.diverged |= ctx.diverged
        return ens.snapshot()

    def consolidate(self, experts: Experts, upto: int, cctx: ConsolidationContext) -> dict[str, Tensor]:
        ens, stream, cfg = cctx.ens, cctx.stream, cctx.cfg
        dev = next(iter(ens.params.values())).device
        S = ens.E
        pool = TrainPool.from_stream(stream)
        gens = transfer_gens(cctx.seeds, upto)
        tr: TransferSet = build_transfer(stream, pool, upto, self.cfg.transfer, self.cfg.M, gens,
                                         cfg.benchmark.tokenizer, cfg.benchmark.max_len)
        self.stored_samples = len(tr) if tr.gidx is not None else 0
        table = stream.class_table().to(dev)
        if tr.gidx is not None:
            b = pool.gather(tr.gidx)
            x, am, task_of, shared = b.x, b.attn_mask, b.task_id, False
        else:
            x, am, task_of, shared = tr.proxy.x, tr.proxy.attn_mask, None, True
        init = self._init(experts, upto, cctx)
        logits = self._teacher_logits(experts, list(range(upto + 1)), x, am, shared, cctx)  # cached for all epochs
        routing = self._routing(stream, task_of is not None)
        mixed = torch.stack([mix_teachers(logits[:, e], table, routing, self.cfg.temperature,
                                          None if task_of is None else task_of[e]) for e in range(S)])  # [S, N, C]
        masks = self._masks(stream, table, upto, task_of, S, len(tr))
        targets = _student_targets(mixed, masks)
        if shared:
            x = x.expand(S, *x.shape)
            am = None if am is None else am.expand(S, *am.shape)
        return self._fit(init, x, am, masks, targets, gens, self.cfg.epochs, upto, cctx)


class ContinualDistillConsolidator(DistillConsolidator):
    """Teacher→student continual distillation (§5.6): one teacher arrives per task and is dropped after use.

    At task t the student is trained, for train.epochs passes, on task t's training data plus a replay buffer of
    earlier tasks (budget consolidate.M, split evenly). Every sample's task is known, so its target comes from its
    owner: the new teacher for task t's samples (knowledge distillation), a frozen copy of the student from task
    t−1 for buffered samples (LwF-style self-distillation). The student may be smaller than the teachers
    (method.params.student).
    """

    keeps_experts = False

    def __init__(self, cfg, params=None):
        super().__init__(cfg, params)
        self._prev: dict[str, Tensor] | None = None  # student after the previous task [S, ...]

    def consolidate(self, experts: Experts, upto: int, cctx: ConsolidationContext) -> dict[str, Tensor]:
        ens, stream, cfg = cctx.ens, cctx.stream, cctx.cfg
        dev = next(iter(ens.params.values())).device
        S = ens.E
        if self._prev is None:  # θ₀, or the student's own initialisation (consolidate.init is not used here)
            self._prev = cctx.ens.snapshot() if cctx.teacher is not None else \
                {k: v.clone() for k, v in experts.base.items()}
        pool = TrainPool.from_stream(stream)
        gens = transfer_gens(cctx.seeds, upto)
        lo, hi = int(pool.offsets[upto]), int(pool.offsets[upto + 1])
        cur = torch.arange(lo, hi)
        per = self.cfg.M // upto if upto else 0
        rows = []
        for g in gens:
            old = [torch.randperm(pool.task_size(t), generator=g)[:per] + int(pool.offsets[t]) for t in range(upto)]
            rows.append(torch.cat([cur, *old]))
        gidx = torch.stack(rows).to(pool.x.device)
        self.stored_samples = gidx.shape[1] - len(cur)  # an earlier task smaller than its share keeps all it has
        b = pool.gather(gidx)
        table = stream.class_table().to(dev)
        teacher = self._teacher_logits(experts, [upto], b.x, b.attn_mask, False, cctx)[0]  # [S, N, C]
        if upto == 0:
            logits, sources, src_of = teacher[None], table[:1], torch.zeros_like(b.task_id)
        else:
            old = _member_logits(ens, self._prev, b.x, b.attn_mask, False, cfg.train.eval_batch_size)
            logits = torch.stack([old, teacher])
            sources = torch.stack([table[:upto].any(0), table[upto]])
            src_of = (b.task_id == upto).long()  # 0: earlier task (old student), 1: current task (teacher)
        mixed = torch.stack([mix_teachers(logits[:, e], sources, "routed", self.cfg.temperature, src_of[e])
                             for e in range(S)])
        masks = self._masks(stream, table, upto, b.task_id, S, gidx.shape[1])
        targets = _student_targets(mixed, masks)
        self._prev = self._fit(self._prev, b.x, b.attn_mask, masks, targets, gens, cfg.train.epochs, upto, cctx)
        return self._prev
