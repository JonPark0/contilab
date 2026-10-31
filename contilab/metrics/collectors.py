"""Training-time raw-data collection (§6.2). Collectors only record; metrics are computed later from RawLog."""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from contilab.core.config import EngineConfig, MetricsConfig, OptimConfig, TrainConfig
from contilab.core.scenario import sample_mask
from contilab.core.types import RunContext, Stream, TensorSplit
from contilab.engine.device import peak_memory_bytes
from contilab.engine.evaluate import EvalResult, evaluate_split
from contilab.metrics.rawlog import RawLog
from contilab.strategies.base import Strategy

RUN_END = 10**6  # "task" value of run-level records so resume truncation drops them

_CORE = {"init_acc", "acc_matrix", "pred_hist", "cost", "val_final"}
_EXTENDED = _CORE | {"stream_acc", "lca_curve", "features", "activations", "general_ppl"}
_FULL = _EXTENDED | {"param_snapshots", "barrier_curve", "grad_cosine", "recovery"}
RAW_KEYS_BY_TIER: dict[str, set[str]] = {"core": _CORE, "extended": _EXTENDED, "full": _FULL}


def _nbytes(ts) -> int:
    return sum(t.numel() * t.element_size() for t in ts if isinstance(t, torch.Tensor))


class Collectors:
    def __init__(self, tier: str, logs: list[RawLog], stream: Stream, strategy: Strategy, metrics_cfg: MetricsConfig,
                 train_cfg: TrainConfig, optim_cfg: OptimConfig | None = None, eval_stream: Stream | None = None):
        if tier not in RAW_KEYS_BY_TIER:
            raise ValueError(f"unknown metrics tier: {tier}")
        if strategy is None:
            from contilab.strategies.naive import Naive

            strategy = Naive()
        self.keys = RAW_KEYS_BY_TIER[tier]
        self.logs, self.stream, self.strategy = logs, stream, strategy
        self.consolidation = False  # set by run_consolidation: no training stream to evaluate continually
        self.cfg, self.train_cfg = metrics_cfg, train_cfg
        self.optim_cfg = optim_cfg or OptimConfig()
        self.eval_stream = eval_stream or stream
        self.step = None
        self._task_t0 = 0.0
        self._samples = 0
        self.extra_cost: dict[str, float] = {}
        self._subsets: dict[str, list[TensorSplit]] = {}
        self._task_steps = 0
        self._prev_params: dict[str, torch.Tensor] | None = None

    # --- helpers -------------------------------------------------------------
    def attach_step(self, step) -> None:
        self.step = step
        step.count_flops = "cost" in self.keys

    def _eval_cur(self, ctx: RunContext) -> int:
        return ctx.task_idx if self.eval_stream is self.stream else len(self.eval_stream.tasks) - 1

    def _eval_tasks(self, ctx: RunContext, params=None, keep_logits=False) -> list[EvalResult]:
        ens = ctx.ens
        params = ens.params if params is None else params
        cur = self._eval_cur(ctx)
        out = []
        for j, task in enumerate(self.eval_stream.tasks):
            self.strategy.eval_setup(ctx, j)
            mask = sample_mask(self.eval_stream, torch.tensor(j), cur)
            out.append(evaluate_split(ens, params, task.test, mask, self.train_cfg.eval_batch_size, keep_logits))
        self.strategy.eval_setup(ctx, None)
        return out

    def _subset(self, kind: str, n: int) -> list[TensorSplit]:
        """Fixed per-task test subsets (seeded), shared by every member and evaluation time."""
        if kind not in self._subsets:
            out = []
            for j, task in enumerate(self.eval_stream.tasks):
                g = torch.Generator().manual_seed(1000 + j)
                idx = torch.randperm(len(task.test), generator=g)[: min(n, len(task.test))].sort().values
                out.append(task.test.subset(idx.to(task.test.y.device)))
            self._subsets[kind] = out
        return self._subsets[kind]

    def _acc_on(self, ctx: RunContext, j: int, split: TensorSplit, params=None) -> torch.Tensor:
        """Accuracy [E] of every member on one split of eval task j."""
        self.strategy.eval_setup(ctx, j)
        mask = sample_mask(self.eval_stream, torch.tensor(j), self._eval_cur(ctx))
        r = evaluate_split(ctx.ens, ctx.ens.params if params is None else params, split, mask, self.train_cfg.eval_batch_size)
        return r.correct.float().mean(1)

    def _record_stream_acc(self, ctx: RunContext) -> None:
        sub = self._subset("stream", self.cfg.stream_subset)
        # a joint run trains on one merged task, so every evaluation task is "seen" from the start
        seen = range(len(sub)) if self.eval_stream is not self.stream else range(min(ctx.task_idx, len(sub) - 1) + 1)
        accs = torch.stack([self._acc_on(ctx, j, sub[j]) for j in seen], 1)
        self.strategy.eval_setup(ctx, None)
        t = ctx.task_idx
        self._per_member(lambda e, log: log.append("stream_acc", {"task": t, "step": ctx.step, "task_now": t, "accs": accs[e].tolist()}))

    def _record_lca(self, ctx: RunContext, batch: int) -> None:
        t = ctx.task_idx
        if t >= len(self.eval_stream.tasks) or self.eval_stream is not self.stream:
            return
        acc = self._acc_on(ctx, t, self._subset("stream", self.cfg.stream_subset)[t])
        self.strategy.eval_setup(ctx, None)
        self._per_member(lambda e, log: log.append("lca_curve", {"task": t, "batch": batch, "acc": acc[e].item()}))

    @torch.no_grad()
    def _record_features(self, ctx: RunContext) -> None:
        t, ens = ctx.task_idx, ctx.ens
        probe = self._subset("probe", self.cfg.probe_size)
        acts_in = []
        for j, sp in enumerate(probe):
            self.strategy.eval_setup(ctx, j)
            if "features" in self.keys:
                f = ens.apply(ens.params, sp.x, sp.attn_mask, mode="features", shared_input=True)

                def put(e, log, f=f, j=j, sp=sp):
                    log.put_tensor("features", f"t{t}.task{j}", f[e].float())
                    log.put_tensor("features", f"t{t}.task{j}.y", sp.y)

                self._per_member(put)
            acts_in.append(sp)
        if "activations" in self.keys:
            sums = None
            n = 0
            for j, sp in enumerate(acts_in):
                self.strategy.eval_setup(ctx, j)
                acts = ens.apply(ens.params, sp.x, sp.attn_mask, mode="acts", shared_input=True)
                cur = [a.abs().float().sum(1) for a in acts]  # [E, units] per layer
                sums = cur if sums is None else [s_ + c for s_, c in zip(sums, cur)]
                n += len(sp)
            self._per_member(lambda e, log: [log.put_tensor("activations", f"t{t}.layer{l}", s_[e] / n) for l, s_ in enumerate(sums)])
        self.strategy.eval_setup(ctx, None)

    @torch.no_grad()
    def _record_ppl(self, ctx: RunContext, task: int) -> None:
        lm = self.stream.lm_eval
        if lm is None:
            return
        ens = ctx.ens
        n = min(self.cfg.ppl_n, len(lm))
        nll = torch.zeros(ens.E, device=lm.x.device)
        count = 0
        for i in range(0, n, 8):
            x, am = lm.x[i : min(i + 8, n)], lm.attn_mask[i : min(i + 8, n)]
            logits = ens.apply(ens.params, x, am, mode="lm", shared_input=True).float()  # [E, B, L, V]
            tgt, valid = x[:, 1:], am[:, 1:].bool()
            lp = logits[:, :, :-1].log_softmax(-1).gather(-1, tgt.expand(ens.E, -1, -1)[..., None]).squeeze(-1)
            nll -= (lp * valid).sum((1, 2))
            count += int(valid.sum())
        ppl = torch.exp(nll / max(count, 1))
        self._per_member(lambda e, log: log.append("general_ppl", {"task": task, "ppl": ppl[e].item()}))

    def _member_ce(self, ctx: RunContext, params, j: int, split: TensorSplit) -> torch.Tensor:
        """Mean CE [E] on a split of task j (masked as at the current task)."""
        mask = sample_mask(self.eval_stream, torch.tensor(j), self._eval_cur(ctx)).to(split.x.device)
        z = ctx.ens.apply(params, split.x, split.attn_mask, shared_input=True).float().masked_fill(~mask, -1e9)
        return F.cross_entropy(z.flatten(0, 1), split.y.repeat(ctx.ens.E), reduction="none").view(ctx.ens.E, -1).mean(1)

    @torch.no_grad()
    def _record_barrier(self, ctx: RunContext) -> None:
        t = ctx.task_idx
        cur = {k: v.detach().clone() for k, v in ctx.ens.params.items()}
        if self._prev_params is not None and t >= 1:
            k = t - 1
            val = self.eval_stream.tasks[k].val
            alphas = [i / (self.cfg.barrier_points - 1) for i in range(self.cfg.barrier_points)]
            self.strategy.eval_setup(ctx, k)
            losses = torch.stack([self._member_ce(ctx, {n: (1 - a) * self._prev_params[n] + a * cur[n] for n in cur}, k, val)
                                  for a in alphas], 1)
            self.strategy.eval_setup(ctx, None)
            self._per_member(lambda e, log: log.append("barrier_curve", {"task": t, "k": k, "alphas": alphas, "losses": losses[e].tolist()}))
        self._prev_params = cur

    def _record_grad_cosine(self, ctx: RunContext) -> None:
        ens, t = ctx.ens, ctx.task_idx
        names = [n for n in ens.params if n not in ens.template.frozen_names()]
        flat = []
        for j in range(min(t, len(self.eval_stream.tasks) - 1) + 1):
            val = self.eval_stream.tasks[j].val
            sp = val.subset(torch.arange(min(256, len(val)), device=val.y.device))
            self.strategy.eval_setup(ctx, j)
            loss = self._member_ce(ctx, ens.params, j, sp)
            gs = torch.autograd.grad(loss.sum(), [ens.params[n] for n in names], allow_unused=True)
            flat.append(torch.cat([(g if g is not None else torch.zeros_like(ens.params[n])).reshape(ens.E, -1)
                                   for n, g in zip(names, gs)], 1))
        self.strategy.eval_setup(ctx, None)
        G = F.normalize(torch.stack(flat, 1), dim=-1)  # [E, k, P]
        cos = G @ G.transpose(1, 2)
        self._per_member(lambda e, log: log.append("grad_cosine", {"task": t, "matrix": cos[e].tolist()}))

    def _record_recovery(self, ctx: RunContext) -> None:
        from contilab.engine.step import TrainStep
        from contilab.strategies.naive import Naive

        ens, T = ctx.ens, len(self.eval_stream.tasks)
        if self.eval_stream is not self.stream:
            return
        snap = ens.snapshot()
        before = torch.stack([self._acc_on(ctx, j, self.eval_stream.tasks[j].test) for j in range(T)], 1)
        for j in range(T - 1):
            ens.load_params(snap)
            self.strategy.eval_setup(ctx, j)
            step = TrainStep(ens, Naive(), self.stream, EngineConfig(compile="off", precision="fp32"),
                             self.optim_cfg)
            step.set_task(j)
            g = torch.Generator().manual_seed(2000 + j)
            n = min(self.cfg.recovery_n, ctx.pool.task_size(j))
            gidx = (torch.randperm(ctx.pool.task_size(j), generator=g)[:n] + ctx.pool.offsets[j]).to(ctx.pool.x.device)
            batch = ctx.pool.gather(gidx.expand(ens.E, -1))
            batch.mask = sample_mask(self.stream, batch.task_id, T - 1)
            for _ in range(self.cfg.recovery_steps):
                step(batch, None, ctx)
            after = self._acc_on(ctx, j, self.eval_stream.tasks[j].test)

            def put(e, log, j=j, after=after):
                at_learn = {r["task"]: r["accs"] for r in log.records("acc_matrix")}.get(j, [math.nan] * T)[j]
                log.append("recovery", {"task": RUN_END, "k": j, "acc_before": before[e, j].item(),
                                        "acc_after": after[e].item(), "acc_at_learn": at_learn})

            self._per_member(put)
        ens.load_params(snap)
        self.strategy.eval_setup(ctx, None)

    def _per_member(self, fn) -> None:
        for e, log in enumerate(self.logs):
            fn(e, log)

    def resume(self, ctx: RunContext) -> None:
        """Re-arm per-run state after loading a checkpoint taken at the end of ctx.task_idx."""
        if "barrier_curve" in self.keys:
            self._prev_params = {k: v.detach().clone() for k, v in ctx.ens.params.items()}

    # --- hooks ---------------------------------------------------------------
    def on_run_start(self, ctx: RunContext) -> None:
        if "init_acc" in self.keys:
            res = self._eval_tasks(ctx)
            self._per_member(lambda e, log: log.append("init_acc", {"accs": [r.correct[e].float().mean().item() for r in res]}))
        if "general_ppl" in self.keys:
            self._record_ppl(ctx, -1)
        if "barrier_curve" in self.keys:
            self._prev_params = None
        if self.consolidation:
            for key in ("stream_acc", "lca_curve"):
                if key in self.keys:
                    self._per_member(lambda e, log, key=key: log.append(key, {"task": 0, "not_applicable": True}))

    def on_task_start(self, ctx: RunContext) -> None:
        self._task_steps = 0
        if "stream_acc" in self.keys and ctx.step == 0 and not self.consolidation:
            self._record_stream_acc(ctx)
        if "lca_curve" in self.keys and not self.consolidation:
            self._record_lca(ctx, 0)
        self._task_t0 = time.perf_counter()
        self._samples = 0

    def on_step(self, ctx: RunContext, out) -> None:
        self._samples += out.logits.shape[1]
        self._task_steps += 1
        if "lca_curve" in self.keys and self._task_steps <= self.cfg.beta:
            self._record_lca(ctx, self._task_steps)
        if "stream_acc" in self.keys and ctx.step % self.cfg.rho == 0:
            self._record_stream_acc(ctx)

    def on_task_end(self, ctx: RunContext) -> None:
        t = ctx.task_idx
        wall = time.perf_counter() - self._task_t0
        res = self._eval_tasks(ctx)
        if "acc_matrix" in self.keys:
            self._per_member(lambda e, log: log.append("acc_matrix", {"task": t, "accs": [r.correct[e].float().mean().item() for r in res]}))
        if "pred_hist" in self.keys:
            ys = torch.cat([task.test.y for task in self.eval_stream.tasks]).cpu()
            tids = torch.cat([torch.full((len(task.test),), j) for j, task in enumerate(self.eval_stream.tasks)])

            def put(e, log):
                log.put_tensor("pred_hist", f"t{t}.pred", torch.cat([r.pred[e] for r in res]))
                log.put_tensor("pred_hist", f"t{t}.conf", torch.cat([r.conf[e] for r in res]))
                log.put_tensor("pred_hist", f"t{t}.y", ys)
                log.put_tensor("pred_hist", f"t{t}.task", tids)

            self._per_member(put)
        if "cost" in self.keys:
            rec = self._cost_record(ctx, t, wall)
            self._per_member(lambda e, log: log.append("cost", rec))
        if "stream_acc" in self.keys and not self.consolidation:
            self._record_stream_acc(ctx)
        if "features" in self.keys or "activations" in self.keys:
            self._record_features(ctx)
        if "general_ppl" in self.keys:
            self._record_ppl(ctx, t)
        if "param_snapshots" in self.keys:
            self._per_member(lambda e, log: [log.put_tensor("param_snapshots", f"t{t}.{k}", v[e]) for k, v in ctx.ens.params.items()])
        if "barrier_curve" in self.keys:
            self._record_barrier(ctx)
        if "grad_cosine" in self.keys:
            self._record_grad_cosine(ctx)
        self._per_member(lambda e, log: log.flush())

    def on_run_end(self, ctx: RunContext) -> None:
        if "val_final" in self.keys:  # validation accuracy of the final model: the hyperparameter-selection score
            accs = torch.stack([self._acc_on(ctx, j, task.val) for j, task in enumerate(self.eval_stream.tasks)], 1)
            self.strategy.eval_setup(ctx, None)
            self._per_member(lambda e, log: log.append("val_final", {"task": RUN_END, "accs": accs[e].tolist()}))
        if "recovery" in self.keys:
            self._record_recovery(ctx)
        if "cost" in self.keys:
            rec = {"task": RUN_END, "flops_per_step": (self.step.flops_per_step or 0) if self.step else 0,
                   "steps": ctx.step, **self.extra_cost}
            self._per_member(lambda e, log: log.append("cost", rec))
        self._per_member(lambda e, log: log.flush())

    def _cost_record(self, ctx: RunContext, t: int, wall: float) -> dict:
        ens, E = ctx.ens, ctx.ens.E
        names = [n for n in self.strategy.trainable_names(list(ens.params), t) if n not in ens.template.frozen_names()]
        sd = self.strategy.state_dict()
        buf = getattr(self.strategy, "buffer", None)
        # a buffer of indices stands in for stored samples: cost = items × (sample + per-item extras such as logits)
        items, per_item = 0, 0
        if buf is not None:
            items = int(buf.size()[0])
            extras = sum(math.prod(v.shape[2:]) * v.element_size() for k, v in buf.tensors.items() if k != "gidx")
            per_item = ctx.pool.sample_bytes() + extras
        return {
            "task": t,
            "wall_s": wall,
            "samples": self._samples,
            "samples_per_s": self._samples * E / max(wall, 1e-9),
            "peak_mem": peak_memory_bytes(ens.device),
            "trainable_params": sum(ens.params[n][0].numel() for n in names),
            "buffer_items": items,
            "buffer_bytes": items * per_item,
            "state_bytes": _nbytes(sd["tensors"].values()) // E,
            "model_bytes": (_nbytes(ens.params.values()) + _nbytes(ens.buffers.values())) // E,
        }
