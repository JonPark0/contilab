"""Training-time raw-data collection (§6.2). Collectors only record; metrics are computed later from RawLog."""

from __future__ import annotations

import time

import torch

from contilab.core.config import MetricsConfig, OptimConfig, TrainConfig
from contilab.core.scenario import sample_mask
from contilab.core.types import RunContext, Stream
from contilab.engine.device import peak_memory_bytes
from contilab.engine.evaluate import EvalResult, evaluate_split
from contilab.metrics.rawlog import RawLog
from contilab.strategies.base import Strategy

RUN_END = 10**6  # "task" value of run-level records so resume truncation drops them

_CORE = {"init_acc", "acc_matrix", "pred_hist", "cost"}
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
        self.keys = RAW_KEYS_BY_TIER[tier]
        self.logs, self.stream, self.strategy = logs, stream, strategy
        self.cfg, self.train_cfg = metrics_cfg, train_cfg
        self.optim_cfg = optim_cfg or OptimConfig()
        self.eval_stream = eval_stream or stream
        self.step = None
        self._task_t0 = 0.0
        self._samples = 0
        self.extra_cost: dict[str, float] = {}

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

    def _per_member(self, fn) -> None:
        for e, log in enumerate(self.logs):
            fn(e, log)

    # --- hooks ---------------------------------------------------------------
    def on_run_start(self, ctx: RunContext) -> None:
        if "init_acc" in self.keys:
            res = self._eval_tasks(ctx)
            self._per_member(lambda e, log: log.append("init_acc", {"accs": [r.correct[e].float().mean().item() for r in res]}))

    def on_task_start(self, ctx: RunContext) -> None:
        self._task_t0 = time.perf_counter()
        self._samples = 0

    def on_step(self, ctx: RunContext, out) -> None:
        self._samples += out.logits.shape[1]

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
        self._per_member(lambda e, log: log.flush())

    def on_run_end(self, ctx: RunContext) -> None:
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
        buf_bytes = _nbytes(buf.tensors.values()) if buf is not None else 0
        return {
            "task": t,
            "wall_s": wall,
            "samples": self._samples,
            "samples_per_s": self._samples * E / max(wall, 1e-9),
            "peak_mem": peak_memory_bytes(ens.device),
            "trainable_params": sum(ens.params[n][0].numel() for n in names),
            "buffer_bytes": buf_bytes // E,
            "state_bytes": _nbytes(sd["tensors"].values()) // E,
            "model_bytes": (_nbytes(ens.params.values()) + _nbytes(ens.buffers.values())) // E,
        }
