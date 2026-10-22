"""Cost metrics (§6.3 효율) from the per-task and run-end cost records."""

from __future__ import annotations

from contilab.metrics.collectors import RUN_END
from contilab.metrics.registry import MetricUnavailable, metric


def _tasks(log):
    return [r for r in log.records("cost") if r["task"] != RUN_END]


def _end(log):
    ends = [r for r in log.records("cost") if r["task"] == RUN_END]
    if not ends:
        raise MetricUnavailable("run did not finish")
    return ends[-1]


@metric("wall_time", ("cost",), "core")
def wall_time(log, mctx):
    return float(sum(r["wall_s"] for r in _tasks(log)) + _end(log).get("expert_wall_s", 0.0))


@metric("samples_per_s", ("cost",), "core")
def samples_per_s(log, mctx):
    rs = _tasks(log)
    w = sum(r["wall_s"] for r in rs)
    return float(sum(r["samples_per_s"] * r["wall_s"] for r in rs) / w) if w > 0 else 0.0


@metric("peak_mem", ("cost",), "core")
def peak_mem(log, mctx):
    return max(r["peak_mem"] for r in _tasks(log))


@metric("trainable_params", ("cost",), "core")
def trainable_params(log, mctx):
    return max(r["trainable_params"] for r in _tasks(log))


@metric("buffer_bytes", ("cost",), "core")
def buffer_bytes(log, mctx):
    return _tasks(log)[-1]["buffer_bytes"]


@metric("model_bytes", ("cost",), "core")
def model_bytes(log, mctx):
    return _tasks(log)[-1]["model_bytes"] + _end(log).get("stored_experts_bytes", 0)


@metric("total_flops", ("cost",), "core")
def total_flops(log, mctx):
    e = _end(log)
    return int(e["flops_per_step"] * e["steps"] + e.get("expert_flops", 0))
