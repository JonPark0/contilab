"""Few-shot recovery (true vs. spurious forgetting; Zheng et al., ICLR 2025)."""

from __future__ import annotations

import numpy as np

from contilab.metrics.registry import MetricUnavailable, metric

MIN_DROP = 0.01


def _ratios(log) -> dict[int, float]:
    out = {}
    for r in log.records("recovery"):
        drop = r["acc_at_learn"] - r["acc_before"]
        if drop > MIN_DROP:
            out[r["k"]] = (r["acc_after"] - r["acc_before"]) / drop
    return out


@metric("recovery", ("recovery",), "full")
def recovery(log, mctx):
    r = _ratios(log)
    if not r:
        raise MetricUnavailable("no task forgot more than 1 point")
    return float(np.mean(list(r.values())))


@metric("recovery_per_task", ("recovery",), "full")
def recovery_per_task(log, mctx):
    return {"per_task": {str(k): v for k, v in sorted(_ratios(log).items())}}
