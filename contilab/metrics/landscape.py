"""Loss landscape / interference: linear-interpolation loss barrier (Mirzadeh 2020), gradient cosine."""

from __future__ import annotations

import numpy as np

from contilab.metrics.registry import MetricUnavailable, metric


@metric("loss_barrier", ("barrier_curve",), "full")
def loss_barrier(log, mctx):
    vals = []
    for r in log.records("barrier_curve"):
        a, L = np.array(r["alphas"]), np.array(r["losses"])
        vals.append(float((L - ((1 - a) * L[0] + a * L[-1])).max()))
    if not vals:
        raise MetricUnavailable("needs at least 2 tasks")
    return float(np.mean(vals))


def _last_matrix(log) -> np.ndarray:
    rows = log.records("grad_cosine")
    if not rows:
        raise MetricUnavailable("no gradient records")
    return np.array(max(rows, key=lambda r: r["task"])["matrix"])


@metric("grad_interference", ("grad_cosine",), "full")
def grad_interference(log, mctx):
    M = _last_matrix(log)
    if len(M) < 2:
        raise MetricUnavailable("needs at least 2 tasks")
    return float(M[~np.eye(len(M), dtype=bool)].mean())


@metric("grad_cosine_matrix", ("grad_cosine",), "full")
def grad_cosine_matrix(log, mctx):
    return {"matrix": _last_matrix(log).tolist()}
