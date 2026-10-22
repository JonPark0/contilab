"""Accuracy-matrix metrics (§6.3): ACC, AIA, BWT, FWT, forgetting, LA, intransigence."""

from __future__ import annotations

import numpy as np

from contilab.metrics.collectors import RUN_END
from contilab.metrics.rawlog import RawLog
from contilab.metrics.registry import MetricContext, MetricUnavailable, metric


def acc_matrix_of(log: RawLog) -> np.ndarray:
    rows = sorted((r for r in log.records("acc_matrix") if r["task"] != RUN_END), key=lambda r: r["task"])
    return np.array([r["accs"] for r in rows], dtype=float)


def init_acc_of(log: RawLog) -> np.ndarray:
    return np.array(log.records("init_acc")[0]["accs"], dtype=float)


def _square(log: RawLog, min_tasks: int = 1) -> np.ndarray:
    R = acc_matrix_of(log)
    if R.shape[0] != R.shape[1]:
        raise MetricUnavailable("needs one accuracy-matrix row per task (not a joint run)")
    if R.shape[0] < min_tasks:
        raise MetricUnavailable(f"needs at least {min_tasks} tasks")
    return R


@metric("acc", ("acc_matrix",), "core")
def acc(log, mctx):
    return float(acc_matrix_of(log)[-1].mean())


@metric("aia", ("acc_matrix",), "core")
def aia(log, mctx):
    R = _square(log)
    return float(np.mean([R[i, : i + 1].mean() for i in range(len(R))]))


@metric("bwt", ("acc_matrix",), "core")
def bwt(log, mctx):
    R = _square(log, 2)
    T = len(R)
    return float(np.mean([R[T - 1, j] - R[j, j] for j in range(T - 1)]))


@metric("fwt", ("acc_matrix", "init_acc"), "core")
def fwt(log, mctx):
    if mctx.scenario == "class":
        raise MetricUnavailable("FWT is undefined in Class-IL: a task's classes are masked out until it is learned")
    R, b = _square(log, 2), init_acc_of(log)
    return float(np.mean([R[j - 1, j] - b[j] for j in range(1, len(R))]))


def _forgetting(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = len(R)
    best = np.array([R[j : T - 1, j].max() for j in range(T - 1)])
    return best - R[T - 1, : T - 1], best


@metric("forgetting", ("acc_matrix",), "core")
def forgetting(log, mctx):
    f, _ = _forgetting(_square(log, 2))
    return float(f.mean())


@metric("relative_forgetting", ("acc_matrix",), "core")
def relative_forgetting(log, mctx):
    f, best = _forgetting(_square(log, 2))
    ok = best > 0
    if not ok.any():
        raise MetricUnavailable("no task ever had positive accuracy")
    return float((f[ok] / best[ok]).mean())


@metric("learning_acc", ("acc_matrix",), "core")
def learning_acc(log, mctx):
    return float(np.diag(_square(log)).mean())


@metric("intransigence", ("acc_matrix",), "core")
def intransigence(log, mctx: MetricContext):
    if mctx.joint_acc is None:
        raise MetricUnavailable("no Joint run for this preset/seed")
    R = _square(log)
    return float(np.mean([mctx.joint_acc[j] - R[j, j] for j in range(len(R))]))
