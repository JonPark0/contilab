"""Continual-evaluation metrics (De Lange et al. 2023; Caccia et al. 2022; Chaudhry et al. 2019)."""

from __future__ import annotations

import numpy as np

from contilab.metrics.registry import MetricUnavailable, metric


def _stream(log):
    rows = sorted(log.records("stream_acc"), key=lambda r: (r["step"], r["task_now"]))
    if not rows or rows[0].get("not_applicable"):
        raise MetricUnavailable("no continual-evaluation stream for this run")
    return rows


def _task_end_steps(rows) -> dict[int, int]:
    ends: dict[int, int] = {}
    for r in rows:
        ends[r["task_now"]] = r["step"]
    return ends


def _num_tasks(rows) -> int:
    return max(r["task_now"] for r in rows) + 1


def _after(rows, k):
    """(step, acc of task k) for every record from the end of task k onwards."""
    end = _task_end_steps(rows)[k]
    return [(r["step"], r["accs"][k]) for r in rows if r["step"] >= end and len(r["accs"]) > k]


@metric("min_acc", ("stream_acc",), "extended")
def min_acc(log, mctx):
    rows = _stream(log)
    T = _num_tasks(rows)
    if T < 2:
        raise MetricUnavailable("needs at least 2 tasks")
    vals = []
    for k in range(T - 1):
        later = [r["accs"][k] for r in rows if r["task_now"] > k]
        if later:
            vals.append(min(later))
    return float(np.mean(vals))


@metric("wc_acc", ("stream_acc",), "extended")
def wc_acc(log, mctx):
    rows = _stream(log)
    T = _num_tasks(rows)
    return float(rows[-1]["accs"][T - 1] / T + (1 - 1 / T) * min_acc(log, mctx))


@metric("aaa", ("stream_acc",), "extended")
def aaa(log, mctx):
    return float(np.mean([np.mean(r["accs"]) for r in _stream(log)]))


@metric("stability_gap", ("stream_acc",), "extended")
def stability_gap(log, mctx):
    rows = _stream(log)
    T = _num_tasks(rows)
    if T < 2:
        raise MetricUnavailable("needs at least 2 tasks")
    gaps = []
    for k in range(T - 1):
        at_end = [r["accs"][k] for r in rows if r["task_now"] == k][-1]
        during_next = [r["accs"][k] for r in rows if r["task_now"] == k + 1]
        if during_next:
            gaps.append(at_end - min(during_next))
    return float(np.mean(gaps))


@metric("windowed_forgetting", ("stream_acc",), "extended")
def windowed_forgetting(log, mctx):
    rows = _stream(log)
    T = _num_tasks(rows)
    if T < 2:
        raise MetricUnavailable("needs at least 2 tasks")
    ends = _task_end_steps(rows)
    starts = [0] + [ends[t] for t in range(T - 1)]
    window = mctx.cfg.window_frac * np.mean([ends[t] - starts[t] for t in range(T)])
    out = []
    for k in range(T - 1):
        pts = _after(rows, k)
        best = 0.0
        for i, (s1, a1) in enumerate(pts):
            for s2, a2 in pts[i + 1 :]:
                if s2 - s1 <= window:
                    best = max(best, a1 - a2)
        out.append(best)
    return float(np.mean(out))


@metric("lca", ("lca_curve",), "extended")
def lca(log, mctx):
    rows = log.records("lca_curve")
    if not rows or rows[0].get("not_applicable"):
        raise MetricUnavailable("no learning-curve records for this run")
    by_task: dict[int, dict[int, float]] = {}
    for r in rows:
        by_task.setdefault(r["task"], {})[r["batch"]] = r["acc"]
    common = sorted(set.intersection(*(set(d) for d in by_task.values())))
    common = [b for b in common if b <= mctx.cfg.beta]
    return float(np.mean([np.mean([d[b] for d in by_task.values()]) for b in common]))
