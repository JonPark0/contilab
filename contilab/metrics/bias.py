"""Prediction bias and calibration: task-recency bias (Wu 2019; Masana 2022), task confusion, ECE."""

from __future__ import annotations

import re

import numpy as np
import torch

from contilab.metrics.registry import MetricUnavailable, metric

_T = re.compile(r"^t(\d+)\.pred$")


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def _final(log):
    ph = log.tensors("pred_hist")
    ts = [int(_T.match(k).group(1)) for k in ph if _T.match(k)]
    if not ts:
        raise MetricUnavailable("no predictions recorded")
    t = max(ts)
    return {k: ph[f"t{t}.{k}"].numpy() for k in ("pred", "y", "conf", "task")}


def _class_to_task(table: torch.Tensor) -> np.ndarray:
    """Each class → the most recent task that owns it."""
    T, C = table.shape
    owner = np.full(C, -1)
    for t in range(T):
        owner[table[t].numpy()] = t
    return owner


@metric("recency_bias", ("pred_hist",), "core")
def recency_bias(log, mctx):
    if mctx.scenario != "class":
        raise MetricUnavailable("task-recency bias is defined for class-IL only")
    d = _final(log)
    owner = _class_to_task(mctx.class_table)
    last = d["task"].max()
    old = d["task"] < last
    if not old.any():
        raise MetricUnavailable("needs at least 2 tasks")
    return float((owner[d["pred"][old]] == last).mean())


@metric("task_confusion", ("pred_hist",), "core")
def task_confusion(log, mctx):
    d = _final(log)
    owner = _class_to_task(mctx.class_table)
    T = int(d["task"].max()) + 1
    M = np.zeros((T, T))
    for true_t, pred_t in zip(d["task"], owner[d["pred"]]):
        M[true_t, pred_t] += 1
    M = M / M.sum(1, keepdims=True).clip(min=1)
    return {"matrix": M.round(6).tolist()}


@metric("ece", ("pred_hist",), "core")
def ece(log, mctx):
    d = _final(log)
    return expected_calibration_error(d["conf"], (d["pred"] == d["y"]).astype(float), mctx.cfg.ece_bins)


@metric("ece_per_task", ("pred_hist",), "core")
def ece_per_task(log, mctx):
    d = _final(log)
    correct = (d["pred"] == d["y"]).astype(float)
    return {"per_task": [expected_calibration_error(d["conf"][d["task"] == t], correct[d["task"] == t], mctx.cfg.ece_bins)
                         for t in range(int(d["task"].max()) + 1)]}
