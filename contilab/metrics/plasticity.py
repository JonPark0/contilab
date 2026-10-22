"""Plasticity-loss diagnostics: dormant neurons (Sokar 2023), effective rank (Roy & Vetterli 2007),
srank (Kumar 2021), weight norms. Report them next to LA / LCA (optimisation-based plasticity)."""

from __future__ import annotations

import re

import numpy as np

from contilab.metrics.registry import MetricUnavailable, metric
from contilab.metrics.representation import _features

_ACT = re.compile(r"^t(\d+)\.layer(\d+)$")
_SNAP = re.compile(r"^t(\d+)\.(.+)$")


def dormant_fraction(mean_abs: np.ndarray, tau: float) -> float:
    m = mean_abs.mean()
    if m <= 0:
        return 1.0
    return float(((mean_abs / m) <= tau).mean())


def effective_rank(F: np.ndarray) -> float:
    s = np.linalg.svd(F, compute_uv=False)
    s = s[s > 0]
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def srank(F: np.ndarray, delta: float) -> int:
    s = np.linalg.svd(F, compute_uv=False)
    c = np.cumsum(s) / s.sum()
    return int(np.searchsorted(c, 1 - delta - 1e-12) + 1)


def _dormant_curve(log, tau) -> list[float]:
    acts: dict[int, dict[int, np.ndarray]] = {}
    for k, v in log.tensors("activations").items():
        m = _ACT.match(k)
        acts.setdefault(int(m.group(1)), {})[int(m.group(2))] = v.float().numpy()
    if not acts:
        raise MetricUnavailable("no activations recorded")
    out = []
    for t in sorted(acts):
        layers = acts[t]
        dormant = sum(dormant_fraction(layers[l], tau) * len(layers[l]) for l in layers)
        out.append(dormant / sum(len(a) for a in layers.values()))
    return out


@metric("dormant_ratio", ("activations",), "extended")
def dormant_ratio(log, mctx):
    return _dormant_curve(log, mctx.cfg.dormant_tau)[-1]


@metric("dormant_ratio_curve", ("activations",), "extended")
def dormant_ratio_curve(log, mctx):
    return {"per_task": _dormant_curve(log, mctx.cfg.dormant_tau)}


def _all_features_at(log) -> list[np.ndarray]:
    feats, _, T = _features(log)
    nt = max(j for _, j in feats) + 1
    return [np.concatenate([feats[(t, j)] for j in range(nt)]) for t in range(T)]


@metric("effective_rank", ("features",), "extended")
def effective_rank_metric(log, mctx):
    return effective_rank(_all_features_at(log)[-1])


@metric("srank", ("features",), "extended")
def srank_metric(log, mctx):
    return srank(_all_features_at(log)[-1], mctx.cfg.srank_delta)


@metric("rank_curve", ("features",), "extended")
def rank_curve(log, mctx):
    at = _all_features_at(log)
    return {"effective_rank": [effective_rank(f) for f in at], "srank": [srank(f, mctx.cfg.srank_delta) for f in at]}


@metric("weight_norms", ("param_snapshots",), "full")
def weight_norms(log, mctx):
    out: dict[str, list[float]] = {}
    snaps = log.tensors("param_snapshots")
    for t in sorted({int(_SNAP.match(k).group(1)) for k in snaps}):
        for k, v in snaps.items():
            m = _SNAP.match(k)
            if int(m.group(1)) == t and m.group(2).endswith("weight"):
                out.setdefault(m.group(2), []).append(float(v.float().norm()))
    return out
