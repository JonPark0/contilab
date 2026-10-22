"""Representation-level metrics: linear probes (Davari 2022), CKA (Kornblith 2019), feature drift."""

from __future__ import annotations

import re

import numpy as np

from contilab.metrics.accuracy import acc_matrix_of
from contilab.metrics.registry import MetricUnavailable, metric

_KEY = re.compile(r"^t(\d+)\.task(\d+)$")


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(hsic / den) if den > 0 else 0.0


def probe_accuracy(F: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y)) < 2:
        raise MetricUnavailable("probe set has a single class")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]
    sc = StandardScaler().fit(F[tr])
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(F[tr]), y[tr])
    return float(clf.score(sc.transform(F[te]), y[te]))


def _features(log) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray], int]:
    feats, labels = {}, {}
    for k, v in log.tensors("features").items():
        if k.endswith(".y"):
            m = _KEY.match(k[:-2])
            labels[(int(m.group(1)), int(m.group(2)))] = v.numpy()
        elif (m := _KEY.match(k)) is not None:
            feats[(int(m.group(1)), int(m.group(2)))] = v.float().numpy()
    if not feats:
        raise MetricUnavailable("no features recorded")
    T = max(t for t, _ in feats) + 1
    return feats, labels, T


def _probe_matrix(log) -> np.ndarray:
    feats, labels, T = _features(log)
    nt = max(j for _, j in feats) + 1
    return np.array([[probe_accuracy(feats[(t, j)], labels[(t, j)]) for j in range(nt)] for t in range(T)])


@metric("probe_acc", ("features",), "extended")
def probe_acc(log, mctx):
    return {"matrix": _probe_matrix(log).tolist()}


def _repr_drop(P: np.ndarray) -> np.ndarray:
    T = P.shape[0]
    if T < 2:
        raise MetricUnavailable("needs at least 2 tasks")
    return np.array([P[j, j] - P[T - 1, j] for j in range(T - 1)])


@metric("repr_forgetting", ("features",), "extended")
def repr_forgetting(log, mctx):
    return float(_repr_drop(_probe_matrix(log)).mean())


@metric("head_forgetting", ("features", "acc_matrix"), "extended")
def head_forgetting(log, mctx):
    P, R = _probe_matrix(log), acc_matrix_of(log)
    T = P.shape[0]
    total = np.array([R[j, j] - R[T - 1, j] for j in range(T - 1)])
    return float((total - _repr_drop(P)).mean())


@metric("cka_final", ("features",), "extended")
def cka_final(log, mctx):
    feats, _, T = _features(log)
    return float(np.mean([linear_cka(feats[(j, j)], feats[(T - 1, j)]) for j in range(T)]))


@metric("cka_matrix", ("features",), "extended")
def cka_matrix(log, mctx):
    feats, _, T = _features(log)
    nt = max(j for _, j in feats) + 1
    at = [np.concatenate([feats[(t, j)] for j in range(nt)]) for t in range(T)]
    return {"matrix": [[linear_cka(at[a], at[b]) for b in range(T)] for a in range(T)]}


@metric("feature_drift", ("features",), "extended")
def feature_drift(log, mctx):
    feats, _, T = _features(log)
    if T < 2:
        raise MetricUnavailable("needs at least 2 tasks")
    out = []
    for j in range(T - 1):
        a, b = feats[(j, j)], feats[(T - 1, j)]
        out.append(np.mean(np.linalg.norm(b - a, axis=1) / (np.linalg.norm(a, axis=1) + 1e-12)))
    return float(np.mean(out))
