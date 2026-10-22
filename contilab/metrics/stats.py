"""Statistics over seeds and benchmarks (§6.4): bootstrap CI, paired tests, Friedman + Nemenyi CD (Demšar 2006)."""

from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np
from scipy import stats

LOW_POWER_N = 5
WARNING = "통계 검정력 경고: 시드 수 {n}개 (< 5)"


def summarize(xs: Sequence[float], n_boot: int = 10_000, seed: int = 0) -> dict:
    a = np.asarray([x for x in xs if x is not None and not np.isnan(x)], dtype=float)
    n = len(a)
    if n == 0:
        return {"mean": None, "std": None, "ci_low": None, "ci_high": None, "n": 0}
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    if n == 1:
        return {"mean": mean, "std": std, "ci_low": mean, "ci_high": mean, "n": 1}
    rng = np.random.default_rng(seed)
    boots = a[rng.integers(0, n, size=(n_boot, n))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": mean, "std": std, "ci_low": float(lo), "ci_high": float(hi), "n": n}


def _sign_flip_p(d: np.ndarray) -> float:
    obs = abs(d.sum())
    if obs == 0:
        return 1.0
    total = 0
    hits = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        total += 1
        hits += abs((d * signs).sum()) >= obs - 1e-12
    return hits / total


def paired_test(a: Sequence[float], b: Sequence[float]) -> dict:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) >= 6:
        if np.allclose(d, 0):
            return {"method": "wilcoxon", "statistic": 0.0, "p": 1.0}
        r = stats.wilcoxon(d)
        return {"method": "wilcoxon", "statistic": float(r.statistic), "p": float(r.pvalue)}
    return {"method": "permutation", "statistic": float(d.sum()), "p": _sign_flip_p(d)}


def nemenyi_q(k: int, alpha: float = 0.05) -> float:
    return float(stats.studentized_range.ppf(1 - alpha, k, 1e6) / math.sqrt(2))


def friedman_nemenyi(scores: np.ndarray, higher_is_better: bool = True, alpha: float = 0.05) -> dict:
    """scores: [N benchmarks, k methods]. Rank 1 = best."""
    s = np.asarray(scores, dtype=float)
    N, k = s.shape
    ranks = np.vstack([stats.rankdata(-row if higher_is_better else row) for row in s])
    avg = ranks.mean(0)
    p = None
    if N >= 2 and k >= 3:
        p = float(stats.friedmanchisquare(*s.T).pvalue)
    cd = nemenyi_q(k, alpha) * math.sqrt(k * (k + 1) / (6 * N)) if k >= 2 else None
    return {"avg_ranks": avg, "friedman_p": p, "cd": cd}


def low_power_warning(n: int) -> str | None:
    return WARNING.format(n=n) if n < LOW_POWER_N else None
