"""Markdown tables over results.csv: main table, per-axis tables, cross-benchmark ranking (§7.6)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from contilab.metrics.stats import friedman_nemenyi, paired_test, summarize

METHOD_ORDER = ["naive", "joint", "ewc", "lwf", "er", "derpp", "agem", "lora_per_task",
                "consolidate:ta", "consolidate:ties", "consolidate:distill"]
MAIN_METRICS = ["acc", "aia", "bwt", "fwt", "forgetting", "learning_acc", "min_acc", "wc_acc", "wall_time", "peak_mem"]
AXES = {
    "stability": ["bwt", "forgetting", "relative_forgetting", "min_acc", "wc_acc", "stability_gap", "windowed_forgetting"],
    "plasticity": ["fwt", "learning_acc", "intransigence", "lca", "aaa"],
    "representation": ["repr_forgetting", "head_forgetting", "cka_final", "feature_drift"],
    "plasticity_loss": ["dormant_ratio", "effective_rank", "srank", "learning_acc", "lca"],
    "bias_calibration": ["recency_bias", "ece"],
    "recovery": ["recovery", "loss_barrier", "grad_interference", "general_ppl_delta"],
    "efficiency": ["wall_time", "samples_per_s", "peak_mem", "trainable_params", "buffer_bytes", "model_bytes", "total_flops"],
}
LOWER_IS_BETTER = {"forgetting", "relative_forgetting", "stability_gap", "windowed_forgetting", "intransigence",
                   "repr_forgetting", "head_forgetting", "feature_drift", "dormant_ratio", "recency_bias", "ece",
                   "loss_barrier", "general_ppl_delta", "wall_time", "peak_mem", "trainable_params", "buffer_bytes",
                   "model_bytes", "total_flops"}


def with_labels(df: pd.DataFrame) -> pd.DataFrame:
    """`label` = method, plus its variant (extra matrix axes / sweep values) when there is one; ✓ marks the
    variant chosen on validation accuracy (§5.3)."""
    variant = df["variant"].fillna("").astype(str) if "variant" in df.columns else pd.Series("", index=df.index)
    sel = _selected(df)
    return df.assign(label=[(m if not v else f"{m} [{v}]") + (" ✓" if s is True else "")
                            for m, v, s in zip(df["method"], variant, sel)])


def _selected(df: pd.DataFrame) -> list:
    if "selected" not in df.columns:
        return [None] * len(df)
    return [True if str(s) == "True" else False if str(s) == "False" else None for s in df["selected"]]


def only_selected(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the sweep variants that lost the validation selection (cells without a sweep are kept)."""
    return df[[s is not False for s in _selected(df)]]


def methods_in(df: pd.DataFrame) -> list[str]:
    """Labels in canonical method order (variants of a method stay together)."""
    col = "label" if "label" in df.columns else "method"
    present = list(dict.fromkeys(df[col]))
    rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    base = dict(zip(df[col], df["method"]))
    return sorted(present, key=lambda l: (rank.get(base[l], len(METHOD_ORDER)), present.index(l)))


def _fmt(metric: str, v: float) -> str:
    if metric == "peak_mem" or metric.endswith("_bytes"):
        return f"{v / 2**20:.1f}MB"
    if metric == "wall_time":
        return f"{v:.1f}s"
    if metric in ("total_flops", "trainable_params", "samples_per_s"):
        return f"{v:.3g}"
    return f"{v:.3f}"


def _cell(df: pd.DataFrame, label: str, metric: str, base: pd.DataFrame | None) -> str:
    if metric not in df.columns:
        return "—"
    sub = df[(df.label == label) & df[metric].notna()]
    if sub.empty:
        return "—"
    s = summarize(sub[metric].tolist())
    txt = _fmt(metric, s["mean"]) + (f" ± {_fmt(metric, s['std'])}" if s["n"] > 1 else "")
    if base is not None and label != base.label.iloc[0] and metric in base.columns:
        pairs = sub.merge(base[["seed", metric]], on="seed", suffixes=("", "_b")).dropna(subset=[metric, f"{metric}_b"])
        if len(pairs) >= 2 and paired_test(pairs[metric], pairs[f"{metric}_b"])["p"] < 0.05:
            txt += " *"
    return txt


def _row_name(d: pd.DataFrame, label: str) -> str:
    st = d[d.label == label].status
    n_div = int((st == "diverged").sum())
    return f"{label} (diverged {n_div}/{len(st)})" if n_div else label


def _table(df: pd.DataFrame, preset: str, metrics: list[str], baseline: str) -> str:
    d = with_labels(df[(df.preset == preset) & df.status.isin(["done", "diverged"])])
    base = d[d.label == baseline] if baseline in set(d.label) else None
    head = "| method | " + " | ".join(metrics) + " |\n|---|" + "---|" * len(metrics) + "\n"
    rows = [f"| {_row_name(d, m)} | " + " | ".join(_cell(d, m, k, base) for k in metrics) + " |" for m in methods_in(d)]
    return head + "\n".join(rows) + "\n"


def main_table(df: pd.DataFrame, preset: str, baseline: str = "naive") -> str:
    return _table(df, preset, MAIN_METRICS, baseline)


def axis_tables(df: pd.DataFrame, preset: str, baseline: str = "naive") -> dict[str, str]:
    d = df[df.preset == preset]
    out = {}
    for axis, ms in AXES.items():
        present = [m for m in ms if m in d.columns and d[m].notna().any()]
        if present:
            out[axis] = _table(df, preset, present, baseline)
    return out


def rank_table(df: pd.DataFrame, metric: str = "acc") -> tuple[str, dict]:
    d = with_labels(only_selected(df[df.status.isin(["done", "diverged"])]))
    piv = d.groupby(["preset", "label"])[metric].mean().unstack("label")
    piv = piv.dropna(axis=1)
    methods = [m for m in methods_in(d) if m in piv.columns]
    if piv.empty or not methods:
        return "", {"avg_ranks": {}, "friedman_p": None, "cd": None}
    st = friedman_nemenyi(piv[methods].to_numpy(), higher_is_better=metric not in LOWER_IS_BETTER)
    ranks = dict(zip(methods, st["avg_ranks"].tolist()))
    lines = ["| method | avg rank |", "|---|---|"] + [f"| {m} | {r:.2f} |" for m, r in sorted(ranks.items(), key=lambda x: x[1])]
    p = st["friedman_p"]
    cd = st["cd"]
    lines.append("")
    lines.append(f"Friedman p = {p:.3g}" if p is not None else "Friedman p = n/a (needs ≥ 2 benchmarks and ≥ 3 methods)")
    if cd is not None and not math.isnan(cd):
        lines.append(f"Nemenyi CD (α=0.05) = {cd:.2f} over {len(piv)} benchmark(s)")
    return "\n".join(lines) + "\n", {"avg_ranks": ranks, "friedman_p": p, "cd": cd}
