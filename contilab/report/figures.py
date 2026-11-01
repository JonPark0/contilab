"""Static report figures (matplotlib, Agg). Small multiples per method: never more series than palette slots."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from contilab.metrics.accuracy import acc_matrix_of  # noqa: E402
from contilab.metrics.rawlog import RawLog  # noqa: E402

# Reference palette (dataviz skill): light surface, text inks, categorical slot 1, sequential blue, diverging pair.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES1 = "#2a78d6"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#9cc3ee", "#2a78d6", "#123a6b"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#d9d8d4", "#e34948"])
TASK_RAMP = LinearSegmentedColormap.from_list("tasks", ["#9cc3ee", "#123a6b"])

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 8,
                     "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2})


def _grid(n: int, w: float = 2.6, h: float = 2.2):
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols, h * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:n])


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _metrics(run: Path) -> dict:
    p = run / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _heat(ax, M, title, cmap=SEQ, vmin=0.0, vmax=1.0, annotate=True):
    M = np.asarray(M, dtype=float)
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.grid(False)
    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))
    ax.set_title(title)
    if annotate and M.size <= 100:
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color=SURFACE if (v - vmin) / max(vmax - vmin, 1e-9) > 0.55 else INK)


def acc_heatmap(runs: dict[str, Path], out: Path) -> Path | None:
    mats = {m: acc_matrix_of(RawLog(p)) for m, p in runs.items()}
    mats = {m: R for m, R in mats.items() if R.size}
    if not mats:
        return None
    fig, axes = _grid(len(mats))
    for ax, (m, R) in zip(axes, mats.items()):
        _heat(ax, R, m)
        ax.set_xlabel("evaluated task")
        ax.set_ylabel("after task")
    return _save(fig, out / "acc_heatmap.png")


def stream_curves(runs: dict[str, Path], out: Path) -> Path | None:
    data = {}
    for m, p in runs.items():
        rows = RawLog(p).records("stream_acc")
        if rows and not rows[0].get("not_applicable"):
            data[m] = rows
    if not data:
        return None
    fig, axes = _grid(len(data), 3.0, 2.2)
    for ax, (m, rows) in zip(axes, data.items()):
        T = max(r["task_now"] for r in rows) + 1
        ends = {}
        for r in rows:
            ends[r["task_now"]] = r["step"]
        for k in range(T):
            pts = [(r["step"], r["accs"][k]) for r in rows if len(r["accs"]) > k]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=TASK_RAMP(k / max(T - 1, 1)), linewidth=1.5, marker="o", markersize=3,
                        label=f"task {k}")
        for k in range(T - 1):  # task boundaries: a dip right after one is the stability gap
            ax.axvline(ends[k], color=INK2, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.set_ylim(0, 1)
        ax.set_title(m)
        ax.set_xlabel("step")
    axes[0].legend(fontsize=6, frameon=False)
    return _save(fig, out / "stream_curves.png")


def _matrix_fig(runs, out, metric, name, title_key="matrix", cmap=SEQ, vmin=0.0, vmax=1.0):
    mats = {m: _metrics(p).get(metric, {}).get("value") for m, p in runs.items()}
    mats = {m: v[title_key] for m, v in mats.items() if isinstance(v, dict) and v.get(title_key)}
    if not mats:
        return None
    fig, axes = _grid(len(mats))
    for ax, (m, M) in zip(axes, mats.items()):
        _heat(ax, M, m, cmap, vmin, vmax)
    return _save(fig, out / f"{name}.png")


def task_confusion(runs, out):
    return _matrix_fig(runs, out, "task_confusion", "task_confusion")


def cka_heatmap(runs, out):
    return _matrix_fig(runs, out, "cka_matrix", "cka_heatmap")


def grad_interference(runs, out):
    return _matrix_fig(runs, out, "grad_cosine_matrix", "grad_interference", cmap=DIV, vmin=-1.0, vmax=1.0)


_SNAP = re.compile(r"^t(\d+)\.(.+)$")


def weight_drift(runs, out):
    data = {}
    for m, p in runs.items():
        snaps = RawLog(p).tensors("param_snapshots")
        if not snaps:
            continue
        T = max(int(_SNAP.match(k).group(1)) for k in snaps)
        layers = sorted({_SNAP.match(k).group(2) for k in snaps if k.endswith("weight") and k.startswith("t0.")})
        data[m] = {l: float((snaps[f"t{T}.{l}"] - snaps[f"t0.{l}"]).norm() / (snaps[f"t0.{l}"].norm() + 1e-12)) for l in layers}
    if not data:
        return None
    fig, axes = _grid(len(data), 3.0, 2.2)
    for ax, (m, d) in zip(axes, data.items()):
        ax.barh(range(len(d)), list(d.values()), color=SERIES1, height=0.6)
        ax.set_yticks(range(len(d)), [l[-28:] for l in d], fontsize=5)
        ax.set_title(m)
        ax.set_xlabel("‖θ_T − θ_1‖ / ‖θ_1‖")
    return _save(fig, out / "weight_drift.png")


def _final_features(p: Path):
    feats = RawLog(p).tensors("features")
    keys = [k for k in feats if re.match(r"^t\d+\.task\d+$", k)]
    if not keys:
        return None
    T = max(int(k.split(".")[0][1:]) for k in keys)
    X, lab = [], []
    for k in sorted(k for k in keys if k.startswith(f"t{T}.")):
        X.append(feats[k].numpy())
        lab += [int(k.split("task")[1])] * len(feats[k])
    return np.concatenate(X), np.array(lab)


def _embed_fig(runs, out, name, embed):
    data = {m: f for m, p in runs.items() if (f := _final_features(p)) is not None}
    if not data:
        return None
    fig, axes = _grid(len(data))
    for ax, (m, (X, lab)) in zip(axes, data.items()):
        Z = embed(X)
        T = lab.max() + 1
        for k in range(T):
            sel = lab == k
            ax.scatter(Z[sel, 0], Z[sel, 1], s=8, color=TASK_RAMP(k / max(T - 1, 1)), edgecolors=SURFACE, linewidths=0.3,
                       label=f"task {k}")
        ax.set_title(m)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(fontsize=6, frameon=False)
    return _save(fig, out / f"{name}.png")


def feature_pca(runs, out):
    def pca(X):
        Xc = X - X.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt[:2].T

    return _embed_fig(runs, out, "feature_pca", pca)


def feature_tsne(runs, out):
    from sklearn.manifold import TSNE

    def tsne(X):
        idx = np.random.default_rng(0).permutation(len(X))[:600]
        Z = np.zeros((len(X), 2))
        Z[idx] = TSNE(2, init="pca", random_state=0, perplexity=min(30, max(5, len(idx) // 4))).fit_transform(X[idx])
        keep = np.zeros(len(X), dtype=bool)
        keep[idx] = True
        Z[~keep] = np.nan
        return Z

    return _embed_fig(runs, out, "feature_tsne", tsne)


def loss_barrier(runs, out):
    data = {m: RawLog(p).records("barrier_curve") for m, p in runs.items()}
    data = {m: r for m, r in data.items() if r}
    if not data:
        return None
    fig, axes = _grid(len(data), 3.0, 2.2)
    for ax, (m, rows) in zip(axes, data.items()):
        n = len(rows)
        for i, r in enumerate(rows):
            ax.plot(r["alphas"], r["losses"], color=TASK_RAMP(i / max(n - 1, 1)), linewidth=1.5, label=f"θ{r['k']}→θ{r['k'] + 1}")
        ax.set_title(m)
        ax.set_xlabel("α")
        ax.set_ylabel(f"val loss on task k")
    axes[0].legend(fontsize=6, frameon=False)
    return _save(fig, out / "loss_barrier.png")


def cd_diagram(ranks: dict[str, float], cd: float | None, out: Path) -> Path | None:
    if len(ranks) < 2 or cd is None or math.isnan(cd):
        return None
    items = sorted(ranks.items(), key=lambda x: x[1])
    k = len(items)
    fig, ax = plt.subplots(figsize=(7, 0.35 * k + 1.2))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-1, k + 0.5)
    ax.invert_xaxis()
    ax.grid(False)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("average rank (1 = best)")
    for i, (m, r) in enumerate(items):
        ax.plot([r, r], [k - i - 0.5, k - i], color=INK2, linewidth=1)
        ax.text(r, k - i + 0.05, f"{m} ({r:.2f})", ha="center", va="bottom", fontsize=7, color=INK)
    best = items[0][1]
    ax.plot([best, best + cd], [-0.6, -0.6], color=SERIES1, linewidth=3)
    ax.text(best + cd / 2, -0.95, f"CD = {cd:.2f}", ha="center", va="top", fontsize=7, color=INK2)
    return _save(fig, out / "cd_diagram.png")


def bench_bars(bench_json: Path, out: Path) -> Path | None:
    if not bench_json.exists():
        return None
    rows = json.loads(bench_json.read_text())
    labels = [f"{r['model']} E={r['E']} {'compile' if r['compile_used'] else 'eager'} {r.get('precision_used', 'fp32')}"
              for r in rows]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(rows) + 1))
    ax.barh(range(len(rows)), [r["samples_per_s"] for r in rows], color=SERIES1, height=0.6)
    ax.set_yticks(range(len(rows)), labels, fontsize=6)
    ax.set_xlabel("samples / s (all members)")
    for i, r in enumerate(rows):
        ax.text(r["samples_per_s"], i, f" {r['samples_per_s']:.0f}", va="center", fontsize=6, color=INK2)
    return _save(fig, out / "bench_bars.png")


PRESET_FIGURES = [acc_heatmap, stream_curves, task_confusion, cka_heatmap, weight_drift, grad_interference,
                  feature_pca, feature_tsne, loss_barrier]
