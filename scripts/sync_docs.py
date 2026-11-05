"""Writes the docs files that are derived from other files, or checks that they are current (--check, run in CI):

  docs/assets/results/split-mnist-class-acc.svg   final accuracy of each method on Split MNIST (Class-IL),
                                                  from docs/assets/results/data/cpu_main.csv
  docs/changelog.md                               CHANGELOG.md, as a page of the docs site

    python3 scripts/sync_docs.py            # rewrite
    python3 scripts/sync_docs.py --check    # exit 1 if a file is stale

Standard library only, so that CI can run it without installing the package.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from statistics import mean

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS = os.path.join(ROOT, "docs", "assets", "results", "data", "cpu_main.csv")
CHART = os.path.join(ROOT, "docs", "assets", "results", "split-mnist-class-acc.svg")
CHANGELOG = os.path.join(ROOT, "CHANGELOG.md")
CHANGELOG_PAGE = os.path.join(ROOT, "docs", "changelog.md")
CHANGELOG_FRONT = """---
title: "Changelog: what changed in each contilab release"
description: "Release notes of contilab: new continual-learning methods, metrics, benchmarks, results, speed-ups and fixes in each version."
nav_order: 16
nav_title: "Changelog"
---
<!-- Generated from CHANGELOG.md by scripts/sync_docs.py; edit CHANGELOG.md instead. -->

"""

NAMES = {"naive": "Naive", "joint": "Joint (upper bound)", "ewc": "EWC", "lwf": "LwF", "er": "Experience Replay",
         "er_ace": "ER-ACE", "derpp": "DER++", "agem": "A-GEM", "gem": "GEM", "lora_per_task": "LoRA-per-task",
         "olora": "O-LoRA", "consolidate:ta": "Task Arithmetic", "consolidate:ties": "TIES-Merging",
         "consolidate:magmax": "MagMax", "consolidate:distill": "Multi-teacher distillation",
         "consolidate:continual_distill": "Continual distillation"}


def accuracies(path: str = RESULTS, preset: str = "split_mnist_class") -> tuple[list[tuple[str, float]], int]:
    """[(method, mean final accuracy over the finished seeds)], best first, and the number of seeds."""
    by: dict[str, list[float]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["preset"] == preset and r["status"] == "done" and r["acc"]:
                by.setdefault(r["method"], []).append(float(r["acc"]))
    if not by:
        sys.exit(f"no finished {preset} runs in {path}")
    rows = sorted(((m, mean(v)) for m, v in by.items()), key=lambda t: -t[1])
    return rows, max(len(v) for v in by.values())


def _bar(x0: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    """Path of a bar that is square at the baseline and rounded at the data end."""
    r = min(r, w, h / 2)
    return (f"M{x0:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}"
            f"a{r},{r} 0 0 1 {-r},{r}h{-(w - r):.1f}z")


def svg_chart(rows: list[tuple[str, float]], seeds: int) -> str:
    W, left, right, top = 760, 210, 56, 92
    row_h, bar_h = 26, 16
    H = top + len(rows) * row_h + 40
    sx = (W - left - right)
    x = lambda v: left + v * sx  # noqa: E731
    y_end = top + len(rows) * row_h
    desc = "; ".join(f"{NAMES.get(m, m)}: {v:.3f}" for m, v in rows)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        'aria-labelledby="acc-title acc-desc" font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, '
        'Arial, sans-serif">',
        '<title id="acc-title">Final accuracy of continual-learning methods on Split MNIST (Class-IL)</title>',
        f'<desc id="acc-desc">Mean over {seeds} seeds. {desc}</desc>',
        "<style>",
        ".bg{fill:#fff;stroke:#e5e7eb}.t{fill:#1f2937}.m{fill:#6b7280}.g{stroke:#e5e7eb}.b{fill:#2a78d6}",
        "@media (prefers-color-scheme:dark){.bg{fill:#1f2023;stroke:#3a3b40}.t{fill:#e5e7eb}.m{fill:#9ca3af}"
        ".g{stroke:#374151}.b{fill:#3987e5}}",
        "</style>",
        f'<rect class="bg" x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="8"/>',
        '<text class="t" x="20" y="32" font-size="17" font-weight="600">Final accuracy on Split MNIST '
        '(class-incremental)</text>',
        f'<text class="m" x="20" y="54" font-size="13">Mean over {seeds} seeds · 5 tasks of 2 digits, one model for all '
        '10 classes, MLP, one epoch per task</text>',
        '<text class="m" x="20" y="72" font-size="13">Joint trains on all tasks at once; Naive fine-tunes on each task '
        'in turn</text>',
    ]
    for i in range(5):
        v = i / 4
        out.append(f'<line class="g" x1="{x(v):.1f}" y1="{top - 6}" x2="{x(v):.1f}" y2="{y_end}"/>')
        out.append(f'<text class="m" x="{x(v):.1f}" y="{y_end + 18}" font-size="12" text-anchor="middle">{v:.2f}'
                   '</text>')
    for k, (m, v) in enumerate(rows):
        y = top + k * row_h
        by = y + (row_h - bar_h) / 2
        out.append(f'<text class="t" x="{left - 12}" y="{y + row_h / 2 + 4:.1f}" font-size="13" text-anchor="end">'
                   f'{NAMES.get(m, m)}</text>')
        out.append(f'<path class="b" d="{_bar(left, by, v * sx, bar_h)}"/>')
        out.append(f'<text class="t" x="{x(v) + 6:.1f}" y="{by + bar_h - 3:.1f}" font-size="12">{v:.3f}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def changelog_page() -> str:
    with open(CHANGELOG, encoding="utf-8") as f:
        return CHANGELOG_FRONT + f.read()


def targets() -> dict[str, str]:
    rows, seeds = accuracies()
    return {CHART: svg_chart(rows, seeds), CHANGELOG_PAGE: changelog_page()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if a derived file is not current")
    a = ap.parse_args()
    stale = []
    for path, text in targets().items():
        try:
            with open(path, encoding="utf-8") as f:
                current = f.read()
        except FileNotFoundError:
            current = None
        if current == text:
            continue
        if a.check:
            stale.append(os.path.relpath(path, ROOT))
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            print("wrote", os.path.relpath(path, ROOT))
    if stale:
        sys.exit("stale (run python3 scripts/sync_docs.py): " + ", ".join(stale))


if __name__ == "__main__":
    main()
