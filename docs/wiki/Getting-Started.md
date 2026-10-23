# Getting Started

## Install

```bash
git clone https://github.com/JonPark0/contilab && cd contilab
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

Python ≥ 3.11 and PyTorch 2.x are required. A GPU is optional: the `cpu` profile is designed to finish on a
4-core machine, the `gpu` profile scales the same experiments up.

## First run: every method on synthetic data

```bash
.venv/bin/contilab run configs/experiments/smoke_all.yaml
.venv/bin/contilab report runs/smoke_all
```

This runs 11 methods × 3 synthetic streams (class-IL, task-IL, text) × 2 seeds with the full metric tier and
writes:

- `runs/smoke_all/results.csv` — one row per run, one column per scalar metric
- `runs/smoke_all/report.md` and `report.html` — tables, statistics and figures

## A real benchmark

```bash
# Split MNIST, class-incremental, all methods, 3 seeds
.venv/bin/contilab run configs/experiments/cpu_main.yaml "matrix.preset=[split_mnist_class]"
.venv/bin/contilab report runs/cpu_main
```

Datasets download once into `$CONTILAB_CACHE` (default `~/.cache/contilab`). Use
`contilab data prepare <benchmark>` to fetch them ahead of time.

## Reading the results

- **ACC** — average accuracy over all tasks after the last task.
- **BWT / forgetting** — how much earlier tasks lost (negative BWT = forgetting).
- **min-ACC / stability gap** — the worst dip of an old task during later training (continual evaluation).
- **recency bias** — in class-IL, how often old-task samples are predicted as the newest task's classes.

Every metric is defined in [Metrics](Metrics.md). Runs marked `diverged` (loss became NaN or exploded) are kept and
flagged in the tables as `(diverged k/n)`.

## Next steps

- Change scale, model or hyperparameters: [Configuration](Configuration.md)
- Compare merging and distillation: [Consolidation](Consolidation.md)
- Add your own method or metric: [Extending](Extending.md)
