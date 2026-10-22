"""Group seed-only variants into E-member runs and schedule them (§7.4)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from contilab.core.config import RunConfig
from contilab.runner.config import LoadedExperiment, expand, group_key, run_id
from contilab.runner.execute import execute_group
from contilab.runner.finalize import finalize_experiment
from contilab.runner.storage import RunDir

TEXT_BENCHMARKS = {"text5", "synthetic_text"}
TERMINAL = {"done", "diverged"}  # a diverged run is a result, not a failure: never re-run it


def group_runs(cfgs: list[RunConfig], max_E: int) -> list[list[RunConfig]]:
    by_key: dict[str, list[RunConfig]] = {}
    for c in cfgs:
        by_key.setdefault(group_key(c), []).append(c)
    out = []
    for members in by_key.values():
        members = sorted(members, key=lambda c: c.seed)
        out.extend(members[i : i + max_E] for i in range(0, len(members), max_E))
    return out


def _init_worker(threads: int) -> None:
    torch.set_num_threads(threads)


def _run_groups(groups: list[list[RunConfig]], exp_dir: Path, workers: int, threads: int) -> None:
    if not groups:
        return
    if workers <= 1:
        prev = torch.get_num_threads()
        torch.set_num_threads(threads)
        try:
            for g in groups:
                execute_group(g, exp_dir)
        finally:
            torch.set_num_threads(prev)
        return
    import multiprocessing as mp

    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1, mp_context=mp.get_context("spawn"),
                             initializer=_init_worker, initargs=(threads,)) as pool:
        for f in [pool.submit(execute_group, g, exp_dir) for g in groups]:
            f.result()


def run_experiment(exp: LoadedExperiment, root: Path) -> Path:
    exp_dir = Path(root) / exp.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "experiment.yaml").write_text(yaml.safe_dump(
        {k: v for k, v in asdict(exp).items() if k not in ("defaults", "presets")}, sort_keys=True))
    cfgs = expand(exp)
    pending = [c for c in cfgs if RunDir(exp_dir / run_id(c)).status() not in TERMINAL]
    groups: list[list[RunConfig]] = []
    by_E: dict[int, list[RunConfig]] = {}
    for c in pending:
        by_E.setdefault(c.engine.max_E, []).append(c)
    for max_E, cs in by_E.items():
        groups.extend(group_runs(cs, max_E))
    cores = os.cpu_count() or 1
    workers = max(1, exp.workers)
    threads = exp.threads_per_worker or max(1, cores // workers)
    vision = [g for g in groups if g[0].benchmark.name not in TEXT_BENCHMARKS]
    text = [g for g in groups if g[0].benchmark.name in TEXT_BENCHMARKS]
    _run_groups(vision, exp_dir, workers, threads)
    _run_groups(text, exp_dir, 1, cores)  # language models get the whole machine (§7.4: text 1×cores)
    finalize_experiment(exp_dir)
    return exp_dir
