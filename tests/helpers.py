"""Shared test helpers (imported by test modules; conftest only handles options/markers)."""

from __future__ import annotations

import torch

from contilab.core.types import Stream, Task, TensorSplit


def make_stream(classes: list[list[int]], scenario: str = "class", C: int | None = None, n: int = 4, dim: int = 3) -> Stream:
    """Dummy stream whose tasks own the given global class ids."""
    C = C if C is not None else max(max(c) for c in classes) + 1
    tasks = []
    for i, cls in enumerate(classes):
        y = torch.tensor([cls[j % len(cls)] for j in range(n)])
        split = TensorSplit(torch.randn(n, dim), y)
        tasks.append(Task(id=i, train=split, val=split, test=split, classes=torch.tensor(cls)))
    return Stream(name="dummy", tasks=tasks, scenario=scenario, num_classes=C, modality="vision")


CPU = torch.device("cpu")


def make_ctx(stream: Stream, seeds: list[int], ens=None, pool=None):
    from contilab.core.types import RunContext

    return RunContext(task_idx=0, step=0, device=CPU, E=len(seeds),
                      gens=[torch.Generator().manual_seed(s) for s in seeds], stream=stream,
                      active=torch.ones(len(seeds), dtype=torch.bool), ens=ens, pool=pool)


def mlp_factory(in_dim: int = 32, C: int = 10, hidden: int = 16):
    from contilab.models.mlp import MLP

    def fac(seed: int):
        torch.manual_seed(seed)
        return MLP(in_dim, C, hidden=hidden)

    return fac


STREAM = None  # lazily built default synthetic stream (see default_stream)


def default_stream():
    global STREAM
    if STREAM is None:
        from contilab.data.synthetic import build_synthetic

        STREAM = build_synthetic()
    return STREAM


def model_factory(stream: Stream, strategy, model: str = "mlp"):
    def fac(seed: int):
        torch.manual_seed(seed)
        if model == "mlp":
            from contilab.models.mlp import MLP

            m = MLP(stream.tasks[0].train.x.shape[1], stream.num_classes, hidden=16)
        else:
            from contilab.models.hf import HFClassifier

            m = HFClassifier("tiny-gptneox", stream.num_classes)
        return strategy.prepare_model(m, stream)

    return fac


def train_on(stream, strategy, seeds, epochs=1, tasks=None, poison_member=None, model="mlp", lr=0.1, hooks=None):
    from contilab.core.config import EngineConfig, OptimConfig, TrainConfig
    from contilab.data.batching import TrainPool
    from contilab.engine.ensemble import Ensemble
    from contilab.engine.trainer import NullHooks, train_stream

    ens = Ensemble(model_factory(stream, strategy, model), seeds, CPU)
    if poison_member is not None:
        with torch.no_grad():
            next(iter(ens.params.values()))[poison_member].fill_(float("nan"))
    ctx = make_ctx(stream, seeds, ens=ens, pool=TrainPool.from_stream(stream))
    train_stream(ens, strategy, stream, TrainConfig(epochs=epochs, batch_size=32),
                 EngineConfig(compile="off", precision="fp32"), OptimConfig(lr=lr),
                 hooks or NullHooks(), ctx, end_task=tasks)
    return ens, ctx


def params_equal(a, b, atol=1e-6) -> bool:
    return all(torch.allclose(a.params[k], b.params[k], atol=atol) for k in a.params)


def per_task_final_acc(stream, strategy, seeds, epochs=1) -> list[float]:
    from contilab.core.scenario import sample_mask
    from contilab.engine.evaluate import evaluate_split

    ens, ctx = train_on(stream, strategy, seeds, epochs=epochs)
    T = len(stream.tasks)
    out = []
    for j, t in enumerate(stream.tasks):
        strategy.eval_setup(ctx, j)
        mask = sample_mask(stream, torch.tensor(j), T - 1)
        out.append(evaluate_split(ens, ens.params, t.test, mask, 256).correct.float().mean().item())
    return out


def final_acc(stream, strategy, seeds, epochs=1) -> float:
    accs = per_task_final_acc(stream, strategy, seeds, epochs)
    return sum(accs) / len(accs)


def run_with_collectors(stream, strategy, seeds, tier, logs, rho=50, beta=10, model="mlp", epochs=1):
    from contilab.core.config import MetricsConfig, OptimConfig, TrainConfig
    from contilab.metrics.collectors import Collectors

    hooks = Collectors(tier, logs, stream, strategy, MetricsConfig(tier=tier, rho=rho, beta=beta),
                       TrainConfig(epochs=epochs, batch_size=32), OptimConfig(lr=0.1))
    return train_on(stream, strategy, seeds, epochs=epochs, model=model, hooks=hooks)


def fake_log(tmp_path, R, B=None, cost=None):
    from contilab.metrics.rawlog import RawLog

    log = RawLog(tmp_path)
    if B is not None:
        log.append("init_acc", {"accs": list(B)})
    for i, row in enumerate(R):
        log.append("acc_matrix", {"task": i, "accs": list(row)})
    for rec in cost or []:
        log.append("cost", rec)
    return log


def values(out: dict) -> dict:
    return {k: v["value"] for k, v in out.items()}


def mctx(T: int, scenario: str = "class", joint_acc=None, cpt: int = 2, **cfg):
    from contilab.core.config import MetricsConfig
    from contilab.metrics.registry import MetricContext

    table = torch.zeros(T, T * cpt, dtype=torch.bool)
    for t in range(T):
        table[t, t * cpt:(t + 1) * cpt] = True
    return MetricContext(scenario=scenario, class_table=table, joint_acc=joint_acc, cfg=MetricsConfig(**cfg))


def smoke_cfg(**over):
    from contilab.core.config import (BenchmarkConfig, ConsolidateConfig, EngineConfig, MethodConfig, ModelConfig,
                                      OptimConfig, RunConfig, TrainConfig)

    cfg = RunConfig(benchmark=BenchmarkConfig(name="synthetic"), model=ModelConfig(name="mlp", hidden=16),
                    optim=OptimConfig(lr=0.1), train=TrainConfig(epochs=2, batch_size=32),
                    engine=EngineConfig(device="cpu", compile="off", precision="fp32"),
                    method=MethodConfig(name="consolidate:ta"), consolidate=ConsolidateConfig(M=60))
    for k, v in over.items():
        obj = cfg
        *path, leaf = k.split(".")
        for p in path:
            obj = getattr(obj, p)
        setattr(obj, leaf, v)
    return cfg


def run_consolidation_for_test(stream, method, seeds, log, experts_lora=False, transfer="buffer", tier="core", M=60):
    from contilab.consolidate.consolidator import run_consolidation
    from contilab.core.config import MetricsConfig
    from contilab.metrics.collectors import Collectors

    cfg = smoke_cfg(**{"method.name": method, "consolidate.experts_lora": experts_lora, "consolidate.transfer": transfer,
                       "consolidate.M": M})
    hooks = Collectors(tier, [log] if not isinstance(log, list) else log, stream, None, MetricsConfig(tier=tier),
                       cfg.train, cfg.optim)
    ctx = make_ctx(stream, seeds)
    run_consolidation(stream, cfg, seeds, hooks, ctx)
    return ctx


def tiny_experiment(methods, tier="core", seeds=(0, 1), name="tiny", cli=None, sweep=None):
    from pathlib import Path

    import yaml

    from contilab.runner.config import LoadedExperiment

    prof = yaml.safe_load((Path(__file__).parent.parent / "configs/profiles/smoke.yaml").read_text())
    return LoadedExperiment(name=name, profile="smoke", matrix={"preset": ["syn_class"], "method.name": list(methods),
                                                               "seed": list(seeds)},
                            overrides={}, sweep=sweep or {}, workers=1, defaults=prof["defaults"], presets=prof["presets"],
                            cli={"metrics.tier": tier, **(cli or {})})


def statuses(exp_dir) -> dict:
    import json

    out: dict = {}
    for d in sorted(p for p in exp_dir.iterdir() if (p / "status.json").exists()):
        st = json.loads((d / "status.json").read_text())
        out.setdefault(d.name.split("-")[1], set()).add(st["status"])
    return out


def read_status(exp_dir, method) -> dict:
    import json

    d = next(p for p in sorted(exp_dir.iterdir()) if p.name.split("-")[1:2] == [method])
    return json.loads((d / "status.json").read_text())


def acc_matrix(exp_dir) -> dict:
    from contilab.metrics.rawlog import RawLog

    return {d.name: RawLog(d).records("acc_matrix") for d in sorted(exp_dir.iterdir()) if (d / "status.json").exists()}


def mtimes(exp_dir) -> dict:
    return {str(p): p.stat().st_mtime_ns for d in exp_dir.iterdir() if (d / "status.json").exists()
            for p in [d / "status.json", *sorted((d / "raw").glob("*"))]}


def crash_after_task(t, monkeypatch):
    monkeypatch.setenv("CONTILAB_TEST_CRASH_AFTER", str(t))


def clear_crash(monkeypatch):
    monkeypatch.delenv("CONTILAB_TEST_CRASH_AFTER", raising=False)
