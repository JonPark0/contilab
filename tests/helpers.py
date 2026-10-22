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
