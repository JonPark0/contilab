"""torch.compile keeps one graph across batch sizes, and allows one graph per task (LoRA-per-task)."""

import pytest
import torch
from torch._dynamo.utils import counters

from contilab.core.config import EngineConfig, OptimConfig
from contilab.core.scenario import sample_mask
from contilab.data.batching import TrainPool
from contilab.data.synthetic import build_synthetic
from contilab.engine.ensemble import Ensemble
from contilab.engine.step import TrainStep, maybe_compile
from contilab.strategies.lora_per_task import LoRAPerTask
from contilab.strategies.naive import Naive
from helpers import CPU, make_ctx, mlp_factory, train_on


@pytest.fixture(autouse=True)
def fresh_dynamo():
    torch._dynamo.reset()
    counters.clear()
    yield
    torch._dynamo.reset()


def graphs() -> int:
    return counters["stats"]["unique_graphs"]


def test_batch_and_replay_sizes_share_one_graph():
    s = build_synthetic()
    ens = Ensemble(mlp_factory(), [0], CPU)
    pool = TrainPool.from_stream(s)
    ctx = make_ctx(s, [0], ens=ens, pool=pool)
    step = TrainStep(ens, Naive(), s, EngineConfig(compile="on", precision="fp32"), OptimConfig(lr=0.1))
    step.set_task(0)
    # more distinct sizes than Dynamo's default limit of 8 graphs; the replay batch (like an A-GEM reference
    # batch) changes size independently
    for n, m in zip((16, 15, 13, 9, 12, 7, 11, 10, 14, 6), (40, 39, 38, 37, 36, 35, 34, 33, 32, 31)):
        b = pool.gather(torch.arange(n).expand(1, n))
        b.mask = sample_mask(s, b.task_id, 0)
        r = pool.gather(torch.arange(100, 100 + m).expand(1, m))
        r.mask = sample_mask(s, r.task_id, 0)
        step(b, r, ctx)
    if not step.compile_decision["used"]:
        pytest.skip(f"compile unavailable: {step.compile_decision['reason']}")
    assert graphs() == 1


def test_recompile_limit_is_raised_per_function():
    table = {f"t{i}": torch.tensor(float(i)) for i in range(10)}

    def f(x, key):  # a string key is always specialised: one graph per key, like LoRA's active adapter
        return x * table[key]

    g, decision = maybe_compile(f, "on", recompile_limit=12)
    x = torch.ones(3)
    for i in range(10):
        assert torch.equal(g(x, f"t{i}"), x * i)
    assert decision["used"] and graphs() == 10  # the default limit (8) would stop at 8 graphs


def test_lora_compiles_every_task_beyond_eight():
    s = build_synthetic(n_tasks=10, n_train=40)
    train_on(s, LoRAPerTask(), [0], compile="on")
    assert graphs() >= 10  # one graph per task's adapter; eager after 8 tasks before the limit was raised


def test_lora_compiles_every_task_and_batch_size_with_vmapped_seeds():
    # With several seeds under vmap the batch size is specialised, so each task's adapter takes one graph for full
    # batches and one for the task's last, partial batch: 20 graphs for 10 tasks, more than tasks + 8 (= 18).
    s = build_synthetic(n_tasks=10, n_train=40)
    train_on(s, LoRAPerTask(), [0, 1], compile="on")
    assert graphs() >= 20
