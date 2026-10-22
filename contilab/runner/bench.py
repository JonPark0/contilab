"""Throughput benchmark: eager vs compile × ensemble width E (§7.1 `contilab bench`)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from contilab.core.config import EngineConfig, ModelConfig, OptimConfig
from contilab.core.scenario import sample_mask
from contilab.core.types import RunContext, Stream, Task, TensorSplit
from contilab.data.batching import TrainPool
from contilab.engine.device import select_device
from contilab.engine.ensemble import Ensemble
from contilab.engine.step import TrainStep
from contilab.models.registry import build_model
from contilab.strategies.naive import Naive

SHAPES = {"mlp": ((784,), 32, ModelConfig(name="mlp")), "resnet18r": ((3, 32, 32), 32, ModelConfig(name="resnet18r")),
          "hf": ((64,), 16, ModelConfig(name="hf", hf_name="tiny-gptneox"))}


def _stream(model: str, n: int, device) -> Stream:
    shape, _, _ = SHAPES[model]
    g = torch.Generator().manual_seed(0)
    if model == "hf":
        x = torch.randint(1, 120, (n, *shape), generator=g)
        split = TensorSplit(x, torch.randint(0, 10, (n,), generator=g), torch.ones_like(x))
    else:
        split = TensorSplit(torch.randn(n, *shape, generator=g), torch.randint(0, 10, (n,), generator=g))
    split = split.to(device)
    return Stream("bench", [Task(0, split, split, split, torch.arange(10))], "class", 10, "text" if model == "hf" else "vision")


def _measure(model: str, E: int, steps: int, compile_mode: str, device) -> dict:
    shape, B, mcfg = SHAPES[model]
    stream = _stream(model, B * (steps + 5), device)
    pool = TrainPool.from_stream(stream)

    def fac(seed):
        torch.manual_seed(seed)
        return build_model(mcfg, stream)

    ens = Ensemble(fac, list(range(E)), device)
    step = TrainStep(ens, Naive(), stream, EngineConfig(compile=compile_mode, precision="fp32"), OptimConfig(lr=0.01))
    step.set_task(0)
    ctx = RunContext(0, 0, device, E, [torch.Generator().manual_seed(s) for s in range(E)], stream,
                     torch.ones(E, dtype=torch.bool), pool=pool, ens=ens)

    def batch(i):
        idx = torch.arange(i * B, (i + 1) * B, device=pool.x.device).expand(E, B)
        b = pool.gather(idx)
        b.mask = sample_mask(stream, b.task_id, 0)
        return b

    for i in range(5):  # warm-up (and compilation)
        step(batch(i), None, ctx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(steps):
        step(batch(5 + i % steps), None, ctx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"model": model, "E": E, "compile": compile_mode, "compile_used": step.compile_decision["used"],
            "compile_reason": step.compile_decision["reason"], "samples_per_s": steps * B * E / dt,
            "step_ms": 1000 * dt / steps, "threads": torch.get_num_threads(), "device": str(device)}


def run_bench(model: str, E_list: list[int], steps: int, compile_modes: list[str], out: Path, device: str = "auto") -> list[dict]:
    dev = select_device(device)
    rows = [_measure(model, E, steps, cm, dev) for cm in compile_modes for E in E_list]
    eager1 = next((r["samples_per_s"] for r in rows if r["E"] == 1 and r["compile"] == "off"), None)
    for r in rows:
        same1 = next((x["samples_per_s"] for x in rows if x["E"] == 1 and x["compile"] == r["compile"]), None)
        r["vmap_speedup"] = r["samples_per_s"] / same1 if same1 else None  # vs E sequential E=1 runs
        r["speedup_vs_E1_eager"] = r["samples_per_s"] / eager1 if eager1 else None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bench.json"
    prev = json.loads(path.read_text()) if path.exists() else []
    keep = [p for p in prev if p["model"] != model]
    path.write_text(json.dumps(keep + rows, indent=1))
    return rows
