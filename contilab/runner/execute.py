"""Execute one group of runs that differ only by seed as a single E-member ensemble (§7.3–7.5)."""

from __future__ import annotations

import hashlib
import os
import shutil
import traceback
from dataclasses import replace
from pathlib import Path

import torch

from contilab.core.config import RunConfig
from contilab.core.types import RunContext, Stream
from contilab.data.batching import TrainPool
from contilab.data.registry import build_stream
from contilab.data.transforms import make_joint_stream
from contilab.engine.device import reset_peak_memory, select_device, select_precision
from contilab.engine.ensemble import Ensemble
from contilab.engine.trainer import train_stream
from contilab.metrics.collectors import Collectors
from contilab.models.registry import build_model
from contilab.runner.config import run_id
from contilab.runner.storage import RunDir, env_info
from contilab.strategies.base import build_strategy

CRASH_ENV = "CONTILAB_TEST_CRASH_AFTER"


class VmapFallback(Exception):
    """The group failed before its first optimisation step; members are retried one by one."""


def _ckpt_dir(exp_dir: Path, cfgs: list[RunConfig]) -> Path:
    h = hashlib.sha1("|".join(run_id(c) for c in cfgs).encode()).hexdigest()[:16]
    return exp_dir / "_ckpt" / h


def _stream_info(stream: Stream) -> dict:
    return {"name": stream.name, "scenario": stream.scenario, "num_classes": stream.num_classes,
            "class_table": stream.class_table().int().tolist(), "modality": stream.modality}


def _lora_optim(cfg: RunConfig):
    """LoRA's own lr (§5.3: 1e-3) is an Adam-style rate; SGD keeps the run's lr."""
    if cfg.model.lora.lr is not None and cfg.optim.name == "adamw":
        return replace(cfg.optim, lr=cfg.model.lora.lr)
    return cfg.optim


def _clear_raw(rdirs: list[RunDir]) -> None:
    """A run that starts from scratch must not append to records of an earlier, abandoned attempt."""
    for rd in rdirs:
        shutil.rmtree(rd.path / "raw", ignore_errors=True)
        (rd.path / "metrics.json").unlink(missing_ok=True)
        rd.log = type(rd.log)(rd.path)


def _save_ckpt(path: Path, t: int, ens: Ensemble, strategy, ctx: RunContext) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"task": t, "params": {k: v.detach().cpu() for k, v in ens.params.items()},
                "buffers": {k: v.cpu() for k, v in ens.buffers.items()}, "strategy": strategy.state_dict(),
                "gens": [g.get_state() for g in ctx.gens], "step": ctx.step, "active": ctx.active.clone(),
                "diverged": sorted(ctx.diverged)}, path / f"task_{t}.pt")
    for old in path.glob("task_*.pt"):
        if old.name != f"task_{t}.pt":
            old.unlink()
    if os.environ.get(CRASH_ENV) == str(t):
        raise RuntimeError(f"injected crash after task {t}")


def _load_ckpt(path: Path) -> dict | None:
    cks = sorted(path.glob("task_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return torch.load(cks[-1], weights_only=False) if cks else None


def _run_training(cfgs: list[RunConfig], rdirs: list[RunDir], exp_dir: Path, stream: Stream, device) -> dict:
    cfg = cfgs[0]
    seeds = [c.seed for c in cfgs]
    method = cfg.method.name
    joint = method == "joint"
    train_s = make_joint_stream(stream) if joint else stream
    strategy = build_strategy("naive" if joint else method, cfg.method.params, stream.modality)
    optim = _lora_optim(cfg) if method == "lora_per_task" else cfg.optim

    def factory(seed: int):
        torch.manual_seed(seed)
        return strategy.prepare_model(build_model(cfg.model, train_s), train_s)

    ens = Ensemble(factory, seeds, device)
    ck_dir = _ckpt_dir(exp_dir, cfgs)
    ck = _load_ckpt(ck_dir)
    if ck is None:
        _clear_raw(rdirs)
    hooks = Collectors(cfg.metrics.tier, [rd.log for rd in rdirs], train_s, strategy, cfg.metrics, cfg.train, optim,
                       eval_stream=stream if joint else None)
    ctx = RunContext(task_idx=0, step=0, device=device, E=len(seeds), gens=[torch.Generator().manual_seed(s) for s in seeds],
                     stream=train_s, active=torch.ones(len(seeds), dtype=torch.bool), pool=TrainPool.from_stream(train_s), ens=ens)
    start = 0
    if ck is not None:
        ens.load_params(ck["params"])
        for k, v in ck["buffers"].items():
            ens.buffers[k].copy_(v)
        strategy.load_state_dict(ck["strategy"])
        for g, s in zip(ctx.gens, ck["gens"]):
            g.set_state(s)
        ctx.step, ctx.active, ctx.diverged = ck["step"], ck["active"], set(ck["diverged"])
        for rd in rdirs:
            rd.log.truncate_after_task(ck["task"])
        start = ck["task"] + 1
        ctx.task_idx = ck["task"]
        hooks.resume(ctx)
    step_holder = {}

    def on_done(t: int) -> None:
        _save_ckpt(ck_dir, t, ens, strategy, ctx)

    try:
        step = train_stream(ens, strategy, train_s, cfg.train, cfg.engine, optim, hooks, ctx, start_task=start, on_task_done=on_done)
        step_holder["step"] = step
    except Exception:
        if ctx.step == 0 and len(seeds) > 1:
            raise VmapFallback(traceback.format_exc())
        raise
    shutil.rmtree(ck_dir, ignore_errors=True)
    step = step_holder["step"]
    return {"compile": step.compile_decision, "precision": step.precision_reason, "diverged": sorted(ctx.diverged),
            "resumed_from_task": start - 1 if start else None}


def _run_consolidation(cfgs: list[RunConfig], rdirs: list[RunDir], stream: Stream, device) -> dict:
    from contilab.consolidate.consolidator import run_consolidation

    cfg = cfgs[0]
    if cfg.consolidate.experts_lora:
        cfg = replace(cfg, optim=_lora_optim(cfg))
    seeds = [c.seed for c in cfgs]
    _clear_raw(rdirs)
    hooks = Collectors(cfg.metrics.tier, [rd.log for rd in rdirs], stream, None, cfg.metrics, cfg.train, cfg.optim)
    ctx = RunContext(task_idx=0, step=0, device=device, E=len(seeds), gens=[torch.Generator().manual_seed(s) for s in seeds],
                     stream=stream, active=torch.ones(len(seeds), dtype=torch.bool), pool=TrainPool.from_stream(stream))
    run_consolidation(stream, cfg, seeds, hooks, ctx)
    _, reason = select_precision(cfg.engine.precision, device)
    return {"compile": {"requested": cfg.engine.compile, "used": False, "reason": "consolidation (eager)"},
            "precision": reason, "diverged": []}


def execute_group(cfgs: list[RunConfig], exp_dir: Path) -> None:
    exp_dir = Path(exp_dir)
    name = cfgs[0].method.name
    if len(cfgs) > 1 and not name.startswith("consolidate:") and name != "joint":
        from contilab.strategies.base import STRATEGIES, build_strategy as _b  # noqa: F401  (registers built-ins)

        cls = STRATEGIES.get(name)
        if cls is not None and not cls.vmappable:
            for c in cfgs:
                execute_group([c], exp_dir)
            return
    rdirs = [RunDir(exp_dir / run_id(c)) for c in cfgs]
    for rd, c in zip(rdirs, cfgs):
        rd.write_config(c)
        rd.set_status("running")
    try:
        cfg = cfgs[0]
        device = select_device(cfg.engine.device)
        reset_peak_memory(device)
        stream = build_stream(cfg.benchmark).to(device)
        for rd in rdirs:
            rd.write_json("stream.json", _stream_info(stream))
        if cfg.method.name.startswith("consolidate:"):
            info = _run_consolidation(cfgs, rdirs, stream, device)
        else:
            info = _run_training(cfgs, rdirs, exp_dir, stream, device)
        env = env_info({"device": str(device), "E": len(cfgs), **info})
        for e, rd in enumerate(rdirs):
            rd.write_json("env.json", env)
            rd.set_status("diverged" if e in info["diverged"] else "done")
    except VmapFallback as fb:
        for rd in rdirs:
            shutil.rmtree(rd.path / "raw", ignore_errors=True)
        for c in cfgs:
            execute_group([c], exp_dir)
            rd = RunDir(exp_dir / run_id(c))
            if (rd.path / "env.json").exists():
                env = rd.read_json("env.json")
                env["vmap_fallback"] = str(fb).strip().splitlines()[-1][:300]
                rd.write_json("env.json", env)
    except Exception:
        tb = traceback.format_exc()
        for rd in rdirs:
            rd.set_status("failed", tb)
