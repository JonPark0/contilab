"""Per-task experts from a shared θ₀ (§5.4), trained in vmapped groups and cached on disk."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from contilab.core.config import RunConfig
from contilab.core.types import Batch, RunContext, Stream
from contilab.data.batching import TrainPool, epoch_perms
from contilab.data.cache import cache_dir, key_hash
from contilab.engine.device import select_device
from contilab.engine.ensemble import Ensemble
from contilab.engine.step import TrainStep
from contilab.models.lora import inject_lora, set_active
from contilab.models.registry import build_model
from contilab.strategies.naive import Naive


@dataclass
class Experts:
    base: dict[str, Tensor]  # [S, ...] θ₀ (identical across seeds)
    tuned: list[dict[str, Tensor]]  # per task, [S, ...]
    lora_modules: list[str] | None
    wall_s: float = 0.0
    flops: int = 0

    def nbytes(self) -> int:
        return sum(v.numel() * v.element_size() for d in self.tuned for v in d.values()) // max(1, next(iter(self.base.values())).shape[0])


class ExpertStrategy(Naive):
    """Naive fine-tuning restricted to the expert's trainable parameters (LoRA adapter 0 + head when LoRA)."""

    def __init__(self, lora: bool):
        super().__init__()
        self.lora = lora

    def trainable_names(self, names, task_idx):
        if not self.lora:
            return names
        return [n for n in names if (("lora_A" in n or "lora_B" in n) and n.endswith(".t0")) or n.startswith("head.")]


def expert_factory(stream: Stream, cfg: RunConfig):
    lora = cfg.consolidate.experts_lora
    holder: dict = {}

    def fac(_seed: int):
        torch.manual_seed(cfg.consolidate.init_seed)
        m = build_model(cfg.model, stream)
        if lora:
            holder["modules"] = inject_lora(m, cfg.model.lora.targets, cfg.model.lora.r, cfg.model.lora.alpha, 1, stream.modality)
        return m

    return fac, holder


def _expert_key(stream: Stream, cfg: RunConfig, seeds: list[int]) -> dict:
    return {"stream": stream.name, "tasks": [len(t.train) for t in stream.tasks], "scenario": stream.scenario,
            "benchmark": asdict(cfg.benchmark), "model": asdict(cfg.model), "optim": asdict(cfg.optim),
            "train": asdict(cfg.train), "seeds": seeds, "init_seed": cfg.consolidate.init_seed,
            "lora": cfg.consolidate.experts_lora}


def plan_expert_groups(task_sizes: list[int], seeds: list[int], max_E: int) -> list[list[tuple[int, int]]]:
    """(task, seed) members trained together: equal-size tasks only (same steps per epoch), at most max_E per group."""
    by_size: dict[int, list[int]] = {}
    for t, n in enumerate(task_sizes):
        by_size.setdefault(n, []).append(t)
    groups = []
    for tasks in by_size.values():
        members = [(t, s) for t in tasks for s in seeds]
        groups.extend(members[i : i + max_E] for i in range(0, len(members), max(1, max_E)))
    return groups


def _train_group(stream: Stream, pool: TrainPool, members: list[tuple[int, int]], cfg: RunConfig, device) -> tuple[Ensemble, int]:
    """Returns the trained ensemble and the total FLOPs of all its members."""
    fac, holder = expert_factory(stream, cfg)
    E = len(members)
    ens = Ensemble(fac, [cfg.consolidate.init_seed] * E, device)
    if holder.get("modules"):
        set_active(ens.template, [0])
    gens = [torch.Generator().manual_seed(s) for _, s in members]
    task_of = torch.tensor([t for t, _ in members], device=pool.x.device)
    ctx = RunContext(task_idx=0, step=0, device=device, E=E, gens=gens, stream=stream,
                     active=torch.ones(E, dtype=torch.bool), pool=pool, ens=ens)
    step = TrainStep(ens, ExpertStrategy(bool(holder.get("modules"))), stream, cfg.engine, cfg.optim)
    step.count_flops = True
    step.set_task(0)
    table = stream.class_table().to(pool.x.device)
    n = pool.task_size(members[0][0])
    flops = 0
    for _ in range(cfg.train.epochs):
        perms = epoch_perms(n, gens).to(pool.x.device) + pool.offsets.to(pool.x.device)[task_of][:, None]
        for i in range(0, n, cfg.train.batch_size):
            b: Batch = pool.gather(perms[:, i : i + cfg.train.batch_size])
            b.mask = torch.ones_like(table[b.task_id]) if stream.scenario == "domain" else table[b.task_id]
            step(b, None, ctx)
            flops += (step.flops_per_step or 0) * E
    return ens, flops


def train_experts(stream: Stream, cfg: RunConfig, seeds: list[int]) -> Experts:
    key = _expert_key(stream, cfg, seeds)
    d = cache_dir() / "experts" / key_hash(key)
    fac, holder = expert_factory(stream, cfg)
    if (d / "experts.safetensors").exists():
        t = load_file(str(d / "experts.safetensors"))
        meta = json.loads((d / "meta.json").read_text())
        base = {k[5:]: v for k, v in t.items() if k.startswith("base.")}
        tuned = [{k[len(f"t{i}."):]: v for k, v in t.items() if k.startswith(f"t{i}.")} for i in range(len(stream.tasks))]
        return Experts(base, tuned, meta["lora_modules"], meta["wall_s"], meta["flops"])

    device = select_device(cfg.engine.device)
    s_stream = stream.to(device)
    pool = TrainPool.from_stream(s_stream)
    S = len(seeds)
    t0 = time.perf_counter()
    base_ens = Ensemble(fac, [cfg.consolidate.init_seed] * S, device)
    base = {k: v.detach().cpu().clone() for k, v in base_ens.params.items()}
    per_member: dict[tuple[int, int], dict[str, Tensor]] = {}
    flops = 0
    sizes = [pool.task_size(t) for t in range(len(stream.tasks))]
    for members in plan_expert_groups(sizes, seeds, cfg.engine.max_E):
        ens, f = _train_group(s_stream, pool, members, cfg, device)
        flops += f
        for i, m in enumerate(members):
            per_member[m] = {k: v[i].detach().cpu().clone() for k, v in ens.params.items()}
        del ens
    tuned = [{k: torch.stack([per_member[(t, s)][k] for s in seeds]) for k in per_member[(t, seeds[0])]}
             for t in range(len(stream.tasks))]
    flops //= S  # per seed
    wall = time.perf_counter() - t0
    ex = Experts(base, tuned, holder.get("modules"), wall, flops)
    d.mkdir(parents=True, exist_ok=True)
    flat = {f"base.{k}": v.contiguous() for k, v in base.items()}
    for i, p in enumerate(tuned):
        flat.update({f"t{i}.{k}": v.contiguous() for k, v in p.items()})
    save_file(flat, str(d / "experts.safetensors"))
    (d / "meta.json").write_text(json.dumps({"lora_modules": ex.lora_modules, "wall_s": wall, "flops": flops, "key": key}, default=str))
    return ex
