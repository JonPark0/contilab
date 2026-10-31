"""Run/experiment configuration dataclasses (loading and merging live in contilab.runner.config)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkConfig:
    name: str = "synthetic"
    scenario: str = "class"
    n_train: int | None = None
    n_test: int | None = None
    n_tasks: int | None = None
    max_len: int = 128
    tokenizer: str | None = None
    task_order_seed: int | None = None


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    targets: list[str] | None = None
    lr: float | None = 1e-3  # optimizer lr whenever only LoRA adapters (+ head) are trained (§5.3)


@dataclass
class ModelConfig:
    name: str = "mlp"
    hidden: int = 400
    depth: int = 2
    nf: int = 20
    hf_name: str | None = None
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class OptimConfig:
    name: str = "sgd"
    lr: float = 0.03
    weight_decay: float = 0.0
    momentum: float = 0.0
    clip: float | None = None


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 32
    eval_batch_size: int = 256


@dataclass
class EngineConfig:
    device: str = "auto"
    compile: str = "auto"
    precision: str = "auto"
    max_E: int = 8


@dataclass
class MethodConfig:
    name: str = "naive"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidateConfig:
    lam_grid: list[float] = field(default_factory=lambda: [round(0.1 * i, 1) for i in range(1, 11)])
    ties_k: float = 0.2
    init: str = "base"
    routing: str = "entropy"
    transfer: str = "buffer"
    M: int = 500
    temperature: float = 2.0
    epochs: int = 1
    experts_lora: bool = False
    init_seed: int = 0


@dataclass
class MetricsConfig:
    tier: str = "core"
    rho: int = 50
    stream_subset: int = 200
    beta: int = 10
    probe_size: int = 256
    recovery_n: int = 50
    recovery_steps: int = 20
    window_frac: float = 0.1
    ece_bins: int = 15
    dormant_tau: float = 0.025
    srank_delta: float = 0.01
    ppl_n: int = 200
    barrier_points: int = 11


@dataclass
class RunConfig:
    experiment: str = "default"
    preset: str = "default"
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    consolidate: ConsolidateConfig = field(default_factory=ConsolidateConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    seed: int = 0


@dataclass
class ExperimentConfig:
    name: str
    profile: str
    matrix: dict[str, list] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    sweep: dict[str, list] = field(default_factory=dict)
    method_sweep: dict[str, dict[str, list]] = field(default_factory=dict)  # per-method grids: {ewc: {key: [...]}}
    workers: int = 1
    threads_per_worker: int | None = None
    baseline: str = "naive"
