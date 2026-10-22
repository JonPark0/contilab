"""Benchmark name → stream builder."""

from __future__ import annotations

from dataclasses import replace

import torch

from contilab.core.config import BenchmarkConfig
from contilab.core.types import Stream
from contilab.data import synthetic, vision


def _opt(**kw):
    return {k: v for k, v in kw.items() if v is not None}


def _build(cfg: BenchmarkConfig) -> Stream:
    n = cfg.name
    if n == "synthetic":
        return synthetic.build_synthetic(scenario=cfg.scenario, **_opt(n_tasks=cfg.n_tasks, n_train=cfg.n_train, n_test=cfg.n_test))
    if n == "synthetic_text":
        return synthetic.build_synthetic_text(scenario=cfg.scenario, **_opt(n_tasks=cfg.n_tasks, n_train=cfg.n_train, n_test=cfg.n_test))
    if n == "split_mnist":
        return vision.build_split_mnist(cfg.scenario, cfg.n_train, cfg.n_test)
    if n == "permuted_mnist":
        return vision.build_permuted_mnist(**_opt(n_tasks=cfg.n_tasks), n_train=cfg.n_train, n_test=cfg.n_test)
    if n == "split_cifar10":
        return vision.build_split_cifar10(cfg.scenario, cfg.n_train, cfg.n_test)
    if n == "split_cifar100":
        return vision.build_split_cifar100(cfg.scenario, cfg.n_train, cfg.n_test)
    if n == "text5":
        from contilab.data import text

        return text.build_text5(cfg.tokenizer or "EleutherAI/pythia-70m", cfg.scenario,
                                **_opt(n_train=cfg.n_train, n_test=cfg.n_test), max_len=cfg.max_len)
    raise ValueError(f"unknown benchmark: {n}")


def build_stream(cfg: BenchmarkConfig) -> Stream:
    stream = _build(cfg)
    if cfg.task_order_seed is not None:
        order = torch.randperm(len(stream.tasks), generator=torch.Generator().manual_seed(cfg.task_order_seed))
        stream = replace(stream, tasks=[replace(stream.tasks[i], id=k) for k, i in enumerate(order.tolist())])
    return stream
