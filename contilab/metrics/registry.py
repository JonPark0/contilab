"""Metric registry: every metric is a pure function of a member's RawLog (§6.1)."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

from torch import Tensor

from contilab.core.config import MetricsConfig
from contilab.metrics.collectors import RAW_KEYS_BY_TIER
from contilab.metrics.rawlog import RawLog

TIERS = ("core", "extended", "full")
_MODULES = ("accuracy", "efficiency", "continual", "representation", "plasticity", "bias", "recovery", "landscape", "llm")


@dataclass
class MetricContext:
    scenario: str
    class_table: Tensor  # Bool[T, C]
    joint_acc: list[float] | None
    cfg: MetricsConfig


class MetricUnavailable(Exception):
    """Raised by a metric when it is undefined for this run (value → None with the reason)."""


@dataclass
class MetricSpec:
    name: str
    fn: Callable[[RawLog, MetricContext], Any]
    requires: tuple[str, ...]
    tier: str


METRICS: dict[str, MetricSpec] = {}


def metric(name: str, requires: tuple[str, ...], tier: str):
    def deco(fn):
        METRICS[name] = MetricSpec(name, fn, tuple(requires), tier)
        return fn

    return deco


def raw_key_tier(key: str) -> str:
    return next(t for t in TIERS if key in RAW_KEYS_BY_TIER[t])


def load_all() -> None:
    for m in _MODULES:
        importlib.import_module(f"contilab.metrics.{m}")


def compute_metrics(log: RawLog, mctx: MetricContext, tier: str, only: list[str] | None = None) -> dict[str, dict]:
    load_all()
    level = TIERS.index(tier)
    out: dict[str, dict] = {}
    for name, spec in METRICS.items():
        if only is not None:
            if name not in only:
                continue
        elif TIERS.index(spec.tier) > level:
            continue
        missing = [k for k in spec.requires if not log.has(k)]
        if missing:
            out[name] = {"value": None, "error": f"missing raw '{missing[0]}' (tier '{raw_key_tier(missing[0])}')"}
            continue
        try:
            out[name] = {"value": spec.fn(log, mctx), "error": None}
        except MetricUnavailable as e:
            out[name] = {"value": None, "error": str(e)}
        except Exception as e:  # one broken metric never sinks the rest (§7.5)
            out[name] = {"value": None, "error": f"{type(e).__name__}: {e}"}
    return out
