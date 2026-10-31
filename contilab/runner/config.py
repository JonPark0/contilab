"""Experiment loading (§7.2): dataclass defaults ← profile.defaults ← preset ← experiment.overrides
← matrix scalar axes ← sweep ← CLI key=val. Unknown keys are errors."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import itertools
import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from contilab.core.config import ExperimentConfig, RunConfig


def _set_path(d: dict, path: str, value: Any) -> None:
    *head, leaf = path.split(".")
    for k in head:
        d = d.setdefault(k, {})
    d[leaf] = value


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _dotted(d: dict) -> dict:
    """{'a.b': 1} → {'a': {'b': 1}} (keys without dots pass through)."""
    out: dict = {}
    for k, v in d.items():
        _set_path(out, k, v)
    return out


def _build(cls, data: dict, prefix: str = ""):
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    kwargs = {}
    for k, v in data.items():
        path = f"{prefix}{k}"
        if k not in known:
            raise ValueError(f"unknown config key: {path}")
        sub = hints[k]
        if is_dataclass(sub) and isinstance(v, dict):
            kwargs[k] = _build(sub, v, f"{path}.")
        else:
            kwargs[k] = v
    return cls(**kwargs)


def parse_overrides(items: list[str]) -> dict[str, Any]:
    out = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"override must be key=value: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def _profile_path(exp_path: Path, profile: str) -> Path:
    p = Path(profile)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    return exp_path.resolve().parent.parent / "profiles" / f"{profile}.yaml"


@dataclasses.dataclass
class LoadedExperiment(ExperimentConfig):
    defaults: dict = dataclasses.field(default_factory=dict)
    presets: dict = dataclasses.field(default_factory=dict)
    cli: dict = dataclasses.field(default_factory=dict)


def load_experiment(path: Path | str, cli_overrides: list[str]) -> LoadedExperiment:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    prof = yaml.safe_load(_profile_path(path, raw["profile"]).read_text()) or {}
    cli = parse_overrides(cli_overrides)
    matrix = dict(raw.get("matrix", {}))
    run_cli = {}
    for k, v in cli.items():
        if k.startswith("matrix."):
            matrix[k[len("matrix."):]] = v if isinstance(v, list) else [v]
        elif k in {f.name for f in fields(ExperimentConfig)}:
            raw[k] = v
        else:
            run_cli[k] = v
    _build(RunConfig, _dotted(run_cli))  # validate CLI keys early
    exp = LoadedExperiment(name=raw["name"], profile=raw["profile"], matrix=matrix, overrides=raw.get("overrides", {}),
                           sweep=raw.get("sweep", {}), method_sweep=raw.get("method_sweep", {}),
                           workers=raw.get("workers", 1),
                           threads_per_worker=raw.get("threads_per_worker"), baseline=raw.get("baseline", "naive"),
                           defaults=prof.get("defaults", {}), presets=prof.get("presets", {}), cli=run_cli)
    for name in exp.matrix.get("preset", []):
        if name not in exp.presets:
            raise ValueError(f"unknown preset '{name}' in profile '{exp.profile}'")
    return exp


def expand(exp: LoadedExperiment) -> list[RunConfig]:
    axes = {k: v for k, v in exp.matrix.items()}
    presets = axes.pop("preset", ["default"])
    grid_keys = list(axes) + list(exp.sweep)
    grid_vals = list(axes.values()) + list(exp.sweep.values())
    runs = []
    for preset in presets:
        for combo in itertools.product(*grid_vals) if grid_keys else [()]:
            d = deep_merge(exp.defaults, exp.presets.get(preset, {}))
            d = deep_merge(d, _dotted(exp.overrides))
            d = deep_merge(d, _dotted(dict(zip(grid_keys, combo))))
            msweep = exp.method_sweep.get((d.get("method") or {}).get("name", "naive"), {})
            for mcombo in itertools.product(*msweep.values()) if msweep else [()]:
                dm = deep_merge(d, _dotted(dict(zip(msweep, mcombo))))
                dm = deep_merge(dm, _dotted(exp.cli))
                dm.update(experiment=exp.name, preset=preset)
                runs.append(_build(RunConfig, dm))
    return runs


def _canonical(cfg: RunConfig, drop_seed: bool = False) -> str:
    d = asdict(cfg)
    if drop_seed:
        d["seed"] = None
    return json.dumps(d, sort_keys=True, default=str)


def run_id(cfg: RunConfig) -> str:
    h = hashlib.sha1(_canonical(cfg).encode()).hexdigest()[:8]
    return f"{cfg.preset}-{cfg.method.name.replace(':', '_')}-s{cfg.seed}-{h}"


def group_key(cfg: RunConfig) -> str:
    return hashlib.sha1(_canonical(cfg, drop_seed=True).encode()).hexdigest()


def run_config_from_dict(d: dict) -> RunConfig:
    return _build(RunConfig, d)
