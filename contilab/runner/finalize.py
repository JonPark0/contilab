"""Post-hoc metric computation for every run of an experiment → metrics.json + results.csv."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from contilab.metrics.accuracy import acc_matrix_of
from contilab.metrics.registry import MetricContext, compute_metrics
from contilab.runner.config import run_config_from_dict
from contilab.runner.storage import RunDir

ID_COLS = ["run_id", "preset", "method", "variant", "seed", "status"]
FIXED_AXES = {"preset", "method.name", "seed"}


def _variant_keys(exp_dir: Path) -> list[str]:
    """Config paths that vary inside one (preset, method) cell: extra matrix axes and sweep keys."""
    p = exp_dir / "experiment.yaml"
    if not p.exists():
        return []
    import yaml

    exp = yaml.safe_load(p.read_text()) or {}
    return sorted([k for k in (exp.get("matrix") or {}) if k not in FIXED_AXES] + list(exp.get("sweep") or {}))


def _get(d: dict, path: str):
    for k in path.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


def run_dirs(exp_dir: Path) -> list[RunDir]:
    return [RunDir(p) for p in sorted(Path(exp_dir).iterdir()) if (p / "config.yaml").exists()]


def _joint_lookup(rds: list[RunDir]) -> dict[tuple[str, int], list[float]]:
    out = {}
    for rd in rds:
        c = rd.read_config()
        if c["method"]["name"] == "joint" and rd.status() == "done":
            R = acc_matrix_of(rd.log)
            if len(R):
                out[(c["preset"], c["seed"])] = R[-1].tolist()
    return out


def finalize_experiment(exp_dir: Path, add: list[str] | None = None, tier: str | None = None) -> Path:
    exp_dir = Path(exp_dir)
    rds = run_dirs(exp_dir)
    joint = _joint_lookup(rds)
    vkeys = _variant_keys(exp_dir)
    rows = []
    for rd in rds:
        raw_cfg = rd.read_config()
        cfg = run_config_from_dict(raw_cfg)
        variant = ", ".join(f"{k}={_get(raw_cfg, k)}" for k in vkeys)
        row = {"run_id": rd.path.name, "preset": cfg.preset, "method": cfg.method.name, "variant": variant,
               "seed": cfg.seed, "status": rd.status()}
        mpath = rd.path / "metrics.json"
        metrics = json.loads(mpath.read_text()) if mpath.exists() else {}
        if row["status"] in ("done", "diverged") and (rd.path / "stream.json").exists():
            info = rd.read_json("stream.json")
            mctx = MetricContext(scenario=info["scenario"], class_table=torch.tensor(info["class_table"], dtype=torch.bool),
                                 joint_acc=joint.get((cfg.preset, cfg.seed)), cfg=cfg.metrics)
            if add:
                metrics.update(compute_metrics(rd.log, mctx, tier or "full", only=add))
            else:
                metrics = compute_metrics(rd.log, mctx, tier or cfg.metrics.tier)
            mpath.write_text(json.dumps(metrics, indent=1))
        for k, v in metrics.items():
            if isinstance(v.get("value"), (int, float)) and not isinstance(v.get("value"), bool):
                row[k] = v["value"]
        rows.append(row)
    cols = ID_COLS + sorted({k for r in rows for k in r} - set(ID_COLS))
    out = exp_dir / "results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return out
