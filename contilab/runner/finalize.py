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

ID_COLS = ["run_id", "preset", "method", "seed", "status"]


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
    rows = []
    for rd in rds:
        cfg = run_config_from_dict(rd.read_config())
        row = {"run_id": rd.path.name, "preset": cfg.preset, "method": cfg.method.name, "seed": cfg.seed, "status": rd.status()}
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
