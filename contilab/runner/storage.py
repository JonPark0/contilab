"""Run directory layout (§7.3)."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from contilab.core.config import RunConfig
from contilab.core.io import atomic_write_text
from contilab.metrics.rawlog import RawLog


class RunDir:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.log = RawLog(self.path)

    def status(self) -> str:
        p = self.path / "status.json"
        return json.loads(p.read_text())["status"] if p.exists() else "pending"

    def set_status(self, status: str, error: str | None = None) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path / "status.json", json.dumps({"status": status, "error": error, "time": time.time()}))

    def write_config(self, cfg: RunConfig) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path / "config.yaml", yaml.safe_dump(asdict(cfg), sort_keys=True))

    def read_config(self) -> dict:
        return yaml.safe_load((self.path / "config.yaml").read_text())

    def write_json(self, name: str, data: dict) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path / name, json.dumps(data, indent=1, default=str))

    def read_json(self, name: str) -> dict:
        return json.loads((self.path / name).read_text())


def env_info(decisions: dict) -> dict:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        rev = None
    return {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(), "threads": torch.get_num_threads(),
            "cuda": torch.cuda.is_available(), "git": rev, **decisions}
