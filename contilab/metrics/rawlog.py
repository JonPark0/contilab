"""Raw-data store for one run member: JSONL records + safetensors tensors under <run>/raw/ (§6.1)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from safetensors.torch import load_file, save_file
from torch import Tensor

_TASK_KEY = re.compile(r"^t(\d+)\.")


class RawLog:
    def __init__(self, run_dir: Path | str):
        self.dir = Path(run_dir) / "raw"
        self._pending: dict[str, dict[str, Tensor]] = {}

    def _jsonl(self, key: str) -> Path:
        return self.dir / f"{key}.jsonl"

    def _st(self, key: str) -> Path:
        return self.dir / f"{key}.safetensors"

    def append(self, key: str, record: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self._jsonl(key), "a") as f:
            f.write(json.dumps(record) + "\n")

    def records(self, key: str) -> list[dict]:
        p = self._jsonl(key)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def put_tensor(self, key: str, name: str, t: Tensor) -> None:
        self._pending.setdefault(key, {})[name] = t.detach().cpu().clone().contiguous()

    def flush(self) -> None:
        if not self._pending:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        for key, new in self._pending.items():
            merged = {**self.tensors(key), **new}
            save_file(merged, str(self._st(key)))
        self._pending = {}

    def tensors(self, key: str) -> dict[str, Tensor]:
        p = self._st(key)
        out = load_file(str(p)) if p.exists() else {}
        out.update(self._pending.get(key, {}))
        return out

    def has(self, key: str) -> bool:
        return self._jsonl(key).exists() or self._st(key).exists() or key in self._pending

    def keys(self) -> set[str]:
        if not self.dir.exists():
            return set(self._pending)
        return {p.stem for p in self.dir.iterdir() if p.suffix in (".jsonl", ".safetensors")} | set(self._pending)

    def truncate_after_task(self, t: int) -> None:
        """Drop everything recorded after task t finished (used when resuming from task t's checkpoint)."""
        self._pending = {}
        if not self.dir.exists():
            return
        for p in self.dir.glob("*.jsonl"):
            keep = [r for r in self.records(p.stem) if r.get("task", -1) <= t]
            p.write_text("".join(json.dumps(r) + "\n" for r in keep))
        for p in self.dir.glob("*.safetensors"):
            kept = {k: v for k, v in load_file(str(p)).items() if not (_TASK_KEY.match(k) and int(_TASK_KEY.match(k).group(1)) > t)}
            if kept:
                save_file(kept, str(p))
            else:
                p.unlink()
