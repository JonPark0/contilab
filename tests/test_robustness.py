"""Robustness fixes from the deferred review minors (M1, M2, M5, M14)."""

import json

import pytest
import torch

from contilab.core.config import EngineConfig, OptimConfig
from contilab.data.synthetic import build_synthetic
from contilab.runner.scheduler import run_experiment
from contilab.strategies.base import register_strategy
from contilab.strategies.naive import Naive
from helpers import clear_crash, crash_after_task, make_ctx, model_factory, statuses, tiny_experiment


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path / "cache"))


@register_strategy("loss_jump")
class LossJump(Naive):
    """Reported loss × 1e4 from task 1 on (gradients unchanged): diverged relative to the run's first loss only."""

    def before_task(self, ctx, task):
        self.tensors["scale"] = torch.full((ctx.E,), 1.0 if task.id == 0 else 1e4)

    def state(self):
        return {"scale": self.tensors["scale"]}

    def loss(self, p, s, batch, replay, f):
        loss, z = super().loss(p, s, batch, replay, f)
        return loss + (s["scale"] - 1) * loss.detach(), z


# M1 ---------------------------------------------------------------------------------------------
def test_divergence_baseline_survives_resume(tmp_path, monkeypatch):
    full = run_experiment(tiny_experiment(["loss_jump"], seeds=[0]), tmp_path / "a")
    assert statuses(full)["loss_jump"] == {"diverged"}
    crash_after_task(0, monkeypatch)
    run_experiment(tiny_experiment(["loss_jump"], seeds=[0]), tmp_path / "b")
    clear_crash(monkeypatch)
    resumed = run_experiment(tiny_experiment(["loss_jump"], seeds=[0]), tmp_path / "b")
    assert statuses(resumed)["loss_jump"] == {"diverged"}


# M2 ---------------------------------------------------------------------------------------------
def test_compile_failure_after_first_call_falls_back_to_eager(monkeypatch):
    import contilab.engine.step as step_mod
    from contilab.data.batching import TrainPool, task_batches
    from contilab.engine.ensemble import Ensemble

    def flaky_compile(fn, mode):
        calls = {"n": 0}

        def wrapped(*a):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("recompile failed")
            return fn(*a)

        return wrapped, {"requested": mode, "used": True, "reason": "pending first call"}

    monkeypatch.setattr(step_mod, "maybe_compile", flaky_compile)
    s = build_synthetic(n_tasks=2)
    ens = Ensemble(model_factory(s, Naive()), [0], torch.device("cpu"))
    ctx = make_ctx(s, [0], ens=ens, pool=TrainPool.from_stream(s))
    step = step_mod.TrainStep(ens, Naive(), s, EngineConfig(compile="on"), OptimConfig(lr=0.1))
    step.set_task(0)
    batches = task_batches(ctx.pool, 0, 32, ctx.gens)
    for _ in range(3):
        b = next(batches)
        b.mask = torch.ones(b.y.shape + (s.num_classes,), dtype=torch.bool)
        out = step(b, None, ctx)
        assert torch.isfinite(out.loss).all()
    assert step.compile_decision["used"] is False and "fallback to eager" in step.compile_decision["reason"]


# M5 ---------------------------------------------------------------------------------------------
def test_atomic_write_keeps_old_file_on_failure(tmp_path):
    from contilab.core.io import atomic_write

    p = tmp_path / "f.bin"
    p.write_bytes(b"old")

    def boom(tmp):
        tmp.write_bytes(b"partial")
        raise OSError("disk full")

    with pytest.raises(OSError):
        atomic_write(p, boom)
    assert p.read_bytes() == b"old" and sorted(x.name for x in tmp_path.iterdir()) == ["f.bin"]


def test_checkpoint_write_is_atomic(tmp_path, monkeypatch):
    import contilab.runner.execute as ex
    from contilab.engine.ensemble import Ensemble

    s = build_synthetic(n_tasks=2)
    ens = Ensemble(model_factory(s, Naive()), [0], torch.device("cpu"))
    ctx = make_ctx(s, [0], ens=ens)
    ex._save_ckpt(tmp_path, 0, ens, Naive(), ctx)

    def failing_save(obj, f, *a, **k):
        open(f, "wb").write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(ex.torch, "save", failing_save)
    with pytest.raises(OSError):
        ex._save_ckpt(tmp_path, 1, ens, Naive(), ctx)
    monkeypatch.undo()
    assert ex._load_ckpt(tmp_path)["task"] == 0 and sorted(p.name for p in tmp_path.iterdir()) == ["task_0.pt"]


def test_incomplete_expert_cache_is_rebuilt(tmp_path):
    from contilab.consolidate.experts import _expert_key, train_experts
    from contilab.data.cache import cache_dir, key_hash
    from helpers import smoke_cfg

    s = build_synthetic(n_tasks=2)
    cfg = smoke_cfg()
    d = cache_dir() / "experts" / key_hash(_expert_key(s, cfg, [0]))
    d.mkdir(parents=True)
    (d / "experts.safetensors").write_bytes(b"truncated")  # crash before meta.json was written
    ex = train_experts(s, cfg, [0])
    assert len(ex.tuned) == 2 and json.loads((d / "meta.json").read_text())["flops"] > 0


# M14 --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["consolidate:ta", "consolidate:distill"])
def test_consolidation_divergence_is_marked(tmp_path, method):
    out = run_experiment(tiny_experiment([method], seeds=[0], cli={"optim.lr": 1e4}), tmp_path)
    assert statuses(out)[method.replace(":", "_")] == {"diverged"}


def test_distill_divergence_is_marked_when_experts_are_fine(tmp_path, monkeypatch):
    import contilab.consolidate.distill as distill_mod

    real = distill_mod.DistillStrategy.loss

    def exploding(self, p, s, batch, replay, f):
        loss, z = real(self, p, s, batch, replay, f)
        return loss * float("nan"), z

    monkeypatch.setattr(distill_mod.DistillStrategy, "loss", exploding)
    out = run_experiment(tiny_experiment(["consolidate:distill"], seeds=[0]), tmp_path)
    assert statuses(out)["consolidate_distill"] == {"diverged"}
