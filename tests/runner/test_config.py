import copy
from dataclasses import replace
from pathlib import Path

import pytest

from contilab.runner.config import expand, group_key, load_experiment, run_id

SMOKE = Path("configs/experiments/smoke_all.yaml")
CPU = Path("configs/experiments/cpu_main.yaml")


def test_merge_order_and_expand():
    exp = load_experiment(SMOKE, ["train.epochs=2"])
    runs = expand(exp)
    assert len(runs) == 3 * 11 * 2
    r = next(r for r in runs if r.preset == "syn_text" and r.method.name == "ewc")
    assert r.train.epochs == 2 and r.model.hf_name == "tiny-gptneox" and r.metrics.tier == "full"
    assert r.experiment == "smoke_all"


def test_matrix_axis_override_from_cli():
    runs = expand(load_experiment(SMOKE, ["matrix.method.name=[naive, er]", "matrix.seed=[3]"]))
    assert {r.method.name for r in runs} == {"naive", "er"} and {r.seed for r in runs} == {3}


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown config key: train.epoch"):
        load_experiment(SMOKE, ["train.epoch=2"])


def test_run_id_stable_and_seed_sensitive():
    a, b = expand(load_experiment(SMOKE, []))[:2]
    assert run_id(a) == run_id(copy.deepcopy(a)) and run_id(a) != run_id(b)
    assert group_key(a) == group_key(replace(a, seed=99)) and run_id(a).startswith(f"{a.preset}-")


def test_cpu_profile_presets():
    runs = expand(load_experiment(CPU, []))
    assert len(runs) == 5 * 11 * 3
    full = {r.preset for r in runs if r.metrics.tier == "full"}
    assert full == {"split_mnist_class"}
    t = next(r for r in runs if r.preset == "text5_class")
    assert t.model.hf_name == "EleutherAI/pythia-70m" and t.optim.name == "adamw" and t.train.batch_size == 16
    c = next(r for r in runs if r.preset == "split_cifar10_class")
    assert c.model.name == "resnet18r" and c.train.epochs == 3


def test_sweep_expands_grid(tmp_path):
    p = tmp_path / "experiments" / "s.yaml"
    p.parent.mkdir()
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "p.yaml").write_text("defaults: {}\npresets: {a: {benchmark: {name: synthetic}}}\n")
    p.write_text("name: s\nprofile: p\nmatrix: {preset: [a], method.name: [ewc]}\nsweep: {method.params.lam: [1, 10]}\n")
    runs = expand(load_experiment(p, []))
    assert sorted(r.method.params["lam"] for r in runs) == [1, 10]
