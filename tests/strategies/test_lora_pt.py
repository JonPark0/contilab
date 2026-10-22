import torch

from contilab.data.synthetic import build_synthetic
from contilab.models.lora import _LoRABase
from contilab.strategies.lora_per_task import LoRAPerTask
from helpers import per_task_final_acc, train_on


def test_base_frozen_and_only_active_adapter_trains():
    s = build_synthetic(scenario="task")
    before, _ = train_on(s, LoRAPerTask(r=2, alpha=4), [0], tasks=0)
    ens, _ = train_on(s, LoRAPerTask(r=2, alpha=4), [0], tasks=2)
    for k, v in ens.params.items():
        if "lora" not in k and not k.startswith("head"):
            assert torch.equal(v, before.params[k]), k
        if "lora_B.t4" in k:
            assert (v == 0).all()
        if "lora_B.t1" in k:
            assert (v != 0).any()


def test_task_il_adapter_selection_beats_chance():
    s = build_synthetic(scenario="task")
    assert min(per_task_final_acc(s, LoRAPerTask(r=4, alpha=8), [0], epochs=3)) > 0.6


def test_class_il_eval_merges_adapters_and_restores_training_mode():
    s = build_synthetic(scenario="class", n_tasks=3)
    strat = LoRAPerTask(r=2, alpha=4)
    ens, ctx = train_on(s, strat, [0])
    strat.eval_setup(ctx, 0)
    mods = [m for m in ens.template.modules() if isinstance(m, _LoRABase)]
    scale = ens.buffers[f"{strat.lora_modules[0]}.member_scale"]
    assert mods[0].active == [0, 1, 2] and scale.tolist() == strat.meta["lam"]
    strat.eval_setup(ctx, None)
    assert mods[0].active == [2] and scale.tolist() == [1.0]
