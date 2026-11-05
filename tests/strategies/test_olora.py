import torch

from contilab.data.synthetic import build_synthetic
from contilab.models.lora import _LoRABase
from contilab.strategies.olora import OLoRA
from helpers import train_on


def _A(ens, strat, t, m=None):
    m = m or strat.lora_modules[0]
    return ens.params[f"{m}.lora_A.t{t}"][0]


def test_backbone_and_previous_adapters_stay_frozen():
    s = build_synthetic(n_tasks=3)
    one, _ = train_on(s, OLoRA(r=2, alpha=4), [0], tasks=1)
    strat = OLoRA(r=2, alpha=4)
    ens, _ = train_on(s, strat, [0], tasks=2)
    for k, v in ens.params.items():
        if "lora" not in k and not k.startswith("head"):
            assert torch.equal(v, one.params[k]), k
        if k.endswith(".t0"):
            assert torch.equal(v, one.params[k]), k
        if "lora_B.t1" in k:
            assert (v != 0).any(), k
        if "lora_B.t2" in k:
            assert (v == 0).all(), k


def test_orthogonality_penalty_shrinks_the_overlap_with_earlier_adapters():
    s = build_synthetic(n_tasks=2)

    def overlap(lam):
        strat = OLoRA(r=2, alpha=4, lam_orth=lam)
        ens, _ = train_on(s, strat, [0], epochs=10)
        return sum((ens.params[f"{m}.lora_A.t0"][0] @ ens.params[f"{m}.lora_A.t1"][0].T).abs().sum().item()
                   for m in strat.lora_modules)

    assert overlap(0.5) < 0.5 * overlap(0.0)


def test_penalty_is_zero_on_the_first_task():
    s = build_synthetic(n_tasks=2)
    strat = OLoRA(r=2, alpha=4)
    ens, _ = train_on(s, strat, [0], tasks=0)
    strat._t = 0
    p = {k: v[0] for k, v in ens.params.items()}
    assert strat.penalty(p).item() == 0.0
    strat._t = 1
    assert strat.penalty(p).item() > 0.0


def test_inference_uses_every_seen_adapter():
    s = build_synthetic(n_tasks=3, scenario="task")
    strat = OLoRA(r=2, alpha=4)
    ens, ctx = train_on(s, strat, [0])
    mods = [m for m in ens.template.modules() if isinstance(m, _LoRABase)]
    strat.eval_setup(ctx, 0)
    assert mods[0].active == [0, 1, 2]
    strat.eval_setup(ctx, None)
    assert mods[0].active == [0, 1, 2]
