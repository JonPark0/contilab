import torch

from contilab.engine.device import peak_memory_bytes, select_device, select_precision
from contilab.engine.ensemble import Ensemble
from contilab.models.hf import HFClassifier
from contilab.models.mlp import MLP

CPU = torch.device("cpu")


def fac(s):
    torch.manual_seed(s)
    return MLP(8, 4, hidden=16)


def test_members_match_individual_modules():
    ens = Ensemble(fac, [1, 2, 3], CPU)
    x = torch.randn(5, 8)
    out = ens.apply(ens.params, x, shared_input=True)
    for i, s in enumerate([1, 2, 3]):
        assert torch.allclose(out[i], fac(s)(x), atol=1e-6)


def test_e1_fast_path_and_per_member_inputs():
    e1 = Ensemble(fac, [7], CPU)
    x = torch.randn(1, 5, 8)
    assert torch.allclose(e1.apply(e1.params, x)[0], fac(7)(x[0]), atol=1e-6)
    e2 = Ensemble(fac, [7, 8], CPU)
    xs = torch.randn(2, 5, 8)
    assert e2.apply(e2.params, xs).shape == (2, 5, 4)
    acts = e2.apply(e2.params, xs, mode="acts")
    assert len(acts) == 2 and acts[0].shape == (2, 5, 16)


def test_member_state_roundtrip_and_module():
    ens = Ensemble(fac, [1, 2], CPU)
    sd = ens.member_state(1)
    ens.load_member_state(0, sd)
    x = torch.randn(3, 8)
    out = ens.apply(ens.params, x, shared_input=True)
    assert torch.allclose(out[0], out[1])
    assert torch.allclose(ens.member_module(1)(x), fac(2)(x), atol=1e-6)


def test_hf_model_with_attention_mask():
    ens = Ensemble(lambda s: (torch.manual_seed(s), HFClassifier("tiny-gptneox", 4))[1], [0], CPU)
    ids = torch.randint(1, 100, (1, 2, 6))
    mask = torch.ones(1, 2, 6, dtype=torch.long)
    assert ens.apply(ens.params, ids, mask).shape == (1, 2, 4)


def test_select_device_cpu():
    assert select_device("cpu").type == "cpu"
    assert select_precision("fp32", CPU)[0] is None
    assert select_precision("auto", CPU)[1]
    assert peak_memory_bytes(CPU) > 0
