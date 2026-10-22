import copy

import torch
from torch import nn

from contilab.models.hf import HFClassifier
from contilab.models.lora import LoRAConv2d, inject_lora, lora_param_names, lora_task_vector, set_active
from contilab.models.mlp import MLP


def test_lora_starts_as_identity_and_merges():
    torch.manual_seed(0)
    m = MLP(8, 4, hidden=16)
    ref = copy.deepcopy(m)
    x = torch.randn(3, 8)
    names = inject_lora(m, None, r=2, alpha=4, n_adapters=2, modality="vision")
    assert names == ["layers.0", "layers.1"]
    set_active(m, [0])
    assert torch.allclose(m(x), ref(x))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B.t0" in n:
                p.normal_()
    p = {k: v.unsqueeze(0) for k, v in m.state_dict().items()}
    tv = lora_task_vector(p, 0, names, r=2, alpha=4)
    merged = copy.deepcopy(m)
    set_active(merged, [])
    with torch.no_grad():
        for k, d in tv.items():
            merged.get_parameter(k).add_(d[0])
    assert torch.allclose(m(x), merged(x), atol=1e-5)


def test_lora_conv_shape_and_merge():
    c = LoRAConv2d(nn.Conv2d(3, 5, 3, padding=1), r=2, alpha=4, n_adapters=1)
    x = torch.randn(1, 3, 8, 8)
    assert c(x).shape == (1, 5, 8, 8)
    with torch.no_grad():
        c.lora_B["t0"].normal_()
    c.active = [0]
    p = {k: v.unsqueeze(0) for k, v in c.state_dict().items()}
    dw = lora_task_vector(p, 0, [""], r=2, alpha=4)["base.weight"][0]
    plain = nn.Conv2d(3, 5, 3, padding=1)
    plain.load_state_dict({"weight": c.base.weight + dw, "bias": c.base.bias})
    assert torch.allclose(c(x), plain(x), atol=1e-5)


def test_text_targets_and_param_names():
    m = HFClassifier("tiny-gptneox", 4)
    names = inject_lora(m, None, r=2, alpha=4, n_adapters=3, modality="text")
    assert names and all("query_key_value" in n for n in names)
    assert all("lora_" in n and ".t1" in n for n in lora_param_names(m, 1))
    assert len(lora_param_names(m, 1)) == 2 * len(names)
