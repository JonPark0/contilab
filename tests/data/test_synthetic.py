import torch

from contilab.data.splits import train_val_split
from contilab.data.synthetic import build_synthetic, build_synthetic_text
from contilab.data.transforms import make_joint_stream
from contilab.core.types import TensorSplit


def test_joint_stream():
    j = make_joint_stream(build_synthetic())
    assert len(j.tasks) == 1 and len(j.tasks[0].classes) == 10 and j.name == "synthetic-joint"
    assert len(j.tasks[0].test) == 500


def test_synthetic_deterministic_and_class_layout():
    a, b = build_synthetic(seed=3), build_synthetic(seed=3)
    assert torch.equal(a.tasks[1].train.x, b.tasks[1].train.x)
    assert a.tasks[1].classes.tolist() == [2, 3] and a.num_classes == 10
    d = build_synthetic(scenario="domain")
    assert d.num_classes == 2 and all(t.classes.tolist() == [0, 1] for t in d.tasks)


def test_synthetic_text_padding_and_lm_eval():
    s = build_synthetic_text()
    tr = s.tasks[0].train
    assert tr.x.shape[1] == 16 and tr.attn_mask is not None and s.modality == "text"
    assert ((tr.x == 0) == (tr.attn_mask == 0)).all()  # pad id 0 exactly where mask is 0
    assert tr.attn_mask[:, 0].all() and s.lm_eval.x.shape == (64, 16)


def test_train_val_split_fraction_and_disjoint():
    sp = TensorSplit(torch.arange(100.0).view(100, 1), torch.arange(100))
    tr, va = train_val_split(sp, 0.1, seed=0)
    assert len(tr) == 90 and len(va) == 10 and not set(tr.y.tolist()) & set(va.y.tolist())
