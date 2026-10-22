import torch

from contilab.core.types import TensorSplit


def test_tensorsplit_subset_len():
    sp = TensorSplit(torch.arange(10.0).view(10, 1), torch.arange(10))
    assert len(sp.subset(torch.tensor([1, 3]))) == 2 and sp.subset(torch.tensor([3])).y.item() == 3


def test_tensorsplit_subset_keeps_attn_mask():
    sp = TensorSplit(torch.zeros(4, 3, dtype=torch.long), torch.arange(4), torch.ones(4, 3, dtype=torch.long))
    assert sp.subset(torch.tensor([0, 2])).attn_mask.shape == (2, 3)
