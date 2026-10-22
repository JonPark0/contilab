import torch

from contilab.core.scenario import sample_mask
from helpers import make_stream


def test_class_table_and_masks():
    s = make_stream(classes=[[0, 1], [2, 3], [4, 5]], scenario="class", C=6)
    assert s.class_table().tolist()[1] == [False, False, True, True, False, False]
    m = sample_mask(s, torch.tensor([0, 2]), current_task=1)
    assert m.shape == (2, 6) and m[0].tolist() == [True] * 4 + [False] * 2 and torch.equal(m[0], m[1])
    s.scenario = "task"
    m = sample_mask(s, torch.tensor([0, 2]), current_task=2)
    assert m[0].tolist() == [True, True, False, False, False, False] and m[1, 4:].all()
    s.scenario = "domain"
    assert sample_mask(s, torch.tensor([[0, 1]]), 0).all()
