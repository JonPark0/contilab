import torch

from contilab.consolidate.heads import assemble_head, body_names
from contilab.consolidate.merge import apply_tv, merge_ta, merge_ties, select_lambda, task_vector


def test_single_expert_ta_is_expert():
    base = {"w": torch.zeros(1, 4)}
    e = {"w": torch.tensor([[1.0, -2.0, 3.0, 0.0]])}
    assert torch.equal(apply_tv(base, merge_ta([task_vector(base, e, ["w"])]), 1.0)["w"], e["w"])


def test_ties_full_k_single_expert_equals_ta():
    base = {"w": torch.zeros(1, 4)}
    tv = [task_vector(base, {"w": torch.tensor([[1.0, -2.0, 3.0, 0.5]])}, ["w"])]
    assert torch.equal(merge_ties(tv, k=1.0)["w"], merge_ta(tv)["w"])


def test_ties_sign_election_and_disjoint_mean():
    a = {"w": torch.tensor([[2.0, 1.0, -1.0]])}
    b = {"w": torch.tensor([[4.0, -3.0, -1.0]])}
    assert merge_ties([a, b], k=1.0)["w"].tolist() == [[3.0, -3.0, -1.0]]


def test_ties_trim_per_member_across_tensors():
    t = {"w": torch.tensor([[1.0, 5.0, 2.0, 4.0, 3.0]])}
    assert merge_ties([t], k=0.4)["w"].tolist() == [[0.0, 5.0, 0.0, 4.0, 0.0]]
    t2 = {"a": torch.tensor([[1.0, 9.0], [5.0, 1.0]]), "b": torch.tensor([[8.0], [1.0]])}
    out = merge_ties([t2], k=0.34)
    assert out["a"].tolist() == [[0.0, 9.0], [5.0, 0.0]] and out["b"].tolist() == [[0.0], [0.0]]


def test_apply_tv_per_member_lambda():
    base = {"w": torch.zeros(2, 1)}
    out = apply_tv(base, {"w": torch.ones(2, 1)}, torch.tensor([0.5, 2.0]))
    assert out["w"].view(-1).tolist() == [0.5, 2.0]


def test_select_lambda_per_member():
    def score(p):
        return -(p["w"][:, 0] - torch.tensor([0.3, 0.8])).abs()

    lam, p = select_lambda(lambda l: {"w": torch.full((2, 1), l)}, score, [round(0.1 * i, 1) for i in range(1, 11)])
    assert torch.allclose(lam, torch.tensor([0.3, 0.8])) and torch.allclose(p["w"][:, 0], lam)


def test_assemble_head_rows_and_domain_mean():
    e0 = {"head.weight": torch.zeros(1, 4, 2), "head.bias": torch.zeros(1, 4)}
    e1 = {"head.weight": torch.ones(1, 4, 2), "head.bias": torch.ones(1, 4)}
    tbl = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.bool)
    h = assemble_head([e0, e1], tbl, 1, "class")
    assert h["head.weight"][0, :2].sum() == 0 and h["head.weight"][0, 2:].sum() == 4 and h["head.bias"][0].tolist() == [0, 0, 1, 1]
    d = assemble_head([e0, e1], torch.ones(2, 4, dtype=torch.bool), 1, "domain")
    assert torch.allclose(d["head.weight"], torch.full((1, 4, 2), 0.5))
    assert body_names({"head.weight": 0, "layers.0.weight": 0}) == ["layers.0.weight"]
