import numpy as np
import pytest

from contilab.metrics.stats import friedman_nemenyi, low_power_warning, nemenyi_q, paired_test, summarize


def test_nemenyi_q_matches_demsar_table():
    assert nemenyi_q(3) == pytest.approx(2.343, abs=1e-3) and nemenyi_q(10) == pytest.approx(3.164, abs=1e-3)


def test_permutation_exact_small_n():
    r = paired_test([1, 2, 3], [0, 0, 0])
    assert r["method"] == "permutation" and r["p"] == pytest.approx(0.25)


def test_wilcoxon_for_six_or_more():
    r = paired_test([1, 2, 3, 4, 5, 6], [0] * 6)
    assert r["method"] == "wilcoxon" and r["p"] < 0.05


def test_paired_identical_is_not_significant():
    assert paired_test([0.5] * 4, [0.5] * 4)["p"] == pytest.approx(1.0)


def test_friedman_ranks():
    out = friedman_nemenyi(np.array([[0.9, 0.5, 0.1], [0.8, 0.6, 0.2], [0.7, 0.4, 0.3]]))
    assert out["avg_ranks"].tolist() == [1.0, 2.0, 3.0] and out["friedman_p"] < 0.1
    assert out["cd"] == pytest.approx(2.343 * np.sqrt(3 * 4 / (6 * 3)), abs=1e-3)


def test_friedman_lower_is_better_and_small_n():
    out = friedman_nemenyi(np.array([[0.1, 0.5]]), higher_is_better=False)
    assert out["avg_ranks"].tolist() == [1.0, 2.0] and out["friedman_p"] is None


def test_summarize():
    s = summarize([0.5])
    assert s["std"] == 0 and s["n"] == 1 and s["ci_low"] == s["ci_high"] == 0.5
    s = summarize([0.1, 0.2, 0.3, 0.4])
    assert s["mean"] == pytest.approx(0.25) and s["ci_low"] <= 0.25 <= s["ci_high"]


def test_low_power_warning():
    assert "3" in low_power_warning(3) and low_power_warning(5) is None
