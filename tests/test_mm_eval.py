import numpy as np, pytest
from idrdesign.mm_eval import upper_tri, pearson_r, diff_raw, diff_norm, seq_identity

def test_helpers_on_known():
    A = np.array([[0., 2., 4.], [2., 0., 6.], [4., 6., 0.]])
    B = A.copy()
    assert pearson_r(A, B) == pytest.approx(1.0)
    assert diff_raw(A, B) == pytest.approx(0.0)
    C = A + 1.0  # constant offset on off-diagonal -> r still 1, diff_raw=1
    assert pearson_r(A, C) == pytest.approx(1.0)
    assert diff_raw(A, C) == pytest.approx(1.0)
    # diff_norm normalizes by the SECOND arg (the WT reference): mean|upper_tri(C)| = mean([3,5,7]) = 5
    assert diff_norm(A, C) == pytest.approx(1.0 / 5.0)
    assert diff_norm(A, C) == pytest.approx(diff_raw(A, C) / np.mean(np.abs(upper_tri(C))))

def test_seq_identity():
    assert seq_identity("AAAA", "AAAA") == pytest.approx(1.0)
    assert seq_identity("AAAA", "AAAB") == pytest.approx(0.75)

from idrdesign.mm_eval import greedy_diverse_select

def test_greedy_diverse_select_skips_too_similar():
    seqs = {0: "AAAAAAAAAA", 1: "AAAAAAAAAB",  # 0.9 identical to 0
            2: "BBBBBBBBBB", 3: "CCCCCCCCCC"}
    # order best-first = [0,1,2,3]; cap 0.25 must skip 1 (too close to 0)
    picked = greedy_diverse_select([0, 1, 2, 3], seqs, cap=0.25, n=10)
    assert picked == [0, 2, 3]

def test_greedy_diverse_select_respects_n():
    seqs = {i: ("ACDEFGHIKL"[i % 10]) * 10 for i in range(5)}
    picked = greedy_diverse_select([0, 1, 2, 3, 4], seqs, cap=0.25, n=2)
    assert picked == [0, 1]
