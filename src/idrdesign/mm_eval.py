import numpy as np
from scipy.stats import pearsonr

def upper_tri(M):
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]

def pearson_r(A, B):
    return float(pearsonr(upper_tri(A), upper_tri(B))[0])

def diff_raw(A, B):
    return float(np.mean(np.abs(upper_tri(A) - upper_tri(B))))

def diff_norm(A, B):
    """Mean abs difference normalized by the magnitude of B (the WT reference map)."""
    return float(diff_raw(A, B) / np.mean(np.abs(upper_tri(B))))

def seq_identity(a, b):
    return float(np.mean([x == y for x, y in zip(a, b)]))

def greedy_diverse_select(order, seqs, cap=0.25, n=10):
    """Pick up to n indices from `order` (candidate indices ranked best-first),
    keeping an index only if its pairwise sequence identity to every already
    picked index is <= cap. `seqs` maps index -> sequence string."""
    picked = []
    for i in order:
        if all(seq_identity(seqs[i], seqs[j]) <= cap for j in picked):
            picked.append(i)
            if len(picked) >= n:
                break
    return picked
