"""Unit tests for idrdesign.compare pure helpers (CPU-only, no GPU)."""
import numpy as np
import pytest

from idrdesign import compare

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def test_aa_frequency_vector_sums_to_one_and_counts_correctly():
    v = compare.aa_frequency_vector("AAAC")  # 3 A, 1 C
    assert v.shape == (20,)
    assert np.isclose(v.sum(), 1.0)
    assert np.isclose(v[AA20.index("A")], 0.75)
    assert np.isclose(v[AA20.index("C")], 0.25)


def test_composition_l1_zero_for_identical():
    assert compare.composition_l1("ACDEFG", "ACDEFG") == 0.0


def test_composition_l1_zero_for_a_permutation():
    # The exact-composition control: a shuffle has identical composition.
    assert np.isclose(compare.composition_l1("ACDEFGACDEFG", "GFEDCAGFEDCA"), 0.0)


def test_composition_l1_detects_single_substitution():
    # "AAAA" vs "AAAC": freq A 1.0 -> 0.75 (|Δ|=0.25); C 0 -> 0.25 (|Δ|=0.25) => 0.5
    assert np.isclose(compare.composition_l1("AAAA", "AAAC"), 0.5)


def test_composition_jsd_zero_for_identical():
    assert np.isclose(compare.composition_jsd("ACDACD", "ACDACD"), 0.0)


def test_composition_jsd_zero_for_permutation():
    assert np.isclose(compare.composition_jsd("ACDACD", "DCADCA"), 0.0)


def test_composition_jsd_one_for_disjoint_alphabets():
    # Fully disjoint compositions -> max base-2 JSD = 1 bit.
    assert np.isclose(compare.composition_jsd("AAAA", "CCCC"), 1.0)


def test_mean_abs_error_zero_for_identical():
    m = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert compare.mean_abs_error(m, m) == 0.0


def test_mean_abs_error_known_value():
    a = np.array([[1.0, 2.0]])
    b = np.array([[2.0, 4.0]])  # |1-2|=1, |2-4|=2 -> mean 1.5
    assert np.isclose(compare.mean_abs_error(a, b), 1.5)


def test_pearson_r_one_for_identical_maps():
    rng = np.random.default_rng(0)
    m = rng.random((5, 5))
    assert np.isclose(compare.pearson_r(m, m), 1.0)


def test_pearson_r_minus_one_for_anticorrelated():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(compare.pearson_r(a, -a), -1.0)


def _toy_ens(rg, ree, dmap, cmap):
    return {
        "rg": np.array(rg, dtype=float),
        "ree": np.array(ree, dtype=float),
        "mean_dmap": np.array(dmap, dtype=float),
        "mean_cmap": np.array(cmap, dtype=float),
    }


def test_compute_comparison_rows_wiring():
    """Sanity-check the combine glue (delta signs, MAE, composition contrast)
    on synthetic in-memory ensembles — guards the GPU run from wiring bugs."""
    target_seq = "ACDEFGACDEFG"  # A2 C2 D2 E2 F2 G2
    dmap_t = np.array([[0.0, 1.0], [1.0, 0.0]])
    cmap_t = np.array([[1.0, 0.5], [0.5, 1.0]])
    target = {"ens": _toy_ens([10, 12], [20, 22], dmap_t, cmap_t), "asphericity_mean": 0.5}
    designs = [
        {
            "name": "free_00", "group": "free", "sequence": "AAAAAAAAAAAA",
            "composition_mode": "none", "final_cosine_similarity": 0.96,
            "convergence_status": "converged", "iterations": 100,
            "asphericity_mean": 0.55, "asphericity_std": 0.1,
            "ens": _toy_ens([11, 13], [20, 22], [[0.0, 2.0], [2.0, 0.0]], cmap_t),
        },
        {
            "name": "exact_00", "group": "exact", "sequence": "GFEDCAGFEDCA",
            "composition_mode": "exact", "final_cosine_similarity": 0.97,
            "convergence_status": "converged", "iterations": 200,
            "asphericity_mean": 0.5, "asphericity_std": 0.1,
            "ens": _toy_ens([10, 12], [20, 22], dmap_t, cmap_t),
        },
    ]
    free, exact = compare.compute_comparison_rows(target, designs, target_seq)

    assert np.isclose(free["delta_rg_mean"], 1.0)   # 12 vs 11
    assert np.isclose(exact["delta_rg_mean"], 0.0)
    assert np.isclose(free["distance_map_mae"], 0.5)  # |Δ|=[[0,1],[1,0]] -> 0.5
    assert np.isclose(exact["distance_map_mae"], 0.0)
    assert np.isclose(exact["contact_map_pearson_r"], 1.0)
    # exact_00 is a permutation of WT -> composition identical; free is all-A -> diverged
    assert np.isclose(exact["composition_l1_vs_wt"], 0.0)
    assert free["composition_l1_vs_wt"] > 0.0
