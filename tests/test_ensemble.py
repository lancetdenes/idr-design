"""Tests for idrdesign.ensemble."""
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.slow
def test_generate_ensembles_returns_named_ensembles():
    from idrdesign.ensemble import generate_ensembles

    seqs = {"target": "G" * 30, "design_01": "S" * 30}
    ensembles = generate_ensembles(seqs, conformations=10)
    assert set(ensembles.keys()) == {"target", "design_01"}

    # Returned objects expose radius_of_gyration method.
    for name, ens in ensembles.items():
        rg = ens.radius_of_gyration()
        assert hasattr(rg, "__len__")
        assert len(rg) == 10


def test_generate_ensembles_rejects_too_long_sequence():
    """Sequences longer than STARLING's MAX_SEQUENCE_LENGTH should be rejected."""
    from idrdesign.ensemble import validate_sequence_length, MAX_SEQUENCE_LENGTH

    with pytest.raises(ValueError, match="exceeds MAX_SEQUENCE_LENGTH"):
        validate_sequence_length("A" * (MAX_SEQUENCE_LENGTH + 1))


def test_validate_sequence_length_ok_at_limit():
    from idrdesign.ensemble import validate_sequence_length, MAX_SEQUENCE_LENGTH
    validate_sequence_length("A" * MAX_SEQUENCE_LENGTH)  # no raise


def test_validate_sequence_length_ok_for_runx2():
    from idrdesign.ensemble import validate_sequence_length
    validate_sequence_length("A" * 296)  # RUNX2 WT length


def _fake_coords(n_conf: int, n_res: int, seed: int = 0) -> np.ndarray:
    """Fake (n_conf, n_res, 3) coords with known structure for metric sanity."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_conf, n_res, 3)).astype(np.float32)


def test_compute_rg_from_coords_shape():
    from idrdesign.ensemble import compute_rg_from_coords
    coords = _fake_coords(5, 20)
    rg = compute_rg_from_coords(coords)
    assert rg.shape == (5,)
    assert np.all(rg > 0)


def test_compute_rg_from_coords_collapsed_sequence_is_zero():
    """All atoms at origin -> Rg = 0."""
    from idrdesign.ensemble import compute_rg_from_coords
    coords = np.zeros((3, 10, 3), dtype=np.float32)
    rg = compute_rg_from_coords(coords)
    assert np.allclose(rg, 0.0)


def test_compute_end_to_end_from_coords():
    from idrdesign.ensemble import compute_end_to_end_from_coords
    # Residue 0 at origin, last residue at (3, 4, 0) → distance 5
    coords = np.zeros((2, 5, 3), dtype=np.float32)
    coords[:, -1] = np.array([3.0, 4.0, 0.0])
    ree = compute_end_to_end_from_coords(coords)
    assert np.allclose(ree, 5.0)


def test_compute_asphericity_spherical_is_zero_ish():
    """A uniform spherical distribution should give asphericity near 0."""
    from idrdesign.ensemble import compute_asphericity_from_coords
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((1, 5000, 3)).astype(np.float32)
    a = compute_asphericity_from_coords(coords)
    assert abs(a[0]) < 0.1


@pytest.mark.gpu
@pytest.mark.slow
def test_extract_metrics_real_ensemble_shapes():
    """Sanity: metrics extracted from a real 10-conformer STARLING ensemble have the right shapes."""
    from idrdesign.ensemble import generate_ensembles, extract_metrics

    seqs = {"short": "GSGSGSGSGSGSGSGSGSGS"}
    ensembles = generate_ensembles(seqs, conformations=10)
    metrics = extract_metrics(ensembles["short"])
    assert metrics["rg"].shape == (10,)
    assert metrics["end_to_end"].shape == (10,)
    L = len(seqs["short"])
    assert metrics["mean_distance_map"].shape == (L, L)
    assert metrics["contact_map"].shape == (L, L)
    assert metrics["asphericity"].shape == (10,)
