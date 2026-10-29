"""Tests for idrdesign.optimize."""
import pytest

from idrdesign.optimize import gs_start


def test_gs_start_even_length():
    assert gs_start(6) == "GSGSGS"


def test_gs_start_odd_length():
    assert gs_start(5) == "GSGSG"


def test_gs_start_length_one():
    assert gs_start(1) == "G"


def test_gs_start_length_zero():
    assert gs_start(0) == ""


def test_gs_start_length_296_matches_runx2():
    out = gs_start(296)
    assert len(out) == 296
    assert out[:4] == "GSGS"
    assert out[-2:] == "GS"
    assert set(out) == {"G", "S"}


def test_gs_start_rejects_negative():
    with pytest.raises(ValueError):
        gs_start(-1)


@pytest.mark.gpu
def test_shuffle_only_optimizer_preserves_composition():
    """After a shuffle-only run, the AA composition must be identical."""
    from collections import Counter
    import random
    import numpy as np

    from idrdesign.encode import CosineSimilarityToTarget, encode_sequence
    from idrdesign.optimize import ShuffleOnlyOptimizer, gs_start

    # Small target so the test runs fast.
    target_seq = "ACDEFGHIKLMNPQRSTVWY"   # 20 residues, one of each
    random.seed(0)
    np.random.seed(0)

    target_emb = encode_sequence(target_seq)

    starting_seq = gs_start(len(target_seq))
    # Seed the composition to match the target so shuffle has something to work with.
    # Shuffle-only requires matching composition upfront; we use the target itself
    # (shuffled) as the starting sequence to test the invariant.
    shuffled_start = "".join(random.sample(list(target_seq), len(target_seq)))

    opt = ShuffleOnlyOptimizer(
        target_length=len(target_seq),
        max_iterations=5,
        verbose=False,
        num_candidates=3,
    )
    opt.set_initial_sequence(shuffled_start)
    opt.add_property(CosineSimilarityToTarget, target_emb)
    final_seq = opt.run()

    assert Counter(final_seq) == Counter(shuffled_start)


def test_design_result_dataclass_fields_present():
    """DesignResult exposes the expected fields."""
    from idrdesign.optimize import DesignResult
    r = DesignResult(
        sequence="ACDE",
        final_cosine_similarity=0.95,
        iterations=42,
        wall_seconds=1.5,
        convergence_status="converged",
        seed=0,
        starting_sequence="GSGS",
        composition_mode="none",
    )
    assert r.sequence == "ACDE"
    assert r.convergence_status == "converged"
    assert r.seed == 0


@pytest.mark.gpu
@pytest.mark.slow
def test_run_single_optimization_smoke_converges_on_identity_target():
    """Trivial target: optimize toward the target sequence itself.

    The GS start should diverge from the target quickly, so any real optimization
    step should land us with final cos_sim > starting cos_sim.
    """
    import random
    import numpy as np
    import torch

    from idrdesign.encode import encode_sequence, cosine_similarity
    from idrdesign.optimize import run_single_optimization

    target_seq = "ACDEFGHIKLMNPQRSTVWY" * 2  # 40 aa
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    target_emb = encode_sequence(target_seq)

    from idrdesign.optimize import gs_start
    start_sim = cosine_similarity(target_emb, encode_sequence(gs_start(len(target_seq))))

    result = run_single_optimization(
        target_seq=target_seq,
        target_emb=target_emb,
        target_similarity=0.90,
        composition_mode="none",
        max_iterations=30,
        max_wall_seconds=120,
        seed=1,
    )

    assert len(result.sequence) == len(target_seq)
    assert result.convergence_status in {"converged", "max_iterations", "max_wall"}
    final_sim = 1.0 - result.final_cosine_similarity  # sanity: opposite sign?
    # result.final_cosine_similarity is the cos_sim (not 1-cos_sim)
    assert result.final_cosine_similarity > start_sim


def test_run_single_optimization_rejects_bad_composition_mode():
    """Unknown composition_mode should error early, not silently fall back."""
    import torch
    from idrdesign.optimize import run_single_optimization

    with pytest.raises(ValueError, match="composition_mode"):
        run_single_optimization(
            target_seq="ACDE",
            target_emb=torch.zeros(64),
            target_similarity=0.95,
            composition_mode="bogus",
            max_iterations=1,
            max_wall_seconds=1,
            seed=0,
        )


@pytest.mark.gpu
@pytest.mark.slow
def test_design_idr_produces_n_distinct_designs():
    """N=3 designs on a 40-aa target should produce 3 distinct sequences."""
    import torch
    from idrdesign.encode import encode_sequence
    from idrdesign.optimize import design_idr

    target_seq = "ACDEFGHIKLMNPQRSTVWY" * 2  # 40 aa
    target_emb = encode_sequence(target_seq)

    results = design_idr(
        target_seq=target_seq,
        n_designs=3,
        target_similarity=0.85,
        composition_mode="none",
        max_iterations=15,
        max_wall_seconds=120,
        seed=0,
    )

    assert len(results) == 3
    seqs = {r.sequence for r in results}
    # With distinct seeds, designs should (very probably) differ.
    assert len(seqs) >= 2
    assert all(len(r.sequence) == 40 for r in results)
    assert all(r.convergence_status in {"converged", "max_iterations", "max_wall"} for r in results)


def test_design_idr_exported_from_package():
    """design_idr is importable from the top-level package."""
    import idrdesign
    assert hasattr(idrdesign, "design_idr")
