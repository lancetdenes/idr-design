"""Regression tests for the two fixes that are easy to silently lose.

Both guard against a failure mode that leaves the code *looking* correct:
a seed that is accepted but does nothing, and an inference path that quietly
keeps training-time behaviour.
"""
import pytest

from idrdesign.seeding import seed_everything, seed_libc


def test_libc_is_reachable():
    """If this fails the search is unseeded, whatever seed is passed."""
    assert seed_libc(0) is True


def test_seed_everything_reports_libc():
    assert seed_everything(0) is True


def test_goose_shuffler_is_reproducible_after_seeding():
    """The point of seeding libc: GOOSE's candidate generator draws from rand().

    random.seed() and numpy.random.seed() alone do NOT make this reproducible,
    which is the bug this guards against.
    """
    from goose.backend.fast_mutations import cython_shuffle_sequence

    seq = "HRQKLDDSKPSLFSDRLSDLGRIPHPSMRVGVPPQNPRPSLNSAPSPFNPQGQSQITDPRQ"

    def draw(seed):
        seed_everything(seed)
        return [
            cython_shuffle_sequence(
                seq, window_size=15, global_shuffle_probability=0.4
            )
            for _ in range(4)
        ]

    assert draw(42) == draw(42), "same seed must reproduce the same shuffles"
    assert draw(42) != draw(43), "different seeds must diverge"


def test_seeding_only_python_and_numpy_is_not_enough():
    """Documents *why* seed_everything exists, by showing the naive way fails."""
    import random

    import numpy as np
    from goose.backend.fast_mutations import cython_shuffle_sequence

    seq = "HRQKLDDSKPSLFSDRLSDLGRIPHPSMRVGVPPQNPRPSLNSAPSPFNPQGQSQITDPRQ"

    def draw_naive(seed):
        random.seed(seed)
        np.random.seed(seed)
        return [
            cython_shuffle_sequence(
                seq, window_size=15, global_shuffle_probability=0.4
            )
            for _ in range(8)
        ]

    # If this ever starts passing, GOOSE changed its RNG and seeding.py can be
    # revisited -- but until then, seeding Python and numpy is not sufficient.
    assert draw_naive(42) != draw_naive(42)


@pytest.mark.slow
def test_starling_models_are_in_eval_mode():
    """STARLING loads its models in training mode; ensure_eval_mode fixes it.

    Left unfixed, SequenceEncoder.forward zeroes the ionic strength on ~20% of
    calls, making the encoder nondeterministic.
    """
    from idrdesign.encode import ensure_eval_mode

    ensure_eval_mode(None)

    from starling.inference.generation import model_manager

    vae, diffusion = model_manager.get_models(device=None)
    still_training = [n for n, m in diffusion.named_modules() if m.training]

    assert not vae.training
    assert not diffusion.training
    assert still_training == [], f"{len(still_training)} submodules still training"


@pytest.mark.slow
def test_encoder_is_deterministic():
    """The observable consequence of the eval-mode fix."""
    from idrdesign.encode import encode_sequence

    seq = "HRQKLDDSKPSLFSDRLSDLGRIPHPSMRVGVPPQNPRPSLNSAPSPFNPQGQSQITDPRQ"
    ref = encode_sequence(seq, ionic_strength=150).float()
    for _ in range(5):
        got = encode_sequence(seq, ionic_strength=150).float()
        assert float((got - ref).abs().max()) == 0.0


@pytest.mark.slow
def test_ionic_strength_still_conditions_the_encoder():
    """Guards the fix against over-reach: dropout is off, conditioning is not."""
    from idrdesign.encode import encode_sequence

    seq = "HRQKLDDSKPSLFSDRLSDLGRIPHPSMRVGVPPQNPRPSLNSAPSPFNPQGQSQITDPRQ"
    lo = encode_sequence(seq, ionic_strength=0).float()
    hi = encode_sequence(seq, ionic_strength=500).float()
    assert float((lo - hi).abs().max()) > 0.0
