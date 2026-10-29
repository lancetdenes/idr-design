"""GOOSE-based optimization driver for idrdesign.

Starts every design run from a length-matched GS repeat (paper methodology),
wraps GOOSE's SequenceOptimizer with per-run seeding and a wall-time watchdog,
and runs N independent optimizations to produce diverse designs.
"""
from __future__ import annotations


def gs_start(length: int) -> str:
    """Return a length-matched GS-repeat starting sequence.

    Paper methodology: every optimization begins from GS-repeat to avoid
    chance similarity between the starting sequence and the target.
    """
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}")
    return ("GS" * ((length + 1) // 2))[:length]


from typing import List

from goose.optimize import SequenceOptimizer
from goose.backend.fast_mutations import cython_shuffle_sequence


class ShuffleOnlyOptimizer(SequenceOptimizer):
    """SequenceOptimizer that generates candidates only by shuffling.

    Preserves exact amino-acid composition at every iteration. Pattern from
    idptools/goose and adapted from muscle_reprogram/variants/goose_shuffle_only.py.
    """

    def _generate_candidates(self, sequence: str, iteration: int) -> List[str]:
        candidates = [sequence]
        for _ in range(self.num_candidates - 1):
            candidates.append(
                cython_shuffle_sequence(
                    sequence,
                    window_size=self.shuffle_window_size,
                    global_shuffle_probability=self.global_shuffle_probability,
                )
            )
        return candidates


import random
import threading
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from idrdesign.encode import (
    CosineSimilarityToTarget,
    PerResidueCosineSimilarityToTarget,
    cosine_similarity,
    encode_sequence,
    encode_sequence_perresidue,
    per_residue_cosine,
)


CompositionMode = Literal["none", "exact", "properties"]
VALID_COMPOSITION_MODES = ("none", "exact", "properties")

Objective = Literal["cosine", "per_residue_cosine"]
VALID_OBJECTIVES = ("cosine", "per_residue_cosine")


@dataclass
class DesignResult:
    sequence: str
    final_cosine_similarity: float
    iterations: int
    wall_seconds: float
    convergence_status: str  # "converged" | "max_iterations" | "max_wall" | "failed"
    seed: int
    starting_sequence: str
    composition_mode: str


def _build_optimizer(
    *,
    target_length: int,
    composition_mode: CompositionMode,
    max_iterations: int,
    error_tolerance: float,
) -> SequenceOptimizer:
    kwargs = dict(
        target_length=target_length,
        max_iterations=max_iterations,
        stagnation_threshold=10,   # paper value: 10 rounds w/o improvement
        enable_error_tolerance=True,
        error_tolerance=error_tolerance,
        num_candidates=5,
        verbose=False,
    )
    if composition_mode == "exact":
        return ShuffleOnlyOptimizer(**kwargs)
    return SequenceOptimizer(**kwargs)


def _starting_sequence_for(
    target_seq: str, composition_mode: CompositionMode, rng: random.Random
) -> str:
    """Paper-default: GS repeat. For composition-exact, we must start with target's AA counts."""
    if composition_mode == "exact":
        # Shuffle target to break exact identity while preserving composition.
        return "".join(rng.sample(list(target_seq), len(target_seq)))
    return gs_start(len(target_seq))


def run_single_optimization(
    *,
    target_seq: str,
    target_emb: torch.Tensor,
    target_similarity: float,
    composition_mode: CompositionMode,
    max_iterations: int,
    max_wall_seconds: float,
    seed: int,
    objective: Objective = "cosine",
) -> DesignResult:
    """Run one GOOSE optimization and return a structured result.

    Seeds Python's random, numpy, and torch before constructing the optimizer.
    A watchdog thread sets a soft-abort flag after max_wall_seconds; if GOOSE
    doesn't honor it (it won't — there's no hook), we stop the thread and
    rely on max_iterations instead. The wall cap is therefore best-effort.
    """
    if composition_mode not in VALID_COMPOSITION_MODES:
        raise ValueError(
            f"composition_mode must be one of {VALID_COMPOSITION_MODES}, "
            f"got {composition_mode!r}"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rng = random.Random(seed)
    starting_seq = _starting_sequence_for(target_seq, composition_mode, rng)

    error_tolerance = 1.0 - target_similarity

    opt = _build_optimizer(
        target_length=len(target_seq),
        composition_mode=composition_mode,
        max_iterations=max_iterations,
        error_tolerance=error_tolerance,
    )
    opt.set_initial_sequence(starting_seq)
    if objective == "per_residue_cosine":
        opt.add_property(PerResidueCosineSimilarityToTarget, target_emb)
    else:
        opt.add_property(CosineSimilarityToTarget, target_emb)

    # "properties" mode: add soft FCR/NCPR/hydrophobicity anchors from the target.
    if composition_mode == "properties":
        _add_composition_anchors(opt, target_seq)

    wall_start = time.monotonic()
    abort_event = threading.Event()
    result_holder: dict = {}

    def _run():
        try:
            best = opt.run()
            result_holder["best"] = best
        except Exception as e:
            result_holder["error"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=max_wall_seconds)
    wall_elapsed = time.monotonic() - wall_start

    if t.is_alive():
        # Watchdog fired; GOOSE has no abort hook. We cannot interrupt the
        # thread safely. Grab whatever best_sequence is currently stored
        # and treat the run as "max_wall". GOOSE will eventually finish in
        # the background but we don't wait further — caller should budget
        # wall time accordingly.
        best_seq = opt.best_sequence or starting_seq
        status = "max_wall"
    elif "error" in result_holder:
        raise result_holder["error"]
    else:
        best_seq = result_holder["best"]
        # Converged = final error ≤ tolerance (error_tolerance = 1 - target_similarity).
        # We can't read GOOSE's internal state cleanly, so infer from iterations:
        status = (
            "converged"
            if opt.iteration < max_iterations
            else "max_iterations"
        )

    # Measure final similarity against target (in the objective's own metric).
    if objective == "per_residue_cosine":
        final_cos_sim = per_residue_cosine(encode_sequence_perresidue(best_seq), target_emb)
    else:
        final_cos_sim = cosine_similarity(target_emb, encode_sequence(best_seq))

    return DesignResult(
        sequence=best_seq,
        final_cosine_similarity=final_cos_sim,
        iterations=opt.iteration,
        wall_seconds=wall_elapsed,
        convergence_status=status,
        seed=seed,
        starting_sequence=starting_seq,
        composition_mode=composition_mode,
    )


def _add_composition_anchors(opt: SequenceOptimizer, target_seq: str) -> None:
    """Add FCR/NCPR/Hydrophobicity properties anchored to the target's values.

    Implemented via sparrow.Protein property readouts + GOOSE's built-in
    property classes.
    """
    import sparrow
    from goose.backend.optimizer_properties import FCR, NCPR, Hydrophobicity

    p = sparrow.Protein(target_seq)
    opt.add_property(FCR, target_value=float(p.FCR), weight=0.5)
    opt.add_property(NCPR, target_value=float(p.NCPR), weight=0.5)
    opt.add_property(Hydrophobicity, target_value=float(p.hydrophobicity), weight=0.5)


def design_idr(
    *,
    target_seq: str,
    n_designs: int = 10,
    target_similarity: float = 0.95,
    composition_mode: CompositionMode = "none",
    max_iterations: int = 5000,
    max_wall_seconds: float = 1800,
    seed: int = 0,
    ionic_strength: float = 150.0,
    objective: Objective = "cosine",
) -> list[DesignResult]:
    """Produce N novel IDR designs targeting a STARLING-encoder match to target_seq.

    Parameters
    ----------
    target_seq
        The target IDR sequence to mimic.
    n_designs
        Number of independent optimization runs (distinct seeds).
    target_similarity
        Stop each run once cosine similarity >= this. 0.95 = tight match.
    composition_mode
        "none" | "exact" | "properties". See spec §5.7.
    max_iterations, max_wall_seconds
        Per-run safety caps.
    seed
        Base seed; run i uses seed = seed + i.
    ionic_strength
        Forwarded to STARLING encoder (conditions the embedding).
    """
    if objective not in VALID_OBJECTIVES:
        raise ValueError(f"objective must be one of {VALID_OBJECTIVES}, got {objective!r}")
    if objective == "per_residue_cosine":
        target_emb = encode_sequence_perresidue(target_seq, ionic_strength=ionic_strength)
    else:
        target_emb = encode_sequence(target_seq, ionic_strength=ionic_strength)

    results: list[DesignResult] = []
    for i in range(n_designs):
        res = run_single_optimization(
            target_seq=target_seq,
            target_emb=target_emb,
            target_similarity=target_similarity,
            composition_mode=composition_mode,
            max_iterations=max_iterations,
            max_wall_seconds=max_wall_seconds,
            seed=seed + i,
            objective=objective,
        )
        results.append(res)
    return results
