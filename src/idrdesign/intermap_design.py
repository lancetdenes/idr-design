"""Design driver for the FINCHES intermap objectives.

Parallel to `idrdesign.optimize`, which designs against STARLING encoder
embeddings. This module designs against inter-residue interaction maps from
FINCHES' Mpipi and CALVADOS forcefields instead.

Differences that matter:

* Scoring is deterministic. The intermaps are analytical, so the same sequence
  always scores identically. The STARLING objective is not: its models load in
  training mode, leaving a 20% ionic-strength dropout active in the sequence
  encoder, so repeated calls on one sequence disagree.
* CPU-only. There is no model to place on a GPU; cost is numpy over an
  (L-w+1, L-w+1) matrix per candidate per forcefield.
* Unbounded objective. Cosine similarity lives in [-1, 1]; intermap RMSE is in
  the forcefield's own energy units, so "how close is close" is scale-dependent
  and differs between the two forcefields. See `normalized_weights`.

This module imports neither starling nor torch directly. Whether they end up
loaded depends on what goose and sparrow pull in; `idrdesign.__init__` is lazy
so that merely importing this module does not drag in `idrdesign.optimize`.

Reproducibility
---------------
`goose.optimize.SequenceOptimizer` takes no seed, and its candidate generator
`cython_shuffle_sequence` draws from the C library's `rand()`, which neither
`random.seed()` nor `np.random.seed()` touches. Seeding it requires `srand()`
via ctypes -- see `idrdesign.seeding`, which this module uses.

With that in place a run IS reproducible: same sequence, same settings, same
seed gives the same design. `IntermapDesignResult.reproducible` records whether
libc could actually be reached; if it is False the search was unseeded and the
result should not be treated as reproducible.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np

from goose.optimize import SequenceOptimizer
from goose.backend.fast_mutations import cython_shuffle_sequence

from idrdesign.seeding import seed_everything
from idrdesign.intermap_objective import (
    CalvadosIntermapRMSEToTarget,
    IntermapRMSEToTarget,
    calvados_intermap,
    mpipi_intermap,
)

Forcefield = Literal["mpipi", "calvados"]
VALID_FORCEFIELDS = ("mpipi", "calvados")

CompositionMode = Literal["none", "exact", "properties"]
VALID_COMPOSITION_MODES = ("none", "exact", "properties")

# Window sizes the campaigns used. 31 costs roughly 2.2x what 17 does.
VALID_WINDOW_SIZES = (17, 31)

_MAP_FN: dict[str, Callable[[str, int], np.ndarray]] = {
    "mpipi": mpipi_intermap,
    "calvados": calvados_intermap,
}
_OBJECTIVE: dict[str, type] = {
    "mpipi": IntermapRMSEToTarget,
    "calvados": CalvadosIntermapRMSEToTarget,
}


@dataclass
class IntermapDesignResult:
    sequence: str
    starting_sequence: str
    iterations: int
    wall_seconds: float
    convergence_status: str   # "converged" | "max_iterations" | "max_wall" | "failed"
    seed: int
    composition_mode: str
    window_size: int
    forcefields: list[str]
    # False means libc's rand() could not be seeded, so the search was not
    # reproducible regardless of the seed that was passed.
    reproducible: bool = True
    # Per-forcefield scores against the target map.
    rmse: dict[str, float] = field(default_factory=dict)
    rmse_start: dict[str, float] = field(default_factory=dict)
    pearson: dict[str, float] = field(default_factory=dict)


def gs_start(length: int) -> str:
    """Length-matched GS repeat.

    Paper methodology: every optimization starts here so no similarity to the
    target is inherited from the starting point.
    """
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}")
    return ("GS" * ((length + 1) // 2))[:length]


class ShuffleOnlyOptimizer(SequenceOptimizer):
    """SequenceOptimizer whose candidates are shuffles of the current sequence.

    Preserves exact amino-acid composition at every iteration. Mirrors the
    class of the same name in idrdesign.optimize.
    """

    def _generate_candidates(self, sequence: str, iteration: int) -> list[str]:
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


def target_maps(
    target_seq: str, forcefields: Sequence[str], window_size: int
) -> dict[str, np.ndarray]:
    """Intermap of the target sequence, per forcefield."""
    return {ff: _MAP_FN[ff](target_seq, window_size) for ff in forcefields}


def normalized_weights(maps: dict[str, np.ndarray]) -> dict[str, float]:
    """Weights that put both forcefields on comparable footing.

    CALVADOS maps run roughly twice the magnitude of Mpipi ones, so equal
    weights would let CALVADOS dominate a joint objective. Each forcefield is
    weighted by the reciprocal of its target map's mean absolute value, then
    scaled so the weights average 1.0.

    This is a project choice, not something the methodology prescribes.
    """
    scale = {ff: 1.0 / max(float(np.mean(np.abs(m))), 1e-12) for ff, m in maps.items()}
    mean = float(np.mean(list(scale.values())))
    return {ff: v / mean for ff, v in scale.items()}


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r over the strict upper triangle of two square maps."""
    iu = np.triu_indices(a.shape[0], k=1)
    x, y = a[iu], b[iu]
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.sqrt((xc**2).sum() * (yc**2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def _starting_sequence(
    target_seq: str, composition_mode: str, rng: random.Random
) -> str:
    if composition_mode == "exact":
        # Must start from the target's exact composition; shuffle to break identity.
        return "".join(rng.sample(list(target_seq), len(target_seq)))
    return gs_start(len(target_seq))


def _add_composition_anchors(opt: SequenceOptimizer, target_seq: str) -> None:
    """Soft FCR/NCPR/hydrophobicity anchors read off the target."""
    import sparrow
    from goose.backend.optimizer_properties import FCR, NCPR, Hydrophobicity

    p = sparrow.Protein(target_seq)
    opt.add_property(FCR, target_value=float(p.FCR), weight=0.5)
    opt.add_property(NCPR, target_value=float(p.NCPR), weight=0.5)
    opt.add_property(Hydrophobicity, target_value=float(p.hydrophobicity), weight=0.5)


def design_intermap(
    *,
    target_seq: str,
    forcefields: Sequence[str] = VALID_FORCEFIELDS,
    window_size: int = 17,
    composition_mode: str = "none",
    max_iterations: int = 150,
    max_wall_seconds: float = 600.0,
    error_tolerance: float = 0.05,
    seed: int = 0,
    num_candidates: int = 5,
) -> IntermapDesignResult:
    """Run one GOOSE optimization against the intermap objective(s).

    Returns a single design. Cost is roughly
    `num_candidates * sum(per-map cost) * iterations`; at window 17 with both
    forcefields that is about 2 s per iteration on a few CPU cores.
    """
    forcefields = list(forcefields)
    if not forcefields:
        raise ValueError("at least one forcefield is required")
    bad = [ff for ff in forcefields if ff not in VALID_FORCEFIELDS]
    if bad:
        raise ValueError(f"unknown forcefield(s) {bad}; valid: {VALID_FORCEFIELDS}")
    if composition_mode not in VALID_COMPOSITION_MODES:
        raise ValueError(
            f"composition_mode must be one of {VALID_COMPOSITION_MODES}, "
            f"got {composition_mode!r}"
        )

    # Seeds libc's rand() too; without that the GOOSE search is unseeded.
    reproducible = seed_everything(seed)

    rng = random.Random(seed)
    starting_seq = _starting_sequence(target_seq, composition_mode, rng)

    targets = target_maps(target_seq, forcefields, window_size)
    weights = normalized_weights(targets)

    kwargs = dict(
        target_length=len(target_seq),
        max_iterations=max_iterations,
        stagnation_threshold=10,      # paper value
        enable_error_tolerance=True,
        error_tolerance=error_tolerance,
        num_candidates=num_candidates,
        verbose=False,
    )
    opt = (
        ShuffleOnlyOptimizer(**kwargs)
        if composition_mode == "exact"
        else SequenceOptimizer(**kwargs)
    )
    opt.set_initial_sequence(starting_seq)

    for ff in forcefields:
        opt.add_property(
            _OBJECTIVE[ff], targets[ff], window_size=window_size, weight=weights[ff]
        )

    if composition_mode == "properties":
        _add_composition_anchors(opt, target_seq)

    # GOOSE offers no abort hook, so the wall cap is best-effort: we stop
    # waiting and read back whatever best_sequence it has reached.
    wall_start = time.monotonic()
    holder: dict = {}

    def _run():
        try:
            holder["best"] = opt.run()
        except Exception as exc:    # surfaced to the caller below
            holder["error"] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=max_wall_seconds)
    wall_elapsed = time.monotonic() - wall_start

    if t.is_alive():
        best_seq = opt.best_sequence or starting_seq
        status = "max_wall"
    elif "error" in holder:
        raise holder["error"]
    else:
        best_seq = holder["best"]
        status = "converged" if opt.iteration < max_iterations else "max_iterations"

    design_maps = {ff: _MAP_FN[ff](best_seq, window_size) for ff in forcefields}
    start_maps = {ff: _MAP_FN[ff](starting_seq, window_size) for ff in forcefields}

    return IntermapDesignResult(
        sequence=best_seq,
        starting_sequence=starting_seq,
        iterations=opt.iteration,
        wall_seconds=wall_elapsed,
        convergence_status=status,
        seed=seed,
        composition_mode=composition_mode,
        window_size=window_size,
        forcefields=forcefields,
        reproducible=reproducible,
        rmse={ff: rmse(design_maps[ff], targets[ff]) for ff in forcefields},
        rmse_start={ff: rmse(start_maps[ff], targets[ff]) for ff in forcefields},
        pearson={ff: pearson(design_maps[ff], targets[ff]) for ff in forcefields},
    )
