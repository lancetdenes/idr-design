"""Direct distance-map design objective (the 'beyond-paper' extension).

Instead of the published cosine-embedding proxy, this optimizes the STARLING
mean distance map of the candidate directly against a target (WT) distance map.
Each evaluation generates a small STARLING ensemble with return_structures=False
(distance-maps only, skips the expensive MDS 3D reconstruction) so it is far
cheaper than full ensemble generation, though still much heavier than the encoder
embedding. Mirrors idrdesign.encode.CosineSimilarityToTarget.
"""
from __future__ import annotations

import numpy as np
import starling

from goose.backend.optimizer_properties import ProteinProperty


# Per-evaluation conformer count for the in-loop objective. Larger = less noisy
# objective (closer to the 400-conformer ground truth) but proportionally slower.
# 64 was shown to be noise-dominated (GOOSE overfits sampling noise); set 400 for
# a fair, low-noise objective. The pilot overrides this before building the optimizer.
DEFAULT_OBJ_CONFORMATIONS = 64


def starling_mean_dmap(seq: str, *, conformations: int = 64,
                       ionic_strength: float = 150.0, device: str = "cuda") -> np.ndarray:
    """Mean STARLING distance map for `seq` (distance-maps only; no 3D structures)."""
    out = starling.generate(
        {"q": seq},
        conformations=conformations,
        ionic_strength=ionic_strength,
        device=device,
        return_structures=False,
        return_data=True,
        verbose=False,
        show_progress_bar=False,
        show_per_step_progress_bar=False,
    )
    ens = out["q"] if isinstance(out, dict) else out
    return np.asarray(ens.distance_maps(return_mean=True), dtype=float)


class DistanceMapRMSEToTarget(ProteinProperty):
    """GOOSE objective: RMSE between the candidate's STARLING mean distance map
    and a fixed target (WT) mean distance map. GOOSE minimizes raw_value -> 0.

    conformations is kept small for speed; the optimization-time RMSE is therefore
    a noisy estimate (re-evaluate the winner with a full 400-conformer ensemble).
    """

    can_be_linear_profile = False
    calculate_in_batch = False

    def __init__(self, target_dmap, *, weight: float = 1.0, conformations: int | None = None,
                 ionic_strength: float = 150.0, device: str = "cuda"):
        super().__init__(target_value=0.0, weight=weight)
        self._target = np.asarray(target_dmap, dtype=float)
        self._k = DEFAULT_OBJ_CONFORMATIONS if conformations is None else conformations
        self._is = ionic_strength
        self._dev = device

    def calculate_raw_value(self, protein) -> float:  # protein: sparrow.Protein
        dm = starling_mean_dmap(
            protein.sequence, conformations=self._k,
            ionic_strength=self._is, device=self._dev,
        )
        return float(np.sqrt(np.mean((dm - self._target) ** 2)))
