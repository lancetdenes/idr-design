"""Mpipi inter-residue interaction-map ('intermap') design objective.

Parallels idrdesign.dmap_objective but uses FINCHES' Mpipi forcefield instead of
STARLING. The intermap is the self-interaction matrix of a sequence at a given
sliding-window size; it is deterministic and CPU-only.
"""
from __future__ import annotations

import functools
import numpy as np

from goose.backend.optimizer_properties import ProteinProperty


@functools.lru_cache(maxsize=1)
def _frontend():
    from finches.frontend.mpipi_frontend import Mpipi_frontend
    return Mpipi_frontend()


def mpipi_intermap(seq: str, window_size: int) -> np.ndarray:
    """Symmetric Mpipi self-interaction map, shape (len(seq)-window_size+1)^2.

    FINCHES returns ((matrix, idx1, idx2), mask1, mask2); we want matrix.
    """
    out = _frontend().intermolecular_idr_matrix(
        seq, seq, window_size=window_size,
        disorder_1=None, disorder_2=None,  # treat whole sequence as IDR
    )
    return np.asarray(out[0][0], dtype=np.float64)


class IntermapRMSEToTarget(ProteinProperty):
    """GOOSE objective: RMSE between the candidate's Mpipi intermap and a fixed
    target intermap at a given window. GOOSE minimizes raw_value -> 0. Deterministic.
    """
    can_be_linear_profile = False
    calculate_in_batch = False

    def __init__(self, target_intermap, *, window_size: int, weight: float = 1.0):
        super().__init__(target_value=0.0, weight=weight)
        self._target = np.asarray(target_intermap, dtype=np.float64)
        self._w = int(window_size)

    def calculate_raw_value(self, protein) -> float:
        m = mpipi_intermap(protein.sequence, self._w)
        return float(np.sqrt(np.mean((m - self._target) ** 2)))


@functools.lru_cache(maxsize=1)
def _calvados_frontend():
    from finches.frontend.calvados_frontend import CALVADOS_frontend
    return CALVADOS_frontend()


def calvados_intermap(seq: str, window_size: int) -> np.ndarray:
    """Symmetric CALVADOS self-interaction map, mirroring mpipi_intermap.

    FINCHES returns ((matrix, idx1, idx2), mask1, mask2); we want matrix.
    """
    out = _calvados_frontend().intermolecular_idr_matrix(
        seq, seq, window_size=window_size,
        disorder_1=None, disorder_2=None,  # treat whole sequence as IDR
    )
    return np.asarray(out[0][0], dtype=np.float64)


class CalvadosIntermapRMSEToTarget(ProteinProperty):
    """GOOSE objective: RMSE between the candidate's CALVADOS intermap and a
    fixed target intermap at a given window. GOOSE minimizes raw_value -> 0.
    Deterministic, CPU. Parallels IntermapRMSEToTarget.
    """
    can_be_linear_profile = False
    calculate_in_batch = False

    def __init__(self, target_intermap, *, window_size: int, weight: float = 1.0):
        super().__init__(target_value=0.0, weight=weight)
        self._target = np.asarray(target_intermap, dtype=np.float64)
        self._w = int(window_size)

    def calculate_raw_value(self, protein) -> float:
        m = calvados_intermap(protein.sequence, self._w)
        return float(np.sqrt(np.mean((m - self._target) ** 2)))
