"""Seeding that actually reaches every RNG this pipeline depends on.

GOOSE generates candidate sequences with `cython_shuffle_sequence`, which draws
from the C standard library's `rand()`. Neither `random.seed()` nor
`numpy.random.seed()` touches that generator, so seeding only those two leaves
the search unseeded and runs unreproducible -- while still appearing to accept a
seed. Reaching it requires calling `srand()` through ctypes.

`seed_everything` seeds Python, numpy, torch (when importable) and libc, and
reports whether libc was reachable so callers can record honestly whether a run
is reproducible.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import random

import numpy as np

# Resolved once: None means "not yet tried", False means "unavailable".
_LIBC: ctypes.CDLL | None | bool = None


def _libc() -> ctypes.CDLL | None:
    global _LIBC
    if _LIBC is None:
        name = ctypes.util.find_library("c")
        try:
            _LIBC = ctypes.CDLL(name) if name else False
        except OSError:
            _LIBC = False
    return _LIBC or None


def seed_libc(seed: int) -> bool:
    """Seed the C `rand()` that GOOSE's shuffler uses. True if it worked."""
    lib = _libc()
    if lib is None:
        return False
    try:
        lib.srand(ctypes.c_uint(seed & 0xFFFFFFFF))
    except (AttributeError, OSError):
        return False
    return True


def seed_everything(seed: int) -> bool:
    """Seed every RNG in play.

    Returns True when libc could be seeded, i.e. when the GOOSE search is
    genuinely reproducible. A False return means Python/numpy/torch were seeded
    but the candidate generator was not, so the run is not reproducible.
    """
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    return seed_libc(seed)
