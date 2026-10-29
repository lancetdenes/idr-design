"""Sequence-overlap penalty objective for composition-controlled design.

MinimizeIdentityToWT rewards sequences that share FEW positions with a fixed
reference (WT), pushing fixed-composition shuffles away from the WT primary
sequence. GOOSE minimizes raw_value -> target_value (0), so raw_value is the
positional identity fraction to the reference.
"""
from __future__ import annotations

from goose.backend.optimizer_properties import ProteinProperty


def identity_to(seq: str, ref: str) -> float:
    """Fraction of positions where seq matches ref (must be equal length)."""
    if len(seq) != len(ref):
        raise ValueError(f"length mismatch: {len(seq)} vs {len(ref)}")
    if not seq:
        return 0.0
    return sum(a == b for a, b in zip(seq, ref)) / len(seq)


class MinimizeIdentityToWT(ProteinProperty):
    """GOOSE objective = positional identity fraction to a fixed reference.
    target_value=0 => GOOSE drives identity toward 0 (minimize overlap).
    """
    can_be_linear_profile = False
    calculate_in_batch = False

    def __init__(self, reference_seq: str, *, weight: float = 1.0):
        super().__init__(target_value=0.0, weight=weight)
        self._ref = str(reference_seq)

    def calculate_raw_value(self, protein) -> float:
        return identity_to(protein.sequence, self._ref)
