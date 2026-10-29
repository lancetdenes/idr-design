# tests/test_seqpenalty.py
import pytest
from idrdesign.seqpenalty import identity_to, MinimizeIdentityToWT

def test_identity_to_basic():
    assert identity_to("ABCD", "ABCD") == 1.0
    assert identity_to("ABCD", "ABCE") == 0.75
    assert identity_to("ABCD", "WXYZ") == 0.0

def test_identity_to_length_mismatch_raises():
    with pytest.raises(ValueError):
        identity_to("ABC", "ABCD")

class _Prot:
    def __init__(self, s): self.sequence = s

def test_penalty_returns_identity_fraction():
    obj = MinimizeIdentityToWT("ABCD")
    assert obj.calculate_raw_value(_Prot("ABCD")) == 1.0
    assert obj.calculate_raw_value(_Prot("WXYZ")) == 0.0
    assert obj.calculate_raw_value(_Prot("ABCE")) == 0.75
