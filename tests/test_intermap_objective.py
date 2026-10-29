import numpy as np
import pytest
from idrdesign.intermap_objective import mpipi_intermap

SEQ = "MKQEPSTGRD" * 9  # 90 aa

@pytest.mark.parametrize("w,dim", [(17, 90 - 17 + 1), (31, 90 - 31 + 1), (71, 90 - 71 + 1)])
def test_intermap_shape_symmetry(w, dim):
    m = mpipi_intermap(SEQ, w)
    assert m.shape == (dim, dim)
    assert np.allclose(m, m.T)          # self-map is symmetric
    assert m.dtype == np.float64

def test_intermap_deterministic():
    a = mpipi_intermap(SEQ, 31)
    b = mpipi_intermap(SEQ, 31)
    assert np.array_equal(a, b)


from idrdesign.intermap_objective import IntermapRMSEToTarget

class _FakeProtein:
    def __init__(self, seq): self.sequence = seq

def test_rmse_zero_against_self():
    tgt = mpipi_intermap(SEQ, 31)
    prop = IntermapRMSEToTarget(tgt, window_size=31)
    assert prop.calculate_raw_value(_FakeProtein(SEQ)) == pytest.approx(0.0, abs=1e-9)

def test_rmse_positive_against_different():
    tgt = mpipi_intermap(SEQ, 31)
    other = "GSGSGSGSGS" * 9  # 90 aa, different composition
    prop = IntermapRMSEToTarget(tgt, window_size=31)
    assert prop.calculate_raw_value(_FakeProtein(other)) > 0.0
