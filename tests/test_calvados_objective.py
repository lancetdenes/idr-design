# tests/test_calvados_objective.py
import numpy as np
from idrdesign.intermap_objective import calvados_intermap, CalvadosIntermapRMSEToTarget

SEQ = "GSGSRRDDEEKKGGPPQQNNSSRRDDEEKKGGPPQQNNSS"  # len 40

def test_calvados_intermap_shape_and_symmetry():
    m = calvados_intermap(SEQ, 17)
    assert m.shape == (len(SEQ) - 17 + 1, len(SEQ) - 17 + 1)
    assert np.allclose(m, m.T)

class _Prot:
    def __init__(self, s): self.sequence = s

def test_calvados_rmse_zero_against_self_positive_against_other():
    tgt = calvados_intermap(SEQ, 17)
    obj = CalvadosIntermapRMSEToTarget(tgt, window_size=17)
    assert obj.calculate_raw_value(_Prot(SEQ)) < 1e-9
    other = "".join(reversed(SEQ))
    assert obj.calculate_raw_value(_Prot(other)) > 1e-6
