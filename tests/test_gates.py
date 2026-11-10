import numpy as np

from vec_stagescope.gates import empty_fraction, variance_ratio


def test_empty_fraction():
    x = np.zeros((4, 3), np.float32)
    x[0, 0] = 1
    assert empty_fraction(x) == 0.75


def test_variance_ratio_collapsed_is_zero(stages):
    s = stages["S1"]
    collapsed = np.repeat(s.mean(0, keepdims=True), 500, axis=0)
    assert variance_ratio(collapsed, s) < 1e-6
    assert 0.5 < variance_ratio(s[:1500], s[1500:]) < 2.0
