import numpy as np

from vec_stagescope.distance import energy_distance


def test_energy_distance_zero_for_identical_and_positive_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (300, 10)).astype(np.float32)
    b = rng.normal(0, 1, (300, 10)).astype(np.float32)
    c = rng.normal(1.5, 1, (300, 10)).astype(np.float32)
    assert energy_distance(a, a) < 0.05
    assert abs(energy_distance(a, b)) < 0.2
    assert energy_distance(a, c) > 2.0


def test_energy_distance_matches_bruteforce():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, (40, 5))
    b = rng.normal(0.5, 1, (50, 5))
    d = lambda u, v: np.linalg.norm(u[:, None] - v[None], axis=2)
    ref = 2 * d(a, b).mean() - d(a, a).sum() / (40 * 39) - d(b, b).sum() / (50 * 49)
    assert np.isclose(energy_distance(a, b), ref, rtol=1e-6)
