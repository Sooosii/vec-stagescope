import numpy as np

from vec_stagescope.depth import cell_depth, match_pair, thin_to_depth


def test_cell_depth_counts_nonzero_genes():
    x = np.array([[0, 1, 2], [0, 0, 0], [3, 3, 3]], dtype=np.float32)
    assert cell_depth(x).tolist() == [2, 0, 3]


def test_thin_reduces_median_depth_and_keeps_library_size(stages):
    x = stages["S3"]
    rng = np.random.default_rng(0)
    target = float(np.median(cell_depth(stages["S1"])))
    y = thin_to_depth(x, target, rng)
    assert abs(np.median(cell_depth(y)) - target) <= 0.15 * target
    lib = np.expm1(y.astype(np.float64)).sum(1)
    assert np.allclose(lib[lib > 0], 1e4, rtol=1e-3)


def test_thin_is_noop_when_already_shallow(stages):
    x = stages["S1"]
    y = thin_to_depth(x, 1e9, np.random.default_rng(0))
    assert np.array_equal(x, y)


def test_match_pair_equalises_depth(stages):
    a, b = match_pair(stages["S1"], stages["S3"], np.random.default_rng(0))
    da, db = np.median(cell_depth(a)), np.median(cell_depth(b))
    assert abs(da - db) <= 0.15 * max(da, db)
    assert np.array_equal(a, stages["S1"])
