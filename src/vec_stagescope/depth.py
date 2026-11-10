"""Detection-depth matching ("dropout matching").

Depth = number of detected (non-zero) genes per cell. The deeper population is brought to the
shallower one's median depth by dropping detected entries uniformly at random, then each cell is
re-normalised to 1e4 and log1p-transformed again (the challenge's log1p-CP10k convention).
This is not count-level binomial thinning: expm1 of log1p-CP10k values are not counts.
"""
from __future__ import annotations

import numpy as np

TARGET_SUM = 1e4


def cell_depth(x: np.ndarray) -> np.ndarray:
    return np.count_nonzero(x, axis=1)


def median_depth(x: np.ndarray) -> float:
    return float(np.median(cell_depth(x)))


def _renormalise(c: np.ndarray) -> np.ndarray:
    tot = c.sum(1, keepdims=True)
    scale = np.divide(TARGET_SUM, tot, out=np.zeros_like(tot), where=tot > 0)
    return np.log1p(c * scale).astype(np.float32)


def thin_to_depth(x: np.ndarray, target_median: float, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of ``x`` whose median depth is about ``target_median``; ``x`` is not modified."""
    current = median_depth(x)
    if current <= target_median or current == 0:
        return x
    keep = target_median / current
    c = np.expm1(x.astype(np.float64))
    mask = (c > 0) & (rng.random(c.shape) < keep)
    return _renormalise(np.where(mask, c, 0.0))


def match_pair(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    """Thin whichever of ``a``/``b`` is deeper down to the other's median depth."""
    da, db = median_depth(a), median_depth(b)
    if da > db:
        return thin_to_depth(a, db, rng), b
    if db > da:
        return a, thin_to_depth(b, da, rng)
    return a, b
