"""Energy distance, same definition as veckit ``common/core_metrics.energy_distance`` (full gene
space, Euclidean, n(n-1) denominators for the within-sample terms, BLAS matmul identity)."""
from __future__ import annotations

import numpy as np


def _pairwise_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.einsum("ij,ij->i", a, a)
    bb = np.einsum("ij,ij->i", b, b)
    d2 = aa[:, None] + bb[None, :] - 2.0 * (a @ b.T)
    np.maximum(d2, 0, out=d2)
    return np.sqrt(d2, out=d2)


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = a.shape[0], b.shape[0]
    daa = _pairwise_euclidean(a, a)
    np.fill_diagonal(daa, 0.0)
    dbb = _pairwise_euclidean(b, b)
    np.fill_diagonal(dbb, 0.0)
    return float(2 * _pairwise_euclidean(a, b).mean()
                 - daa.sum() / (na * (na - 1)) - dbb.sum() / (nb * (nb - 1)))
