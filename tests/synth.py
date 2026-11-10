"""Synthetic 'stages' that reproduce the real data's depth trap: stages differ in a shifting expression
program AND in detection depth. No challenge data is used anywhere in the test suite."""
from __future__ import annotations

import numpy as np

N_GENES = 80
GENES = [f"g{i:03d}" for i in range(N_GENES)]

# (program position 0..1, mean counts per cell) - depth differs by stage like the real MERFISH data
STAGES = {"S1": (0.0, 60.0), "S2": (0.5, 90.0), "S3": (1.0, 300.0)}


def _profile(pos: float) -> np.ndarray:
    g = np.arange(N_GENES)
    base = np.exp(-((g - 20 - 40 * pos) ** 2) / (2 * 12.0**2)) + 0.05
    return base / base.sum()


def log_norm(counts: np.ndarray) -> np.ndarray:
    tot = counts.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(tot > 0, counts / tot * 1e4, 0.0)
    return np.log1p(x).astype(np.float32)


def make_stage(name: str, n_cells: int, seed: int) -> np.ndarray:
    pos, depth = STAGES[name]
    rng = np.random.default_rng(seed)
    p = _profile(pos)
    lib = rng.lognormal(np.log(depth), 0.3, n_cells)
    counts = rng.poisson(lib[:, None] * p[None, :] * rng.lognormal(0, 0.4, (n_cells, N_GENES)))
    return log_norm(counts)


def mix(a: np.ndarray, b: np.ndarray, alpha: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nb = int(round(n * alpha))
    ia = rng.choice(a.shape[0], n - nb, replace=False)
    ib = rng.choice(b.shape[0], nb, replace=False)
    return np.vstack([a[ia], b[ib]])


def uniform_dropout(x: np.ndarray, keep: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = np.expm1(x.astype(np.float64))
    c = np.where((c > 0) & (rng.random(c.shape) < keep), c, 0.0)
    return log_norm(c)


def expression_dependent_dropout(x: np.ndarray, mean_keep: float, seed: int) -> np.ndarray:
    """Foreign corruption (a)(i): low values drop more often than high ones."""
    rng = np.random.default_rng(seed)
    c = np.expm1(x.astype(np.float64))
    nz = c > 0
    ranks = np.zeros_like(c)
    for i in range(c.shape[0]):
        idx = np.flatnonzero(nz[i])
        if idx.size:
            ranks[i, idx] = (np.argsort(np.argsort(c[i, idx])) + 1) / idx.size
    p_keep = np.clip(2 * mean_keep * ranks, 0, 1)
    c = np.where(nz & (rng.random(c.shape) < p_keep), c, 0.0)
    return log_norm(c)


def count_like_thinning(x: np.ndarray, keep: float, seed: int) -> np.ndarray:
    """Foreign corruption (a)(ii): round to integers on a small library, binomial thinning."""
    rng = np.random.default_rng(seed)
    c = np.expm1(x.astype(np.float64)) / 1e4 * 100.0
    c = rng.binomial(np.rint(c).astype(np.int64), keep).astype(np.float64)
    return log_norm(c)


def add_noise(x: np.ndarray, sd: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(x + rng.normal(0, sd, x.shape), 0, None).astype(np.float32)
