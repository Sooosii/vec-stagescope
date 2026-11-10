"""Guards that stop placement for inputs that are not a cell population. Other community tools
check the same failures; here they only prevent a misleading proximity reading."""
from __future__ import annotations

import numpy as np

EMPTY_FRACTION_LIMIT = 0.5
COLLAPSE_LIMIT = 0.1   # veckit variance_ratio: "far below ~0.1" = not a population


def empty_fraction(x: np.ndarray) -> float:
    return float(np.mean(~np.any(x != 0, axis=1)))


def variance_ratio(pred: np.ndarray, ref: np.ndarray) -> float:
    """veckit definition: mean per-gene variance of prediction / of reference."""
    vb = float(np.asarray(ref, np.float64).var(0).mean())
    va = float(np.asarray(pred, np.float64).var(0).mean())
    return va / vb if vb > 0 else float("nan")
