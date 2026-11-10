"""Depth-matched proximity of a prediction to each published training stage.

r = ED(prediction, base) / ED(base, nearest other stage), all depth-matched. A noise floor
(two disjoint samples of the base stage) guards the copy label: a board whose stages are not
farther apart than a few noise floors gets no label at all.

Thresholds are provisional until the pre-registered calibration (plan §9 c) fixes them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .depth import match_pair, median_depth
from .distance import energy_distance
from .gates import COLLAPSE_LIMIT, EMPTY_FRACTION_LIMIT, empty_fraction, variance_ratio
from .io import sample_dense

COPY_R = 0.10          # r below this (and within K_FLOOR noise floors) -> "copy_like"
NEAR_R = 0.30          # r below this -> "very_close"
K_FLOOR = 2.0
MIN_SEPARATION = 3.0   # denominator must exceed this many noise floors for any label
FLOOR_MIN_FRACTION = 0.01  # the floor estimate is itself noisy around 0; never use less than 1% of
#                            the stage separation as the floor in the copy rule
DEPTH_WARN = 0.5


@dataclass(frozen=True)
class Result:
    base: str
    n: int
    repeats: int
    depth: dict
    gate: str | None = None
    label: str | None = None
    r_base: float | None = None
    r_by_stage: dict = field(default_factory=dict)
    ed_by_stage: dict = field(default_factory=dict)   # stage -> (min, mean, max) over repeats
    ed_floor: float | None = None
    denominator: float | None = None
    r_floor: float | None = None
    variance_ratio: float | None = None
    warnings: tuple = ()


def _ed(a, b, rng, depth_matched):
    if depth_matched:
        a, b = match_pair(a, b, rng)
    return energy_distance(a, b)


def _depth_table(pred, stages):
    sample = np.random.default_rng(0)
    table = {"prediction": median_depth(sample_dense(pred, 3000, sample))}
    for label, x in stages.items():
        table[label] = median_depth(sample_dense(x, 3000, sample))
    return table


def _one_repeat(pred, stages, base, n, rng, depth_matched):
    p = sample_dense(pred, n, rng)
    eds, vr = {}, None
    for label, x in stages.items():
        s = sample_dense(x, n, rng)
        a, b = match_pair(p, s, rng) if depth_matched else (p, s)
        eds[label] = energy_distance(a, b)
        if label == base:
            # dropout matching can add spread to a collapsed profile, so gate on the smaller ratio
            vr = min(variance_ratio(a, b), variance_ratio(p, s))
    base_x = stages[base]
    perm = rng.permutation(base_x.shape[0])
    nf = min(n, base_x.shape[0] // 2)
    b1 = sample_dense(base_x, 0, rng, idx=np.sort(perm[:nf]))
    b2 = sample_dense(base_x, 0, rng, idx=np.sort(perm[nf:2 * nf]))
    floor = _ed(b1, b2, rng, depth_matched)
    b = sample_dense(base_x, n, rng)
    denom = min(_ed(b, sample_dense(x, n, rng), rng, depth_matched)
                for label, x in stages.items() if label != base)
    return eds, vr, floor, denom


def _label(ed_base, floor, denom, r):
    if denom <= MIN_SEPARATION * max(floor, 1e-12):
        return "no_separation"
    if ed_base <= K_FLOOR * max(floor, FLOOR_MIN_FRACTION * denom) and r < COPY_R:
        return "copy_like"
    if r < NEAR_R:
        return "very_close"
    return "info"


def _depth_warnings(depth):
    pred = depth["prediction"]
    others = [v for k, v in depth.items() if k != "prediction" and v > 0]
    if others and all(abs(pred - v) / v > DEPTH_WARN for v in others):
        return ("Prediction depth differs from every stage by more than 50%; "
                "comparisons rely on dropout matching.",)
    return ()


def diagnose(pred, stages: dict, base: str, n: int = 1500, repeats: int = 5, seed: int = 0,
             depth_matched: bool = True) -> Result:
    if base not in stages or len(stages) < 2:
        raise ValueError("base must be one of at least two stages")
    depth = _depth_table(pred, stages)
    common = dict(base=base, n=n, repeats=repeats, depth=depth)
    if empty_fraction(sample_dense(pred, 5000, np.random.default_rng(seed))) > EMPTY_FRACTION_LIMIT:
        return Result(**common, gate="empty",
                      warnings=("More than half of the cells are all-zero; no placement.",))

    runs = [_one_repeat(pred, stages, base, n, np.random.default_rng([seed, i]), depth_matched)
            for i in range(repeats)]
    ed_by_stage = {}
    for label in stages:
        vals = [r[0][label] for r in runs]
        ed_by_stage[label] = (min(vals), float(np.mean(vals)), max(vals))
    vr = float(np.mean([r[1] for r in runs]))
    floor = float(np.mean([r[2] for r in runs]))
    denom = float(np.mean([r[3] for r in runs]))
    warnings = list(_depth_warnings(depth))
    if vr < COLLAPSE_LIMIT:
        warnings.append(f"Variance ratio vs {base} is {vr:.3g} (< {COLLAPSE_LIMIT}): the prediction "
                        "has collapsed towards a mean profile; no placement.")
        return Result(**common, gate="collapsed", ed_by_stage=ed_by_stage, variance_ratio=vr,
                      warnings=tuple(warnings))

    r_by_stage = {k: v[1] / denom for k, v in ed_by_stage.items()}
    r_base = r_by_stage[base]
    label = _label(ed_by_stage[base][1], floor, denom, r_base)
    if label == "copy_like":
        warnings.append(f"Prediction looks like a copy of the input stage {base} after depth matching.")
    elif label == "very_close":
        warnings.append(f"Prediction is very close to the input stage {base}.")
    return Result(**common, label=label, r_base=r_base, r_by_stage=r_by_stage,
                  ed_by_stage=ed_by_stage, ed_floor=floor, denominator=denom,
                  r_floor=floor / denom, variance_ratio=vr, warnings=tuple(warnings))
