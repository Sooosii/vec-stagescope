import numpy as np
import pytest

from synth import (add_noise, count_like_thinning, expression_dependent_dropout, make_stage, mix,
                   uniform_dropout)
from vec_stagescope.proximity import diagnose

KW = dict(n=500, repeats=2, seed=0)


def _run(pred, stages, base="S3", **kw):
    return diagnose(pred, stages, base=base, **{**KW, **kw})


def test_exact_copy_is_copy_like(stages):
    res = _run(stages["S3"][:2000].copy(), stages)
    assert res.r_base < 0.10
    assert res.label == "copy_like"


@pytest.mark.parametrize("keep", [0.7, 0.5, 0.3])
def test_hidden_copy_is_caught_after_depth_matching(stages, keep):
    pred = uniform_dropout(stages["S3"], keep, seed=7)
    res = _run(pred, stages)
    assert res.r_base < 0.10, res.r_base
    assert res.label == "copy_like"


def test_raw_distance_misses_heavily_thinned_copy(stages):
    pred = uniform_dropout(stages["S3"], 0.3, seed=7)
    raw = _run(pred, stages, depth_matched=False)
    assert raw.r_base > 0.10


def test_noisy_copy_is_close(stages):
    res = _run(add_noise(stages["S3"], 0.1, seed=3), stages)
    assert res.r_base < 0.30
    assert res.label in {"copy_like", "very_close"}


def test_different_stage_is_far(stages):
    other = make_stage("S1", 2000, seed=99)
    res = _run(other, stages)
    assert res.r_base > 0.50
    assert res.label == "info"


# (a) foreign corruptions: not produced by the mechanism the method undoes
def test_expression_dependent_dropout_still_close(stages):
    res = _run(expression_dependent_dropout(stages["S3"], 0.5, seed=5), stages)
    assert res.r_base < 0.30, res.r_base


def test_count_like_thinning_still_close(stages):
    res = _run(count_like_thinning(stages["S3"], 0.5, seed=5), stages)
    assert res.r_base < 0.30, res.r_base


def test_copy_with_neighbour_mixture_still_close(stages):
    res = _run(mix(stages["S3"], stages["S2"], 0.15, 2000, seed=5), stages)
    assert res.r_base < 0.30, res.r_base


def test_floor_and_r_per_stage_reported(stages):
    res = _run(stages["S2"][:1500].copy(), stages, base="S2")
    assert res.ed_floor > -0.05   # unbiased estimator: may dip just below 0
    assert set(res.r_by_stage) == {"S1", "S2", "S3"}
    assert res.r_floor == pytest.approx(res.ed_floor / res.denominator)
    lo, mean, hi = res.ed_by_stage["S2"]
    assert lo <= mean <= hi


def test_no_label_when_stages_do_not_separate():
    a = make_stage("S1", 2000, seed=1)
    b = make_stage("S1", 2000, seed=2)
    res = diagnose(a[:1000].copy(), {"A": a, "B": b}, base="A", **KW)
    assert res.label == "no_separation"


def test_collapsed_prediction_is_gated(stages):
    pred = np.repeat(stages["S3"].mean(0, keepdims=True), 1000, axis=0)
    res = _run(pred, stages)
    assert res.gate == "collapsed"
    assert res.label is None and res.r_base is None


def test_empty_prediction_is_gated(stages):
    res = _run(np.zeros((1000, stages["S3"].shape[1]), np.float32), stages)
    assert res.gate == "empty"
    assert res.label is None


def test_deterministic(stages):
    pred = uniform_dropout(stages["S3"], 0.5, seed=7)
    a, b = _run(pred, stages), _run(pred, stages)
    assert a.r_base == b.r_base and a.ed_by_stage == b.ed_by_stage
