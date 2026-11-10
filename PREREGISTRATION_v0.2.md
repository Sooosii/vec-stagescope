# Pre-registration for vec-stagescope v0.2

Written 10 November 2026, after the v0.1 calibration and **before** any v0.2 rule is run on data.
Boards already looked at during v0.1 (`T2:heart:val_interp`, `T2:embryo:val_interp`, `T1:val`) do not count
as validation for anything below.

## Hypotheses

- **H1 (uniform-dropout copy label).** A prediction built from the input stage by uniform dropout of detected
  genes (keep fraction 0.7, 0.5 or 0.3) has depth-matched `r < 0.10`, and every real other training stage has
  depth-matched `r > 0.30`. If both hold, a label "copy hidden by uniform dropout" ships, worded as covering
  only that construction.
- **H2 (min rule).** `r_min = min(raw r, depth-matched r)` separates *all* v0.1 disguised-copy constructions
  (exact, uniform dropout 0.7/0.5/0.3, additive noise sd 0.1, expression-dependent dropout, count-like
  thinning, copy + 15% neighbouring stage) from real other stages with one threshold: all copies
  `r_min < 0.30` and all real other stages `r_min > 0.50`. If it holds, a general copy label ships based on
  `r_min`.

## Where they are tested

The first board not used in v0.1 for which the needed files are available to the author (in order of
preference: `T2:heart:val_extrap` with input stage E9.5, then any board whose stages become public later).
Caveat: `T2:heart:val_extrap` uses the same stage files as the v0.1 heart board, only with a different input
stage (E9.5 copies were never built in v0.1); it is a weaker test than a fully unseen board, and the report
will say so.
Thresholds above are fixed now and are not tuned on that board. If a hypothesis fails there, its label does
not ship and the failure is reported.

## Fixed settings

n = 1500 cells per side, 5 repeats, seed 0, same corruption generators as `tests/synth.py` with seed 1,
copy built from a 6000-cell sample of the input stage (seed 11).
