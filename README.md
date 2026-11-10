# vec-stagescope

A depth-matched stage-proximity diagnostic for [Virtual Embryo Challenge](https://virtualembryo.ai/challenge)
prediction files (Task 1 and Task 2). Unofficial community tool. **This repository contains no challenge data.**

## The question it answers

> *Did my model learn something, or am I submitting a degraded copy of the input stage?*

The trivial baseline on every board is the input (earlier) stage itself. A prediction that is the input stage
with some genes dropped, noise added or cells subsampled is no longer a byte-identical copy, so diff-style
copy detectors miss it. Worse, in the challenge data the published stages also differ in **detection depth**
(detected genes per cell), and a plain distance mixes that technical difference with biology: a thinned copy
of an early stage can look *closer to a later stage than to its own source*.

vec-stagescope removes that confound first, then measures distance:

1. **Dropout matching.** For every pair it compares, the population with more detected genes per cell is
   brought down to the other's median by dropping detected entries at random, then re-normalised to the
   challenge's log1p-CP10k scale. (This is dropout matching on normalised values, not count-level binomial
   thinning.)
2. **Energy distance** in the full shared gene space, with the same definition as veckit's
   `energy_distance` (no PCA, Euclidean, matmul identity), on `n = 1500` cells per side, repeated 5 times.
3. **Relative ratio** `r = ED(prediction, stage) / ED(input stage, nearest other stage)`, reported against
   *every* training stage, **raw and depth-matched side by side**, next to a **noise floor**: the matched
   distance between two disjoint samples of the input stage.

## Copy label: not shipped

Our pre-registered calibration required every disguised-copy construction to fall below the copy threshold
on real data; uniform detected-gene dropout did (depth-matched r ≈ 0.001 vs ≥ 0.96 for distinct stages), but
additive noise, expression-dependent dropout and count-like thinning did not, so per our own rule the tool
reports raw and depth-matched distances and r side by side without a copy verdict. When raw r is small but
depth-matched r is large, the report flags it: depth matching can discard signal when the depth difference
comes from filled-in zeros rather than lower detection. A depth-matched copy test restricted to uniform
dropout, and a min(raw, matched) rule, are pre-registered for v0.2 (`PREREGISTRATION_v0.2.md`) and will be
evaluated on a board not used here.

**Post-hoc observation, not validated:** on the boards we looked at, depth matching recovered copies hidden
by uniform dropout that the raw distance missed (raw r between about 0.4 and 25 vs matched r near 0; real
other stages about 1 or more). With additive noise the opposite happened: matched r became far larger than
for genuinely different stages (about 4 to 33). On the two 500-gene Task 2 boards raw r stayed small for the
noisy copy, so the divergence flag fires; on the ~32,000-gene Task 1 board noise filled so many zeros that
raw r was also above 1, so **neither column and no flag catches a noisy copy there**. On Task 1 the
count-like construction zeroed so much of the matrix that it hit the collapse gate. Read the two columns
together, and do not read a large r as proof that a prediction is not a copy.

## Output

```
Energy distance (ED) and r to each training stage, raw and depth-matched:
  stage        raw ED   raw r  matched ED               range  matched r
  S1            0.012   0.001       0.004      [0.002, 0.006]      0.001  <- input stage
  ...
noise floor (two disjoint samples of S1, matched): 0.004
```
(Layout only; synthetic numbers, not challenge data.)

Two gates stop placement before any distance is read: more than half of the cells all-zero (`empty`), or a
variance ratio (veckit's definition, mean per-gene variance vs the input stage) below 0.1 (`collapsed`;
the smaller of the raw and depth-matched ratios is used, since dropout matching can add spread to a collapsed
profile). Other community tools already check these; here they only prevent a misleading reading.

## Install and run

```bash
pip install git+https://github.com/Sooosii/vec-stagescope
vec-stagescope check my_prediction.h5ad --board T2:heart:val_interp --data-dir path/to/training_h5ad/
```

`--data-dir` is the folder with the training files you downloaded as a registered participant
(`E6.75.h5ad`, `E7.25.h5ad`, `E8.0.h5ad`, `E8.25_late.h5ad`, `E8.75.h5ad`, `E9.5.h5ad`, `E8.5_RNA.h5ad`,
`E9.5_RNA.h5ad`). Options: `--json out.json`, `--strict` (exit 1 on any warning), `--n`, `--repeats`,
`--seed`. Exit codes: 0 ran, 1 warning with `--strict`, 2 input error. Task 1 files are read in full;
expect ~1.5 GB RAM.

| Board | Stages compared | Input stage |
|---|---|---|
| `T2:embryo:val_interp` / `test_interp` | E6.75, E7.25, E8.0 | E7.25 (assumed on test) |
| `T2:heart:val_interp` | E8.25, E8.75, E9.5 | E8.25 |
| `T2:heart:val_extrap` / `test_extrap` | E8.25, E8.75, E9.5 | E9.5 (assumed on test) |
| `T1:val` / `T1:test` | E8.5, E9.5 | E9.5 (assumed on test) |

Task 3 is out of scope. On test boards the input stage is an assumption; the report says so and shows `r`
against every stage so you can see if the assumption is wrong.

## How the thresholds were chosen, and limits

- Synthetic tests (in `tests/`, no challenge data) cover exact, dropped-out (70/50/30%), noisy copies, and
  three corruptions the method does **not** undo by construction: expression-dependent dropout, count-like
  thinning, and a copy mixed with 15% of a neighbouring stage.
- The calibration rule was written before looking at real data: thresholds chosen on one board, then
  tested untouched on the others, locally. Its outcome was that no copy threshold separated all disguised
  copies from real stage differences, so no label ships (above). Per the challenge terms no table of
  data-derived numbers is published; the few values quoted above are summaries of that calibration.
- Each board has only two "other" stages, so the evidence that the thresholds separate copies from real
  stage differences is thin.
- Each published stage is a single embryo. A difference between stage files is development + embryo +
  dissection + depth. The tool measures **similarity to published stage files**, not developmental time, and
  is not a score or a leaderboard predictor.
- Dropout matching removes cell-level depth differences; gene-specific detection differences remain.
- The repeat ranges cover sampling and dropout randomness only, not embryo-to-embryo variation; they are
  ranges, not confidence intervals.

## Deviations from the written plan

- **Noise floor.** The floor is the unbiased (U-statistic) energy distance between disjoint samples, so on
  real data it is close to zero and can be slightly negative; this is expected, not an error. A rule using
  `max(floor, 0.01 × stage separation)` was chosen after seeing that; it only affected the copy label, which
  does not ship.
- **Collapse gate.** It uses the smaller of the raw and depth-matched variance ratios, not only the
  depth-matched one as planned, because dropout matching adds spread to a collapsed profile.

## Related work

veckit removed its `stage_discrimination` metric and noted that the question is legitimate but needs a
scale-independent distance (`common/core_metrics.py`, `cc3859a`). vec-stagescope does not score that
question; it asks a related one as a diagnostic. VEC Diff and the `copy_last` baseline catch byte-identical
copies and their score; this tool targets copies hidden by dropout, noise or subsampling.

Data: Virtual Embryo Challenge, Neil Chi lab (UC San Diego). Energy distance and variance ratio follow
[veckit](https://github.com/aristoteleo/veckit) (MIT).

## Türkçe özet

Tahmin dosyasının girdi evresinin bozulmuş (gen düşürülmüş, gürültü eklenmiş, alt örneklenmiş) bir kopyası
olup olmadığını, algılama derinliğini eşitledikten sonra enerji uzaklığıyla ölçer. Evre değil, yayımlanmış
evre dosyalarına benzerlik ölçer; puan değildir. Depoda yarışma verisi yoktur.
Kopya etiketi yayımlanmadı: ön kayıtlı kalibrasyon, gizlenmiş kopyaların tümünde ayrım istiyordu; yalnız
tekdüze gen düşürmede sağlandı.

## License

MIT
