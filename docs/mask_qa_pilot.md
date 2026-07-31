# SAM-3 mask QA — 1,000-frame pilot

## Why

`scripts/run_calibration_eval.py` segments every annotated frame with SAM-3 and reduces mask +
metric depth to a predicted subject-to-camera distance. Mask quality gates everything
downstream — Stage-1 alignment, Stage-2 calibration, the exported paired dataset — but nothing
in the pipeline measures it. We do not know how many masks are wrong, in what way, or what it
would cost to find out across the corpus.

This pilot builds the QA funnel on ~1,000 frames and **measures the dials** so the corpus-scale
cost is extrapolated from numbers rather than assumptions:

| Dial | Where it comes from | Why it matters |
|---|---|---|
| `f_v` — fraction needing a vision call | Stage 2 pre-filter | Multiplies the whole vision bill |
| Bad-mask rate + failure mix | Stage 3 triage | Tells us whether QA is worth doing at all |
| Verifier precision/recall | Gold set (Stage 5) | Tells us whether the rate above can be believed |
| Escalation rate | Threshold fitted on the gold set | A first-class cost line, not a rounding error |

**Scope is mask quality as such, not distance accuracy.** Ground-truth `distance` is used only
for sampling and as a free geometric covariate in the pre-filter. No stage of this pilot scores
a predicted distance against ground truth.

**Correction is deliberately out of scope.** Choosing correction strategies before the failure
mix is measured means guessing which ones matter. The decision gate below is what tells us; the
correction stage is a separate, data-driven piece of work that follows it.

## The funnel

```
sample.py ──▶ run_calibration_eval.py ──▶ prefilter.py ──▶ render_overlays.py ──▶ triage.py
   1,100         masks + status              f_v              review panels        verdicts
   frames                                     │                                       │
                                              ├── empty_mask ──▶ auto-fail (£0)       │
                                              └── clean singleton ──▶ auto-pass       │
                                                                                      ▼
                                    report.py goldset ──▶ hand labels ──▶ report.py summarise
```

Every stage emits a CSV keyed by `(video_name, frame_idx, instance_idx)`.

Run order and the SLURM specifics are in [`slurm/README.md`](../slurm/README.md).

## Methodology decisions that carry the result

### Sampling

Drawn from `data/annotations_20260709_with_fps_clean.csv`, the QC'd generation. The older
`annotations_06052026.csv` has `frame_idx` values inconsistent with the probed fps, so some
rendered "frames" would be the wrong frame and the verifier would score a frame-indexing bug as
a mask failure — corrupting the pilot's headline number.

**Only `TIMESTAMP_PAST_END` and `VIDEO_NOT_ON_DISK` flags are excluded.** `qc_annotations.py
--fix` already *drops* `ABSURD_DISTANCE`/`ZERO_DISTANCE` rows and already *repairs*
`FRAME_IDX_FPS_MISMATCH` ones when it writes the clean CSV. Excluding all flagged rows would
discard a large, systematically-chosen slice of good data — every row whose `frame_idx` needed
fixing — and bias the cohort.

**Flags are joined on `frame_timestamp`, not `frame_idx`.** See `scripts/qc_exclusions.py`: the
flags CSV carries the pre-fix index and the clean CSV recomputes it, so a direct `frame_idx`
join cannot match the very rows that most need matching. This was a live bug affecting
`calibrate_depth.py`, `benchmark_calibration.py` and `export_calibrated.py`, not just this
pilot.

**Allocation is equal-per-site (~143 each), not proportional.** Proportional sampling would give
beauvois ~23 frames, from which no per-site rate is estimable. The cost is that small sites are
over-represented, so every row carries an `inclusion_weight` and the report gives both an
unweighted per-site rate and a corpus-weighted aggregate. **The weighted one is what multiplies
into the corpus estimate** — using the unweighted rate there would be a real error.

### Pre-filter

Two choices do the work, both in `scripts/qa/prefilter.py`:

**Area is normalised by distance before it is z-scored.** Within one clip the subject walks
toward and away from the camera — that *is* the calibration protocol — so mask area varies by
close to an order of magnitude and a raw per-video z-score flags correct masks. Apparent area
falls as 1/distance², so `log(area) + 2·log(distance)` is flat across a clip.

**Dispersion uses median/MAD with a floor, not mean/std.** Clips have a median of ~6 annotated
frames; at that n a single grossly-wrong mask inflates the standard deviation enough to hide
itself. The floor (`--min-area-dispersion`) matters just as much in the other direction: MAD
divides, so a clip whose masks barely vary — a subject standing still — would otherwise turn
every one-pixel wobble into a twenty-sigma outlier and flag the steadiest clips in the corpus.

**Dispositions come from the results CSV `status` column, prefix-matched, never from file
presence.** `empty_mask` frames deliberately write no mask JSON, so "no file" cannot distinguish
*empty* from *never run*. `depth_error` frames still have a valid mask and are examined.

### Triage

Batch API only, `claude-haiku-4-5`, structured output. There is no fallback path: routing around
a missing key would mean the measured dials came from a different model on a different prompt,
making the corpus extrapolation non-comparable.

Panels pair a full frame with a zoomed crop. `bleed` and `split` are pixel-boundary judgements
and are read poorly from a full-frame translucent overlay where the subject is 40px tall; the
full frame is what makes `wrong-subject` and `multiple` visible.

## Gold set protocol

The gold set is the only independent trust anchor in the pilot. Three rules make it one.

**1. Draw it stratified, not uniformly.** `report.py goldset` draws:

| Slice | Default n | What it buys |
|---|---:|---|
| Per predicted verdict class | 30 each | Per-class **precision** with a usable per-class n |
| Uniform over triaged panels | 50 | **Recall** — a stratified draw alone cannot give it |
| Pre-filter auto-passes | 40 | Silent pre-filter **false negatives**, which nothing else in the funnel would ever see |

A 150-frame uniform set cannot support this. At a ~15% bad rate that is ~22 bad masks across
five classes — roughly four each — and 3 correct out of 4 is a 95% Wilson interval of about
30–99%. The report prints Wilson intervals throughout for this reason: **read the interval, not
the point estimate.**

**2. Label blind.** `gold_template.csv` contains the overlay path and a blank `gold_verdict`
column, and deliberately *not* the model's verdict. If the labeller can see what the verifier
said, precision and recall measure agreement-under-anchoring rather than accuracy.

**3. Split tune/holdout.** The escalation confidence threshold is fitted on the tune half; every
reported precision and recall is on the holdout half, which the threshold never saw. Fitting and
reporting on the same labels would make the headline trust number an in-sample fit.

Label with exactly one of: `ok`, `empty`, `wrong-subject`, `bleed`, `split`, `multiple`. When
more than one applies, pick the one that would most mislead a downstream distance estimate:
`wrong-subject` > `multiple` > `bleed` > `split`. This is the same tie-break the model is given,
so a disagreement is a real disagreement rather than a difference of convention.

## Decision gate

**Write the thresholds below into this file, with numbers, before running `report.py
summarise`.** Pre-registering them is the whole point: thresholds chosen after seeing the
results are not a gate, they are a rationalisation.

Proceed to the corpus-scale run only if **all** of the following hold on the **holdout** half:

| Criterion | Threshold | Fill in before looking |
|---|---|---|
| ok-vs-bad precision (lower CI bound) | ≥ ____ | |
| ok-vs-bad recall (lower CI bound) | ≥ ____ | |
| `f_v` (corpus-weighted) | ≤ ____ | |
| Projected corpus cost | ≤ £____ | |

Two further judgements, which are not pass/fail but must be recorded before the numbers are
seen — write down what you would conclude in each case:

- **If the bad-mask rate is very low** (say under 5%), the honest conclusion may be that mask QA
  is not worth doing at corpus scale at all, and the pilot has succeeded by telling you so.
- **If the unannotated stratum's mix differs sharply** from the annotated one, the corpus cost
  table is an upper bound on usefulness rather than a forecast, and the population question has
  to be settled before spending.

## Known limitation: the corpus population is not the pilot population

1,850 videos × ~500 frames reaches ~1M only if *every* frame is segmented. This pilot measures
on ~12.5k annotated frames, where a sign-holder is present by construction. On unannotated
frames the sign-holder is frequently absent, so `empty` is often the *correct* mask rather than
a failure, and the measured failure mix will not transfer unchanged.

The 100-frame unannotated stratum exists to size that shift. The report prints both strata and
never blends them. This is an assumption stated, not a measurement made — treat the corpus cost
table accordingly.

## Reproducing the cohort

`scripts/qa/sample.py` is seeded (`--seed`, default `20260731`). Given the same annotations and
flags CSVs it reproduces the cohort exactly. Because `data/` and `outputs/` are both untracked,
the committed record of what was actually drawn is
[`docs/qa_cohort_1000.csv`](qa_cohort_1000.csv) — `video_name, frame_idx, stratum`.
