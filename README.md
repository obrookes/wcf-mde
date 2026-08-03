# wcf-mde

Wild Chimpanzee Foundation camera-trap reference videos and ground-truth
distance annotations, plus a pipeline that evaluates monocular distance
estimation against that ground truth. See `CLAUDE.md` for the dataset layout.

## Pipeline

`scripts/run_calibration_eval.py` walks `data/annotations_06052026.csv`, and
for each video that has annotated frames:

1. resolves `video_name` to an actual video file under `data/` (via
   `data/list_reference_videos.xlsx`, see `scripts/video_lookup.py`)
2. extracts **all** annotated frames for that video by sequential decode
   (`scripts/frame_source.py`)
3. estimates per-pixel metric depth **jointly** across all annotated frames in
   one inference call — both Pi3X and DA3NESTED are multi-view architectures
   whose cross-frame attention and scale estimation improve with N > 1
   (this dataset has a maximum of 16 annotated frames per video, median 6)
4. segments each frame with **SAM-3** using a text prompt (default
   `"person holding sign"`)
5. reduces mask + depth to a single predicted distance (mean-in-mask, centroid)

and writes `video_name, frame_idx, frame_timestamp, distance_gt, depth_model,
status, mask_area_px, depth_mask_mean, depth_centroid` to an output CSV, plus a
summary (status counts, MAE/RMSE/bias vs. ground truth) at the end.

## Environment setup

Requires a CUDA-capable GPU for any non-trivial run (CPU works for smoke-testing
only). Tested with conda env `dap-3_py3-11` (Python 3.10/3.11, CUDA 12.6 torch).

```bash
conda create -n dap-3_py3-11 "python>=3.7,<3.11"
conda activate dap-3_py3-11

# torch + ultralytics (SAM-3)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics openpyxl

# SAM-3's promptable text segmentation needs Ultralytics' CLIP fork
pip install "git+https://github.com/ultralytics/CLIP.git"

# Pi3 (metric monocular depth) — clone and install in editable mode
git clone https://github.com/yyfz/Pi3 third_party/Pi3
pip install -e third_party/Pi3
```

A SAM-3 checkpoint is required. Vanilla `sam3.pt` is gated on Hugging Face;
the default `--sam3-checkpoint` points at the SA-FARI wildlife checkpoint —
adjust the path to wherever you have a checkpoint locally:

```
/home/dl18206/projs/Unmarked-Anything/weights/sam3/safari_checkpoint_hf.pt
```

The depth backend is selected with `--depth-model {pi3x,da3}` (default `pi3x`):

- **Pi3X** — `Pi3X.from_pretrained("yyfz233/Pi3X")` downloads weights from
  Hugging Face on first run (only `Pi3X`, not plain `Pi3`, gives metric-scale
  depth). Requires `third_party/Pi3` cloned and installed (see above).
- **Depth Anything 3 (DA3NESTED)** — clone and install similarly:
  ```bash
  git clone https://github.com/bytedance-seed/depth-anything-3 third_party/depth_anything_3
  pip install -e third_party/depth_anything_3
  ```
  `DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")`
  downloads weights on first run and outputs metric depth in metres natively
  (no external camera-intrinsics calibration needed, unlike `DA3METRIC`).
  Like Pi3X and the SA-FARI checkpoint, it's licensed CC BY-NC 4.0
  (non-commercial).

## Running

```bash
python scripts/run_calibration_eval.py \
  --device cuda \
  --results-csv outputs/calibration_results.csv
```

Useful flags (all have sensible defaults pointing at `data/`):

| Flag | Purpose |
|---|---|
| `--limit N` | process only the first N annotation rows (smoke-testing) |
| `--device {auto,cpu,cuda}` | compute device (default `auto`) |
| `--sam3-checkpoint PATH` | SAM-3 checkpoint to load |
| `--sam3-prompt TEXT` | text prompt for SAM-3 segmentation |
| `--depth-model {pi3x,da3}` | metric depth backend to evaluate (default `pi3x`) |
| `--pi3-model-id ID` | Pi3X model id/path (default `yyfz233/Pi3X`) |
| `--da3-model-id ID` | Depth Anything 3 model id/path (default `depth-anything/DA3NESTED-GIANT-LARGE-1.1`) |
| `--conf FLOAT` | SAM-3 confidence threshold (default `0.25`) |
| `--results-csv PATH` | where to write per-row results |
| `--overlay-dir PATH` | dump frame\|mask\|depth sanity-check PNGs (requires `--limit`) |

Run `python scripts/run_calibration_eval.py --help` for the full list.

### Qualitatively checking outputs

For a small smoke-test run, pass `--overlay-dir` (it requires `--limit`, since
dumping a PNG per frame is only meant for spot-checks, not full runs):

```bash
python scripts/run_calibration_eval.py --device cuda --limit 20 \
  --results-csv outputs/smoke_results.csv --overlay-dir outputs/overlays
```

Each PNG is named `<video_stem>_frame<NNNNNN>.png` and shows the decoded
frame (with the SAM-3 mask outline + bbox-center marker) beside a colourised
depth map with a metres-labelled scale bar, annotated with `status`,
`mask_area_px`, `depth_mask_mean`, `depth_centroid`, and `distance_gt`. The frame index burned into the image is
the one the script itself decoded (`iter_frames_at_indices`'s loop counter),
not a value re-read from the CSV — comparing it against the visible frame
content is a direct check that the correct frame was extracted.

## Per-video depth calibration

The metric depth backends are residually mis-scaled per camera. Because each
annotated frame gives one sparse `(predicted-depth-at-subject, true-distance)`
point, we can fit a small **per-video** (or per-camera) transform from that
video's annotated frames and apply it to its depth maps — the inference-time,
per-deployment idea from
[`timmh/distance-estimation`](https://github.com/timmh/distance-estimation).
This is a three-step workflow — QC the annotations once, run the expensive
depth/SAM-3 inference once, then fit/re-fit the cheap calibration as many times
as you like — so switching `--method` or `--calib-level` never requires
re-running inference.

**Step 0 — QC the annotations** (`scripts/qc_annotations.py`, pandas only, no
torch): flags rows that are impossible under *any* annotation protocol —
`frame_idx` inconsistent with the video's probed fps, timestamps past the
video's end, videos missing from disk, and absurd (>100m by default) or
non-positive distances. It never edits a `distance` value, only flags/drops
rows — sequence-based heuristics (repeated values, implied speed, direction
reversals) are deliberately *not* used, since videos are annotated by 2+
people and sparsely, so such heuristics can't distinguish bugs from valid
labels (see `scripts/qc_annotations.py`'s module docstring).

```bash
python scripts/qc_annotations.py data/annotations_20260709_with_fps.csv --fix
```

Writes `data/qc_flags_<basename>.csv` (one row per flagged `(video_name,
frame_idx)`, with a `reason`) and, with `--fix`, `<basename>_clean.csv` (same
rows minus `ABSURD_DISTANCE`/`ZERO_DISTANCE` ones, `frame_idx` recomputed from
timestamp+fps). `data/annotations_06052026.csv` (CLAUDE.md's originally
documented file, still `run_calibration_eval.py`'s default) predates this QC
pass; `data/annotations_20260709_with_fps.csv` / `_clean.csv` are the newer,
QC'd/fps-corrected generation — prefer the latter, and feed the resulting
`qc_flags_*.csv` into Stage 2 below rather than re-running Stage 1 against it
(Stage 1 is the expensive step; see Step 2).

> **The flags CSV and `_clean.csv` do not share a `frame_idx` key.** `--fix`
> recomputes `frame_idx = round(frame_timestamp × fps)` for *every* on-disk row,
> not just the flagged ones, while the flags CSV keeps the original values. A
> direct `(video_name, frame_idx)` join between them therefore cannot match any
> `FRAME_IDX_FPS_MISMATCH` row — by construction, those are exactly the rows
> whose index changed — and silently under-excludes. Anything consuming
> `--qc-flags` resolves the join through `frame_timestamp` instead
> (`scripts/qc_exclusions.py`), which is why `calibrate_depth.py`,
> `benchmark_calibration.py` and `export_calibrated.py` all take an
> `--annotations-csv`: it tells them which `frame_idx` space the results are in.
> Unit tests: `python scripts/test_qc_exclusions.py`.

**Step 1 — persist the original depth maps** (adds flags to the eval run):

```bash
python scripts/run_calibration_eval.py --device cuda --limit 40 \
  --results-csv outputs/smoke_results.csv \
  --save-depth-dir outputs/depth_orig --save-masks-dir outputs/masks
```

`--save-depth-dir` writes one fp16 `.npy` per *decoded* frame
(`<video_name>_frame<NNNNNN>_orig.npy`, keyed by the flat annotation name so it
never collides across camera folders) — unconditionally, even if SAM-3 finds no
detection in that frame. `--save-masks-dir` writes one `*_masks.json` per frame
*only when SAM-3 detects the prompt* (`status=empty_mask` frames get no mask);
required for `--align ransac` below. Because of this, don't expect the two
output dirs to have matching file counts — that's expected, not a bug. The
results CSV also carries the subject centroid (`center_x`, `center_y`,
`center_y_norm`).

**Step 1.5 — benchmark methods before writing anything** (optional, CPU only,
no torch): `scripts/benchmark_calibration.py` sweeps `--method`/`--degree`/
`--robust`/`--anchor`/`--align`/`--ref-frame-method` combinations in one
process and reports the same leave-one-out (LOO) metric as Step 2 for each —
refit the calibrator on every point but one, predict the held-out point, and
average that held-out error, so the reported accuracy reflects a frame the fit
never saw rather than one it memorized. It never writes a `*_calib.npy` (there
is no `--out-dir`/`--viz` on this script at all), so it's safe to sweep broadly
before committing to a `--method` for Step 2:

```bash
python scripts/benchmark_calibration.py \
  --results-csv outputs/smoke_results.csv \
  --qc-flags data/qc_flags_annotations_20260709_with_fps.csv \
  --depth-dir outputs/depth_orig --masks-dir outputs/masks \
  --aligns none ransac --out outputs/calibration_benchmark.csv
```

Writes one row per combination to `--out` (mean LOO MAE uncalibrated vs.
calibrated, mean improvement, % of groups improved, and the same
close/medium/long distance-bucket breakdown as Step 2) and prints a ranked
top-N table (`--sort-by improvement|mae_cal`) to the console. `--depth-dir`/
`--masks-dir` here are read-only inputs (only needed when `--aligns` includes
`ransac`, to re-derive Stage-1 alignment per group) — never written to.

**Step 2 — fit calibration and write calibrated maps** (CPU only, no torch):

```bash
# --qc-flags excludes annotation rows already flagged by Step 0 (e.g. a
# handful of clips whose ground-truth distance is a data-entry typo, like
# 401410m instead of ~10m) before fitting/reporting -- one bad point can send
# a clip's small-n LOO fit to an extreme slope and swamp the aggregate MAE.
python scripts/calibrate_depth.py \
  --results-csv outputs/smoke_results.csv \
  --depth-dir outputs/depth_orig --out-dir outputs/depth_calib \
  --method linear --calib-level clip \
  --qc-flags data/qc_flags_annotations_20260709_with_fps.csv --viz
```

For every group it fits the chosen transform on that group's sparse points,
applies it to each saved map (`<...>_calib.npy`), and — with `--viz` — writes an
`orig | calib` colourised PNG. It also writes `outputs/calibration_fits.csv`
(per-group method, fitted params, and **leave-one-out** calibrated-vs-uncalibrated
MAE), prints an aggregate improvement summary, and a breakdown of LOO MAE by
ground-truth distance range (close 1-4m / medium 4-8m / long 8m+, pooled across
all points) so accuracy at different ranges is visible, not just one number.

| flag | purpose |
|---|---|
| `--calib-level {clip,cam}` | fit one calibrator per clip (default, e.g. one `DSCF0005.AVI`) or pool all clips under one camera-reference folder into a single fit (`cam`; needs `--video-list-xlsx`/`--data-dir`, both default to `data/`) |
| `--qc-flags PATH` | exclude `(video_name, frame_idx)` rows flagged by Step 0's `qc_annotations.py` before fitting/reporting |
| `--annotations-csv PATH` | the annotations generation Stage 1 was run against, used to resolve `--qc-flags` into that file's `frame_idx` space (see the join note in Step 0). Defaults to `data/annotations_20260709_with_fps_clean.csv` |
| `--align {none,ransac}` | optional cross-frame background alignment (`scripts/alignment.py`) before calibration; default `none` since it isn't known ahead of time whether this helps on top of this dataset's already-joint depth inference — compare `calibration_fits.csv`'s `align_*` columns across runs to find out. Needs `--masks-dir` |
| `--robust` | median-ratio / Theil-Sen fit for `scale`/`linear` instead of least-squares |

`--method` selects the calibration model (`scripts/calibration.py`):

| `--method` | model | notes |
|---|---|---|
| `scale` | `d_cal = s·d` | 1 param, robust; low-N fallback |
| `linear` | `d_cal = a·d + b` | the simple default |
| `disparity` | affine in `1/d` | depth error is often affine in disparity |
| `poly` | degree-k polynomial (`--degree`) | capped to what the point count supports |
| `poly2d` | `f(depth, vertical-position)` | also uses the subject's image row (ground-plane geometry) |

Omit `--depth-dir` to only fit and report leave-one-out MAE from the CSV (handy
for quickly comparing methods before writing any maps). Re-run with a different
`--method`/`--calib-level`/`--qc-flags` on the same `--depth-dir` to compare
without re-running inference.

Unit tests for the calibration maths: `python scripts/test_calibration.py`.
For the QC-flag join: `python scripts/test_qc_exclusions.py`.

**Exporting calibrated depth maps + frames** (`scripts/export_calibrated.py`,
CPU only): bundles Step 2's `*_calib.npy` maps with their source video frames
into a paired dataset, keeping only QC-valid frames — any `(video_name,
frame_idx)` present in the Step 0 flags CSV is dropped, whether or not Step 2
was run with `--qc-flags`.

```bash
python scripts/export_calibrated.py \
  --calib-dir outputs/depth_calib --out-dir outputs/export
```

writes:

```
outputs/export/
  depth_maps/{video_name}_frame{idx:06d}_calib.npy   # copied as-is (fp16)
  frames/{video_name}_frame{idx:06d}.png             # re-decoded from the source video
```

Frames aren't persisted anywhere by the pipeline, so they are re-decoded from
the videos under `data/` (via `list_reference_videos.xlsx`). The two dirs stay
strictly 1:1: a frame that fails to resolve or decode drops its depth map too.
`--frame-format jpg` trades losslessness for size; `--qc-flags` overrides the
default `data/qc_flags_annotations_20260709_with_fps.csv`.

## Mask QA — how the pieces fit

Two interchangeable backends triage SAM-3 masks and converge on one contract, the
verdicts CSV (`VERDICT_FIELDS`, defined in `scripts/qa/verdicts_schema.py`):

```
export (frames+masks)
  ├─ A: prefilter → render_overlays → triage (VLM, paid) ──┐
  └─ B: score_masks → heuristic_verdicts (CPU, free) ──────┴─▶ verdicts.csv
                                                                    │
      review_server → corrections.csv → apply_corrections ◀─────────┘
```

**Path A — VLM triage (Batch API, paid; details: [Mask QA pilot](#mask-qa-pilot)):**

```bash
python scripts/qa/prefilter.py                        # -> outputs/qa/prefilter.csv (f_v)
python scripts/qa/render_overlays.py                   # -> outputs/qa/overlays/, overlay_manifest.csv
python scripts/qa/triage.py estimate                   # free: check the bill first
python scripts/qa/triage.py submit && python scripts/qa/triage.py poll
python scripts/qa/triage.py fetch                      # -> outputs/qa/verdicts_haiku.csv
```

**Path B — CPU heuristics (free, no API key, no GPU; details: [Mask triage](#mask-triage)):**

```bash
python scripts/score_masks.py --export-dir outputs/export --out-dir outputs/triage \
  --workers 12                                         # -> mask_scores.csv, review_queue.csv
python scripts/qa/heuristic_verdicts.py --scores outputs/triage/mask_scores.csv \
  --out outputs/qa/verdicts_heuristic.csv
python scripts/triage_contact_sheet.py --scores outputs/triage/mask_scores.csv \
  --export-dir outputs/export --out outputs/triage/worst.png --top 24  # eyeball the worst
```

B is the free first pass — no API key or GPU, just the mask JSONs and frames already on disk;
A costs money but reads the actual pixels through a VLM and is the only one of the two that can
tell "bleed" from "multiple" apart. Either output slots into the same downstream tooling: any
script that accepts `--verdicts` (`review_server.py`, `make_review_bundle.py`) takes A's or B's
CSV interchangeably.

## Mask QA pilot

See [Mask QA — how the pieces fit](#mask-qa--how-the-pieces-fit) for how this (the VLM path)
relates to the CPU-heuristic path in [Mask triage](#mask-triage).

Mask quality gates everything above — Stage-1 alignment, Stage-2 calibration,
the exported dataset — but nothing in the pipeline measures it.
`scripts/qa/` is a QA funnel that does, on a fixed 1,000-frame cohort, and
**measures the dials** so a corpus-scale cost can be extrapolated from numbers
rather than assumed: how far a free pre-filter cuts the vision-call count
(`f_v`), the bad-mask rate and failure mix, and how well a cheap verifier agrees
with a blind human gold set.

| Stage | Script | Runs on |
|---|---|---|
| 0 sample | `scripts/qa/sample.py` | login node |
| 1 masks | `scripts/run_calibration_eval.py` (unchanged; the sample CSV *is* an annotations CSV) | GPU job |
| 2 pre-filter | `scripts/qa/prefilter.py` — free, no inference | login node |
| 3a render | `scripts/qa/render_overlays.py` | CPU job |
| 3b triage | `scripts/qa/triage.py` — Batch API, `submit`/`poll`/`fetch` | login node (needs internet) |
| 4 review + correct | `scripts/qa/make_review_bundle.py` → `run_review.py` → `scripts/qa/apply_corrections.py` | laptop (review), login node (bundle/apply) |
| 5 report | `scripts/qa/report.py` — `goldset`, then `summarise` | login node |

```bash
python scripts/qa/sample.py                       # -> outputs/qa/sample.csv
bash   slurm/submit.sh stage1_eval                # -> results.csv, masks/
python scripts/qa/prefilter.py                    # -> f_v
bash   slurm/submit.sh stage3a_render             # -> overlays/
python scripts/qa/triage.py estimate              # free: check the bill first
python scripts/qa/triage.py submit && python scripts/qa/triage.py poll
python scripts/qa/triage.py fetch                 # -> verdicts_haiku.csv
python scripts/qa/report.py goldset               # -> blind labelling worksheet
python scripts/qa/report.py summarise --gold outputs/qa/gold_labelled.csv
```

**Reviewing the flagged masks (Stage 4).** `scripts/qa/review_server.py` is a
single-file stdlib HTTP UI over the flagged-bad verdict rows: one large tabbed
image (original mask / auto-fix preview / zoom / raw frame), button-or-keyboard
actions (accept, accept auto-fix, draw a SAM3 re-prompt box, discard, skip),
each decision appended immediately to a `corrections.csv`. Add `--include-ok`
to also spot-check the unflagged (`verdict=ok`) masks — they queue after the
flagged classes. The intended way to run it is **locally**, from a
self-contained tarball:

```bash
# cluster: pack frames + masks + queue + server into one archive
python scripts/qa/make_review_bundle.py \
  --verdicts outputs/qa/verdicts_haiku.csv \
  --frames-dir outputs/export/frames --masks-dir outputs/qa/masks \
  --out outputs/qa/review_bundle.tar.gz --include-ok
# laptop: download, extract, review at http://localhost:8765
tar xzf review_bundle.tar.gz && cd review_bundle
pip install -r requirements.txt && python run_review.py
# cluster: copy corrections.csv back, then apply the deterministic fixes
python scripts/qa/apply_corrections.py morph \
  --corrections corrections.csv --masks-dir outputs/qa/masks \
  --out-masks-dir outputs/qa/masks_corrected
```

Originals are never touched — corrected mask JSONs go to a parallel directory,
with an `applied.csv` log (last decision per key wins). `apply_corrections.py
sam3` (box re-prompts) is a stub until the SAM3 weights arrive. The review tool
displays the model verdict, so it must **never** be used to label the blind
gold set — see the protocol in the docs below.

Methodology, gold-set protocol and the pre-registered decision gate:
[`docs/mask_qa_pilot.md`](docs/mask_qa_pilot.md). SLURM run order, the
no-egress-on-compute-nodes constraint, and artifact sizes:
[`slurm/README.md`](slurm/README.md). The committed record of the drawn cohort
is `docs/qa_cohort_1000.csv`.

Unit tests (no GPU, data, torch, network or API key needed):

```bash
for t in scripts/test_*.py scripts/qa/test_*.py; do python "$t"; done
```
## Mask triage

See [Mask QA — how the pieces fit](#mask-qa--how-the-pieces-fit) for how this (the CPU-heuristic
path) relates to the VLM path in [Mask QA pilot](#mask-qa-pilot).

`scripts/score_masks.py` (CPU only) scores every predicted SAM-3 mask in an
export directory and ranks a human review queue. It answers one question —
**which masks are wrong and therefore need human time?** — from artifacts that
are already on disk: no model forward pass, no GPU, no ground truth, and no new
annotation. It's the cheap tier of a larger triage design; model-side signals
(confidence, test-time-augmentation agreement) need a SAM-3 checkpoint on the
machine and a re-run of inference, neither of which this step requires.

```bash
python scripts/score_masks.py \
  --export-dir outputs/export --out-dir outputs/triage --workers 12
```

Over the 9,588-frame export (12,197 masks, 572 stations) this takes ~5 minutes
on 12 workers and buckets 61.7% `auto_accept` / 38.3% `needs_review`.

Three tables are written:

| file | grain | contents |
|---|---|---|
| `mask_scores.csv` | one row per `(video_name, frame_idx, instance_idx)` | every signal as its own column, the named flags it tripped, a fused `triage_score`, and a bucket |
| `frame_scores.csv` | one row per frame | the **exhaustivity** signal — whether a subject-sized piece of the scene changed with no mask over it |
| `review_queue.csv` | | `mask_scores.csv` minus the accepted masks, ranked worst-first |

Every flag is named and traceable to something visible, so an annotator can see
*why* a mask surfaced rather than being handed an opaque score:

| flag | fires on | rate |
|---|---|---|
| `fragmented` | second connected component ≥ 25% of the largest — mask spans two subjects or shattered | 19.9% |
| `truncated` | ≥ 15% of the boundary on a frame edge | 15.3% |
| `no_residual_support` | < 10% of the mask overlaps any scene change vs. the clip's background | 8.5% |
| `holes` | enclosed background > 10% of mask area | 1.5% |
| `weak_boundary` | boundary gradient below the frame's own mean — the outline isn't on an image edge | 1.5% |
| `anomalous_area` | \|log-area z\| > 2.5 against the station's own area prior | 1.0% |
| `sliver` / `thin` | compactness < 0.05, or < 30% of area survives a 3×3 erosion | 0.2% |
| `tiny` / `engulfs_frame` / `is_banner` | degenerate: < 64 px, > 60% of frame, or mostly status bar | ~0% |

Three properties of this imagery drive the implementation, and getting any of
them wrong quietly breaks the signals:

- **The burnt-in status banner** (`scripts/banner.py`) — a Bushnell strip across
  the bottom ~20–24 of 404 rows. Left in, it makes every mask reaching the frame
  bottom look truncated, inflates gradient statistics with hard-edged text, and
  poisons the background model. Its height varies by camera model, so it's
  detected per station from the fact that its pixels barely change across
  frames; the per-row statistic is the **median** over columns, because the
  ticking clock does change and would defeat a mean.
- **Background scope is the clip, not the station** (`scripts/background.py`) —
  a camera-reference folder collects clips from repeat visits months apart, so a
  median across the whole station is a smear that matches no individual frame.
  Measured over 40 clips, station-scope backgrounds give a median mask/residual
  IoU of 0.05 against 0.27 for clip-scope. Clips too short for a trustworthy
  median get *no* residual signal rather than a misleading one (`background_scope`
  records which); that's 17% of masks here.
- **Three photometric modes**, not two — daylight colour, monochrome IR, and a
  rare magenta false-colour IR (~0.3% of frames). Their pixel statistics are
  unrelated, so the mode is part of the background's grouping key.

Mask quality and **exhaustivity are kept in separate tables on purpose**. They
are different questions in different units, and fusing them would let a frame of
immaculate masks hide a subject nobody segmented. The exhaustivity test is
deliberately not "how much residual is unmasked" — that is ~50% in normal frames
and flagged 43% of the corpus. It asks instead whether one *coherent,
subject-sized* blob was missed (opened to drop the halo around correct masks and
canopy speckle), sized against the frame's own median mask area. That fires on
17.8% of frames and does catch genuine partial masks — though it also fires on
wind-moved vegetation, and it is the weaker of the two checks.

Check the ranking by eye with `scripts/triage_contact_sheet.py`, which renders a
grid of masks with their flags printed on them:

```bash
# the worst of the queue
python scripts/triage_contact_sheet.py --scores outputs/triage/mask_scores.csv \
  --export-dir outputs/export --out outputs/triage/worst.png --top 24

# a random audit of what would be auto-accepted
python scripts/triage_contact_sheet.py --scores outputs/triage/mask_scores.csv \
  --export-dir outputs/export --out outputs/triage/accept_audit.png \
  --bucket auto_accept --sample 24 --seed 0
```

The accept-bucket audit is not optional decoration: accepted masks become
training data, so uncorrected errors there are self-reinforcing while
correction-based metrics keep looking healthy. It is the only instrument here
that detects that.

**The bucket boundaries are not calibrated.** Cutoffs are set from this corpus's
own signal distributions so each flag fires on a tail rather than on the bulk;
turning `auto_accept` into a claim about precision needs a labelled gold set,
which doesn't exist yet. Until then the bucket is a sort order and the flag
reasons are the product.

`review_queue.csv` and the contact sheets are as far as this table format goes on its own —
neither is a verdicts CSV. `scripts/qa/heuristic_verdicts.py` is the bridge into review: it
projects `mask_scores.csv`'s flags into a `VERDICT_FIELDS` row per mask (deriving `verdict` from
the fired flags; see its module docstring for the exact flag → verdict table), so
`review_server.py`, `make_review_bundle.py` and `apply_corrections.py` run on this backend's
output exactly as they do on `triage.py`'s. See [Mask QA — how the pieces
fit](#mask-qa--how-the-pieces-fit) for the copy-pasteable path.

Unit tests: `python scripts/test_mask_signals.py` (synthetic, no GPU/data).
