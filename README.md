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
  --output-csv outputs/calibration_results.csv
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
| `--output-csv PATH` | where to write per-row results |
| `--overlay-dir PATH` | dump frame\|mask\|depth sanity-check PNGs (requires `--limit`) |

Run `python scripts/run_calibration_eval.py --help` for the full list.

### Qualitatively checking outputs

For a small smoke-test run, pass `--overlay-dir` (it requires `--limit`, since
dumping a PNG per frame is only meant for spot-checks, not full runs):

```bash
python scripts/run_calibration_eval.py --device cuda --limit 20 \
  --output-csv outputs/smoke_results.csv --overlay-dir outputs/overlays
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

**Step 1 — persist the original depth maps** (adds flags to the eval run):

```bash
python scripts/run_calibration_eval.py --device cuda --limit 40 \
  --output-csv outputs/smoke_results.csv \
  --save-depth-dir outputs/depth_orig --save-mask-dir outputs/masks
```

`--save-depth-dir` writes one fp16 `.npy` per *decoded* frame
(`<video_name>_frame<NNNNNN>_orig.npy`, keyed by the flat annotation name so it
never collides across camera folders) — unconditionally, even if SAM-3 finds no
detection in that frame. `--save-mask-dir` writes one `*_masks.json` per frame
*only when SAM-3 detects the prompt* (`status=empty_mask` frames get no mask);
required for `--align ransac` below. Because of this, don't expect the two
output dirs to have matching file counts — that's expected, not a bug. The
results CSV also carries the subject centroid (`center_x`, `center_y`,
`center_y_norm`).

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
| `--align {none,ransac}` | optional cross-frame background alignment (`scripts/alignment.py`) before calibration; default `none` since it isn't known ahead of time whether this helps on top of this dataset's already-joint depth inference — compare `calibration_fits.csv`'s `align_*` columns across runs to find out. Needs `--mask-dir` |
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
