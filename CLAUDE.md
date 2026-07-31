# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`wcf-mde` holds Wild Chimpanzee Foundation camera-trap reference videos and ground-truth
distance annotations for a monocular distance-estimation (MDE) effort — estimating
subject-to-camera distance from single-camera footage — **plus the pipeline that evaluates and
calibrates it**. Read `README.md` for the pipeline; this file covers the dataset layout.

There is still no build system or linter, but there *is* a test suite: plain `test_*.py` files
under `scripts/` and `scripts/qa/`, each runnable directly (`python scripts/test_calibration.py`)
and pytest-compatible, with no GPU, data, torch, network or API key needed.

```bash
for t in scripts/test_*.py scripts/qa/test_*.py; do python "$t"; done
```

The heavy stages need a CUDA GPU and the `dap-3_py3-11` conda env (see `README.md`); everything
under `scripts/qa/` except Stage 1 is CPU-only.

## Layout

```
data/<site>/<trial_or_experiment>/<camera_ref_folder>/<clip>.{AVI,MP4,MOV}
```

Reference video clips (~1850 total, mostly `.AVI`/`.MP4`, a few `.MOV`) live under per-site
directories:

- `mafou`, `pss`, `pnt`, `kora`, `beauvois`, `mbnp`, `fello_sounga`

Each site contains one or more trial/experiment subfolders (e.g. `e1`, `E2`, `T_16`, `pnt_p2`),
each of which contains one camera-reference folder per camera (e.g. `16_vid_ref_Cam_184`,
`T47_vid_ref_cam33`, `videos_reference_v15`) holding the actual clips. Naming conventions differ
slightly site to site, but the four-level hierarchy (site → trial → camera-ref folder → clips)
is consistent throughout.

## `data/list_reference_videos.xlsx`

Single sheet (`Sheet1`) with columns `ori, anno, check`:

- `ori` — the video's original relative path under `data/`, e.g.
  `beauvois/T_16/16_vid_ref_Cam_184/DSCF0005.AVI`
- `anno` — the flattened canonical name used to key annotations: the path with `/` replaced by
  `_` and the extension dropped, e.g. `beauvois_T_16_16_vid_ref_Cam_184_DSCF0005`
- `check` — a flag column (observed values are `1`)

This file is the join key between raw video paths and the annotations CSV — `anno` matches
`video_name` in `annotations_06052026.csv`.

## `data/annotations_06052026.csv`

12,459 rows of frame-level ground-truth distance annotations, columns:

```
video_name, frame_idx, frame_timestamp, distance
```

`video_name` corresponds to the `anno` column of `list_reference_videos.xlsx` (not the raw file
path). `frame_timestamp` is in seconds; `distance` is the annotated subject-to-camera distance
at that frame. Row counts per site (by `video_name` prefix): pnt 6017, pss 2353, fello 1453,
kora 934, mafou 710, mbnp 701, beauvois 290 — `pnt` and `pss` are by far the most densely
annotated sites. (Those prefixes come from `scripts/sites.py::site_of`, which splits on the
first underscore — so the `fello_sounga` site appears as `fello`.)

**This is the older generation.** Its `frame_idx` values are inconsistent with the videos'
probed fps, so a frame extracted by index may not be the frame that was annotated. Prefer
`data/annotations_20260709_with_fps_clean.csv`, produced by `scripts/qc_annotations.py --fix`,
which recomputes `frame_idx` from `frame_timestamp × fps` and drops impossible rows. It remains
`run_calibration_eval.py`'s default only for backwards compatibility — see README.md Step 0,
including the note on why the flags CSV cannot be joined to the clean CSV on `frame_idx`.
