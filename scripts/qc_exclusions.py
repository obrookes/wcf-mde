"""Read scripts/qc_annotations.py's flags CSV into a set of rows to exclude.

There are two incompatible `frame_idx` key spaces in this repo, and joining across them
silently loses exclusions:

  * `qc_flags_<basename>.csv` is written from the *input* annotations CSV, so its `frame_idx`
    column holds that file's ORIGINAL values.
  * `<basename>_clean.csv` (written by `qc_annotations.py --fix`) RECOMPUTES
    `frame_idx = round(frame_timestamp * fps_probed)` for every on-disk row -- not only the
    flagged ones, and not only those off by more than the +/-1 tolerance `collect_flags` uses.

So a `(video_name, frame_idx)` join between the two never matches a `FRAME_IDX_FPS_MISMATCH`
row -- by construction, those are exactly the rows whose `frame_idx` changed -- and can miss
others that shifted by one. The exclusions that matter most quietly do nothing.

`frame_timestamp` is the fixed point: `apply_fixes` copies it through untouched, and it is
carried in both the flags CSV and every annotations generation. `resolve_exclusions` therefore
joins on `(video_name, frame_timestamp)` and re-emits `(video_name, frame_idx)` using the
`frame_idx` of the annotations CSV *actually in use* -- the same shape every existing consumer
(`calibrate_depth.load_points`, `export_calibrated`) already takes, so nothing downstream needs
to know this happened.

Timestamps are compared as integer milliseconds rather than floats, so a value that round-trips
through CSV text as "1.2" in one file and "1.20" in another still matches.

Stdlib only (csv + pathlib), matching calibrate_depth.py's "CPU only, no torch" import weight.
"""
from __future__ import annotations

import csv
from pathlib import Path

# Reasons `qc_annotations.py --fix` already resolves when writing `<basename>_clean.csv`:
# ABSURD_DISTANCE / ZERO_DISTANCE rows are DROPPED outright, and FRAME_IDX_FPS_MISMATCH rows
# have their frame_idx REPAIRED. Excluding these against a clean CSV is at best a no-op and at
# worst discards good data -- the repaired rows are usable, not disqualified.
REASONS_HANDLED_BY_FIX = frozenset({"ABSURD_DISTANCE", "ZERO_DISTANCE", "FRAME_IDX_FPS_MISMATCH"})

# Reasons that still disqualify a row in `<basename>_clean.csv`: the frame either doesn't exist
# (timestamp past the end of the video) or can't be resolved at all (video missing from disk,
# whose frame_idx `apply_fixes` blanks to NaN under the default --fix-mode).
REASONS_FOR_CLEAN = frozenset({"TIMESTAMP_PAST_END", "VIDEO_NOT_ON_DISK"})


def _timestamp_key(video_name: str, frame_timestamp: str | float) -> tuple[str, int]:
    """(video_name, milliseconds) -- an integer key, so CSV float formatting can't break the join."""
    return (video_name, int(round(float(frame_timestamp) * 1000)))


def load_flag_rows(qc_flags_csv: Path | str, reasons: frozenset[str] | set[str] | None = None) -> list[dict]:
    """Raw flag rows, optionally filtered to `reasons` (None = every reason)."""
    with open(qc_flags_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if reasons is None:
        return rows
    return [r for r in rows if r.get("reason") in reasons]


def load_flag_timestamps(
    qc_flags_csv: Path | str, reasons: frozenset[str] | set[str] | None = None
) -> set[tuple[str, int]]:
    """(video_name, timestamp_ms) pairs flagged by qc_annotations.py.

    Rows with an unparseable `frame_timestamp` are skipped rather than crashing the load -- a
    malformed flag row should not be able to take down a calibration run.
    """
    flagged: set[tuple[str, int]] = set()
    for row in load_flag_rows(qc_flags_csv, reasons):
        try:
            flagged.add(_timestamp_key(row["video_name"], row["frame_timestamp"]))
        except (KeyError, TypeError, ValueError):
            continue
    return flagged


def resolve_exclusions(
    qc_flags_csv: Path | str,
    annotations_csv: Path | str,
    reasons: frozenset[str] | set[str] | None = None,
) -> set[tuple[str, int]]:
    """(video_name, frame_idx) pairs to drop, keyed in `annotations_csv`'s own frame_idx space.

    Joins flags to annotations on (video_name, frame_timestamp) -- see the module docstring for
    why frame_idx cannot be used -- and emits whatever frame_idx that annotations CSV carries.
    Rows with a blank/NaN frame_idx (which `apply_fixes` writes for videos missing from disk)
    are skipped: there is no index to exclude, and they are unusable downstream regardless.
    """
    flagged = load_flag_timestamps(qc_flags_csv, reasons)
    if not flagged:
        return set()

    exclude: set[tuple[str, int]] = set()
    with open(annotations_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = _timestamp_key(row["video_name"], row["frame_timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            if key not in flagged:
                continue
            raw_idx = row.get("frame_idx")
            if raw_idx in (None, "", "None", "nan", "NaN", "<NA>"):
                continue
            try:
                exclude.add((row["video_name"], int(float(raw_idx))))
            except (TypeError, ValueError):
                continue
    return exclude


def build_exclusions(
    qc_flags_csv: Path | str,
    annotations_csv: Path | str | None,
    reasons: frozenset[str] | set[str] | None = None,
    verbose: bool = True,
) -> set[tuple[str, int]]:
    """CLI helper shared by calibrate_depth / benchmark_calibration / export_calibrated.

    Uses `resolve_exclusions` when the annotations CSV is available, and falls back to the
    frame-idx-keyed `load_qc_exclusions` when it isn't -- printing a warning in that case,
    because the fallback silently under-excludes if the results were produced from a `_clean.csv`.
    """
    if annotations_csv is not None and Path(annotations_csv).exists():
        exclude = resolve_exclusions(qc_flags_csv, annotations_csv, reasons)
        if verbose:
            print(f"loaded {len(exclude)} QC-flagged (video_name, frame_idx) exclusions from "
                  f"{qc_flags_csv}, resolved by timestamp against {annotations_csv}")
        return exclude

    exclude = load_qc_exclusions(qc_flags_csv)
    if verbose:
        print(f"loaded {len(exclude)} QC-flagged (video_name, frame_idx) exclusions from {qc_flags_csv}")
        print(f"  !! WARNING: no annotations CSV at {annotations_csv}; falling back to the flags "
              f"CSV's own frame_idx column. If the results CSV came from a *_clean.csv, its "
              f"frame_idx values were recomputed and these exclusions will silently under-match "
              f"(see scripts/qc_exclusions.py). Pass --annotations-csv to fix.")
    return exclude


def load_qc_exclusions(qc_flags_csv: Path | str) -> set[tuple[str, int]]:
    """(video_name, frame_idx) pairs read directly from the flags CSV's own frame_idx column.

    Correct ONLY against the same annotations generation the flags were computed from (e.g.
    `annotations_20260709_with_fps.csv`, not its `_clean.csv`). Prefer `resolve_exclusions`,
    which works for either. Kept because it is the historical behaviour and the sensible
    fallback when the annotations CSV in use isn't known.
    """
    with open(qc_flags_csv, newline="") as f:
        return {(row["video_name"], int(row["frame_idx"])) for row in csv.DictReader(f)}
