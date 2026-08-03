#!/usr/bin/env python
"""Synthetic unit tests for scripts/qc_exclusions.py.
Run directly:  python scripts/test_qc_exclusions.py  (also pytest-compatible).
No GPU / data / torch / network needed."""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.qc_exclusions import (
    REASONS_FOR_CLEAN,
    REASONS_HANDLED_BY_FIX,
    build_exclusions,
    load_flag_timestamps,
    load_qc_exclusions,
    resolve_exclusions,
)

# One video, three annotated frames. The flags CSV is written from the ORIGINAL annotations,
# whose frame_idx values are wrong; the _clean.csv recomputes them from timestamp * fps (25).
# This is exactly the situation qc_annotations.py --fix produces.
_ORIGINAL_ROWS = [
    # video_name, frame_idx (original/wrong), frame_timestamp, distance
    ("mafou_e1_cam7_DSCF0001", 100, 2.0, 4.0),   # FRAME_IDX_FPS_MISMATCH -> clean idx 50
    ("mafou_e1_cam7_DSCF0001", 200, 4.0, 8.0),   # FRAME_IDX_FPS_MISMATCH -> clean idx 100
    ("mafou_e1_cam7_DSCF0001", 150, 6.0, 12.0),  # TIMESTAMP_PAST_END     -> clean idx 150
]
_CLEAN_ROWS = [
    ("mafou_e1_cam7_DSCF0001", 50, 2.0, 4.0),
    ("mafou_e1_cam7_DSCF0001", 100, 4.0, 8.0),
    ("mafou_e1_cam7_DSCF0001", 150, 6.0, 12.0),
]
_FLAGS = [
    ("mafou_e1_cam7_DSCF0001", 100, 2.0, 4.0, "FRAME_IDX_FPS_MISMATCH"),
    ("mafou_e1_cam7_DSCF0001", 200, 4.0, 8.0, "FRAME_IDX_FPS_MISMATCH"),
    ("mafou_e1_cam7_DSCF0001", 150, 6.0, 12.0, "TIMESTAMP_PAST_END"),
]


def _write_annotations(path: Path, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "frame_idx", "frame_timestamp", "distance"])
        w.writerows(rows)


def _write_flags(path: Path, rows=_FLAGS) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "video_name", "frame_idx", "frame_timestamp", "distance", "reason", "detail"])
        for i, (video, idx, ts, dist, reason) in enumerate(rows):
            w.writerow([i, video, idx, ts, dist, reason, ""])


def _fixture(tmp: Path) -> tuple[Path, Path, Path]:
    flags, original, clean = tmp / "qc_flags.csv", tmp / "anno.csv", tmp / "anno_clean.csv"
    _write_flags(flags)
    _write_annotations(original, _ORIGINAL_ROWS)
    _write_annotations(clean, _CLEAN_ROWS)
    return flags, original, clean


def test_legacy_loader_reads_the_flags_own_frame_idx():
    with tempfile.TemporaryDirectory() as d:
        flags, _, _ = _fixture(Path(d))
        assert load_qc_exclusions(flags) == {
            ("mafou_e1_cam7_DSCF0001", 100),
            ("mafou_e1_cam7_DSCF0001", 200),
            ("mafou_e1_cam7_DSCF0001", 150),
        }


def test_legacy_loader_undermatches_against_clean_csv():
    """The regression this module exists to fix: the flags CSV's frame_idx values 100/200 do
    not appear in the clean CSV at all (they became 50/100), so a direct join drops two of the
    three exclusions and picks up a THIRD row by coincidence -- clean idx 100 is a different
    row (t=4.0) than flagged original idx 100 (t=2.0)."""
    with tempfile.TemporaryDirectory() as d:
        flags, _, clean = _fixture(Path(d))
        legacy = load_qc_exclusions(flags)
        clean_keys = {(r[0], r[1]) for r in _CLEAN_ROWS}
        matched = legacy & clean_keys
        assert matched == {("mafou_e1_cam7_DSCF0001", 100), ("mafou_e1_cam7_DSCF0001", 150)}
        # and the one it "matched" at idx 100 is the wrong annotation row
        assert ("mafou_e1_cam7_DSCF0001", 50) not in legacy


def test_resolve_exclusions_against_clean_csv_uses_clean_frame_idx():
    with tempfile.TemporaryDirectory() as d:
        flags, _, clean = _fixture(Path(d))
        assert resolve_exclusions(flags, clean) == {
            ("mafou_e1_cam7_DSCF0001", 50),
            ("mafou_e1_cam7_DSCF0001", 100),
            ("mafou_e1_cam7_DSCF0001", 150),
        }


def test_resolve_exclusions_against_original_csv_matches_legacy():
    """Joining by timestamp is not a behaviour change when the generations agree -- against the
    CSV the flags were computed from, it reproduces the legacy result exactly."""
    with tempfile.TemporaryDirectory() as d:
        flags, original, _ = _fixture(Path(d))
        assert resolve_exclusions(flags, original) == load_qc_exclusions(flags)


def test_resolve_exclusions_filters_by_reason():
    with tempfile.TemporaryDirectory() as d:
        flags, _, clean = _fixture(Path(d))
        assert resolve_exclusions(flags, clean, REASONS_FOR_CLEAN) == {
            ("mafou_e1_cam7_DSCF0001", 150)
        }
        assert resolve_exclusions(flags, clean, REASONS_HANDLED_BY_FIX) == {
            ("mafou_e1_cam7_DSCF0001", 50),
            ("mafou_e1_cam7_DSCF0001", 100),
        }


def test_timestamp_key_tolerates_float_formatting():
    """"2.0" in one file and "2.00" in another must still join."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        flags, clean = tmp / "f.csv", tmp / "c.csv"
        _write_flags(flags, [("vid", 100, 2.0, 4.0, "TIMESTAMP_PAST_END")])
        with open(clean, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_name", "frame_idx", "frame_timestamp", "distance"])
            w.writerow(["vid", 50, "2.00", 4.0])
        assert resolve_exclusions(flags, clean) == {("vid", 50)}


def test_resolve_exclusions_skips_blank_frame_idx():
    """apply_fixes blanks frame_idx for videos missing from disk; there is no index to exclude
    and int('') would otherwise raise."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        flags, clean = tmp / "f.csv", tmp / "c.csv"
        _write_flags(flags, [("vid", 0, 1.0, 4.0, "VIDEO_NOT_ON_DISK")])
        with open(clean, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_name", "frame_idx", "frame_timestamp", "distance"])
            w.writerow(["vid", "", 1.0, 4.0])
        assert resolve_exclusions(flags, clean) == set()


def test_load_flag_timestamps_skips_malformed_rows():
    with tempfile.TemporaryDirectory() as d:
        flags = Path(d) / "f.csv"
        _write_flags(flags, [("vid", 100, 2.0, 4.0, "TIMESTAMP_PAST_END")])
        with open(flags, "a", newline="") as f:
            csv.writer(f).writerow([1, "vid2", 5, "", 4.0, "TIMESTAMP_PAST_END", ""])
        assert load_flag_timestamps(flags) == {("vid", 2000)}


def test_build_exclusions_falls_back_when_annotations_missing():
    with tempfile.TemporaryDirectory() as d:
        flags, _, _ = _fixture(Path(d))
        missing = Path(d) / "does_not_exist.csv"
        assert build_exclusions(flags, missing, verbose=False) == load_qc_exclusions(flags)
        assert build_exclusions(flags, None, verbose=False) == load_qc_exclusions(flags)


def test_build_exclusions_resolves_when_annotations_present():
    with tempfile.TemporaryDirectory() as d:
        flags, _, clean = _fixture(Path(d))
        assert build_exclusions(flags, clean, verbose=False) == resolve_exclusions(flags, clean)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nall {len(fns)} qc_exclusions tests passed")


if __name__ == "__main__":
    _run_all()
