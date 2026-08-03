#!/usr/bin/env python
"""Synthetic unit tests for the --masks-dir mask-copy behaviour of scripts/export_calibrated.py,
plus a check that old CLI flag spellings still parse as aliases of the same dest (repo-wide
flag-rename item, see CLAUDE.md / the task that added this file).

Run directly:  python scripts/test_export_masks.py  (also pytest-compatible).
No GPU / data / network needed. Imports scripts.export_calibrated (which imports cv2, but not
torch) and scripts.probe_video_fps (chosen over scripts.run_calibration_eval.py for the alias
check because the latter imports torch at module level -- see the module docstring below for
why that matters here).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.export_calibrated import copy_mask_for_frame
from scripts.probe_video_fps import build_parser


# --------------------------------------------------------------------------------------
# copy_mask_for_frame
# --------------------------------------------------------------------------------------

def test_copies_mask_when_present():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        masks_dir = tmp / "masks_in"
        masks_dir.mkdir()
        out_dir = tmp / "export" / "masks"
        src = masks_dir / "mafou_e1_cam7_DSCF0001_frame000042_masks.json"
        src.write_text('[{"instance_idx": 0}]')

        result = copy_mask_for_frame(masks_dir, out_dir, "mafou_e1_cam7_DSCF0001", 42)

        assert result is True
        copied = out_dir / "mafou_e1_cam7_DSCF0001_frame000042_masks.json"
        assert copied.exists()
        assert copied.read_text() == '[{"instance_idx": 0}]'


def test_returns_false_and_skips_silently_when_absent():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        masks_dir = tmp / "masks_in"
        masks_dir.mkdir()
        out_dir = tmp / "export" / "masks"

        result = copy_mask_for_frame(masks_dir, out_dir, "no_such_video", 7)

        assert result is False
        # no output dir should have been created for a frame with nothing to copy
        assert not out_dir.exists()


def test_creates_out_dir_lazily():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        masks_dir = tmp / "masks_in"
        masks_dir.mkdir()
        out_dir = tmp / "export" / "masks"
        assert not out_dir.exists()

        src = masks_dir / "vid_frame000001_masks.json"
        src.write_text("[]")
        copy_mask_for_frame(masks_dir, out_dir, "vid", 1)

        assert out_dir.exists()
        assert out_dir.is_dir()


def test_output_filename_matches_mask_path_naming():
    """The copied file's name must exactly match scripts/masks.py::mask_path's convention, since
    that's what score_masks.py / heuristic tooling looks it up by."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        masks_dir = tmp / "masks_in"
        masks_dir.mkdir()
        out_dir = tmp / "masks_out"

        video_name, frame_idx = "beauvois_T_16_16_vid_ref_Cam_184_DSCF0005", 123
        (masks_dir / f"{video_name}_frame{frame_idx:06d}_masks.json").write_text("[]")

        copy_mask_for_frame(masks_dir, out_dir, video_name, frame_idx)

        expected_name = f"{video_name}_frame{frame_idx:06d}_masks.json"
        assert (out_dir / expected_name).exists()
        assert [p.name for p in out_dir.iterdir()] == [expected_name]


def test_only_copies_the_requested_frame():
    """A masks_dir with several frames' worth of masks: only the one asked for is touched."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        masks_dir = tmp / "masks_in"
        masks_dir.mkdir()
        out_dir = tmp / "masks_out"

        for idx in (1, 2, 3):
            (masks_dir / f"vid_frame{idx:06d}_masks.json").write_text(f"[{idx}]")

        assert copy_mask_for_frame(masks_dir, out_dir, "vid", 2) is True
        assert [p.name for p in out_dir.iterdir()] == ["vid_frame000002_masks.json"]
        assert (out_dir / "vid_frame000002_masks.json").read_text() == "[2]"


# --------------------------------------------------------------------------------------
# CLI flag-rename aliasing: old spellings must still parse to the SAME dest.
#
# run_calibration_eval.py imports torch at module level (confirmed by reading its imports:
# `import torch` right after `import cv2` / `import numpy as np`), so importing its parser here
# would drag in a heavy, possibly-absent dependency for what should be a fast, dependency-light
# unit test. scripts/probe_video_fps.py has no such import (argparse/csv/subprocess/sys/
# concurrent.futures/fractions/pathlib only) and underwent the same kind of rename
# (--data-root -> --data-dir, old spelling kept as an alias on the same `data_root` dest), so it
# stands in for the alias-preservation check instead.
# --------------------------------------------------------------------------------------

def test_probe_video_fps_old_flag_alias_same_dest():
    parser = build_parser()

    new_spelling = parser.parse_args(["--data-dir", "/some/path"])
    old_spelling = parser.parse_args(["--data-root", "/some/path"])

    assert new_spelling.data_root == Path("/some/path")
    assert old_spelling.data_root == Path("/some/path")
    assert new_spelling.data_root == old_spelling.data_root


def test_probe_video_fps_default_unchanged_when_no_flag_given():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.data_root == Path(__file__).resolve().parent.parent / "data"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nall {len(fns)} export_masks tests passed")


if __name__ == "__main__":
    _run_all()
