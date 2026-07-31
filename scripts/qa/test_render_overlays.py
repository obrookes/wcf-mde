#!/usr/bin/env python
"""Synthetic unit tests for scripts/qa/render_overlays.py.
Run directly:  python scripts/qa/test_render_overlays.py  (also pytest-compatible).
No GPU / data / torch / network needed."""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.qa.render_overlays import (
    OTHER_BGR,
    SUBJECT_BGR,
    crop_box,
    draw_mask,
    draw_outline,
    fit_height,
    load_targets,
    overlay_name,
    render_panel,
)

H, W = 200, 300


def _frame() -> np.ndarray:
    return np.full((H, W, 3), 90, dtype=np.uint8)


def _mask(x0=140, x1=170, y0=60, y1=150) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(panel_height=384, crop_pad_frac=0.35, alpha=0.45)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# --------------------------------------------------------------------------------------
# filenames -- the collision this script exists to avoid
# --------------------------------------------------------------------------------------

def test_overlay_name_keys_on_video_name_not_clip_stem():
    """Camera folders reuse clip filenames like DSCF0005 across sites; keying on the stem would
    silently overwrite one site's overlays with another's (see scripts/masks.py)."""
    a = overlay_name("beauvois_T_16_16_vid_ref_Cam_184_DSCF0005", 12, 0)
    b = overlay_name("mafou_E2_videos_reference_v15_DSCF0005", 12, 0)
    assert a != b
    assert a.startswith("beauvois_") and b.startswith("mafou_")


def test_overlay_name_separates_instances_and_zero_pads_frames():
    assert overlay_name("v", 7, 0) != overlay_name("v", 7, 1)
    assert "frame000007" in overlay_name("v", 7, 0)
    # zero-padding keeps a directory listing in frame order
    names = sorted(overlay_name("v", i, 0) for i in (2, 10, 100))
    assert names == [overlay_name("v", i, 0) for i in (2, 10, 100)]


# --------------------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------------------

def test_draw_mask_does_not_mutate_the_source_frame():
    frame = _frame()
    before = frame.copy()
    draw_mask(frame, _mask(), SUBJECT_BGR, 0.45)
    np.testing.assert_array_equal(frame, before)


def test_draw_mask_tints_inside_and_leaves_outside_alone():
    frame = _frame()
    out = draw_mask(frame, _mask(), SUBJECT_BGR, 0.45)
    assert out[100, 150, 1] > frame[100, 150, 1]        # green channel lifted inside the mask
    np.testing.assert_array_equal(out[10, 10], frame[10, 10])  # far corner untouched


def test_draw_mask_handles_an_empty_mask():
    frame = _frame()
    out = draw_mask(frame, np.zeros((H, W), dtype=bool), SUBJECT_BGR, 0.45)
    np.testing.assert_array_equal(out, frame)


def test_draw_mask_resizes_a_mismatched_mask():
    """Masks are stored at the frame size they were computed at; a resolution change between
    Stage 1 and rendering must not crash the panel."""
    frame = _frame()
    small = np.zeros((H // 2, W // 2), dtype=bool)
    small[30:70, 70:85] = True
    out = draw_mask(frame, small, SUBJECT_BGR, 0.45)
    assert out.shape == frame.shape
    assert (out != frame).any()


def test_draw_outline_uses_a_second_colour_for_other_instances():
    frame = _frame()
    out = draw_outline(frame.copy(), _mask(), OTHER_BGR)
    painted = out[(out != frame).any(axis=2)]
    assert len(painted) > 0
    # outline only: the mask interior is not filled
    assert (out[100, 155] == frame[100, 155]).all()


# --------------------------------------------------------------------------------------
# cropping and fitting
# --------------------------------------------------------------------------------------

def test_crop_box_pads_around_the_bbox():
    x0, y0, x1, y1 = crop_box(_mask(), (H, W), 0.35)
    assert x0 < 140 and y0 < 60 and x1 > 170 and y1 > 150


def test_crop_box_stays_in_bounds_for_an_edge_mask():
    m = np.zeros((H, W), dtype=bool)
    m[0:20, 0:20] = True
    x0, y0, x1, y1 = crop_box(m, (H, W), 0.5)
    assert (x0, y0) == (0, 0)
    assert x1 <= W and y1 <= H


def test_crop_box_of_empty_mask_is_the_whole_frame():
    assert crop_box(np.zeros((H, W), dtype=bool), (H, W), 0.35) == (0, 0, W, H)


def test_fit_height_scales_to_the_requested_height():
    out = fit_height(np.zeros((100, 200, 3), dtype=np.uint8), 384)
    assert out.shape[0] == 384


def test_fit_height_caps_upscaling_and_pads_instead():
    """A 20px crop upscaled to 384 would be interpolation noise billed as image tokens."""
    out = fit_height(np.zeros((20, 30, 3), dtype=np.uint8), 384)
    assert out.shape[0] == 384
    assert out.shape[1] <= 30 * 3 + 1  # width reflects the 3x cap, not the full 19.2x


def test_fit_height_handles_a_degenerate_image():
    out = fit_height(np.zeros((0, 0, 3), dtype=np.uint8), 384)
    assert out.shape[0] == 384


# --------------------------------------------------------------------------------------
# panel assembly
# --------------------------------------------------------------------------------------

def _instances() -> list[dict]:
    return [
        {"instance_idx": 0, "mask": _mask(), "center_xy": (155, 105), "area_px": int(_mask().sum())},
        {"instance_idx": 1, "mask": _mask(40, 70), "center_xy": (55, 105),
         "area_px": int(_mask(40, 70).sum())},
    ]


def test_render_panel_produces_a_two_up_panel_of_the_requested_height():
    args = _args()
    meta = {"video_name": "pnt_p2_cam_v", "frame_idx": 42, "flags": "multi_instance"}
    panel = render_panel(_frame(), _instances(), 0, meta, args)
    assert panel is not None
    assert panel.shape[0] == args.panel_height
    assert panel.shape[1] > args.panel_height  # full frame + zoom side by side


def test_render_panel_returns_none_for_a_missing_instance():
    meta = {"video_name": "v", "frame_idx": 1, "flags": ""}
    assert render_panel(_frame(), _instances(), 7, meta, _args()) is None


def test_render_panel_marks_the_subject_and_the_others_differently():
    """`multiple` has to be visible in the panel, or the verdict can't distinguish it."""
    meta = {"video_name": "v", "frame_idx": 1, "flags": ""}
    one = render_panel(_frame(), _instances(), 0, meta, _args())
    two = render_panel(_frame(), _instances(), 1, meta, _args())
    assert not np.array_equal(one, two)


def test_render_panel_token_cost_stays_within_budget():
    """Panel area is the dominant per-frame cost (image tokens ~ w*h/750), so a change that
    quietly inflates it should fail here rather than on the invoice."""
    meta = {"video_name": "v", "frame_idx": 1, "flags": ""}
    panel = render_panel(_frame(), _instances(), 0, meta, _args())
    h, w = panel.shape[:2]
    assert w * h / 750 < 1200, f"panel is {w}x{h} = ~{w * h / 750:.0f} image tokens"


# --------------------------------------------------------------------------------------
# target selection
# --------------------------------------------------------------------------------------

def _write_prefilter(path: Path, rows) -> None:
    fields = ["video_name", "frame_idx", "instance_idx", "site", "stratum",
              "disposition", "prefilter_class", "flags"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_load_targets_selects_only_the_requested_dispositions():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "pf.csv"
        _write_prefilter(path, [
            dict(video_name="a", frame_idx=1, instance_idx=0, site="a", stratum="annotated",
                 disposition="vision", prefilter_class="flagged", flags="low_fill"),
            dict(video_name="a", frame_idx=2, instance_idx=0, site="a", stratum="annotated",
                 disposition="pass", prefilter_class="clean_singleton", flags=""),
        ])
        assert len(load_targets(path, ["vision"], None)) == 1
        assert len(load_targets(path, ["vision", "pass"], None)) == 2


def test_load_targets_skips_frame_level_rows_with_no_instance():
    """empty_mask and pipeline-failure rows have instance_idx blank -- there is no mask to draw."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "pf.csv"
        _write_prefilter(path, [
            dict(video_name="a", frame_idx=1, instance_idx="", site="a", stratum="annotated",
                 disposition="vision", prefilter_class="flagged", flags=""),
        ])
        assert load_targets(path, ["vision"], None) == []


def test_load_targets_is_sorted_and_limited():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "pf.csv"
        _write_prefilter(path, [
            dict(video_name="b", frame_idx=5, instance_idx=0, site="b", stratum="annotated",
                 disposition="vision", prefilter_class="flagged", flags=""),
            dict(video_name="a", frame_idx=9, instance_idx=1, site="a", stratum="annotated",
                 disposition="vision", prefilter_class="flagged", flags=""),
            dict(video_name="a", frame_idx=9, instance_idx=0, site="a", stratum="annotated",
                 disposition="vision", prefilter_class="flagged", flags=""),
        ])
        targets = load_targets(path, ["vision"], None)
        assert [(t["video_name"], t["instance_idx"]) for t in targets] == \
               [("a", 0), ("a", 1), ("b", 0)]
        assert len(load_targets(path, ["vision"], 2)) == 2


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nall {len(fns)} render_overlays tests passed")


if __name__ == "__main__":
    _run_all()
