#!/usr/bin/env python
"""Synthetic unit tests for scripts/qa/review_server.py and apply_corrections.py.
Run directly:  python scripts/qa/test_review.py  (also pytest-compatible).
No GPU / data / torch / network needed -- masks are drawn with numpy, files live in tmp dirs."""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.masks import load_instance_masks, save_instance_masks
from scripts.qa.apply_corrections import APPLIED_FIELDS, apply_morph, write_applied
from scripts.qa.review_server import (
    CORRECTION_FIELDS,
    append_decision,
    autofix_mask,
    build_queue,
    clamp_box,
    key_of,
    load_decisions,
    render_autofix_panel,
)

H, W = 120, 160


def _row(video="pnt_v1_cam1_A", frame=10, inst=0, verdict="split",
         result_type="succeeded", **extra):
    row = {"video_name": video, "frame_idx": str(frame), "instance_idx": str(inst),
           "verdict": verdict, "result_type": result_type, "confidence": "0.8",
           "rationale": "r", "flags": "", "site": "pnt", "overlay_path": "/x.png"}
    row.update(extra)
    return row


def _decision(video="v", frame=5, inst=0, action="autofix", box=("", "", "", "")):
    return {"key": f"{video}|{frame}|{inst}", "video_name": video, "frame_idx": str(frame),
            "instance_idx": str(inst), "haiku_verdict": "split", "action": action,
            "box_x0": box[0], "box_y0": box[1], "box_x1": box[2], "box_y1": box[3],
            "notes": "", "decided_at": "2026-08-02T00:00:00+00:00"}


# --------------------------------------------------------------------------------------
# queue
# --------------------------------------------------------------------------------------


def test_build_queue_filters_ok_and_errored():
    rows = [_row(verdict="ok"), _row(frame=11, verdict="split"),
            _row(frame=12, verdict="bleed", result_type="errored"),
            _row(frame=13, verdict="", result_type="errored")]
    queue = build_queue(rows)
    assert [r["frame_idx"] for r in queue] == ["11"]


def test_build_queue_groups_by_class_then_video_frame():
    rows = [_row(video="b", frame=2, verdict="split"),
            _row(video="a", frame=9, verdict="multiple"),
            _row(video="a", frame=3, verdict="empty"),
            _row(video="a", frame=1, verdict="split"),
            _row(video="a", frame=1, inst=1, verdict="split")]
    queue = build_queue(rows)
    got = [(r["verdict"], r["video_name"], r["frame_idx"], r["instance_idx"]) for r in queue]
    assert got == [("empty", "a", "3", "0"), ("split", "a", "1", "0"),
                   ("split", "a", "1", "1"), ("split", "b", "2", "0"),
                   ("multiple", "a", "9", "0")]


def test_build_queue_numeric_frame_sort():
    rows = [_row(frame=100), _row(frame=20)]
    assert [r["frame_idx"] for r in build_queue(rows)] == ["20", "100"]


def test_build_queue_only_classes():
    rows = [_row(frame=1, verdict="split"), _row(frame=2, verdict="bleed")]
    queue = build_queue(rows, only_classes=["bleed"])
    assert [r["verdict"] for r in queue] == ["bleed"]


def test_key_of():
    assert key_of(_row(video="v", frame=7, inst=2)) == "v|7|2"


# --------------------------------------------------------------------------------------
# autofix morphology
# --------------------------------------------------------------------------------------


def test_autofix_keeps_largest_component():
    m = np.zeros((H, W), bool)
    m[10:60, 10:60] = True   # 2500 px
    m[80:90, 80:90] = True   # 100 px straggler
    fixed = autofix_mask(m)
    assert fixed[30, 30] and not fixed[85, 85]
    assert fixed.sum() == 2500


def test_autofix_fills_holes():
    m = np.zeros((H, W), bool)
    m[10:60, 10:60] = True
    m[25:35, 25:35] = False  # interior hole
    fixed = autofix_mask(m)
    assert fixed[30, 30]
    assert fixed.sum() == 2500


def test_autofix_idempotent_on_clean_mask():
    m = np.zeros((H, W), bool)
    m[10:60, 10:60] = True
    assert (autofix_mask(m) == m).all()
    assert (autofix_mask(autofix_mask(m)) == autofix_mask(m)).all()


def test_autofix_empty_mask_stays_empty():
    m = np.zeros((H, W), bool)
    assert autofix_mask(m).sum() == 0


def test_autofix_mask_touching_corner():
    # foreground covering (0,0) must not hijack the background flood-fill seed
    m = np.zeros((H, W), bool)
    m[0:40, 0:40] = True
    m[10:20, 10:20] = False  # hole inside the corner component
    m[100:105, 100:105] = True  # smaller distractor
    fixed = autofix_mask(m)
    assert fixed[15, 15], "hole in corner-touching component should be filled"
    assert not fixed[102, 102]
    assert fixed.sum() == 1600


def test_autofix_notch_at_border_is_not_a_hole():
    m = np.zeros((H, W), bool)
    m[10:60, 0:50] = True
    m[20:30, 0:10] = False  # notch open to the image border: reachable, must stay empty
    fixed = autofix_mask(m)
    assert not fixed[25, 5]


# --------------------------------------------------------------------------------------
# box clamping
# --------------------------------------------------------------------------------------


def test_clamp_box_orders_corners():
    assert clamp_box([50, 40, 10, 8], W, H) == [10, 8, 50, 40]


def test_clamp_box_clamps_to_image():
    assert clamp_box([-5, -5, 5000, 5000], W, H) == [0, 0, W - 1, H - 1]


def test_clamp_box_rejects_degenerate():
    assert clamp_box([10, 10, 11, 40], W, H) is None    # 1 px wide
    assert clamp_box([-20, 10, -5, 40], W, H) is None   # fully off-image
    assert clamp_box(None, W, H) is None
    assert clamp_box([1, 2, 3], W, H) is None
    assert clamp_box(["a", 0, 1, 1], W, H) is None


# --------------------------------------------------------------------------------------
# corrections CSV round-trip
# --------------------------------------------------------------------------------------


def test_decisions_roundtrip_last_write_wins():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "corrections.csv"
        assert load_decisions(path) == {}
        append_decision(path, _decision(action="skip"))
        append_decision(path, _decision(video="w", frame=1, action="discard"))
        append_decision(path, _decision(action="autofix"))  # re-decide same key
        got = load_decisions(path)
        assert set(got) == {"v|5|0", "w|1|0"}
        assert got["v|5|0"]["action"] == "autofix"
        # exactly one header line
        lines = path.read_text().splitlines()
        assert lines[0] == ",".join(CORRECTION_FIELDS)
        assert len(lines) == 4


def test_append_decision_drops_unknown_fields():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "c.csv"
        append_decision(path, {**_decision(), "rogue": "x"})
        assert "rogue" not in path.read_text()


# --------------------------------------------------------------------------------------
# autofix render panel
# --------------------------------------------------------------------------------------


def test_render_autofix_panel_shape_and_effect():
    frame = np.full((H, W, 3), 120, np.uint8)
    before = np.zeros((H, W), bool)
    before[10:60, 10:60] = True
    before[80:90, 80:90] = True
    out = render_autofix_panel(frame, before, autofix_mask(before))
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert not (out == frame).all()


# --------------------------------------------------------------------------------------
# apply_corrections morph end-to-end
# --------------------------------------------------------------------------------------


def _seed_masks(masks_dir: Path, video: str, frame: int):
    """Frame with two instances: inst 0 clean, inst 1 split (blob + straggler) with a hole."""
    clean = np.zeros((H, W), bool)
    clean[10:30, 10:30] = True
    broken = np.zeros((H, W), bool)
    broken[40:90, 60:110] = True
    broken[55:65, 75:85] = False   # hole
    broken[5:10, 140:150] = True   # straggler component
    instances = [
        {"mask": clean, "center_xy": (20, 20), "area_px": int(clean.sum())},
        {"mask": broken, "center_xy": (85, 65), "area_px": int(broken.sum())},
    ]
    save_instance_masks(masks_dir, video, frame, instances)
    return clean, broken


def test_apply_morph_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        masks_dir, out_dir = Path(td) / "masks", Path(td) / "masks_fixed"
        clean, broken = _seed_masks(masks_dir, "vid", 5)
        decisions = {
            "vid|5|1": _decision(video="vid", frame=5, inst=1, action="autofix"),
            "vid|5|0": _decision(video="vid", frame=5, inst=0, action="accept"),
            "gone|1|0": _decision(video="gone", frame=1, inst=0, action="autofix"),
            "vid|5|9": _decision(video="vid", frame=5, inst=9, action="autofix"),
            "w|2|0": _decision(video="w", frame=2, inst=0, action="discard"),
            "x|3|0": _decision(video="x", frame=3, inst=0, action="box", box=(1, 2, 30, 40)),
            "y|4|0": _decision(video="y", frame=4, inst=0, action="skip"),
        }
        rows = apply_morph(decisions, masks_dir, out_dir)
        by_key = {r["key"]: r for r in rows}
        assert "y|4|0" not in by_key  # skip rows are not logged
        assert by_key["vid|5|0"]["disposition"] == "accepted_as_is"
        assert by_key["w|2|0"]["disposition"] == "excluded"
        assert by_key["x|3|0"]["disposition"] == "pending_sam3"
        assert by_key["gone|1|0"]["disposition"] == "errored"
        assert by_key["vid|5|9"]["disposition"] == "errored"

        fixed_row = by_key["vid|5|1"]
        assert fixed_row["disposition"] == "corrected"
        assert fixed_row["area_before"] == int(broken.sum())
        assert Path(fixed_row["out_path"]).exists()

        got = load_instance_masks(out_dir, "vid", 5)
        assert [g["instance_idx"] for g in got] == [0, 1]
        assert (got[0]["mask"] == clean).all()  # untouched sibling preserved
        fixed = got[1]["mask"]
        assert fixed[60, 80], "hole should be filled"
        assert not fixed[7, 145], "straggler should be dropped"
        assert got[1]["area_px"] == int(fixed.sum()) == 2500
        assert got[1]["center_xy"] == (84, 64)

        # originals untouched
        orig = load_instance_masks(masks_dir, "vid", 5)
        assert (orig[1]["mask"] == broken).all()


def test_write_applied_schema():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "applied.csv"
        write_applied([{f: "" for f in APPLIED_FIELDS}], path)
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            assert r.fieldnames == APPLIED_FIELDS
            assert len(list(r)) == 1


# --------------------------------------------------------------------------------------


def _run_all():
    mod = sys.modules[__name__]
    tests = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
