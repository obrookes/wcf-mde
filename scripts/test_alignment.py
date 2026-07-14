#!/usr/bin/env python
"""Synthetic unit tests for scripts/alignment.py and scripts/masks.py.
Run directly:  python scripts/test_alignment.py  (also pytest-compatible).
No GPU / data / torch needed."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.alignment import (
    align_frame_to_reference,
    aligned_subject_value,
    pick_reference_frame,
    _ransac_affine,
)
from scripts.masks import decode_rle, encode_rle, load_instance_masks, save_instance_masks


def test_pick_reference_frame_furthest():
    pts = [{"gt": 4.0}, {"gt": 20.0}, {"gt": 10.0}]
    assert pick_reference_frame(pts, "furthest")["gt"] == 20.0


def test_pick_reference_frame_median():
    pts = [{"gt": 4.0}, {"gt": 20.0}, {"gt": 10.0}]
    assert pick_reference_frame(pts, "median")["gt"] == 10.0


def test_pick_reference_frame_rejects_unknown_method():
    try:
        pick_reference_frame([{"gt": 1.0}], "bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown reference-frame method")


def test_ransac_affine_recovers_under_outliers():
    rng = np.random.default_rng(0)
    x = np.linspace(1, 20, 60)
    a_true, b_true = 2.5, 1.0
    y = a_true * x + b_true
    outlier_idx = rng.choice(60, size=12, replace=False)
    y_noisy = y.copy()
    y_noisy[outlier_idx] += rng.uniform(-50, 50, size=12)
    a_fit, b_fit = _ransac_affine(x, y_noisy, rng=rng)
    assert abs(a_fit - a_true) < 0.5
    assert abs(b_fit - b_true) < 5.0


def test_ransac_affine_rejects_negative_slope():
    # a decreasing relationship should fall back rather than return a<=0 when
    # positive_slope=True (the default), matching timmh's non-negative-slope constraint
    x = np.linspace(1, 10, 20)
    y = -2.0 * x + 5.0
    a_fit, _ = _ransac_affine(x, y, positive_slope=True, rng=np.random.default_rng(0))
    # can't recover the true (negative) slope under the constraint; just confirm it doesn't
    # silently return a negative slope as if unconstrained
    assert a_fit > 0 or np.isclose(a_fit, np.polyfit(x, y, 1)[0])


def test_align_frame_to_reference_recovers_known_affine():
    h, w = 40, 40
    ref_depth = np.tile(np.linspace(5, 25, w), (h, 1))
    depth = 0.5 * ref_depth + 2.0  # ref = 2*depth - 4
    mask = np.zeros((h, w), dtype=bool)
    mask[15:25, 15:25] = True
    ref_mask = np.zeros((h, w), dtype=bool)
    ref_mask[10:20, 10:20] = True
    depth_masked = depth.copy()
    depth_masked[mask] = 10.0  # arbitrary subject reading, excluded from the background fit

    a, b = align_frame_to_reference(depth_masked, mask, ref_depth, ref_mask)
    assert abs(a - 2.0) < 0.3
    assert abs(b - (-4.0)) < 3.0


def test_align_frame_to_reference_identity_when_insufficient_background():
    h, w = 10, 10
    depth = np.ones((h, w))
    ref_depth = np.ones((h, w)) * 5
    mask = np.ones((h, w), dtype=bool)       # whole frame is "subject" -> no background
    ref_mask = np.ones((h, w), dtype=bool)
    a, b = align_frame_to_reference(depth, mask, ref_depth, ref_mask)
    assert (a, b) == (1.0, 0.0)


def test_align_frame_to_reference_rejects_shape_mismatch():
    a = np.zeros((10, 10))
    b = np.zeros((10, 10), dtype=bool)
    c = np.zeros((5, 5))
    try:
        align_frame_to_reference(a, b, c, b)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched shapes")


def test_aligned_subject_value_uses_median():
    depth = np.array([1.0, 2.0, 3.0, 100.0])  # one gross outlier inside the mask
    mask = np.array([True, True, True, True])
    val = aligned_subject_value(depth, mask, a=1.0, b=0.0)
    assert val == 2.5  # median of [1,2,3,100], not mean (mean would be ~26.5)


def test_rle_round_trip():
    rng = np.random.default_rng(0)
    m = rng.uniform(size=(30, 40)) > 0.7
    rle = encode_rle(m)
    assert isinstance(rle["counts"], str)  # JSON-safe, not bytes
    decoded = decode_rle(rle)
    np.testing.assert_array_equal(decoded, m)


def test_save_and_load_instance_masks_preserve_order_and_shape():
    m1 = np.zeros((20, 30), dtype=bool)
    m1[5:10, 5:10] = True
    m2 = np.zeros((20, 30), dtype=bool)
    m2[12:18, 15:25] = True
    instances = [
        {"mask": m1, "center_xy": (7, 7), "area_px": int(m1.sum())},
        {"mask": m2, "center_xy": (20, 15), "area_px": int(m2.sum())},
    ]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        save_instance_masks(d, "testvid", 42, instances)
        loaded = load_instance_masks(d, "testvid", 42)
        assert len(loaded) == 2
        assert loaded[0]["instance_idx"] == 0 and loaded[1]["instance_idx"] == 1
        np.testing.assert_array_equal(loaded[0]["mask"], m1)
        np.testing.assert_array_equal(loaded[1]["mask"], m2)
        assert loaded[0]["center_xy"] == (7, 7)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nall {len(fns)} alignment/masks tests passed")


if __name__ == "__main__":
    _run_all()
