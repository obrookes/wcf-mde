#!/usr/bin/env python
"""Synthetic unit tests for scripts/calibration.py. Run directly:  python scripts/test_calibration.py
(also pytest-compatible). No GPU / data / torch needed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.calibration import fit_calibration, leave_one_out


def test_linear_recovers_affine():
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    gt = 2.0 * d + 1.0
    cal = fit_calibration(d, gt, "linear")
    assert abs(cal.a - 2.0) < 1e-9 and abs(cal.b - 1.0) < 1e-9
    # apply() to a full map matches the affine pointwise
    m = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(cal.apply(m), 2.0 * m + 1.0, rtol=1e-5)


def test_scale_recovers():
    d = np.array([2.0, 4.0, 6.0, 8.0])
    cal = fit_calibration(d, 3.0 * d, "scale")
    assert abs(cal.s - 3.0) < 1e-9


def test_disparity_recovers():
    a, b = 1.5, 0.2
    d = np.array([2.0, 3.0, 5.0, 8.0, 13.0])
    gt = 1.0 / (a / d + b)  # 1/gt = a*(1/d) + b
    cal = fit_calibration(d, gt, "disparity")
    assert abs(cal.a - a) < 1e-6 and abs(cal.b - b) < 1e-6
    np.testing.assert_allclose(cal.predict(d), gt, rtol=1e-6)


def test_poly_recovers_quadratic():
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    gt = 0.5 * d**2 - 1.0 * d + 2.0
    cal = fit_calibration(d, gt, "poly", degree=2)
    np.testing.assert_allclose(cal.predict(d), gt, rtol=1e-6)


def test_poly2d_uses_vertical_position():
    rng = np.random.default_rng(0)
    d = rng.uniform(2, 20, size=10)
    v = rng.uniform(0, 1, size=10)
    gt = 1.0 + 0.5 * d + 2.0 * v  # truly depends on vertical position
    cal = fit_calibration(d, gt, "poly2d", ys=v)
    np.testing.assert_allclose(cal.predict(d, v), gt, atol=1e-6)


def test_apply_preserves_invalid_pixels():
    cal = fit_calibration(np.array([1.0, 2.0]), np.array([3.0, 5.0]), "linear")  # a=2,b=1
    m = np.array([[2.0, 0.0], [-1.0, np.nan], [np.inf, 4.0]], dtype=np.float32)
    out = cal.apply(m)
    assert out[0, 0] == np.float32(5.0)   # 2*2+1
    assert out[0, 1] == np.float32(0.0)   # non-positive -> unchanged
    assert out[1, 0] == np.float32(-1.0)  # non-positive -> unchanged
    assert np.isnan(out[1, 1])            # nan -> unchanged
    assert np.isinf(out[2, 0])            # inf -> unchanged
    assert out[2, 1] == np.float32(9.0)   # 2*4+1


def test_apply_marks_nonphysical_extrapolation_invalid():
    # fit a decreasing transform so near pixels extrapolate to negative "distance"
    cal = fit_calibration(np.array([3.0, 4.0, 5.0]), np.array([5.0, 4.0, 3.0]), "linear")  # a<0
    m = np.array([[4.0, 100.0]], dtype=np.float32)  # 4 -> ~4m (ok); 100 -> negative -> NaN
    out = cal.apply(m)
    assert out[0, 0] > 0 and np.isfinite(out[0, 0])
    assert np.isnan(out[0, 1])  # non-physical extrapolation marked invalid, not written


def test_piecewise_recovers_known_breakpoint():
    a1, b1, a2, k = 2.0, 1.0, 5.0, 10.0
    d_lo = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # includes the breakpoint itself
    d_hi = np.array([12.0, 15.0, 18.0, 20.0])
    d = np.concatenate([d_lo, d_hi])
    gt_lo = a1 * d_lo + b1
    gt_hi = (a1 * k + b1) + a2 * (d_hi - k)
    gt = np.concatenate([gt_lo, gt_hi])
    cal = fit_calibration(d, gt, "piecewise")
    np.testing.assert_allclose(cal.predict(d), gt, rtol=1e-6, atol=1e-6)


def test_piecewise_is_continuous_at_breakpoint():
    d = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 14.0, 17.0])
    gt = np.where(d <= 8.0, 2.0 * d, 2.0 * 8.0 + 0.3 * (d - 8.0))
    cal = fit_calibration(d, gt, "piecewise")
    just_below = cal.predict(np.array([cal.k - 1e-6]))[0]
    just_above = cal.predict(np.array([cal.k + 1e-6]))[0]
    assert abs(just_below - just_above) < 1e-3


def test_piecewise_low_n_falls_back_to_linear():
    d = np.array([1.0, 2.0, 3.0])
    gt = 2.0 * d + 1.0
    cal = fit_calibration(d, gt, "piecewise")
    np.testing.assert_allclose(cal.predict(d), gt, rtol=1e-6)
    assert cal.k == float("inf")


def test_piecewise_leave_one_out_runs():
    d = np.array([1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0])
    gt = np.where(d <= 5.0, 2.0 * d, 2.0 * 5.0 + 0.5 * (d - 5.0))
    loo = leave_one_out(d, gt, "piecewise")
    assert loo["loo_mae_cal"] is not None
    assert loo["loo_mae_cal"] < loo["loo_mae_uncal"]


def test_poly_single_point_falls_back():
    cal = fit_calibration(np.array([4.0]), np.array([8.0]), "poly", degree=2)
    assert abs(float(cal.predict(np.array([4.0]))[0]) - 8.0) < 1e-9  # scale through the point


def test_leave_one_out_beats_uncalibrated():
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    gt = 2.0 * d + 1.0  # perfectly affine -> LOO calibrated error ~0
    loo = leave_one_out(d, gt, "linear")
    assert loo["loo_mae_cal"] is not None
    assert loo["loo_mae_cal"] < 1e-6
    assert loo["loo_mae_uncal"] > loo["loo_mae_cal"]


def test_low_n_does_not_crash():
    # single point: linear falls back to scale-through-point, LOO declines gracefully
    cal = fit_calibration(np.array([4.0]), np.array([8.0]), "linear")
    assert abs(float(cal.predict(np.array([4.0]))[0]) - 8.0) < 1e-9
    loo = leave_one_out(np.array([4.0]), np.array([8.0]), "linear")
    assert loo["loo_mae_cal"] is None  # too few points to leave one out


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nall {len(fns)} calibration tests passed")


if __name__ == "__main__":
    _run_all()
