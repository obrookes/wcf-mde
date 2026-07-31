#!/usr/bin/env python
"""Per-video depth-map calibration from sparse ground-truth points.

Each annotated frame of a wcf-mde reference video gives ONE sparse supervision point:
the model's predicted depth at the subject (a person holding a sign, localized by SAM-3)
paired with the measured subject-to-camera distance. A video has up to ~16 such points.

This module fits a calibration transform `g` on those points and applies it pointwise to
the full depth map, turning the model's (residually mis-scaled) metric depth into a
calibrated metric distance map. Calibrators range from a single-parameter scale to a
2-D model that also uses the subject's vertical image position (ground-plane geometry).

All maths is plain NumPy. Fits auto-reduce complexity when there are too few points, so a
1-2 frame video never raises -- it just gets a simpler transform.

Public API:
    fit_calibration(pred, gt, method, *, ys=None, degree=2, robust=False) -> Calibrator
    Calibrator.apply(depth_map, ys=None) -> np.ndarray   # pointwise; non-finite/<=0 kept as-is
    Calibrator.predict(d, y=None) -> np.ndarray          # predict distance for subject depths
    Calibrator.params: dict                              # JSON-serializable fitted parameters
    leave_one_out(pred, gt, method, ...) -> dict         # honest cal vs uncal MAE
"""
from __future__ import annotations

import numpy as np

METHODS = ("scale", "linear", "disparity", "poly", "poly2d", "piecewise")

# minimum number of points a method needs to be fully determined (used to gate LOO CV)
_MIN_POINTS = {"scale": 1, "linear": 2, "disparity": 2, "poly": 2, "poly2d": 4, "piecewise": 4}


# --------------------------------------------------------------------------------------
# calibrators
# --------------------------------------------------------------------------------------

class Calibrator:
    """Base: subclasses implement predict(); apply() handles full 2-D maps uniformly."""

    method = "base"

    def __init__(self) -> None:
        self.params: dict = {}

    def predict(self, d, y=None) -> np.ndarray:  # noqa: ARG002 - y used only by poly2d
        raise NotImplementedError

    def apply(self, depth_map: np.ndarray, ys: np.ndarray | None = None) -> np.ndarray:
        """Apply the transform pointwise to a depth map. Pixels that are non-finite or
        non-positive are passed through unchanged (sky/invalid depth stays invalid). For
        poly2d, `ys` is a per-pixel *normalized* vertical position in [0, 1]; if omitted it
        is built from the map's own height, so the same fit works at any resolution."""
        d = np.asarray(depth_map, dtype=np.float64)
        out = d.astype(np.float32).copy()
        valid = np.isfinite(d) & (d > 0)
        if not valid.any():
            return out

        if self.method == "poly2d":
            if ys is None:
                h = d.shape[0]
                rows = (np.arange(h, dtype=np.float64) + 0.5) / max(h, 1)
                ys = np.broadcast_to(rows[:, None], d.shape)
            yv = np.asarray(ys, dtype=np.float64)
            pred = self.predict(d[valid], yv[valid])
        else:
            pred = self.predict(d[valid])

        # The transform is fit on the subject's (narrow) depth range; extrapolating it to
        # pixels far outside that range (sky, far background, near foreground) can yield
        # non-physical distances (negative, or exploding for disparity/poly). Mark those
        # invalid (NaN) rather than writing a misleading number into the calibrated map.
        pred = np.asarray(pred, dtype=np.float32)
        pred = np.where(np.isfinite(pred) & (pred > 0), pred, np.float32(np.nan))
        out[valid] = pred
        return out

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params})"


class ScaleCalibrator(Calibrator):
    """d_cal = s * d. Robust single-parameter fit (median ratio); good for very low N."""

    method = "scale"

    def __init__(self, s: float) -> None:
        super().__init__()
        self.s = float(s)
        self.params = {"s": self.s}

    @classmethod
    def fit(cls, pred: np.ndarray, gt: np.ndarray, robust: bool = True) -> "ScaleCalibrator":
        if robust:
            s = float(np.median(gt / pred))
        else:
            s = float(np.sum(gt * pred) / np.sum(pred * pred))
        return cls(s)

    def predict(self, d, y=None) -> np.ndarray:
        return self.s * np.asarray(d, dtype=np.float64)


class LinearCalibrator(Calibrator):
    """d_cal = a * d + b (affine). The default 'simple' calibration."""

    method = "linear"

    def __init__(self, a: float, b: float) -> None:
        super().__init__()
        self.a = float(a)
        self.b = float(b)
        self.params = {"a": self.a, "b": self.b}

    @classmethod
    def fit(cls, pred: np.ndarray, gt: np.ndarray, robust: bool = False) -> "LinearCalibrator":
        if pred.size == 1:  # underdetermined -> scale through the single point
            return cls(float(gt[0] / pred[0]), 0.0)
        if robust:
            a, b = _theil_sen(pred, gt)
        else:
            a, b = np.polyfit(pred, gt, 1)
        return cls(a, b)

    def predict(self, d, y=None) -> np.ndarray:
        return self.a * np.asarray(d, dtype=np.float64) + self.b


class DisparityCalibrator(Calibrator):
    """Affine in inverse-depth: 1/d_cal = a*(1/d) + b  ->  d_cal = 1/(a/d + b).

    Monocular depth error is often closer to affine in disparity than in depth, so this is
    a theoretically motivated alternative to the plain linear fit."""

    method = "disparity"

    def __init__(self, a: float, b: float) -> None:
        super().__init__()
        self.a = float(a)
        self.b = float(b)
        self.params = {"a": self.a, "b": self.b}

    @classmethod
    def fit(cls, pred: np.ndarray, gt: np.ndarray) -> "DisparityCalibrator":
        x = 1.0 / pred
        ygt = 1.0 / gt
        if pred.size == 1:  # one point, assume b=0 -> a = (1/gt)/(1/pred)
            return cls(float(ygt[0] / x[0]), 0.0)
        a, b = np.polyfit(x, ygt, 1)
        return cls(a, b)

    def predict(self, d, y=None) -> np.ndarray:
        d = np.asarray(d, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = self.a * (1.0 / d) + self.b
            return 1.0 / inv


class PolyCalibrator(Calibrator):
    """d_cal = polynomial_k(d). Degree is capped to what the point count supports."""

    method = "poly"

    def __init__(self, coeffs) -> None:
        super().__init__()
        self.coeffs = np.asarray(coeffs, dtype=np.float64)
        self.params = {"coeffs": self.coeffs.tolist(), "degree": int(self.coeffs.size - 1)}

    @classmethod
    def fit(cls, pred: np.ndarray, gt: np.ndarray, degree: int = 2) -> "PolyCalibrator":
        if pred.size == 1:  # underdetermined -> scale through the point (matches LinearCalibrator)
            return cls([float(gt[0] / pred[0]), 0.0])
        deg = int(min(degree, max(1, pred.size - 1)))
        coeffs = np.polyfit(pred, gt, deg)
        return cls(coeffs)

    def predict(self, d, y=None) -> np.ndarray:
        return np.polyval(self.coeffs, np.asarray(d, dtype=np.float64))


class Poly2DCalibrator(Calibrator):
    """distance = f(d, v) with v the *normalized* vertical image position in [0, 1].

    Uses the subject's row in the frame as an extra cue: on a ground plane, image row
    correlates with distance largely independent of the raw depth value. Feature columns
    are added in order [1, d, v, d*v, d^2, v^2], truncated to what the point count supports
    (so it never over-fits a handful of frames)."""

    method = "poly2d"
    FEATURES = ("1", "d", "v", "d*v", "d^2", "v^2")

    def __init__(self, coeffs, n_features: int) -> None:
        super().__init__()
        self.coeffs = np.asarray(coeffs, dtype=np.float64)
        self.n_features = int(n_features)
        self.params = {"coeffs": self.coeffs.tolist(), "features": list(self.FEATURES[:n_features])}

    @staticmethod
    def _design(d, v, n_features: int) -> np.ndarray:
        d = np.asarray(d, dtype=np.float64).ravel()
        v = np.asarray(v, dtype=np.float64).ravel()
        cols = [np.ones_like(d), d, v, d * v, d * d, v * v]
        return np.stack(cols[:n_features], axis=1)

    @classmethod
    def fit(cls, pred: np.ndarray, gt: np.ndarray, ys: np.ndarray) -> "Poly2DCalibrator":
        n = pred.size
        n_features = min(len(cls.FEATURES), max(2, n - 1), n)
        x = cls._design(pred, ys, n_features)
        coeffs, *_ = np.linalg.lstsq(x, gt, rcond=None)
        return cls(coeffs, n_features)

    def predict(self, d, y=None) -> np.ndarray:
        if y is None:
            raise ValueError("poly2d.predict requires vertical positions `y`")
        shape = np.asarray(d).shape
        x = self._design(d, y, self.n_features)
        return (x @ self.coeffs).reshape(shape)


class PiecewiseCalibrator(Calibrator):
    """Two linear segments meeting continuously at a fitted breakpoint k (predicted-depth
    space): d_cal = a1*d+b1 for d<=k, else a1*k+b1 + a2*(d-k). Matches Markham-25's segmented
    (piecewise/breakpoint) regression, used when a single global linear/poly fit under-serves
    both near and far ranges. Continuity at k is enforced by construction, not fit."""

    method = "piecewise"

    def __init__(self, a1: float, b1: float, a2: float, k: float) -> None:
        super().__init__()
        self.a1 = float(a1)
        self.b1 = float(b1)
        self.a2 = float(a2)
        self.k = float(k)
        self.params = {"a1": self.a1, "b1": self.b1, "a2": self.a2, "k": self.k}

    @classmethod
    def fit(cls, pred: np.ndarray, gt: np.ndarray) -> "PiecewiseCalibrator":
        n = pred.size
        uniq = np.unique(pred)
        if n < 4 or uniq.size < 2:
            # too few points/distinct values to fit two segments -> single-segment fallback,
            # behaves exactly like LinearCalibrator (k=+inf so predict never takes the k<d branch)
            lin = LinearCalibrator.fit(pred, gt)
            return cls(lin.a, lin.b, lin.a, float("inf"))

        candidates = uniq[1:-1]  # interior order statistics -> both sides keep >=2 points
        if candidates.size == 0:
            lin = LinearCalibrator.fit(pred, gt)
            return cls(lin.a, lin.b, lin.a, float("inf"))

        best = None
        for k in candidates:
            lo = pred <= k
            hi = ~lo
            if lo.sum() < 2 or hi.sum() < 1:
                continue
            a1, b1 = np.polyfit(pred[lo], gt[lo], 1)
            # continuity: intercept on the high side is pinned to a1*k+b1 at d=k, so only the
            # slope a2 is free -> least-squares over (d-k) vs (gt - (a1*k+b1))
            x_hi = pred[hi] - k
            y_hi = gt[hi] - (a1 * k + b1)
            if hi.sum() >= 1 and np.any(x_hi != 0):
                a2 = float(np.sum(x_hi * y_hi) / np.sum(x_hi * x_hi))
            else:
                a2 = a1
            pred_lo = a1 * pred[lo] + b1
            pred_hi = (a1 * k + b1) + a2 * x_hi
            sse = float(np.sum((pred_lo - gt[lo]) ** 2) + np.sum((pred_hi - gt[hi]) ** 2))
            if best is None or sse < best[0]:
                best = (sse, a1, b1, a2, float(k))

        if best is None:
            lin = LinearCalibrator.fit(pred, gt)
            return cls(lin.a, lin.b, lin.a, float("inf"))
        _, a1, b1, a2, k = best
        return cls(a1, b1, a2, k)

    def predict(self, d, y=None) -> np.ndarray:
        d = np.asarray(d, dtype=np.float64)
        lo_val = self.a1 * d + self.b1
        if np.isinf(self.k):  # single-segment fallback -- avoid inf-inf nan in the unused branch
            return lo_val
        hi_val = (self.a1 * self.k + self.b1) + self.a2 * (d - self.k)
        return np.where(d <= self.k, lo_val, hi_val)


_REGISTRY = {
    "scale": ScaleCalibrator,
    "linear": LinearCalibrator,
    "disparity": DisparityCalibrator,
    "poly": PolyCalibrator,
    "poly2d": Poly2DCalibrator,
    "piecewise": PiecewiseCalibrator,
}


# --------------------------------------------------------------------------------------
# fitting + evaluation
# --------------------------------------------------------------------------------------

def _theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Robust affine fit: median of pairwise slopes, median intercept. No deps, O(n^2)
    (n <= 16 here)."""
    n = len(x)
    slopes = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(n) for j in range(i + 1, n) if x[j] != x[i]
    ]
    a = float(np.median(slopes)) if slopes else 0.0
    b = float(np.median(y - a * x))
    return a, b


def _clean(pred, gt, ys):
    """Drop points that can't supervise a fit: non-finite, non-positive depth/distance."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    ys_arr = None
    if ys is not None:
        ys_arr = np.asarray(ys, dtype=np.float64)
        mask &= np.isfinite(ys_arr)
        ys_arr = ys_arr[mask]
    return pred[mask], gt[mask], ys_arr


def fit_calibration(
    pred,
    gt,
    method: str = "linear",
    *,
    ys=None,
    degree: int = 2,
    robust: bool = False,
) -> Calibrator:
    """Fit a calibrator mapping subject predicted-depth -> true distance.

    pred, gt : 1-D arrays of (predicted depth at subject, ground-truth distance) per frame.
    ys       : per-frame normalized vertical position in [0, 1] (required for 'poly2d').
    """
    if method not in _REGISTRY:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    pred, gt, ys_arr = _clean(pred, gt, ys)
    if pred.size == 0:
        raise ValueError("no valid (pred, gt) points to fit calibration")

    if method == "scale":
        return ScaleCalibrator.fit(pred, gt, robust=robust)
    if method == "linear":
        return LinearCalibrator.fit(pred, gt, robust=robust)
    if method == "disparity":
        return DisparityCalibrator.fit(pred, gt)
    if method == "poly":
        return PolyCalibrator.fit(pred, gt, degree=degree)
    if method == "piecewise":
        return PiecewiseCalibrator.fit(pred, gt)
    # poly2d
    if ys_arr is None:
        raise ValueError("poly2d requires per-point vertical positions `ys`")
    return Poly2DCalibrator.fit(pred, gt, ys_arr)


def leave_one_out(
    pred,
    gt,
    method: str = "linear",
    *,
    ys=None,
    degree: int = 2,
    robust: bool = False,
) -> dict:
    """Leave-one-out CV: honest calibrated MAE vs the uncalibrated baseline (raw model depth
    at the subject vs ground truth). Returns loo_mae_cal=None when there are too few points
    to leave one out for this method. The *delivered* calibrator should still be refit on all
    points (this is for reporting only)."""
    pred, gt, ys_arr = _clean(pred, gt, ys)
    n = int(pred.size)
    uncal_mae = float(np.mean(np.abs(pred - gt))) if n else None
    if n < _MIN_POINTS[method] + 1:
        return {"n": n, "loo_mae_cal": None, "loo_mae_uncal": uncal_mae, "loo_residuals": None}

    residuals = []
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        ytr = None if ys_arr is None else ys_arr[tr]
        cal = fit_calibration(pred[tr], gt[tr], method, ys=ytr, degree=degree, robust=robust)
        yi = None if ys_arr is None else ys_arr[i:i + 1]
        p = float(np.asarray(cal.predict(pred[i:i + 1], yi)).ravel()[0])
        residuals.append(abs(p - float(gt[i])))
    return {
        "n": n,
        "loo_mae_cal": float(np.mean(residuals)),
        "loo_mae_uncal": uncal_mae,
        "loo_residuals": residuals,
        "loo_residuals_uncal": np.abs(pred - gt).tolist(),
        "loo_gt": gt.tolist(),
    }
