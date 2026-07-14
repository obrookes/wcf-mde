#!/usr/bin/env python
"""Stage 1 (optional): cross-frame background alignment, before Stage-2 calibration.

Both Markham-25 (papers/markham-25.pdf, S2.3.1) and timmh/distance-estimation align each
calibration frame's disparity to a single reference frame (the one with the greatest known
distance) before fitting a final depth->distance calibrator, using the *background* (non-subject)
pixels only. wcf-mde's depth inference is already joint/multi-frame (see
run_calibration_eval.py's docstring), so it isn't known ahead of time whether this extra step
still helps here -- hence "optional": scripts/calibrate_depth.py's --align flag lets --align none
vs --align ransac be compared empirically on the same data (see calibration_fits.csv's
alignment_* columns), rather than assuming the answer.

All maths is plain NumPy -- no scikit-learn dependency in this repo's env. The RANSAC affine fit
below mirrors timmh/distance-estimation's utils.py calibrate() RANSAC branch (a non-negative-slope
linear fit wrapped in RANSACRegressor): random-sample-and-verify over minimal subsets, reject
non-positive slopes, refit least-squares on the inlier set.

Public API:
    pick_reference_frame(points, method="furthest") -> dict
    align_frame_to_reference(depth, mask, ref_depth, ref_mask, method="ransac") -> (a, b)
    aligned_subject_value(depth, mask, a, b) -> float
"""
from __future__ import annotations

import numpy as np

REFERENCE_METHODS = ("furthest", "median")
ALIGN_METHODS = ("ransac",)

MIN_BACKGROUND_PIXELS = 8  # below this, align_frame_to_reference falls back to identity


def pick_reference_frame(points: list[dict], method: str = "furthest") -> dict:
    """Pick one point (dict with a 'gt' key) from `points` to anchor Stage-1 alignment against.

    'furthest' (default; matches timmh/Markham-25): the point with the largest ground-truth
    distance -- extrapolating outward from a far anchor is the safer direction, since monocular
    depth error typically grows with distance.

    'median': the point whose gt is closest to the group's median distance. A single farthest
    frame is a riskier anchor here than in Markham-25's 60-point Oxfordshire grid, since
    per-camera point counts in this dataset are much smaller (up to ~16) -- one bad SAM-3 mask
    on that one frame would corrupt every other frame's alignment. 'median' is a gentler,
    less-extreme default to fall back to if 'furthest' turns out to be too fragile in practice.
    """
    if not points:
        raise ValueError("no points to pick a reference frame from")
    if method == "furthest":
        return max(points, key=lambda p: p["gt"])
    if method == "median":
        med = float(np.median([p["gt"] for p in points]))
        return min(points, key=lambda p: abs(p["gt"] - med))
    raise ValueError(f"unknown reference-frame method {method!r}; choose from {REFERENCE_METHODS}")


def _ransac_affine(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_samples: int = 2,
    max_trials: int = 200,
    residual_threshold: float | None = None,
    positive_slope: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Robust affine fit y = a*x + b via random-sample consensus.

    Mirrors timmh/distance-estimation's utils.py calibrate() RANSAC branch (RANSACRegressor
    wrapping a non-negative-slope LinearRegression) without a scikit-learn dependency: random
    minimal-subset fits, reject non-positive slopes, keep the largest inlier set, refit
    least-squares on it. Falls back to a plain least-squares fit if there aren't enough points
    for random sampling to be meaningful, or if RANSAC never finds a valid model.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n < max(min_samples, 3):
        a, b = np.polyfit(x, y, 1)
        return float(a), float(b)

    if residual_threshold is None:
        # scale-free default: a fraction of y's spread, rather than a hardcoded metre threshold
        # (unlike timmh's fixed t=0.5, tuned for their disparity units) -- this is applied to
        # background depth/disparity values whose scale varies by depth model and units.
        spread = float(np.ptp(y))
        residual_threshold = 0.1 * spread if spread > 0 else 1.0

    rng = rng or np.random.default_rng(0)
    best_inliers = None
    best_count = -1
    for _ in range(max_trials):
        sample_idx = rng.choice(n, size=min_samples, replace=False)
        xs, ys = x[sample_idx], y[sample_idx]
        if np.ptp(xs) == 0:
            continue
        a, b = np.polyfit(xs, ys, 1)
        if positive_slope and a <= 0:
            continue
        residuals = np.abs(y - (a * x + b))
        inliers = residuals < residual_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < min_samples:
        a, b = np.polyfit(x, y, 1)
        return float(a), float(b)

    a, b = np.polyfit(x[best_inliers], y[best_inliers], 1)
    if positive_slope and a <= 0:
        # constraint violated even after refit on inliers -- fall back rather than return a
        # model that inverts near/far ordering
        a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def align_frame_to_reference(
    depth: np.ndarray,
    mask: np.ndarray,
    ref_depth: np.ndarray,
    ref_mask: np.ndarray,
    method: str = "ransac",
) -> tuple[float, float]:
    """Fit (a, b) s.t. a*depth + b ~= ref_depth over background pixels only
    (~mask & ~ref_mask), mirroring Stage 1 of timmh/Markham-25: align each calibration frame to
    one reference frame using everything *except* the subject, so the subsequent subject-depth
    reading is expressed on the reference frame's scale before Stage 2 ever sees it.

    `depth`/`ref_depth` and `mask`/`ref_mask` must already share one resolution -- resize before
    calling (both frames' masks and depth maps can come from clips of different native
    resolutions within the same --calib-level cam group). Returns identity (1.0, 0.0) if there
    isn't enough valid background signal to fit -- callers should treat that as "alignment not
    applied" rather than a hard error, since a single frame's mask covering nearly the whole
    image is a legitimate (if unhelpful-for-alignment) SAM-3 result, not a bug.
    """
    if method not in ALIGN_METHODS:
        raise ValueError(f"unknown alignment method {method!r}; choose from {ALIGN_METHODS}")
    if depth.shape != ref_depth.shape or mask.shape != depth.shape or ref_mask.shape != depth.shape:
        raise ValueError("depth/mask/ref_depth/ref_mask must all share one resolution")

    bg = (~mask) & (~ref_mask) & np.isfinite(depth) & np.isfinite(ref_depth)
    if int(bg.sum()) < MIN_BACKGROUND_PIXELS:
        return 1.0, 0.0

    return _ransac_affine(depth[bg], ref_depth[bg], positive_slope=True)


def aligned_subject_value(depth: np.ndarray, mask: np.ndarray, a: float, b: float) -> float:
    """Median-in-mask of the aligned depth (a*depth + b).

    Median, not mean -- matching both timmh's and Markham-25's robustness choice against
    imperfect SAM-3 masks (Markham-25 S2.3.1 explicitly motivates this: "the median operation
    enhances robustness against imperfect landmark masks"). This differs from
    scripts/calibrate_depth.py's existing `--anchor mask_mean`, which is intentional: alignment
    is a separate, optional stage layered in front of Stage 2's existing anchor choice.
    """
    values = (a * depth.astype(np.float64) + b)[mask]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no valid values inside mask to compute aligned subject value")
    return float(np.median(finite))
