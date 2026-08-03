"""Detect and crop the camera's burnt-in status banner.

Every frame in this dataset carries a manufacturer status bar across the bottom -- a
Bushnell strip (white or black depending on model) showing temperature, date and a
running clock, roughly the last 20-24 of 404 rows (~5%). It is not scene content, and
leaving it in corrupts every signal downstream:

  * background models see a strip whose only variation is the ticking clock;
  * border-contact / truncation flags fire on every mask that reaches the frame bottom,
    because the real bottom of the *scene* is the top of the banner, not row H-1;
  * frame-wide gradient statistics are inflated by hard-edged text;
  * and SAM-3 can segment the banner itself.

Detection is per station rather than by a hardcoded row, because the strip's height
varies by camera model. It exploits the fact that the cameras are static and the banner
is near-constant: a banner row's pixels barely change across frames of the same station.
The clock digits do change, so the per-row statistic is the **median** over columns --
the handful of changing digit columns can't move it -- and rows are consumed from the
bottom up while that median stays near zero.
"""
from __future__ import annotations

import numpy as np

# Fallback for stations with too few frames to measure. Deliberately on the generous
# side of the 20-24 rows observed: over-cropping two rows of scene costs nothing,
# under-cropping leaves banner text inside every signal.
DEFAULT_BANNER_ROWS = 24

MIN_FRAMES_FOR_DETECTION = 5
# The statistic converges long before a big station's 400+ frames, and the float32 cast
# needed to compute it is what puts a worker at risk of being OOM-killed. Subsample.
MAX_FRAMES_FOR_DETECTION = 40
# A banner row's median temporal std, as a fraction of the frame's own median. Observed
# banner rows sit at ~0.00-0.05; the first scene row above them jumps to >1.
_QUIET_ROW_FRACTION = 0.10
_QUIET_ROW_FLOOR = 1.0
# Reject implausible detections rather than silently cropping half the frame away.
_MAX_BANNER_FRACTION = 0.25


def detect_banner_top(frames: np.ndarray) -> int | None:
    """First row index of the banner in a stack of frames from one static station.

    `frames` is (N, H, W) or (N, H, W, C), any dtype. Returns None when the station has
    too few frames, or when the detected strip is implausibly tall (a camera that moved
    mid-deployment, or a station whose frames aren't actually the same scene) -- callers
    should fall back to DEFAULT_BANNER_ROWS rather than trusting a bad measurement.
    """
    frames = np.asarray(frames)
    if frames.ndim == 4:
        frames = frames.mean(axis=-1)
    if frames.ndim != 3:
        raise ValueError(f"expected (N, H, W[, C]) frames, got shape {frames.shape}")
    if len(frames) < MIN_FRAMES_FOR_DETECTION:
        return None
    if len(frames) > MAX_FRAMES_FOR_DETECTION:
        # evenly spread rather than the first N, so a station whose banner changed
        # mid-deployment still shows the change as scene-level variance
        frames = frames[np.linspace(0, len(frames) - 1, MAX_FRAMES_FOR_DETECTION).astype(int)]

    height = frames.shape[1]
    temporal_std = frames.astype(np.float32).std(axis=0)   # (H, W)
    row_stat = np.median(temporal_std, axis=1)             # (H,)

    threshold = max(_QUIET_ROW_FLOOR, _QUIET_ROW_FRACTION * float(np.median(row_stat)))
    row = height - 1
    while row >= 0 and row_stat[row] < threshold:
        row -= 1
    top = row + 1

    n_banner_rows = height - top
    if n_banner_rows == 0 or n_banner_rows > _MAX_BANNER_FRACTION * height:
        return None
    return top


def banner_top_or_default(frames: np.ndarray | None, height: int) -> int:
    """detect_banner_top with the fallback applied -- always returns a usable row."""
    if frames is not None:
        top = detect_banner_top(frames)
        if top is not None:
            return top
    return max(0, height - DEFAULT_BANNER_ROWS)


def crop_banner(image: np.ndarray, banner_top: int) -> np.ndarray:
    """Drop the banner rows. Works on frames (H, W[, C]) and on masks (H, W)."""
    return image[:banner_top]
