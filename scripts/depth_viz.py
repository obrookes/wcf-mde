#!/usr/bin/env python
"""Shared depth-map visualisation helpers (cv2 + numpy only, no torch).

Factored out of run_calibration_eval.py so calibrate_depth.py can render orig|calib panels
without importing the heavy depth-inference stack.
"""
from __future__ import annotations

import cv2
import numpy as np

DEPTH_COLORMAP = cv2.COLORMAP_TURBO


def make_depth_colorbar(height: int, d_min: float, d_max: float,
                        bar_width: int = 30, label_width: int = 80) -> np.ndarray:
    """Vertical scale bar (max at top, min at bottom) in DEPTH_COLORMAP, labelled
    with the metric range (metres) it represents."""
    gradient = np.linspace(255, 0, height, dtype=np.uint8).reshape(-1, 1)
    bar = cv2.applyColorMap(np.repeat(gradient, bar_width, axis=1), DEPTH_COLORMAP)

    canvas = np.zeros((height, bar_width + label_width, 3), dtype=np.uint8)
    canvas[:, :bar_width] = bar
    for value, y in ((d_max, 15), (d_min, height - 8)):
        cv2.putText(canvas, f"{value:.2f}m", (bar_width + 4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def colorize_depth(depth: np.ndarray, d_min: float | None = None, d_max: float | None = None,
                   with_colorbar: bool = True) -> np.ndarray:
    """Colourise a depth/distance map with TURBO plus an optional metric scale bar.
    Returns a BGR image; non-finite pixels collapse to the low end of the range."""
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        h, w = depth.shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)
    if d_min is None:
        d_min = float(finite.min())
    if d_max is None:
        d_max = float(finite.max())
    norm = (depth - d_min) / max(d_max - d_min, 1e-6)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    norm = np.clip(norm, 0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), DEPTH_COLORMAP)
    if with_colorbar:
        vis = cv2.hconcat([vis, make_depth_colorbar(vis.shape[0], d_min, d_max)])
    return vis
