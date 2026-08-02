"""Per-mask geometric and photometric signals -- the cheap tier of mask triage.

Every function here runs on CPU from artifacts already on disk (a decoded frame and a
COCO-RLE mask), needs no model forward pass, no ground truth, and no annotation. That is
the whole point: it gives a ranked review queue before any GPU work happens.

Signals are chosen to be *interpretable* -- each one names a specific, visible failure
mode an annotator can confirm at a glance:

    n_components / second_component_frac   mask leaked onto a second object, or shattered
    hole_area_frac                         mask swiss-cheesed over the subject
    compactness / erosion_survival         sliver or filament, not an object
    border_contact_frac                    subject truncated at the frame edge
    banner_overlap_frac                    mask ate the burnt-in status bar
    boundary_gradient_ratio                mask boundary doesn't sit on an image edge
    area_frac                              degenerate scale

None of them is decisive alone; they are fused and thresholded in scripts/score_masks.py.
All expect the mask and frame to already be banner-cropped (scripts/banner.py) and to
share the same (H, W) -- except `banner_overlap_frac`, which is measured on the *uncropped*
mask and passed in.
"""
from __future__ import annotations

import cv2
import numpy as np


def connected_component_stats(mask: np.ndarray) -> dict:
    """Fragmentation. A clean instance is one blob; two comparable blobs usually means the
    mask spans two subjects, or a subject plus a lookalike patch of background."""
    n_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    # label 0 is background; component sizes for labels 1..n-1
    sizes = np.bincount(labels.ravel())[1:] if n_labels > 1 else np.array([], dtype=int)
    sizes = np.sort(sizes)[::-1]
    largest = int(sizes[0]) if sizes.size else 0
    second = int(sizes[1]) if sizes.size > 1 else 0
    return {
        "n_components": int(sizes.size),
        "largest_component_px": largest,
        "second_component_frac": (second / largest) if largest else 0.0,
    }


def hole_stats(mask: np.ndarray) -> dict:
    """Interior holes: background components that don't reach the image border.

    Touching the border is what separates "hole" from "the rest of the scene", so a mask
    that reaches an edge simply has fewer background components counted as holes.
    """
    inverted = (~mask.astype(bool)).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(inverted, connectivity=4)
    if n_labels <= 1:
        return {"n_holes": 0, "hole_area_frac": 0.0}

    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    sizes = np.bincount(labels.ravel(), minlength=n_labels)
    hole_labels = [i for i in range(1, n_labels) if i not in border_labels]
    hole_area = int(sum(sizes[i] for i in hole_labels))
    area = int(mask.sum())
    return {
        "n_holes": len(hole_labels),
        "hole_area_frac": (hole_area / area) if area else 0.0,
    }


def shape_stats(mask: np.ndarray) -> dict:
    """Compactness and erosion survival -- two independent ways of catching slivers.

    Compactness (4*pi*A / P^2) is 1 for a disc and tends to 0 for a filament, but it is
    also driven down by ragged boundaries on a perfectly good mask. Erosion survival --
    the area fraction left after a 3x3 erode -- is blunter and cares only about thickness,
    so a mask that is thin *everywhere* scores low on both while a merely rough mask
    scores low only on compactness.
    """
    mask_u8 = mask.astype(np.uint8)
    area = int(mask_u8.sum())
    if area == 0:
        return {"perimeter_px": 0.0, "compactness": 0.0, "erosion_survival": 0.0}

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    compactness = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
    return {
        "perimeter_px": perimeter,
        "compactness": float(min(compactness, 1.0)),
        "erosion_survival": float(eroded.sum()) / area,
    }


def border_contact_frac(mask: np.ndarray) -> float:
    """Fraction of the mask's boundary pixels that lie on the frame edge -> truncation.

    Note this is measured *after* the banner crop, so the bottom edge is the last row of
    real scene. That is the intended behaviour: a subject cut off by the banner is just as
    truncated as one cut off by the sensor.
    """
    mask_u8 = mask.astype(np.uint8)
    if not mask_u8.any():
        return 0.0
    boundary = mask_u8 - cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
    n_boundary = int(boundary.sum())
    if n_boundary == 0:
        return 0.0
    on_edge = int(
        mask_u8[0, :].sum() + mask_u8[-1, :].sum() + mask_u8[:, 0].sum() + mask_u8[:, -1].sum()
    )
    return min(on_edge / n_boundary, 1.0)


def boundary_gradient_ratio(mask: np.ndarray, gray: np.ndarray) -> float:
    """Mean image-gradient magnitude in a band around the mask boundary, over the frame's own mean.

    A correct boundary lies on an intensity edge, so the ratio is well above 1. A boundary
    hallucinated across flat ground, or one that cuts through the middle of a uniform
    region, sits near or below 1. Normalising by the frame's own mean gradient is what
    makes this comparable between a cluttered canopy scene and a bare forest floor.
    """
    if not mask.any():
        return 0.0
    grad_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(grad_x, grad_y)

    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = mask.astype(np.uint8)
    band = cv2.dilate(mask_u8, kernel) - cv2.erode(mask_u8, kernel)
    band = band.astype(bool)
    if not band.any():
        return 0.0

    frame_mean = float(magnitude.mean())
    if frame_mean <= 0:
        return 0.0
    return float(magnitude[band].mean() / frame_mean)


def mask_signals(mask: np.ndarray, gray: np.ndarray, banner_overlap_frac: float = 0.0) -> dict:
    """All per-mask signals for one instance. `mask` and `gray` must be banner-cropped.

    `banner_overlap_frac` is the share of the mask's *original* area that fell inside the
    banner rows, which the caller measures before cropping -- a mask that is largely banner
    is a distinct and obvious failure, and cropping destroys the evidence for it.
    """
    mask = mask.astype(bool)
    height, width = mask.shape
    area = int(mask.sum())

    signals = {
        "area_px": area,
        "area_frac": area / float(height * width),
        "banner_overlap_frac": float(banner_overlap_frac),
        "border_contact_frac": border_contact_frac(mask),
        "boundary_gradient_ratio": boundary_gradient_ratio(mask, gray),
    }
    signals.update(connected_component_stats(mask))
    signals.update(hole_stats(mask))
    signals.update(shape_stats(mask))
    return signals


SIGNAL_COLUMNS = [
    "area_px",
    "area_frac",
    "banner_overlap_frac",
    "border_contact_frac",
    "boundary_gradient_ratio",
    "n_components",
    "largest_component_px",
    "second_component_frac",
    "n_holes",
    "hole_area_frac",
    "perimeter_px",
    "compactness",
    "erosion_survival",
]
