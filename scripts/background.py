"""Per-clip background modelling and the foreground residual it yields.

The cameras are static, so the strongest available check on a predicted mask needs no
model and no labels at all: build the scene's background from the station's own frames,
subtract it, and compare the leftover foreground silhouette against the mask. The two are
derived from completely independent evidence -- one from SAM-3, one from pixel statistics
-- so agreement is meaningful and disagreement is a concrete, visible thing to show an
annotator side by side.

Three things stop the naive version from working:

  * **Auto-exposure and auto-white-balance.** Consecutive frames of the *same* empty scene
    differ by a mean absolute 21-60 grey levels because the camera keeps re-metering under
    dappled canopy light. A two-frame difference is therefore useless; the model has to be
    a temporal median over many frames, and each frame has to be gain-normalised onto the
    background before differencing.
  * **The station is not one scene.** A camera-reference folder collects clips from repeat
    visits months apart -- the camera is re-aimed, vegetation grows, the season changes --
    so a median over the whole station comes out as a smear that matches no individual
    frame. Measured over 40 clips, station-scope backgrounds give a median mask/residual
    IoU of 0.05 (useless) against 0.27 for clip-scope ones. The background is therefore
    built **per clip**, and frames whose clip is too short to support one get no residual
    signal at all rather than a misleading one.
  * **Night.** Frames come in three photometric modes -- ordinary daylight colour,
    monochrome IR, and a magenta false-colour IR -- and their pixel statistics have nothing
    to do with each other. Pooling them into one median produces a background that matches
    none of them, so the mode is part of the grouping key, not a covariate.

Even at clip scope this is a *moderate* signal, not a decisive one: dappled forest light
and subjects whose clothing matches the litter keep median recall around 0.43. Its value
for triage is in the tail -- a mask with near-zero residual support is sitting on
something the camera says never moved -- not in discriminating 0.4 from 0.5.
"""
from __future__ import annotations

import cv2
import numpy as np

# Below this many frames a temporal median is dominated by whichever frames happen to
# contain the subject, and the "background" starts including the person.
MIN_FRAMES_FOR_BACKGROUND = 5

# Foreground threshold: max(absolute floor, k * robust sigma of the difference image).
# The floor stops a perfectly still scene from turning sensor noise into foreground.
# k and the floor were swept against the predicted masks over 40 clips; this setting
# maximises mask/residual agreement (median IoU 0.27, recall 0.43) and the surface is
# flat enough nearby that the exact values don't much matter.
_RESIDUAL_SIGMA_K = 3.0
_RESIDUAL_ABS_FLOOR = 10.0
_MAD_TO_SIGMA = 1.4826


def photometric_mode(frame_bgr: np.ndarray) -> str:
    """"day", "ir_mono" or "ir_magenta" -- the frame's illumination regime.

    Monochrome IR is near-zero saturation. The magenta mode (rare, ~0.3% of frames) is the
    opposite extreme: saturation above 110 with a hue in the magenta band, which daylight
    forest scenes never reach. Anything else is daylight colour.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    saturation = float(hsv[..., 1].mean())
    hue = float(np.median(hsv[..., 0]))
    if saturation < 15.0:
        return "ir_mono"
    if 140.0 <= hue <= 175.0 and saturation > 110.0:
        return "ir_magenta"
    return "day"


def normalise_gain(gray: np.ndarray, reference_median: float) -> np.ndarray:
    """Rescale a frame so its median matches the background's, absorbing exposure drift.

    A multiplicative correction (rather than an additive offset) because the camera's
    response to re-metering is a gain change, so bright regions shift more than dark ones.
    """
    gray = gray.astype(np.float32)
    median = float(np.median(gray))
    if median <= 1e-6:
        return gray
    return gray * (reference_median / median)


def build_background(grays: np.ndarray) -> np.ndarray | None:
    """Temporal median background from one clip's frames in one photometric mode.

    `grays` is (N, H, W) float or uint8, already banner-cropped. Returns None when there
    are too few frames to be trustworthy. Each frame is gain-normalised onto a common
    level *before* the median so exposure drift doesn't smear the result.
    """
    grays = np.asarray(grays)
    if len(grays) < MIN_FRAMES_FOR_BACKGROUND:
        return None
    reference_median = float(np.median(grays))
    stack = np.stack([normalise_gain(g, reference_median) for g in grays])
    return np.median(stack, axis=0).astype(np.float32)


def foreground_residual(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Boolean foreground mask: what changed relative to the station's background.

    The threshold is derived from the difference image's own robust spread rather than
    fixed, because residual noise varies hugely between a still scene and one with moving
    canopy shadow. An open-then-close pass removes speckle and fills the interior of the
    subject, which otherwise comes out hollow wherever it happens to match the ground.
    """
    normalised = normalise_gain(gray, float(np.median(background)))
    difference = np.abs(normalised - background)

    median = float(np.median(difference))
    sigma = _MAD_TO_SIGMA * float(np.median(np.abs(difference - median)))
    threshold = max(_RESIDUAL_ABS_FLOOR, median + _RESIDUAL_SIGMA_K * sigma)

    residual = (difference > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, kernel)
    residual = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, kernel)
    return residual.astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = int((a | b).sum())
    if union == 0:
        return 0.0
    return float((a & b).sum()) / union


def residual_signals(mask: np.ndarray, residual: np.ndarray) -> dict:
    """How well an independent pixel-statistics silhouette agrees with the predicted mask.

    Three numbers rather than one, because they fail in different directions and an
    annotator needs to know which:

      residual_iou        overall agreement
      residual_recall     share of the mask backed by actual scene change -- low means the
                          mask covers background the camera says never moved
      residual_precision  share of the change explained by this mask -- low means real
                          movement was left unmasked, which is an exhaustivity hint rather
                          than a boundary-quality one
    """
    mask, residual = mask.astype(bool), residual.astype(bool)
    mask_area, residual_area = int(mask.sum()), int(residual.sum())
    intersection = int((mask & residual).sum())
    return {
        "residual_iou": iou(mask, residual),
        "residual_recall": (intersection / mask_area) if mask_area else 0.0,
        "residual_precision": (intersection / residual_area) if residual_area else 0.0,
        "residual_area_px": residual_area,
    }


def unexplained_residual_frac(masks: list[np.ndarray], residual: np.ndarray) -> float:
    """Share of the frame's foreground change covered by no predicted mask at all.

    This is the frame-level exhaustivity signal: a large unexplained residual means
    something moved that SAM-3 did not segment. It is deliberately kept separate from the
    per-mask quality signals -- a frame can contain one perfect mask and still be missing
    a second subject entirely, and fusing the two would hide exactly that case.
    """
    residual = residual.astype(bool)
    residual_area = int(residual.sum())
    if residual_area == 0:
        return 0.0
    return float(_unexplained(masks, residual).sum()) / residual_area


def _unexplained(masks: list[np.ndarray], residual: np.ndarray) -> np.ndarray:
    covered = np.zeros_like(residual, dtype=bool)
    for mask in masks:
        covered |= mask.astype(bool)
    return residual & ~covered


def largest_unmasked_blob_px(masks: list[np.ndarray], residual: np.ndarray) -> int:
    """Area of the biggest single unmasked patch of scene change.

    The plain fraction above is a poor exhaustivity flag: the residual always carries a
    halo just outside every correct mask plus scattered canopy-shadow speckle, so "half the
    change is unmasked" is the *normal* state and flags 43% of frames. A genuinely missed
    subject is different in kind -- it is one coherent, subject-sized blob. Opening the
    leftover first removes the halo and the speckle, leaving something specific enough to
    act on.
    """
    leftover = _unexplained(masks, residual.astype(bool)).astype(np.uint8)
    if not leftover.any():
        return 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    leftover = cv2.morphologyEx(leftover, cv2.MORPH_OPEN, kernel)
    n_labels, labels = cv2.connectedComponents(leftover, connectivity=8)
    if n_labels <= 1:
        return 0
    return int(np.bincount(labels.ravel())[1:].max())


RESIDUAL_COLUMNS = ["residual_iou", "residual_recall", "residual_precision", "residual_area_px"]
