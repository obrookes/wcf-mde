#!/usr/bin/env python
"""Evaluate SAM-3 segmentation + a metric-depth model against the wcf-mde calibration
ground truth in data/annotations_06052026.csv.

For every annotated row (video_name, frame_idx, frame_timestamp, distance):
  1. resolve video_name -> actual video file under data/ (via list_reference_videos.xlsx)
  2. extract all annotated frames for that video by sequential decode
  3. estimate metric depth jointly across all annotated frames in one inference call
     (both Pi3X and DA3NESTED are multi-view architectures; joint inference gives better
     metric scale than processing each frame independently)
  4. segment each frame with SAM-3 (text prompt, default "person holding sign") -> one mask per
     detected instance (some frames have >1 person holding a placard; this can't be known from
     the annotations CSV ahead of time, only discovered from what SAM-3 actually returns, so
     every frame is treated as potentially-multi-instance)
  5. reduce each instance's mask + precomputed depth to a distance estimate (mean-in-mask,
     centroid); a frame with N detected instances fans its one ground-truth distance out to N
     output rows (one per instance) rather than collapsing them into a single blended value
  6. write predicted vs ground-truth distance to an output CSV, plus a summary

Patterns for SAM-3 invocation, mask extraction, and mask->depth reduction are adapted
from Unmarked-Anything-export-distances/apps/camera_trap/cli/dap3_cli.py.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.frame_source import iter_frames_at_indices
from scripts.video_lookup import load_anno_to_path, resolve_video_path
from scripts.depth_viz import DEPTH_COLORMAP, make_depth_colorbar
from scripts.masks import save_instance_masks

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAM3_CHECKPOINT = Path(
    "/home/dl18206/projs/Unmarked-Anything/weights/sam3/safari_checkpoint_hf.pt"
)
PI3_PIXEL_LIMIT = 255_000  # matches Pi3's load_images_as_tensor default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations-csv", type=Path, default=REPO_ROOT / "data" / "annotations_06052026.csv")
    p.add_argument("--video-list", "--video-list-xlsx", type=Path, dest="video_list_xlsx",
                   default=REPO_ROOT / "data" / "list_reference_videos.xlsx")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--sam3-checkpoint", type=Path, default=DEFAULT_SAM3_CHECKPOINT)
    p.add_argument("--sam3-prompt", type=str, default="person holding sign")
    p.add_argument("--depth-model", choices=["pi3x", "da3"], default="pi3x",
                   help="metric depth backend: Pi3X or Depth Anything 3 (DA3NESTED, metres-native)")
    p.add_argument("--pi3-model-id", type=str, default="yyfz233/Pi3X")
    p.add_argument("--da3-model-id", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    p.add_argument("--results-csv", "--output-csv", type=Path, dest="output_csv",
                   default=REPO_ROOT / "outputs" / "calibration_results.csv")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--limit", type=int, default=None, help="process at most this many CSV rows (for smoke-testing)")
    p.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="dump a frame|mask|depth sanity-check PNG per processed frame here "
             "(labelled with the script-decoded frame index; requires --limit)",
    )
    p.add_argument(
        "--save-depth-dir",
        type=Path,
        default=None,
        help="persist each frame's native-resolution depth map as fp16 .npy here "
             "(keyed by video_name + frame_idx), so calibrate_depth.py can fit/apply a "
             "per-video calibration without re-running depth inference. Safe on full runs.",
    )
    p.add_argument(
        "--save-masks-dir",
        "--save-mask-dir",
        type=Path,
        dest="save_mask_dir",
        default=None,
        help="persist each frame's detected-instance SAM-3 masks as COCO RLE JSON here "
             "(keyed by video_name + frame_idx; one entry per instance, ordered to match "
             "instance_idx in --results-csv), for Stage-1 alignment (scripts/alignment.py) "
             "to reload without re-running SAM-3. Safe on full runs.",
    )
    args = p.parse_args()
    if args.overlay_dir is not None and args.limit is None:
        p.error("--overlay-dir requires --limit (it's a sanity-check aid for small smoke-test runs only)")
    return args


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# --------------------------------------------------------------------------------------
# Pi3X depth
# --------------------------------------------------------------------------------------

def frame_to_pi3_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """Replicate Pi3's load_images_as_tensor preprocessing for a single in-memory frame:
    RGB convert, resize to a multiple-of-14 size under PI3_PIXEL_LIMIT, scale to [0, 1].

    Mirrors the target-size computation in pi3.utils.basic.load_images_as_tensor exactly,
    including its shrink-to-fit refinement loop (simple independent per-axis rounding can
    overshoot PIXEL_LIMIT and land on a different grid than Pi3's own preprocessing)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    w_orig, h_orig = pil.size
    scale = math.sqrt(PI3_PIXEL_LIMIT / (w_orig * h_orig)) if w_orig * h_orig > 0 else 1.0
    w_target, h_target = w_orig * scale, h_orig * scale
    k, m = round(w_target / 14), round(h_target / 14)
    while (k * 14) * (m * 14) > PI3_PIXEL_LIMIT:
        if k / m > w_target / h_target:
            k -= 1
        else:
            m -= 1
    target_w, target_h = max(1, k) * 14, max(1, m) * 14
    resized = pil.resize((target_w, target_h))
    return transforms.ToTensor()(resized)


def run_pi3x_depth_batch(model, frames_bgr: list[np.ndarray], device: torch.device) -> list[np.ndarray]:
    """Joint metric depth inference over N frames from the same video.

    Pi3X is a multi-view architecture: odd decoder blocks apply cross-frame attention and
    metric scale is estimated from relative camera motion — both require N > 1 to be effective.
    All frames are resized to the same (target_h, target_w) grid derived from the first frame
    so they can be stacked into a single (1, N, 3, H, W) tensor.
    """
    # derive target size from first frame; apply to all so the stack is shape-consistent
    ref_tensor = frame_to_pi3_tensor(frames_bgr[0])
    _, h, w = ref_tensor.shape
    tensors = [ref_tensor]
    for f in frames_bgr[1:]:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((w, h))
        tensors.append(transforms.ToTensor()(pil))
    imgs = torch.stack(tensors).unsqueeze(0).to(device)  # (1, N, 3, H, W)
    with torch.no_grad():
        res = model(imgs)
    # local_points[..., 2] is per-pixel metric Z-depth; shape (1, N, H, W)
    depths = res["local_points"][0, :, :, :, 2].detach().cpu().numpy().astype(np.float32)
    return [depths[i] for i in range(depths.shape[0])]


# --------------------------------------------------------------------------------------
# Depth Anything 3 (DA3NESTED) depth -- outputs metric depth in metres natively, no
# external camera-intrinsics calibration needed (unlike DA3METRIC, which would require
# focal lengths we don't have for these camera-trap rigs).
# --------------------------------------------------------------------------------------

def run_da3_depth_batch(model, frames_bgr: list[np.ndarray], device: torch.device) -> list[np.ndarray]:
    """Joint metric depth inference over N frames from the same video.

    DA3NESTED applies cross-view attention and reference-view selection when N > 2, giving
    better metric scale than single-frame inference. Frames should be in temporal order.
    Returns one (H, W) float32 depth map per input frame, in metres.
    """
    rgbs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    with torch.no_grad():
        prediction = model.inference(rgbs)
    # prediction.depth: (N, H, W) — one map per input frame
    depths = np.asarray(prediction.depth, dtype=np.float32)
    assert depths.ndim == 3, f"expected (N, H, W) from DA3 batch inference, got {depths.shape}"
    return [depths[i].squeeze() for i in range(depths.shape[0])]


# --------------------------------------------------------------------------------------
# SAM-3 mask extraction: one boolean mask + bbox centroid per detected instance.
#
# Multi-person frames (>1 placard-holder) can't be flagged ahead of time from the
# annotations CSV -- there's no metadata column for it, only whatever SAM-3 actually detects
# in a given frame. So every frame goes through this same per-instance path regardless of
# whether it's "known" to have one subject or several; the union-mask short-cut this used to
# take silently blended multiple subjects' depths into one meaningless average.
# --------------------------------------------------------------------------------------

def extract_instance_masks(sam_result, frame_shape_hw: tuple[int, int]) -> list[dict]:
    """Return one {"mask", "center_xy", "area_px"} dict per SAM-3 detection, sorted
    left-to-right by center_x for a stable, human-checkable instance order. Empty list if
    SAM-3 found nothing."""
    height, width = frame_shape_hw

    masks = getattr(sam_result, "masks", None)
    if masks is None or masks.data is None:
        return []

    masks_data = masks.data.detach().cpu().numpy()
    if masks_data.size == 0:
        return []

    instances: list[dict] = []
    for det_mask in masks_data > 0:
        if det_mask.shape != (height, width):
            det_mask = cv2.resize(
                det_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        area_px = int(det_mask.sum())
        if area_px == 0:
            continue
        ys, xs = np.where(det_mask)
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        center_xy = (int((xmin + xmax) // 2), int((ymin + ymax) // 2))
        instances.append({"mask": det_mask, "center_xy": center_xy, "area_px": area_px})

    instances.sort(key=lambda inst: inst["center_xy"][0])
    return instances


def union_mask_from_instances(instances: list[dict], frame_shape_hw: tuple[int, int]) -> np.ndarray:
    """Only for overlay visualization -- draws all instances' outlines as one contour set."""
    height, width = frame_shape_hw
    union = np.zeros((height, width), dtype=bool)
    for inst in instances:
        union |= inst["mask"]
    return union


# --------------------------------------------------------------------------------------
# mask + depth -> single distance values (mirrors compute_depth_mask_mean from sibling repo)
# --------------------------------------------------------------------------------------

def _resize_depth_to(depth: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if depth.shape == shape_hw:
        return depth
    return cv2.resize(depth, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_CUBIC)


def compute_depth_mask_mean(depth: np.ndarray, mask: np.ndarray) -> float | None:
    """`depth` must already be resized to `mask`'s resolution (see _resize_depth_to)."""
    values = depth[mask]
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def compute_depth_centroid(depth: np.ndarray, center_xy: tuple[int, int] | None) -> float | None:
    """`depth` must already be resized to the resolution `center_xy` was computed in."""
    if center_xy is None:
        return None
    cx, cy = center_xy
    val = float(depth[cy, cx])
    return val if math.isfinite(val) else None


# --------------------------------------------------------------------------------------
# overlay dumps: frame | mask outline + centroid | depth colormap, for visually
# sanity-checking frame extraction, segmentation, and depth on a small (--limit'ed) run.
# DEPTH_COLORMAP / make_depth_colorbar live in scripts/depth_viz.py (shared with
# calibrate_depth.py).
# --------------------------------------------------------------------------------------


def save_overlay(
    out_path: Path,
    frame_bgr: np.ndarray,
    meta: dict,
    instances: list[dict] | None,
    depth: np.ndarray | None,
    fields_by_instance: list[dict],
) -> None:
    """Write a frame|mask|depth panel labelled with the *script-decoded* frame index
    (`meta["frame_idx"]`, the iter_frames_at_indices loop counter) rather than a value
    re-read from the CSV row, so the image itself can confirm the correct frame was
    extracted, independent of any row/CSV bookkeeping. Draws every detected instance's
    outline + centroid (labelled with its instance index) so multi-person frames are visibly
    distinguishable from single-person ones, not just implied by the printed fields."""
    annotated = frame_bgr.copy()
    if instances:
        union_mask = union_mask_from_instances(instances, frame_bgr.shape[:2])
        contours, _ = cv2.findContours(union_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)
        for idx, inst in enumerate(instances):
            cv2.drawMarker(annotated, inst["center_xy"], (0, 0, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.putText(annotated, str(idx), (inst["center_xy"][0] + 8, inst["center_xy"][1] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    if depth is not None and np.isfinite(depth).any():
        # depth may be at the model's native grid (e.g. Pi3X 672x378) while the frame is
        # 720x404; resize to the frame so the side-by-side panels share a row count
        # (cv2.hconcat requires it). NEAREST avoids interpolating across NaN/invalid pixels.
        if depth.shape[:2] != frame_bgr.shape[:2]:
            depth = cv2.resize(depth, (frame_bgr.shape[1], frame_bgr.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        finite = depth[np.isfinite(depth)]
        d_min, d_max = float(finite.min()), float(finite.max())
        norm = np.clip((depth - d_min) / max(d_max - d_min, 1e-6), 0.0, 1.0)
        depth_vis = cv2.applyColorMap((norm * 255).astype(np.uint8), DEPTH_COLORMAP)
        depth_panel = cv2.hconcat([depth_vis, make_depth_colorbar(depth_vis.shape[0], d_min, d_max)])
    else:
        depth_panel = np.zeros_like(frame_bgr)

    panel = cv2.hconcat([annotated, depth_panel])

    lines = [
        f"video={meta['video_name']}",
        f"frame_idx (script-decoded)={meta['frame_idx']}",
        f"distance_gt={meta['distance_gt']}  ({len(fields_by_instance)} instance(s))",
    ]
    for idx, fields in enumerate(fields_by_instance):
        lines.append(
            f"  [{idx}] status={fields['status']} mask_area_px={fields['mask_area_px']} "
            f"depth_mask_mean={fields['depth_mask_mean']} depth_centroid={fields['depth_centroid']}"
        )
    for n, line in enumerate(lines):
        org = (10, 25 + 22 * n)
        cv2.putText(panel, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def _maybe_save_overlay(
    overlay_path: Path | None,
    meta: dict | None,
    frame_bgr: np.ndarray | None,
    instances: list[dict] | None,
    depth: np.ndarray | None,
    fields_by_instance: list[dict],
) -> None:
    if overlay_path is None or frame_bgr is None:
        return
    try:
        save_overlay(overlay_path, frame_bgr, meta, instances, depth, fields_by_instance)
    except Exception as exc:  # noqa: BLE001 - overlay dumping must never break the run
        print(f"  !! failed to write overlay {overlay_path}: {exc}")


# --------------------------------------------------------------------------------------
# per-frame pipeline: decoded frame -> list of {instance_idx, status, mask_area_px,
# depth_mask_mean, depth_centroid} dicts, one per detected instance (>=1 for a processed
# frame with detections, exactly 1 with instance_idx=None for a frame-level failure/no-detection
# so every annotation row still gets at least one output row).
#
# Run once per decoded frame (not once per annotation row): multiple CSV rows can reference
# the same (video_name, frame_idx), and SAM-3 / Pi3X inference is by far the most expensive
# part of the pipeline, so the result is computed here and then fanned out to every row that
# shares the frame -- and, when a frame has N detected instances, each of those rows is further
# fanned out to N output rows, one per instance, all sharing that row's ground-truth distance.
# --------------------------------------------------------------------------------------

def save_depth_maps(out_dir: Path, video_name: str, depth_by_idx: dict) -> None:
    """Persist each frame's native-resolution depth map as fp16 .npy, keyed by the flat
    annotation name (`video_name`) so files don't collide across camera folders that reuse
    filenames like DSCF0005 (unlike save_overlay, which keys on video_path.stem). These are
    reloaded by calibrate_depth.py to fit/apply a per-video calibration without re-inferring."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, depth in depth_by_idx.items():
        if depth is None:
            continue
        np.save(out_dir / f"{video_name}_frame{frame_idx:06d}_orig.npy", depth.astype(np.float16))


def _center_fields(center_xy: tuple[int, int] | None,
                   frame_shape_hw: tuple[int, int] | None = None) -> dict:
    """Expose the subject (mask-bbox) centroid as flat CSV columns. calibrate_depth.py's
    poly2d calibrator uses `center_y_norm` (vertical position normalized to [0, 1]) to fit
    ground-plane geometry; normalizing here keeps it resolution-independent so it lines up
    with the depth map's own rows at apply time, regardless of either resolution."""
    if center_xy is None:
        return {"center_x": None, "center_y": None, "center_y_norm": None}
    cx, cy = center_xy
    cy_norm = (cy + 0.5) / frame_shape_hw[0] if frame_shape_hw and frame_shape_hw[0] else None
    return {"center_x": cx, "center_y": cy, "center_y_norm": cy_norm}


def process_frame(
    sam3,
    frame_bgr: np.ndarray | None,
    depth: np.ndarray | None,
    prompt: str,
    device: torch.device,
    overlay_path: Path | None = None,
    overlay_meta: dict | None = None,
    mask_dir: Path | None = None,
    mask_video_name: str | None = None,
    mask_frame_idx: int | None = None,
) -> list[dict]:
    if frame_bgr is None:
        results = [{"instance_idx": None, "status": "frame_decode_error", "mask_area_px": None,
                    "depth_mask_mean": None, "depth_centroid": None, **_center_fields(None)}]
        _maybe_save_overlay(overlay_path, overlay_meta, frame_bgr, None, None, results)
        return results

    frame_shape_hw = frame_bgr.shape[:2]

    try:
        sam_results = sam3(source=frame_bgr, text=[prompt])
        instances = extract_instance_masks(sam_results[0], frame_shape_hw)
    except Exception as exc:  # noqa: BLE001 - record and continue
        results = [{"instance_idx": None, "status": f"sam_error: {exc}", "mask_area_px": None,
                    "depth_mask_mean": None, "depth_centroid": None,
                    **_center_fields(None, frame_shape_hw)}]
        _maybe_save_overlay(overlay_path, overlay_meta, frame_bgr, None, depth, results)
        return results

    if mask_dir is not None and instances:
        save_instance_masks(mask_dir, mask_video_name, mask_frame_idx, instances)

    if not instances:
        results = [{"instance_idx": None, "status": "empty_mask", "mask_area_px": 0,
                    "depth_mask_mean": None, "depth_centroid": None,
                    **_center_fields(None, frame_shape_hw)}]
        _maybe_save_overlay(overlay_path, overlay_meta, frame_bgr, instances, depth, results)
        return results

    if depth is None:
        results = [
            {"instance_idx": idx, "status": "depth_error: no depth map (inference failed for this video)",
             "mask_area_px": inst["area_px"], "depth_mask_mean": None, "depth_centroid": None,
             **_center_fields(inst["center_xy"], frame_shape_hw)}
            for idx, inst in enumerate(instances)
        ]
        _maybe_save_overlay(overlay_path, overlay_meta, frame_bgr, instances, depth, results)
        return results

    depth_resized = _resize_depth_to(depth, frame_shape_hw)
    results = []
    for idx, inst in enumerate(instances):
        try:
            depth_mask_mean = compute_depth_mask_mean(depth_resized, inst["mask"])
            depth_centroid = compute_depth_centroid(depth_resized, inst["center_xy"])
            status = "processed"
        except Exception as exc:  # noqa: BLE001 - one bad instance shouldn't sink the others
            depth_mask_mean = depth_centroid = None
            status = f"depth_error: {exc}"
        results.append({
            "instance_idx": idx, "status": status, "mask_area_px": inst["area_px"],
            "depth_mask_mean": depth_mask_mean, "depth_centroid": depth_centroid,
            **_center_fields(inst["center_xy"], frame_shape_hw),
        })
    _maybe_save_overlay(overlay_path, overlay_meta, frame_bgr, instances, depth_resized, results)
    return results


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

OUTPUT_FIELDS = [
    "video_name",
    "frame_idx",
    "frame_timestamp",
    "distance_gt",
    "depth_model",
    "instance_idx",
    "status",
    "mask_area_px",
    "depth_mask_mean",
    "depth_centroid",
    "center_x",
    "center_y",
    "center_y_norm",
]


def load_rows(csv_path: Path, limit: int | None) -> list[dict]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]
    for r in rows:
        r["frame_idx"] = int(r["frame_idx"])
        r["frame_timestamp"] = float(r["frame_timestamp"])
        r["distance_gt"] = float(r["distance"])
    return rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    use_half = device.type == "cuda"
    print(f"device={device}")

    rows = load_rows(args.annotations_csv, args.limit)
    print(f"loaded {len(rows)} annotation rows")

    anno_to_path = load_anno_to_path(args.video_list_xlsx, args.data_dir)

    # group row indices by resolved video path, marking unresolvable ones up front.
    # `results` is appended to, not indexed by input row: a single annotation row can now
    # fan out to N output rows (one per SAM-3 instance detected in that frame), so `filled`
    # tracks which input rows already got at least one output row (used by the video-level
    # error handler below to avoid double-emitting for rows processed before a mid-video
    # exception).
    groups: dict[Path, list[int]] = defaultdict(list)
    results: list[dict] = []
    filled: set[int] = set()
    for i, row in enumerate(rows):
        try:
            video_path = resolve_video_path(row["video_name"], anno_to_path)
        except (KeyError, FileNotFoundError):
            results.append({
                "video_name": row["video_name"],
                "frame_idx": row["frame_idx"],
                "frame_timestamp": row["frame_timestamp"],
                "distance_gt": row["distance_gt"],
                "depth_model": args.depth_model,
                "instance_idx": None,
                "status": "video_missing",
                "mask_area_px": None,
                "depth_mask_mean": None,
                "depth_centroid": None,
            })
            filled.add(i)
            continue
        groups[video_path].append(i)

    n_missing = len(filled)
    print(f"resolved {len(groups)} distinct videos; {n_missing} rows reference videos not present on disk")

    print(f"loading SAM-3 ({args.sam3_checkpoint}) with prompt {args.sam3_prompt!r} ...")
    from ultralytics.models.sam import SAM3SemanticPredictor

    sam3_overrides = dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        model=str(args.sam3_checkpoint),
        half=use_half,
        save=False,
        verbose=False,
        device=str(device),
    )
    sam3 = SAM3SemanticPredictor(overrides=sam3_overrides)

    if args.depth_model == "pi3x":
        print(f"loading Pi3X ({args.pi3_model_id}) ...")
        from pi3.models.pi3x import Pi3X

        depth_model = Pi3X.from_pretrained(args.pi3_model_id).to(device).eval()
        depth_fn = lambda frames: run_pi3x_depth_batch(depth_model, frames, device)
    else:
        print(f"loading Depth Anything 3 ({args.da3_model_id}) ...")
        from depth_anything_3.api import DepthAnything3

        depth_model = DepthAnything3.from_pretrained(args.da3_model_id).to(device).eval()
        depth_fn = lambda frames: run_da3_depth_batch(depth_model, frames, device)

    def row_dict(i: int, fields: dict) -> dict:
        row = rows[i]
        return {
            "video_name": row["video_name"],
            "frame_idx": row["frame_idx"],
            "frame_timestamp": row["frame_timestamp"],
            "distance_gt": row["distance_gt"],
            "depth_model": args.depth_model,
            **fields,
        }

    # The output CSV is opened here -- after the models have loaded, before the expensive loop --
    # and flushed once per video rather than written in one go at the end. Masks and depth maps
    # are already persisted incrementally, so without this a job killed at its wall clock (this
    # pipeline runs under SLURM) would leave every expensive artifact on disk but lose the
    # `status` column, which is the source of truth for which frames produced a mask, forcing a
    # full re-run. Opening it *after* model loading means a failure to load weights leaves any
    # previous run's CSV intact instead of truncating it on startup. Same final file either way.
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_file = open(args.output_csv, "w", newline="")
    writer = csv.DictWriter(out_file, fieldnames=OUTPUT_FIELDS)
    writer.writeheader()
    for result in results:  # the video_missing rows gathered while grouping, above
        writer.writerow(result)
    out_file.flush()

    def emit(result: dict) -> None:
        """Record a result row both in memory (for print_summary) and on disk (for crash safety)."""
        results.append(result)
        writer.writerow(result)

    n_done = 0
    for video_path, row_indices in groups.items():
        frame_indices = [rows[i]["frame_idx"] for i in row_indices]
        idx_to_row_indices: dict[int, list[int]] = defaultdict(list)
        for i in row_indices:
            idx_to_row_indices[rows[i]["frame_idx"]].append(i)

        try:
            # decode all annotated frames for this video (max 16 in this dataset)
            frames_buffer = list(iter_frames_at_indices(video_path, frame_indices))

            # single joint depth inference over all frames — both Pi3X and DA3NESTED are
            # multi-view architectures that give better metric scale with N > 1.
            # Only feed frames that actually decoded: a frame index past the video's end
            # yields None, and one None frame would make the whole batch throw (cvtColor on
            # an empty array), sinking depth for every frame of the video. Run inference on
            # the decoded frames only and map results back; None-decoded indices keep depth
            # None and are reported per-frame as frame_decode_error.
            depth_by_idx: dict[int, np.ndarray | None] = {fid: None for fid, _ in frames_buffer}
            decoded = [(fid, bgr) for fid, bgr in frames_buffer if bgr is not None]
            if decoded:
                try:
                    decoded_depths = depth_fn([bgr for _, bgr in decoded])
                except Exception as exc:  # noqa: BLE001 - depth failure shouldn't abort the video
                    print(f"  !! depth inference failed for {video_path}: {exc}")
                    decoded_depths = [None] * len(decoded)
                for (fid, _), d in zip(decoded, decoded_depths):
                    depth_by_idx[fid] = d

            if args.save_depth_dir is not None:
                save_depth_maps(args.save_depth_dir, rows[row_indices[0]]["video_name"], depth_by_idx)

            for frame_idx, frame_bgr in frames_buffer:
                overlay_path = overlay_meta = None
                if args.overlay_dir is not None:
                    first_row = rows[idx_to_row_indices[frame_idx][0]]
                    overlay_meta = {
                        "frame_idx": frame_idx,  # script-decoded index, not the CSV's
                        "video_name": first_row["video_name"],
                        "distance_gt": first_row["distance_gt"],
                    }
                    overlay_path = args.overlay_dir / f"{video_path.stem}_frame{frame_idx:06d}.png"

                video_name = rows[idx_to_row_indices[frame_idx][0]]["video_name"]
                frame_fields_list = process_frame(
                    sam3, frame_bgr, depth_by_idx.get(frame_idx),
                    args.sam3_prompt, device,
                    overlay_path=overlay_path, overlay_meta=overlay_meta,
                    mask_dir=args.save_mask_dir, mask_video_name=video_name, mask_frame_idx=frame_idx,
                )
                for i in idx_to_row_indices[frame_idx]:
                    for fields in frame_fields_list:
                        emit(row_dict(i, fields))
                    filled.add(i)

                n_done += 1
                if n_done % 25 == 0:
                    print(f"  ... {n_done} frames processed")
        except Exception as exc:  # noqa: BLE001 - record, preserve prior results, move to next video
            print(f"  !! video-level error on {video_path}: {exc}")
            error_fields = {"instance_idx": None, "status": f"video_error: {exc}", "mask_area_px": None,
                            "depth_mask_mean": None, "depth_centroid": None}
            for i in row_indices:
                if i not in filled:
                    emit(row_dict(i, error_fields))
                    filled.add(i)
        finally:
            # one flush per video, not per row: bounds the loss from a kill to the video in
            # flight without paying an fsync-ish cost on every instance
            out_file.flush()

    out_file.close()
    print(f"wrote {len(results)} rows to {args.output_csv}")

    print_summary(results)


def print_summary(results: list[dict]) -> None:
    status_counts: dict[str, int] = defaultdict(int)
    for r in results:
        status = r["status"]
        key = status.split(":", 1)[0] if status.startswith(("sam_error", "depth_error")) else status
        status_counts[key] += 1

    print("\n--- status counts ---")
    for status, count in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {status}: {count}")

    pairs = [
        (r["depth_mask_mean"], r["distance_gt"])
        for r in results
        if r["status"] == "processed" and r["depth_mask_mean"] is not None
    ]
    if pairs:
        pred = np.array([p[0] for p in pairs])
        gt = np.array([p[1] for p in pairs])
        err = pred - gt
        print(f"\n--- depth_mask_mean vs ground truth (n={len(pairs)}) ---")
        print(f"  MAE:  {np.mean(np.abs(err)):.3f} m")
        print(f"  RMSE: {np.sqrt(np.mean(err ** 2)):.3f} m")
        print(f"  bias (pred - gt mean): {np.mean(err):+.3f} m")
        print(f"  pred range: [{pred.min():.2f}, {pred.max():.2f}] m   gt range: [{gt.min():.2f}, {gt.max():.2f}] m")
    else:
        print("\nno rows with both a predicted depth and ground-truth distance to compare")


if __name__ == "__main__":
    main()
