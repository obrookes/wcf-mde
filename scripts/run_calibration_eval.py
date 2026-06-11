#!/usr/bin/env python
"""Evaluate SAM-3 segmentation + a metric-depth model against the wcf-mde calibration
ground truth in data/annotations_06052026.csv.

For every annotated row (video_name, frame_idx, frame_timestamp, distance):
  1. resolve video_name -> actual video file under data/ (via list_reference_videos.xlsx)
  2. extract that frame by sequential decode
  3. segment it with SAM-3 (text prompt, default "person holding sign") -> union mask
  4. estimate metric depth with the selected --depth-model (Pi3X or Depth Anything 3 /
     DA3NESTED, both metric-scale) -> per-pixel depth map (metres)
  5. reduce mask + depth to a single distance estimate (mean-in-mask, centroid)
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

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAM3_CHECKPOINT = Path(
    "/home/dl18206/projs/Unmarked-Anything/weights/sam3/safari_checkpoint_hf.pt"
)
PI3_PIXEL_LIMIT = 255_000  # matches Pi3's load_images_as_tensor default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations-csv", type=Path, default=REPO_ROOT / "data" / "annotations_06052026.csv")
    p.add_argument("--video-list-xlsx", type=Path, default=REPO_ROOT / "data" / "list_reference_videos.xlsx")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--sam3-checkpoint", type=Path, default=DEFAULT_SAM3_CHECKPOINT)
    p.add_argument("--sam3-prompt", type=str, default="person holding sign")
    p.add_argument("--depth-model", choices=["pi3x", "da3"], default="pi3x",
                   help="metric depth backend: Pi3X or Depth Anything 3 (DA3NESTED, metres-native)")
    p.add_argument("--pi3-model-id", type=str, default="yyfz233/Pi3X")
    p.add_argument("--da3-model-id", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    p.add_argument("--output-csv", type=Path, default=REPO_ROOT / "outputs" / "calibration_results.csv")
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


def run_pi3x_depth(model, frame_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Single-frame metric depth map (H, W) in metres, at the model's working resolution."""
    tensor = frame_to_pi3_tensor(frame_bgr).to(device)
    imgs = tensor.unsqueeze(0).unsqueeze(0)  # (B=1, N=1, 3, H, W)
    with torch.no_grad():
        res = model(imgs)
    # local_points = concat([xy * z, z]); z is the per-pixel metric depth (camera-local Z).
    depth = res["local_points"][0, 0, :, :, 2].detach().cpu().numpy().astype(np.float32)
    return depth


# --------------------------------------------------------------------------------------
# Depth Anything 3 (DA3NESTED) depth -- outputs metric depth in metres natively, no
# external camera-intrinsics calibration needed (unlike DA3METRIC, which would require
# focal lengths we don't have for these camera-trap rigs).
# --------------------------------------------------------------------------------------

def run_da3_depth(model, frame_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Single-frame metric depth map (H, W) in metres, at the model's working resolution."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        prediction = model.inference([rgb])
    # squeeze defensively: depth is documented as [N, H, W] for N input images, but a
    # stray singleton channel dim ([N, 1, H, W] or [N, H, W, 1]) would otherwise slip
    # through here and break the (H, W) contract downstream (_resize_depth_to, mask indexing).
    depth = np.asarray(prediction.depth[0], dtype=np.float32).squeeze()
    assert depth.ndim == 2, f"expected (H, W) depth map from DA3, got shape {depth.shape}"
    return depth


# --------------------------------------------------------------------------------------
# SAM-3 mask extraction (single text prompt -> union boolean mask + bbox centroid)
# --------------------------------------------------------------------------------------

def extract_union_mask(sam_result, frame_shape_hw: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int] | None, int]:
    height, width = frame_shape_hw
    union = np.zeros((height, width), dtype=bool)

    masks = getattr(sam_result, "masks", None)
    if masks is None or masks.data is None:
        return union, None, 0

    masks_data = masks.data.detach().cpu().numpy()
    if masks_data.size == 0:
        return union, None, 0

    for det_mask in masks_data > 0:
        if det_mask.shape != (height, width):
            det_mask = cv2.resize(
                det_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        union |= det_mask

    mask_pixels = int(union.sum())
    if mask_pixels == 0:
        return union, None, 0

    ys, xs = np.where(union)
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    center_xy = (int((xmin + xmax) // 2), int((ymin + ymax) // 2))
    return union, center_xy, mask_pixels


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
# --------------------------------------------------------------------------------------

DEPTH_COLORMAP = cv2.COLORMAP_TURBO


def make_depth_colorbar(height: int, d_min: float, d_max: float, bar_width: int = 30, label_width: int = 80) -> np.ndarray:
    """Vertical scale bar (max at top, min at bottom) in DEPTH_COLORMAP, labelled
    with the metric depth range (metres) it represents."""
    gradient = np.linspace(255, 0, height, dtype=np.uint8).reshape(-1, 1)
    bar = cv2.applyColorMap(np.repeat(gradient, bar_width, axis=1), DEPTH_COLORMAP)

    canvas = np.zeros((height, bar_width + label_width, 3), dtype=np.uint8)
    canvas[:, :bar_width] = bar
    for value, y in ((d_max, 15), (d_min, height - 8)):
        cv2.putText(canvas, f"{value:.2f}m", (bar_width + 4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_overlay(
    out_path: Path,
    frame_bgr: np.ndarray,
    meta: dict,
    union_mask: np.ndarray | None,
    center_xy: tuple[int, int] | None,
    depth: np.ndarray | None,
    fields: dict,
) -> None:
    """Write a frame|mask|depth panel labelled with the *script-decoded* frame index
    (`meta["frame_idx"]`, the iter_frames_at_indices loop counter) rather than a value
    re-read from the CSV row, so the image itself can confirm the correct frame was
    extracted, independent of any row/CSV bookkeeping."""
    annotated = frame_bgr.copy()
    if union_mask is not None:
        contours, _ = cv2.findContours(union_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)
    if center_xy is not None:
        cv2.drawMarker(annotated, center_xy, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    if depth is not None and np.isfinite(depth).any():
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
        f"status={fields['status']}",
        f"mask_area_px={fields['mask_area_px']}",
        f"depth_mask_mean={fields['depth_mask_mean']}",
        f"depth_centroid={fields['depth_centroid']}",
        f"distance_gt={meta['distance_gt']}",
    ]
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
    fields: dict,
    union_mask: np.ndarray | None = None,
    center_xy: tuple[int, int] | None = None,
    depth: np.ndarray | None = None,
) -> None:
    if overlay_path is None or frame_bgr is None:
        return
    try:
        save_overlay(overlay_path, frame_bgr, meta, union_mask, center_xy, depth, fields)
    except Exception as exc:  # noqa: BLE001 - overlay dumping must never break the run
        print(f"  !! failed to write overlay {overlay_path}: {exc}")


# --------------------------------------------------------------------------------------
# per-frame pipeline: decoded frame -> {status, mask_area_px, depth_mask_mean, depth_centroid}
#
# Run once per decoded frame (not once per annotation row): multiple CSV rows can reference
# the same (video_name, frame_idx), and SAM-3 / Pi3X inference is by far the most expensive
# part of the pipeline, so the result is computed here and then fanned out to every row that
# shares the frame.
# --------------------------------------------------------------------------------------

def process_frame(
    sam3,
    depth_fn,
    frame_bgr: np.ndarray | None,
    prompt: str,
    device: torch.device,
    overlay_path: Path | None = None,
    overlay_meta: dict | None = None,
) -> dict:
    if frame_bgr is None:
        result = {"status": "frame_decode_error", "mask_area_px": None,
                  "depth_mask_mean": None, "depth_centroid": None}
        _maybe_save_overlay(overlay_path, overlay_meta, frame_bgr, result)
        return result

    frame_shape_hw = frame_bgr.shape[:2]
    union_mask: np.ndarray | None = None
    center_xy: tuple[int, int] | None = None
    depth: np.ndarray | None = None

    try:
        sam_results = sam3(source=frame_bgr, text=[prompt])
        union_mask, center_xy, mask_area_px = extract_union_mask(sam_results[0], frame_shape_hw)
    except Exception as exc:  # noqa: BLE001 - record and continue
        result = {"status": f"sam_error: {exc}", "mask_area_px": None,
                  "depth_mask_mean": None, "depth_centroid": None}
        _maybe_save_overlay(overlay_path, overlay_meta, frame_bgr, result, union_mask, center_xy, depth)
        return result

    if mask_area_px == 0:
        result = {"status": "empty_mask", "mask_area_px": 0,
                  "depth_mask_mean": None, "depth_centroid": None}
        _maybe_save_overlay(overlay_path, overlay_meta, frame_bgr, result, union_mask, center_xy, depth)
        return result

    try:
        depth = depth_fn(frame_bgr)
        depth = _resize_depth_to(depth, frame_shape_hw)
        depth_mask_mean = compute_depth_mask_mean(depth, union_mask)
        depth_centroid = compute_depth_centroid(depth, center_xy)
    except Exception as exc:  # noqa: BLE001
        result = {"status": f"depth_error: {exc}", "mask_area_px": mask_area_px,
                  "depth_mask_mean": None, "depth_centroid": None}
        _maybe_save_overlay(overlay_path, overlay_meta, frame_bgr, result, union_mask, center_xy, depth)
        return result

    result = {"status": "processed", "mask_area_px": mask_area_px,
              "depth_mask_mean": depth_mask_mean, "depth_centroid": depth_centroid}
    _maybe_save_overlay(overlay_path, overlay_meta, frame_bgr, result, union_mask, center_xy, depth)
    return result


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

OUTPUT_FIELDS = [
    "video_name",
    "frame_idx",
    "frame_timestamp",
    "distance_gt",
    "depth_model",
    "status",
    "mask_area_px",
    "depth_mask_mean",
    "depth_centroid",
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

    # group row indices by resolved video path, marking unresolvable ones up front
    groups: dict[Path, list[int]] = defaultdict(list)
    results: list[dict | None] = [None] * len(rows)
    for i, row in enumerate(rows):
        try:
            video_path = resolve_video_path(row["video_name"], anno_to_path)
        except (KeyError, FileNotFoundError):
            results[i] = {
                "video_name": row["video_name"],
                "frame_idx": row["frame_idx"],
                "frame_timestamp": row["frame_timestamp"],
                "distance_gt": row["distance_gt"],
                "depth_model": args.depth_model,
                "status": "video_missing",
                "mask_area_px": None,
                "depth_mask_mean": None,
                "depth_centroid": None,
            }
            continue
        groups[video_path].append(i)

    n_missing = sum(1 for r in results if r is not None)
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
        depth_fn = lambda frame: run_pi3x_depth(depth_model, frame, device)
    else:
        print(f"loading Depth Anything 3 ({args.da3_model_id}) ...")
        from depth_anything_3.api import DepthAnything3

        depth_model = DepthAnything3.from_pretrained(args.da3_model_id).to(device).eval()
        depth_fn = lambda frame: run_da3_depth(depth_model, frame, device)

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

    n_done = 0
    for video_path, row_indices in groups.items():
        frame_indices = [rows[i]["frame_idx"] for i in row_indices]
        idx_to_row_indices: dict[int, list[int]] = defaultdict(list)
        for i in row_indices:
            idx_to_row_indices[rows[i]["frame_idx"]].append(i)

        try:
            for frame_idx, frame_bgr in iter_frames_at_indices(video_path, frame_indices):
                overlay_path = overlay_meta = None
                if args.overlay_dir is not None:
                    first_row = rows[idx_to_row_indices[frame_idx][0]]
                    overlay_meta = {
                        "frame_idx": frame_idx,  # script-decoded index, not the CSV's
                        "video_name": first_row["video_name"],
                        "distance_gt": first_row["distance_gt"],
                    }
                    overlay_path = args.overlay_dir / f"{video_path.stem}_frame{frame_idx:06d}.png"

                frame_fields = process_frame(sam3, depth_fn, frame_bgr, args.sam3_prompt, device,
                                             overlay_path=overlay_path, overlay_meta=overlay_meta)
                for i in idx_to_row_indices[frame_idx]:
                    results[i] = row_dict(i, frame_fields)

                n_done += 1
                if n_done % 25 == 0:
                    print(f"  ... {n_done} frames processed")
        except Exception as exc:  # noqa: BLE001 - record, preserve prior results, move to next video
            print(f"  !! video-level error on {video_path}: {exc}")
            error_fields = {"status": f"video_error: {exc}", "mask_area_px": None,
                            "depth_mask_mean": None, "depth_centroid": None}
            for i in row_indices:
                if results[i] is None:
                    results[i] = row_dict(i, error_fields)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
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
