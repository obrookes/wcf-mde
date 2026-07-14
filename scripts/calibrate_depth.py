#!/usr/bin/env python
"""Stage 2: fit a per-video calibration from the sparse annotated frames and apply it to the
saved depth maps, producing calibrated metric-distance maps.

Consumes the artifacts of run_calibration_eval.py:
  * the results CSV (per-frame subject depth `depth_mask_mean`/`depth_centroid`, ground-truth
    `distance_gt`, and normalized subject vertical position `center_y_norm`)
  * the `--save-depth-dir` of native-resolution original depth maps (`*_orig.npy`)

For each video (one clip, e.g. a single DSCF0005.AVI) it:
  1. gathers that video's sparse (subject_depth, distance) points (+ vertical position)
  2. fits the chosen calibrator (scripts/calibration.py) on ALL of them
  3. reports leave-one-out calibrated vs uncalibrated MAE (honest, no train-on-test)
  4. applies the fit to every saved depth map -> `--out-dir/*_calib.npy` (+ optional orig|calib
     PNG under `--viz-dir`, kept separate so the .npy dir stays free of sanity-check images)

Calibration is cheap CPU work, so you can re-run this with different --method / --degree on the
same saved maps without re-running depth inference. With no --depth-dir it still fits and
reports LOO MAE from the CSV alone (handy for quickly comparing methods).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.calibration import METHODS, fit_calibration, leave_one_out
from scripts.depth_viz import colorize_depth
from scripts.video_lookup import load_anno_to_path
from scripts.alignment import (
    REFERENCE_METHODS,
    align_frame_to_reference,
    aligned_subject_value,
    pick_reference_frame,
)
from scripts.masks import load_instance_masks

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-csv", type=Path, default=REPO_ROOT / "outputs" / "calibration_results.csv",
                   help="per-frame results from run_calibration_eval.py")
    p.add_argument("--depth-dir", type=Path, default=None,
                   help="dir of *_orig.npy maps from run_calibration_eval.py --save-depth-dir; "
                        "omit to only fit+report (no calibrated maps written)")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "depth_calib",
                   help="where to write *_calib.npy")
    p.add_argument("--method", choices=list(METHODS), default="linear",
                   help="calibration model: scale/linear (simple) or disparity/poly/poly2d (richer)")
    p.add_argument("--degree", type=int, default=2, help="polynomial degree for --method poly")
    p.add_argument("--anchor", choices=["mask_mean", "centroid"], default="mask_mean",
                   help="which per-frame subject depth to calibrate against")
    p.add_argument("--robust", action="store_true", help="robust fit (median-ratio / Theil-Sen) for scale/linear")
    p.add_argument("--viz", action="store_true", help="also write an orig|calib colourised PNG per frame")
    p.add_argument("--viz-dir", type=Path, default=REPO_ROOT / "outputs" / "depth_calib_viz",
                   help="where to write orig|calib PNGs when --viz is set (kept separate from "
                        "--out-dir so the .npy dir stays free of sanity-check images)")
    p.add_argument("--limit-videos", type=int, default=None, help="only process the first N videos (start small)")
    p.add_argument("--fits-csv", type=Path, default=REPO_ROOT / "outputs" / "calibration_fits.csv",
                   help="per-video fit summary output")
    p.add_argument("--calib-level", choices=["clip", "cam"], default="clip",
                   help="fit one calibrator per clip (today's default, e.g. one DSCF0005.AVI) "
                        "or pool all clips from the same camera-reference folder into one fit "
                        "(e.g. 16_vid_ref_Cam_184, which typically holds several clips) -- "
                        "'cam' requires --video-list-xlsx/--data-dir to resolve clip->camera")
    p.add_argument("--video-list-xlsx", type=Path, default=REPO_ROOT / "data" / "list_reference_videos.xlsx",
                   help="only needed for --calib-level cam, to resolve video_name -> camera folder")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                   help="only needed for --calib-level cam, to resolve video_name -> camera folder")
    p.add_argument("--align", choices=["none", "ransac"], default="none",
                   help="optional Stage-1 cross-frame background alignment (scripts/alignment.py) "
                        "before Stage-2 calibration; default 'none' since wcf-mde's depth "
                        "inference is already joint/multi-frame per video and it isn't known "
                        "ahead of time whether this extra step still helps here -- compare "
                        "--align none vs --align ransac runs' calibration_fits.csv to find out")
    p.add_argument("--ref-frame-method", choices=list(REFERENCE_METHODS), default="furthest",
                   help="how Stage-1 picks its alignment anchor within each group: 'furthest' "
                        "(matches timmh/Markham-25) or 'median' (gentler for this dataset's "
                        "smaller per-camera point counts). Only used when --align != none")
    p.add_argument("--mask-dir", type=Path, default=None,
                   help="dir of *_masks.json from run_calibration_eval.py --save-mask-dir; "
                        "required when --align != none (Stage-1 needs per-instance masks to "
                        "isolate background pixels)")
    p.add_argument("--align-min-points", type=int, default=4,
                   help="skip Stage-1 alignment for a group with fewer than this many points "
                        "and fall back to unaligned pred -- a handful of points is more likely "
                        "to be hurt than helped by fitting an extra affine on top")
    args = p.parse_args()
    if args.align != "none" and args.mask_dir is None:
        p.error("--align requires --mask-dir (per-instance masks saved by "
                "run_calibration_eval.py --save-mask-dir)")
    return args


def _f(value: str | None) -> float:
    """Parse a CSV cell to float; blank / 'None' -> NaN."""
    if value is None or value == "" or value == "None":
        return float("nan")
    return float(value)


def build_camera_key_fn(video_list_xlsx: Path, data_dir: Path) -> Callable[[str], str]:
    """Resolve a flattened `video_name` (e.g. beauvois_T_16_16_vid_ref_Cam_184_DSCF0005) to its
    camera-reference folder (e.g. beauvois/T_16/16_vid_ref_Cam_184) via list_reference_videos.xlsx.

    Deliberately resolves through the real path rather than string-splitting `video_name`:
    camera folder names (e.g. "16_vid_ref_Cam_184") already contain underscores, so there's no
    unambiguous way to recover the folder boundary from the flattened name alone."""
    anno_to_path = load_anno_to_path(video_list_xlsx, data_dir)

    def camera_key(video_name: str) -> str:
        path = anno_to_path[video_name]  # KeyError surfaces unresolvable video_names loudly
        return str(path.parent.relative_to(data_dir))

    return camera_key


def load_points(
    results_csv: Path,
    anchor: str,
    group_key_fn: Callable[[str], str] | None = None,
) -> dict[str, list[dict]]:
    """Group processed frames into sparse calibration points, keyed by `group_key_fn(video_name)`
    (default: identity, i.e. one group per clip -- pass build_camera_key_fn(...)'s result for
    one group per camera location instead).

    Some frames have more than one person holding a reference placard -- SAM-3 then reports
    multiple instances for that frame (run_calibration_eval.py's `instance_idx` column), each
    with its own subject depth but sharing that frame's single ground-truth `distance_gt`
    (multiple people standing at the same recorded distance). Each instance is therefore its
    own independent (pred, gt) calibration point; only true duplicate rows -- the same
    (video_name, frame_idx, instance_idx) appearing twice, e.g. from multiple annotation rows
    referencing one frame -- are deduped."""
    if group_key_fn is None:
        group_key_fn = lambda video_name: video_name  # noqa: E731 - trivial default
    anchor_col = "depth_mask_mean" if anchor == "mask_mean" else "depth_centroid"
    seen: set[tuple[str, int, int | None]] = set()
    groups: dict[str, list[dict]] = defaultdict(list)
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "processed":
                continue
            video_name = row["video_name"]
            frame_idx = int(row["frame_idx"])
            raw_instance_idx = row.get("instance_idx")
            instance_idx = int(raw_instance_idx) if raw_instance_idx not in (None, "", "None") else None
            key = (video_name, frame_idx, instance_idx)
            if key in seen:
                continue
            seen.add(key)
            groups[group_key_fn(video_name)].append({
                "video_name": video_name,
                "frame_idx": frame_idx,
                "instance_idx": instance_idx,
                "pred": _f(row.get(anchor_col)),
                "gt": _f(row.get("distance_gt")),
                "v": _f(row.get("center_y_norm")),
            })
    return groups


def _resize_depth(depth: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    import cv2  # local import keeps module import light when --align is off
    if depth.shape == shape_hw:
        return depth
    return cv2.resize(depth, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_CUBIC)


def _resize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    import cv2  # local import keeps module import light when --align is off
    if mask.shape == shape_hw:
        return mask
    return cv2.resize(mask.astype(np.uint8), (shape_hw[1], shape_hw[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)


def compute_aligned_preds(
    pts: list[dict],
    depth_dir: Path,
    mask_dir: Path,
    align_method: str,
    ref_frame_method: str,
) -> tuple[np.ndarray, dict] | None:
    """Stage 1: replace each point's CSV-derived `pred` with a Stage-1-aligned subject value,
    anchored to one reference frame within this group (scripts/alignment.py). Returns
    (aligned_pred_array, ref_point) in the same order as `pts`, or None if the reference frame's
    own depth map / mask can't be loaded at all -- callers should fall back to the unaligned pred
    array in that case rather than fail the whole group over one missing artifact."""
    ref_point = pick_reference_frame(pts, method=ref_frame_method)

    depth_cache: dict[tuple[str, int], np.ndarray | None] = {}
    mask_cache: dict[tuple[str, int], list[dict] | None] = {}

    def _load_depth(video_name: str, frame_idx: int) -> np.ndarray | None:
        key = (video_name, frame_idx)
        if key not in depth_cache:
            path = depth_dir / f"{video_name}_frame{frame_idx:06d}_orig.npy"
            depth_cache[key] = np.load(path).astype(np.float32) if path.exists() else None
        return depth_cache[key]

    def _load_masks(video_name: str, frame_idx: int) -> list[dict] | None:
        key = (video_name, frame_idx)
        if key not in mask_cache:
            try:
                mask_cache[key] = load_instance_masks(mask_dir, video_name, frame_idx)
            except FileNotFoundError:
                mask_cache[key] = None
        return mask_cache[key]

    def _instance_mask(video_name: str, frame_idx: int, instance_idx: int | None) -> np.ndarray | None:
        instances = _load_masks(video_name, frame_idx)
        if not instances:
            return None
        if instance_idx is None:
            return instances[0]["mask"]
        for inst in instances:
            if inst["instance_idx"] == instance_idx:
                return inst["mask"]
        return None

    ref_depth = _load_depth(ref_point["video_name"], ref_point["frame_idx"])
    ref_mask = _instance_mask(ref_point["video_name"], ref_point["frame_idx"], ref_point["instance_idx"])
    if ref_depth is None or ref_mask is None:
        return None
    ref_depth = _resize_depth(ref_depth, ref_mask.shape)

    aligned: list[float] = []
    for p in pts:
        if p is ref_point:
            aligned.append(aligned_subject_value(ref_depth, ref_mask, 1.0, 0.0))
            continue
        depth = _load_depth(p["video_name"], p["frame_idx"])
        mask = _instance_mask(p["video_name"], p["frame_idx"], p["instance_idx"])
        if depth is None or mask is None:
            aligned.append(p["pred"])  # fall back to this point's own CSV pred
            continue
        # resize this frame's depth+mask onto the reference's grid (mirrors timmh, which
        # resizes every calibration frame onto the farthest-frame's fixed shape) so background
        # pixels line up 1:1 for the alignment fit.
        depth = _resize_depth(depth, ref_mask.shape)
        mask = _resize_mask(mask, ref_mask.shape)
        a, b = align_frame_to_reference(depth, mask, ref_depth, ref_mask, method=align_method)
        aligned.append(aligned_subject_value(depth, mask, a, b))

    return np.array(aligned, dtype=np.float64), ref_point


def render_pair(video_name: str, frame_idx: int, orig: np.ndarray, calib: np.ndarray,
                gt: float, pred_uncal: float, pred_cal: float, method: str, params: dict) -> np.ndarray:
    """Side-by-side original-depth | calibrated-distance panel, labelled with the subject
    readings (uncalibrated, calibrated, ground truth)."""
    import cv2  # local import keeps module import light when --viz is off

    panel = cv2.hconcat([colorize_depth(orig), colorize_depth(calib)])
    pstr = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in params.items() if k != "coeffs")
    lines = [
        f"{video_name}  frame {frame_idx}",
        f"left=orig depth   right=calibrated ({method} {pstr})".strip(),
        f"subject: uncal={pred_uncal:.2f}m  cal={pred_cal:.2f}m  gt={gt:.2f}m",
    ]
    for n, line in enumerate(lines):
        org = (10, 25 + 22 * n)
        cv2.putText(panel, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


FIT_FIELDS = ["calib_level", "group_key", "method", "anchor", "n_points", "params",
              "loo_mae_uncal", "loo_mae_cal", "improvement",
              "align_method", "align_applied", "align_ref_video", "align_ref_frame_idx",
              "align_ref_instance_idx"]


def main() -> None:
    args = parse_args()
    group_key_fn = None
    if args.calib_level == "cam":
        group_key_fn = build_camera_key_fn(args.video_list_xlsx, args.data_dir)
    groups = load_points(args.results_csv, args.anchor, group_key_fn=group_key_fn)
    group_keys = sorted(groups)
    if args.limit_videos is not None:
        group_keys = group_keys[:args.limit_videos]
    print(f"loaded {sum(len(groups[k]) for k in group_keys)} processed frames/instances "
          f"across {len(group_keys)} {'cameras' if args.calib_level == 'cam' else 'videos'} "
          f"from {args.results_csv} (--calib-level {args.calib_level})")

    if args.depth_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if args.viz:
            args.viz_dir.mkdir(parents=True, exist_ok=True)
    elif args.viz:
        print("  (note: --viz needs --depth-dir; no PNGs will be written)")

    fit_rows: list[dict] = []
    agg_uncal: list[float] = []
    agg_cal: list[float] = []

    for group_key in group_keys:
        pts = groups[group_key]
        gt = np.array([p["gt"] for p in pts], dtype=np.float64)
        v = np.array([p["v"] for p in pts], dtype=np.float64)
        ys = v if args.method == "poly2d" else None

        # Stage 1 (optional): replace the CSV-derived pred with a Stage-1-aligned subject
        # value anchored to one reference frame in this group. --align defaults to "none"
        # because it isn't known ahead of time whether this helps here (wcf-mde's depth
        # inference is already joint/multi-frame per video, unlike timmh/Markham-25's
        # frame-independent DPT) -- compare calibration_fits.csv's align_* columns across
        # --align none vs --align ransac runs to find out empirically.
        align_ref = None
        if args.align != "none" and len(pts) >= args.align_min_points:
            result = compute_aligned_preds(pts, args.depth_dir, args.mask_dir,
                                           args.align, args.ref_frame_method)
            if result is None:
                print(f"  !! {group_key}: alignment reference frame unreadable; "
                      f"falling back to unaligned pred")
                pred = np.array([p["pred"] for p in pts], dtype=np.float64)
            else:
                pred, align_ref = result
        else:
            pred = np.array([p["pred"] for p in pts], dtype=np.float64)

        try:
            cal = fit_calibration(pred, gt, args.method, ys=ys, degree=args.degree, robust=args.robust)
        except ValueError as exc:
            print(f"  !! {group_key}: cannot fit ({exc}); skipping")
            continue
        # NOTE: `pts` holds one point per *instance*, not per frame -- both because multi-
        # instance frames (>1 placard-holder) contribute >1 point sharing one frame_idx/gt,
        # and because --calib-level cam pools points across several clips into one group. LOO
        # here therefore leaves out one instance at a time, which is a slightly optimistic
        # estimate of held-out accuracy versus truly independent frames (a left-in sibling
        # instance from the same held-out frame still informs the fit). Fine for comparing
        # methods, but don't read loo_mae_cal as if every point were an independent observation.
        loo = leave_one_out(pred, gt, args.method, ys=ys, degree=args.degree, robust=args.robust)

        improvement = None
        if loo["loo_mae_cal"] is not None and loo["loo_mae_uncal"] is not None:
            improvement = loo["loo_mae_uncal"] - loo["loo_mae_cal"]
            agg_uncal.append(loo["loo_mae_uncal"])
            agg_cal.append(loo["loo_mae_cal"])
        fit_rows.append({
            "calib_level": args.calib_level,
            "group_key": group_key,
            "method": args.method,
            "anchor": args.anchor,
            "n_points": loo["n"],
            "params": json.dumps(cal.params),
            "loo_mae_uncal": loo["loo_mae_uncal"],
            "loo_mae_cal": loo["loo_mae_cal"],
            "improvement": improvement,
            "align_method": args.align,
            "align_applied": align_ref is not None,
            "align_ref_video": align_ref["video_name"] if align_ref else None,
            "align_ref_frame_idx": align_ref["frame_idx"] if align_ref else None,
            "align_ref_instance_idx": align_ref["instance_idx"] if align_ref else None,
        })

        if args.depth_dir is None:
            continue
        # the depth map (and hence the calibrated map) is per-frame, not per-instance -- a
        # frame with N detected instances shares one *_orig.npy / *_calib.npy. Cache per
        # (video_name, frame_idx) -- not frame_idx alone -- so N instance points don't
        # redundantly reload/reapply/rewrite the same map, and so --calib-level cam (which
        # pools points from several clips, whose frame_idx values are NOT globally unique)
        # doesn't conflate frame 5 of one clip with frame 5 of another in the same group.
        frame_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        for p, p_pred in zip(pts, pred):
            video_name, frame_idx = p["video_name"], p["frame_idx"]
            cache_key = (video_name, frame_idx)
            if cache_key in frame_cache:
                orig, calib = frame_cache[cache_key]
            else:
                orig_path = args.depth_dir / f"{video_name}_frame{frame_idx:06d}_orig.npy"
                if not orig_path.exists():
                    print(f"  !! missing depth map {orig_path.name}; skipping that frame")
                    continue
                orig = np.load(orig_path).astype(np.float32)
                calib = cal.apply(orig)  # poly2d builds its own normalized rows from the map height
                np.save(args.out_dir / f"{video_name}_frame{frame_idx:06d}_calib.npy", calib)
                frame_cache[cache_key] = (orig, calib)
            if args.viz:
                yi = None if ys is None else np.array([p["v"]])
                # use p_pred (the value `cal` was actually fit on -- aligned when --align is on,
                # the raw CSV pred otherwise), not p["pred"], so the displayed "uncal" reading
                # and cal.predict(...) stay consistent with what was actually fit
                pred_cal = float(np.asarray(cal.predict(np.array([p_pred]), yi)).ravel()[0])
                png = render_pair(video_name, frame_idx, orig, calib,
                                  p["gt"], p_pred, pred_cal, args.method, cal.params)
                import cv2
                # instance suffix keeps multi-instance frames' viz PNGs from colliding
                inst_suffix = f"_inst{p['instance_idx']}" if p["instance_idx"] is not None else ""
                cv2.imwrite(str(args.viz_dir / f"{video_name}_frame{frame_idx:06d}{inst_suffix}.png"), png)

    args.fits_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.fits_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIT_FIELDS)
        writer.writeheader()
        writer.writerows(fit_rows)
    print(f"wrote {len(fit_rows)} per-video fits to {args.fits_csv}")

    print_summary(args.method, agg_uncal, agg_cal, fit_rows)


def print_summary(method: str, agg_uncal: list[float], agg_cal: list[float], fit_rows: list[dict]) -> None:
    n_scored = len(agg_cal)
    print(f"\n--- leave-one-out summary ({method}) ---")
    print(f"  videos with >= enough frames for LOO: {n_scored} / {len(fit_rows)}")
    if n_scored:
        mu, mc = float(np.mean(agg_uncal)), float(np.mean(agg_cal))
        improved = sum(1 for u, c in zip(agg_uncal, agg_cal) if c < u)
        print(f"  mean uncalibrated LOO MAE: {mu:.3f} m")
        print(f"  mean calibrated   LOO MAE: {mc:.3f} m")
        print(f"  mean improvement:          {mu - mc:+.3f} m  ({improved}/{n_scored} videos improved)")
    else:
        print("  (no video had enough annotated frames to leave one out for this method)")


if __name__ == "__main__":
    main()
