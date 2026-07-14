#!/usr/bin/env python
"""Stage 2: fit a per-video calibration from the sparse annotated frames and apply it to the
saved depth maps, producing calibrated metric-distance maps.

Consumes the artifacts of run_calibration_eval.py:
  * the results CSV (per-frame subject depth `depth_mask_mean`/`depth_centroid`, ground-truth
    `distance_gt`, and normalized subject vertical position `center_y_norm`)
  * the `--save-depth-dir` of native-resolution original depth maps (`*_orig.npy`)

For each video (= one camera location) it:
  1. gathers that video's sparse (subject_depth, distance) points (+ vertical position)
  2. fits the chosen calibrator (scripts/calibration.py) on ALL of them
  3. reports leave-one-out calibrated vs uncalibrated MAE (honest, no train-on-test)
  4. applies the fit to every saved depth map -> `*_calib.npy` (+ optional orig|calib PNG)

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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.calibration import METHODS, fit_calibration, leave_one_out
from scripts.depth_viz import colorize_depth

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-csv", type=Path, default=REPO_ROOT / "outputs" / "calibration_results.csv",
                   help="per-frame results from run_calibration_eval.py")
    p.add_argument("--depth-dir", type=Path, default=None,
                   help="dir of *_orig.npy maps from run_calibration_eval.py --save-depth-dir; "
                        "omit to only fit+report (no calibrated maps written)")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "depth_calib",
                   help="where to write *_calib.npy (and PNGs if --viz)")
    p.add_argument("--method", choices=list(METHODS), default="linear",
                   help="calibration model: scale/linear (simple) or disparity/poly/poly2d (richer)")
    p.add_argument("--degree", type=int, default=2, help="polynomial degree for --method poly")
    p.add_argument("--anchor", choices=["mask_mean", "centroid"], default="mask_mean",
                   help="which per-frame subject depth to calibrate against")
    p.add_argument("--robust", action="store_true", help="robust fit (median-ratio / Theil-Sen) for scale/linear")
    p.add_argument("--viz", action="store_true", help="also write an orig|calib colourised PNG per frame")
    p.add_argument("--limit-videos", type=int, default=None, help="only process the first N videos (start small)")
    p.add_argument("--fits-csv", type=Path, default=REPO_ROOT / "outputs" / "calibration_fits.csv",
                   help="per-video fit summary output")
    return p.parse_args()


def _f(value: str | None) -> float:
    """Parse a CSV cell to float; blank / 'None' -> NaN."""
    if value is None or value == "" or value == "None":
        return float("nan")
    return float(value)


def load_points(results_csv: Path, anchor: str) -> dict[str, list[dict]]:
    """Group processed frames by video into sparse calibration points, deduped per
    (video_name, frame_idx) — multiple annotation rows can share a frame."""
    anchor_col = "depth_mask_mean" if anchor == "mask_mean" else "depth_centroid"
    seen: set[tuple[str, int]] = set()
    groups: dict[str, list[dict]] = defaultdict(list)
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "processed":
                continue
            video_name = row["video_name"]
            frame_idx = int(row["frame_idx"])
            key = (video_name, frame_idx)
            if key in seen:
                continue
            seen.add(key)
            groups[video_name].append({
                "frame_idx": frame_idx,
                "pred": _f(row.get(anchor_col)),
                "gt": _f(row.get("distance_gt")),
                "v": _f(row.get("center_y_norm")),
            })
    return groups


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


FIT_FIELDS = ["video_name", "method", "anchor", "n_points", "params", "loo_mae_uncal", "loo_mae_cal", "improvement"]


def main() -> None:
    args = parse_args()
    groups = load_points(args.results_csv, args.anchor)
    video_names = sorted(groups)
    if args.limit_videos is not None:
        video_names = video_names[:args.limit_videos]
    print(f"loaded {sum(len(groups[v]) for v in video_names)} processed frames "
          f"across {len(video_names)} videos from {args.results_csv}")

    if args.depth_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    elif args.viz:
        print("  (note: --viz needs --depth-dir; no PNGs will be written)")

    fit_rows: list[dict] = []
    agg_uncal: list[float] = []
    agg_cal: list[float] = []

    for video_name in video_names:
        pts = groups[video_name]
        pred = np.array([p["pred"] for p in pts], dtype=np.float64)
        gt = np.array([p["gt"] for p in pts], dtype=np.float64)
        v = np.array([p["v"] for p in pts], dtype=np.float64)
        ys = v if args.method == "poly2d" else None

        try:
            cal = fit_calibration(pred, gt, args.method, ys=ys, degree=args.degree, robust=args.robust)
        except ValueError as exc:
            print(f"  !! {video_name}: cannot fit ({exc}); skipping")
            continue
        loo = leave_one_out(pred, gt, args.method, ys=ys, degree=args.degree, robust=args.robust)

        improvement = None
        if loo["loo_mae_cal"] is not None and loo["loo_mae_uncal"] is not None:
            improvement = loo["loo_mae_uncal"] - loo["loo_mae_cal"]
            agg_uncal.append(loo["loo_mae_uncal"])
            agg_cal.append(loo["loo_mae_cal"])
        fit_rows.append({
            "video_name": video_name,
            "method": args.method,
            "anchor": args.anchor,
            "n_points": loo["n"],
            "params": json.dumps(cal.params),
            "loo_mae_uncal": loo["loo_mae_uncal"],
            "loo_mae_cal": loo["loo_mae_cal"],
            "improvement": improvement,
        })

        if args.depth_dir is None:
            continue
        for p in pts:
            orig_path = args.depth_dir / f"{video_name}_frame{p['frame_idx']:06d}_orig.npy"
            if not orig_path.exists():
                print(f"  !! missing depth map {orig_path.name}; skipping that frame")
                continue
            orig = np.load(orig_path).astype(np.float32)
            calib = cal.apply(orig)  # poly2d builds its own normalized rows from the map height
            np.save(args.out_dir / f"{video_name}_frame{p['frame_idx']:06d}_calib.npy", calib)
            if args.viz:
                yi = None if ys is None else np.array([p["v"]])
                pred_cal = float(np.asarray(cal.predict(np.array([p["pred"]]), yi)).ravel()[0])
                png = render_pair(video_name, p["frame_idx"], orig, calib,
                                  p["gt"], p["pred"], pred_cal, args.method, cal.params)
                import cv2
                cv2.imwrite(str(args.out_dir / f"{video_name}_frame{p['frame_idx']:06d}.png"), png)

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
