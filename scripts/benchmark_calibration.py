#!/usr/bin/env python
"""Benchmark-only sweep across calibration methods/permutations -- never writes depth maps.

`calibrate_depth.py` fits ONE method/permutation per run; comparing several (e.g. `linear` vs
`poly2d` vs `piecewise`, with/without `--align ransac`, `--robust` on/off, several `--degree`
values) means re-running that CLI once per combination and diffing output by hand. This script
instead sweeps the whole grid in one process and writes a single comparison CSV, reusing exactly
the same leave-one-out (LOO) metric `calibrate_depth.py` reports per group: for each annotated
point, refit the calibrator on every *other* point in its group and predict the held-out one, so
the reported accuracy reflects a frame the fit never saw (not a memorized one). Averaging that
held-out error across a group's points gives an honest per-group LOO MAE, compared against the
uncalibrated baseline (raw model depth vs. ground truth, no fit at all).

There is deliberately no --out-dir/--viz/--depth-dir-as-output-flag here: this script has no way
to write a *_calib.npy, by construction, so it's safe to sweep broadly before committing to a
--method for the real (npy-producing) calibrate_depth.py run.

--depth-dir/--mask-dir are read-only inputs here (source of *_orig.npy / *_masks.json), needed
only when --aligns includes "ransac" (Stage-1 alignment re-reads them per group).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.calibration import METHODS, fit_calibration, leave_one_out
from scripts.calibrate_depth import (
    DISTANCE_BUCKETS,
    build_camera_key_fn,
    compute_aligned_preds,
    load_points,
    load_qc_exclusions,
)
from scripts.alignment import REFERENCE_METHODS

REPO_ROOT = Path(__file__).resolve().parent.parent

# only these methods actually use --robust / --degree; everything else gets one placeholder pass
_ROBUST_METHODS = ("scale", "linear")
_DEGREE_METHODS = ("poly",)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-csv", type=Path, default=REPO_ROOT / "outputs" / "calibration_results.csv")
    p.add_argument("--qc-flags", type=Path, default=None,
                   help="optional scripts/qc_annotations.py flags CSV; same semantics as "
                        "calibrate_depth.py --qc-flags")
    p.add_argument("--calib-level", choices=["clip", "cam"], default="clip")
    p.add_argument("--video-list-xlsx", type=Path, default=REPO_ROOT / "data" / "list_reference_videos.xlsx",
                   help="only needed for --calib-level cam")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                   help="only needed for --calib-level cam")
    p.add_argument("--depth-dir", type=Path, default=None,
                   help="read-only dir of *_orig.npy from run_calibration_eval.py --save-depth-dir; "
                        "only needed when --aligns includes ransac (never written to)")
    p.add_argument("--mask-dir", type=Path, default=None,
                   help="read-only dir of *_masks.json from run_calibration_eval.py --save-mask-dir; "
                        "required when --aligns includes ransac")
    p.add_argument("--align-min-points", type=int, default=4,
                   help="skip Stage-1 alignment for a group with fewer than this many points "
                        "(same semantics as calibrate_depth.py)")
    p.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS),
                   help="calibration methods to sweep (default: all)")
    p.add_argument("--degrees", nargs="+", type=int, default=[1, 2, 3],
                   help="polynomial degrees to sweep; only applies to --methods poly")
    p.add_argument("--anchors", nargs="+", choices=["mask_mean", "centroid"],
                   default=["mask_mean", "centroid"], help="subject-depth anchors to sweep")
    p.add_argument("--aligns", nargs="+", choices=["none", "ransac"], default=["none"],
                   help="Stage-1 alignment options to sweep; 'ransac' requires --depth-dir/--mask-dir")
    p.add_argument("--ref-frame-methods", nargs="+", choices=list(REFERENCE_METHODS), default=["furthest"],
                   help="Stage-1 reference-frame picks to sweep; only used when --aligns includes ransac")
    p.add_argument("--limit-videos", type=int, default=None,
                   help="only consider the first N videos/cameras (start small)")
    p.add_argument("--out-csv", type=Path, default=REPO_ROOT / "outputs" / "calibration_benchmark.csv",
                   help="per-combination metrics CSV (metrics only -- no depth data)")
    p.add_argument("--sort-by", choices=["improvement", "mae_cal"], default="improvement",
                   help="ranking key for the printed top-N table")
    p.add_argument("--top-n", type=int, default=15, help="how many ranked rows to print to console")
    args = p.parse_args()
    if "ransac" in args.aligns and (args.depth_dir is None or args.mask_dir is None):
        p.error("--aligns ransac requires --depth-dir and --mask-dir (per-instance masks and "
                "original depth maps saved by run_calibration_eval.py)")
    return args


def _bucket_maes(gt: list[float], uncal_res: list[float], cal_res: list[float]) -> dict[str, tuple[float | None, float | None, int]]:
    """Per-distance-bucket (uncal_mae, cal_mae, n), pooled across LOO'd points -- mirrors
    calibrate_depth.py's print_distance_summary but returns values instead of printing."""
    out: dict[str, tuple[float | None, float | None, int]] = {}
    if not gt:
        for _, _, label in DISTANCE_BUCKETS:
            out[label] = (None, None, 0)
        return out
    gt_arr = np.asarray(gt, dtype=np.float64)
    uncal_arr = np.asarray(uncal_res, dtype=np.float64)
    cal_arr = np.asarray(cal_res, dtype=np.float64)
    for lo, hi, label in DISTANCE_BUCKETS:
        mask = (gt_arr >= lo) & (gt_arr < hi)
        n = int(mask.sum())
        if n == 0:
            out[label] = (None, None, 0)
        else:
            out[label] = (float(uncal_arr[mask].mean()), float(cal_arr[mask].mean()), n)
    return out


def build_group_arrays(
    results_csv: Path,
    anchor: str,
    align: str,
    ref_frame_method: str | None,
    group_key_fn,
    exclude: set[tuple[str, int]] | None,
    depth_dir: Path | None,
    mask_dir: Path | None,
    align_min_points: int,
    limit_videos: int | None,
) -> dict[str, dict]:
    """The expensive, method/degree/robust-independent step: load this (anchor, align,
    ref_frame_method) combination's per-group (pred, gt, v) arrays, applying Stage-1 alignment
    once per group when align == 'ransac'. Cache and reuse across the cheap inner sweep."""
    groups = load_points(results_csv, anchor, group_key_fn=group_key_fn, exclude=exclude)
    group_keys = sorted(groups)
    if limit_videos is not None:
        group_keys = group_keys[:limit_videos]

    out: dict[str, dict] = {}
    for group_key in group_keys:
        pts = groups[group_key]
        gt = np.array([p["gt"] for p in pts], dtype=np.float64)
        v = np.array([p["v"] for p in pts], dtype=np.float64)
        align_ref = None
        if align != "none" and len(pts) >= align_min_points:
            result = compute_aligned_preds(pts, depth_dir, mask_dir, align, ref_frame_method)
            if result is None:
                pred = np.array([p["pred"] for p in pts], dtype=np.float64)
            else:
                pred, align_ref = result
        else:
            pred = np.array([p["pred"] for p in pts], dtype=np.float64)
        out[group_key] = {"pred": pred, "gt": gt, "v": v, "align_ref": align_ref, "n_pts": len(pts)}
    return out


def evaluate_combination(
    group_arrays: dict[str, dict],
    method: str,
    degree: int | None,
    robust: bool | None,
) -> dict:
    """Cheap inner-sweep step: fit_calibration + leave_one_out per group, reusing cached arrays."""
    agg_uncal: list[float] = []
    agg_cal: list[float] = []
    dist_gt: list[float] = []
    dist_uncal: list[float] = []
    dist_cal: list[float] = []
    n_groups_total = len(group_arrays)

    for arrays in group_arrays.values():
        pred, gt, v = arrays["pred"], arrays["gt"], arrays["v"]
        ys = v if method == "poly2d" else None
        try:
            loo = leave_one_out(pred, gt, method, ys=ys, degree=degree or 2, robust=bool(robust))
        except ValueError:
            continue
        if loo["loo_mae_cal"] is not None and loo["loo_mae_uncal"] is not None:
            agg_uncal.append(loo["loo_mae_uncal"])
            agg_cal.append(loo["loo_mae_cal"])
            dist_gt.extend(loo["loo_gt"])
            dist_uncal.extend(loo["loo_residuals_uncal"])
            dist_cal.extend(loo["loo_residuals"])

    n_scored = len(agg_cal)
    mean_uncal = float(np.mean(agg_uncal)) if n_scored else None
    mean_cal = float(np.mean(agg_cal)) if n_scored else None
    mean_improvement = (mean_uncal - mean_cal) if n_scored else None
    pct_improved = (100.0 * sum(1 for u, c in zip(agg_uncal, agg_cal) if c < u) / n_scored) if n_scored else None
    buckets = _bucket_maes(dist_gt, dist_uncal, dist_cal)

    row = {
        "method": method,
        "degree": degree if degree is not None else "",
        "robust": robust if robust is not None else "",
        "n_groups_total": n_groups_total,
        "n_groups_scored": n_scored,
        "mean_loo_mae_uncal": mean_uncal,
        "mean_loo_mae_cal": mean_cal,
        "mean_improvement": mean_improvement,
        "pct_groups_improved": pct_improved,
    }
    for _, _, label in DISTANCE_BUCKETS:
        uncal_mae, cal_mae, n = buckets[label]
        key = label.split()[0]  # "close 1-4m" -> "close"
        row[f"{key}_n"] = n
        row[f"{key}_mae_uncal"] = uncal_mae
        row[f"{key}_mae_cal"] = cal_mae
    return row


def main() -> None:
    args = parse_args()
    group_key_fn = None
    if args.calib_level == "cam":
        group_key_fn = build_camera_key_fn(args.video_list_xlsx, args.data_dir)
    exclude = None
    if args.qc_flags is not None:
        exclude = load_qc_exclusions(args.qc_flags)
        print(f"loaded {len(exclude)} QC-flagged (video_name, frame_idx) exclusions from {args.qc_flags}")

    rows: list[dict] = []
    for anchor in args.anchors:
        for align in args.aligns:
            ref_methods = args.ref_frame_methods if align == "ransac" else [None]
            for ref_frame_method in ref_methods:
                print(f"loading points: anchor={anchor} align={align} ref_frame_method={ref_frame_method} ...")
                group_arrays = build_group_arrays(
                    args.results_csv, anchor, align, ref_frame_method, group_key_fn, exclude,
                    args.depth_dir, args.mask_dir, args.align_min_points, args.limit_videos,
                )
                for method in args.methods:
                    degrees = args.degrees if method in _DEGREE_METHODS else [None]
                    robusts = [False, True] if method in _ROBUST_METHODS else [None]
                    for degree, robust in itertools.product(degrees, robusts):
                        row = evaluate_combination(group_arrays, method, degree, robust)
                        row.update({
                            "anchor": anchor,
                            "align": align,
                            "ref_frame_method": ref_frame_method or "",
                            "calib_level": args.calib_level,
                        })
                        rows.append(row)

    fieldnames = [
        "method", "degree", "robust", "anchor", "align", "ref_frame_method", "calib_level",
        "n_groups_total", "n_groups_scored", "mean_loo_mae_uncal", "mean_loo_mae_cal",
        "mean_improvement", "pct_groups_improved",
    ]
    for _, _, label in DISTANCE_BUCKETS:
        key = label.split()[0]
        fieldnames += [f"{key}_n", f"{key}_mae_uncal", f"{key}_mae_cal"]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} method/permutation combinations to {args.out_csv}")

    sort_key = (lambda r: (r["mean_improvement"] is None, -(r["mean_improvement"] or 0.0))) \
        if args.sort_by == "improvement" \
        else (lambda r: (r["mean_loo_mae_cal"] is None, r["mean_loo_mae_cal"] or float("inf")))
    ranked = sorted(rows, key=sort_key)

    print(f"\n--- top {min(args.top_n, len(ranked))} combinations (sorted by {args.sort_by}) ---")
    header = f"{'method':<10} {'deg':<4} {'rob':<6} {'anchor':<10} {'align':<7} {'ref':<9} {'lvl':<5} {'n':<4} {'uncal':>7} {'cal':>7} {'impr':>7} {'%imp':>6}"
    print(header)
    for row in ranked[:args.top_n]:
        uncal = f"{row['mean_loo_mae_uncal']:.3f}" if row["mean_loo_mae_uncal"] is not None else "n/a"
        cal = f"{row['mean_loo_mae_cal']:.3f}" if row["mean_loo_mae_cal"] is not None else "n/a"
        impr = f"{row['mean_improvement']:+.3f}" if row["mean_improvement"] is not None else "n/a"
        pct = f"{row['pct_groups_improved']:.0f}%" if row["pct_groups_improved"] is not None else "n/a"
        print(f"{row['method']:<10} {str(row['degree']):<4} {str(row['robust']):<6} {row['anchor']:<10} "
              f"{row['align']:<7} {row['ref_frame_method']:<9} {row['calib_level']:<5} "
              f"{row['n_groups_scored']:<4} {uncal:>7} {cal:>7} {impr:>7} {pct:>6}")


if __name__ == "__main__":
    main()
