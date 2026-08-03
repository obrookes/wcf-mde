#!/usr/bin/env python3
"""Score every predicted SAM-3 mask in an export directory and rank a human review queue.

This is the CPU tier of mask triage: it answers "which of these masks are wrong?" using
only the frames and COCO-RLE masks already on disk -- no model forward pass, no GPU, no
ground truth, no new annotation. Three tables come out:

    <out>/mask_scores.csv     one row per (video_name, frame_idx, instance_idx), every
                              signal its own column, plus the flags it tripped, a fused
                              triage_score and a bucket
    <out>/frame_scores.csv    one row per frame, carrying the **exhaustivity** signal --
                              whether a subject-sized piece of the scene changed without
                              any mask covering it. Deliberately a separate table: "is
                              this mask right?" and "is anything missing?" are different
                              questions, and fusing them lets a frame full of immaculate
                              masks hide a subject nobody segmented
    <out>/review_queue.csv    mask_scores.csv restricted to non-accepted masks, ranked

Work is grouped by **station** (camera-reference folder, see scripts/stations.py) because
the banner geometry and the area prior are properties of one fixed camera deployment. The
background model is finer-grained still -- per clip -- for the reason given in
scripts/background.py.

    python scripts/score_masks.py --export-dir /scratch/.../export_test --out-dir outputs/triage

IMPORTANT -- the bucket thresholds below are *starting points*, not calibrated ones.
Turning `auto_accept` into a claim about precision needs a labelled gold set; until that
exists, treat the bucket as a sort order and the flag reasons as the product.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.background import (
    RESIDUAL_COLUMNS,
    build_background,
    foreground_residual,
    largest_unmasked_blob_px,
    photometric_mode,
    residual_signals,
    unexplained_residual_frac,
)
from scripts.banner import DEFAULT_BANNER_ROWS, banner_top_or_default, crop_banner
from scripts.mask_signals import SIGNAL_COLUMNS, mask_signals
from scripts.masks import decode_rle
from scripts.stations import parse_frame_name, parse_mask_name, site_of, station_of

# --- Flag thresholds -------------------------------------------------------------------
# Hard flags are degenerate masks: nothing an annotator can salvage by nudging a boundary.
MIN_AREA_PX = 64
MAX_AREA_FRAC = 0.60
MAX_BANNER_OVERLAP_FRAC = 0.50
# Soft flags are "look at this": a specific, nameable thing looks wrong. Cutoffs are set
# from the observed distribution over this corpus so each one fires on a tail rather than
# on the bulk -- a flag that fires on 80% of masks sorts nothing.
MAX_SECOND_COMPONENT_FRAC = 0.25
MAX_HOLE_AREA_FRAC = 0.10
MIN_COMPACTNESS = 0.05
MIN_EROSION_SURVIVAL = 0.30
MAX_BORDER_CONTACT_FRAC = 0.15
MIN_BOUNDARY_GRADIENT_RATIO = 0.80
# Residual *recall*, not IoU: the question is whether any of the mask is backed by scene
# change, and a person half-matching the leaf litter legitimately scores ~0.4. Only
# near-zero support is evidence of a mask sitting on something that never moved.
MIN_RESIDUAL_RECALL = 0.10
MAX_AREA_LOG_Z = 2.5
# Frame-level exhaustivity: a coherent unmasked blob at least this large, and at least
# this share of the frame's own median mask area, means something subject-sized was missed.
MIN_UNMASKED_BLOB_PX = 600
UNMASKED_BLOB_AREA_RATIO = 0.40

# Weights for the fused score. Roughly ordered by how directly the signal implicates the
# mask rather than the scene; re-fit these against a gold set before trusting the number.
SOFT_FLAG_WEIGHTS = {
    "no_residual_support": 1.0,
    "fragmented": 0.9,
    "weak_boundary": 0.8,
    "holes": 0.6,
    "sliver": 0.6,
    "thin": 0.5,
    "truncated": 0.4,
    "anomalous_area": 0.4,
}


def _flag_mask(row: dict) -> tuple[list[str], list[str]]:
    """Named hard and soft flags for one mask. Names are user-facing -- they are what an
    annotator reads to understand why the mask surfaced."""
    hard, soft = [], []

    if row["area_px"] < MIN_AREA_PX:
        hard.append("tiny")
    if row["area_frac"] > MAX_AREA_FRAC:
        hard.append("engulfs_frame")
    if row["banner_overlap_frac"] > MAX_BANNER_OVERLAP_FRAC:
        hard.append("is_banner")

    if row["second_component_frac"] > MAX_SECOND_COMPONENT_FRAC:
        soft.append("fragmented")
    if row["hole_area_frac"] > MAX_HOLE_AREA_FRAC:
        soft.append("holes")
    if row["compactness"] < MIN_COMPACTNESS:
        soft.append("sliver")
    if row["erosion_survival"] < MIN_EROSION_SURVIVAL:
        soft.append("thin")
    if row["border_contact_frac"] > MAX_BORDER_CONTACT_FRAC:
        soft.append("truncated")
    if row["boundary_gradient_ratio"] < MIN_BOUNDARY_GRADIENT_RATIO:
        soft.append("weak_boundary")
    residual_recall = row.get("residual_recall")
    if residual_recall is not None and residual_recall < MIN_RESIDUAL_RECALL:
        soft.append("no_residual_support")
    area_log_z = row.get("area_log_z")
    if area_log_z is not None and abs(area_log_z) > MAX_AREA_LOG_Z:
        soft.append("anomalous_area")

    return hard, soft


def _bucket_and_score(hard: list[str], soft: list[str]) -> tuple[str, float]:
    if hard:
        return "reject", 10.0 + len(hard)
    score = sum(SOFT_FLAG_WEIGHTS.get(name, 0.5) for name in soft)
    return ("needs_review" if soft else "auto_accept"), score


def process_station(station: str, entries: list[tuple[str, int]], export_dir: Path) -> tuple[list[dict], list[dict]]:
    """Score every mask belonging to one static camera deployment.

    A station has to be held in memory together, because banner detection is a statistic
    *over* the station rather than a property of one frame. Only the greyscale channel is
    kept: the colour frame is needed once, to read the photometric mode, and is released
    immediately. Holding colour for the largest station (426 frames) alongside its float32
    derivatives is what OOM-killed workers in the first full run.

    Parallelism is across stations, so OpenCV's own thread pool is pinned to one thread per
    worker; left alone it opens a pool per core (288 on this node) inside every process and
    the run dies on thread exhaustion.
    """
    cv2.setNumThreads(1)
    frames_dir, masks_dir = export_dir / "frames", export_dir / "masks"

    loaded: list[tuple[str, int, np.ndarray]] = []   # (video_name, frame_idx, uint8 gray)
    modes: list[str] = []
    orphan_masks: list[tuple[str, int]] = []
    for video_name, frame_idx in entries:
        frame_path = frames_dir / f"{video_name}_frame{frame_idx:06d}.png"
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR) if frame_path.exists() else None
        if image is None:
            orphan_masks.append((video_name, frame_idx))
            continue
        # The mode is read before the banner row is known, so a conservative fixed crop is
        # used here; the banner is ~5% of the frame and unsaturated, and the day/IR call is
        # nowhere near close enough for a couple of rows to change it.
        modes.append(photometric_mode(image[:max(1, image.shape[0] - DEFAULT_BANNER_ROWS)]))
        loaded.append((video_name, frame_idx, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))

    mask_rows: list[dict] = []
    frame_rows: list[dict] = []

    # Masks whose frame was dropped by the QC-filtered export can't be scored: every
    # signal needs the pixels. Record them rather than silently losing them.
    for video_name, frame_idx in orphan_masks:
        mask_rows.append({
            "video_name": video_name, "frame_idx": frame_idx, "instance_idx": None,
            "station": station, "site": site_of(video_name), "photometric_mode": None,
            "status": "frame_missing", "bucket": "needs_review", "triage_score": 1.0,
            "flags": "frame_missing",
        })

    if not loaded:
        return mask_rows, frame_rows

    height = loaded[0][2].shape[0]
    banner_top = banner_top_or_default(np.stack([gray for _, _, gray in loaded]), height)

    # Background scope is the **clip**, not the station: a station's clips come from repeat
    # visits months apart, and a median across them is a smear (see scripts/background.py).
    # Illumination mode joins the key because daylight, mono-IR and magenta-IR frames have
    # unrelated pixel statistics.
    by_clip_mode: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, (video_name, _, _) in enumerate(loaded):
        by_clip_mode[(video_name, modes[i])].append(i)

    grays = [crop_banner(gray, banner_top).astype(np.float32) for _, _, gray in loaded]
    # A clip too short for a trustworthy median gets no background and therefore no
    # residual signal -- reported as null, never as a plausible-looking bad number.
    backgrounds: dict[tuple[str, str], np.ndarray | None] = {
        key: build_background(np.stack([grays[i] for i in indices]))
        for key, indices in by_clip_mode.items()
    }

    # Per-station log-area prior. Areas span orders of magnitude with distance, so the
    # prior is fitted in log space; a robust (median/MAD) centre and spread stop the very
    # outliers we are hunting from inflating their own denominator.
    all_areas: list[float] = []
    per_frame_instances: list[list[dict]] = []
    for video_name, frame_idx, _ in loaded:
        mask_file = masks_dir / f"{video_name}_frame{frame_idx:06d}_masks.json"
        instances = []
        if mask_file.exists():
            with open(mask_file) as f:
                instances = json.load(f)
        per_frame_instances.append(instances)
        all_areas.extend(inst["area_px"] for inst in instances if inst["area_px"] > 0)

    if len(all_areas) >= 5:
        log_areas = np.log(np.asarray(all_areas, dtype=np.float64))
        area_centre = float(np.median(log_areas))
        area_spread = 1.4826 * float(np.median(np.abs(log_areas - area_centre)))
    else:
        area_centre = area_spread = None

    for i, (video_name, frame_idx, _gray) in enumerate(loaded):
        mode = modes[i]
        gray = grays[i]
        background = backgrounds.get((video_name, mode))
        residual = foreground_residual(gray, background) if background is not None else None
        background_scope = "clip" if background is not None else "none"

        cropped_masks: list[np.ndarray] = []
        instances = per_frame_instances[i]
        for inst in instances:
            full_mask = decode_rle(inst["rle"])
            full_area = int(full_mask.sum())
            banner_area = int(full_mask[banner_top:].sum())
            mask = crop_banner(full_mask, banner_top)
            cropped_masks.append(mask)

            row = {
                "video_name": video_name,
                "frame_idx": frame_idx,
                "instance_idx": inst["instance_idx"],
                "station": station,
                "site": site_of(video_name),
                "photometric_mode": mode,
                "status": "scored",
                "background_scope": background_scope,
                "banner_top": banner_top,
                "center_x": inst["center_xy"][0],
                "center_y": inst["center_xy"][1],
            }
            row.update(mask_signals(
                mask, gray,
                banner_overlap_frac=(banner_area / full_area) if full_area else 0.0,
            ))

            if residual is not None:
                row.update(residual_signals(mask, residual))
            else:
                row.update({column: None for column in RESIDUAL_COLUMNS})

            if area_centre is not None and area_spread and row["area_px"] > 0:
                row["area_log_z"] = (np.log(row["area_px"]) - area_centre) / area_spread
            else:
                row["area_log_z"] = None

            hard, soft = _flag_mask(row)
            bucket, score = _bucket_and_score(hard, soft)
            row["flags"] = ";".join(hard + soft)
            row["bucket"] = bucket
            row["triage_score"] = score
            mask_rows.append(row)

        if residual is not None:
            unexplained = unexplained_residual_frac(cropped_masks, residual)
            biggest_gap = largest_unmasked_blob_px(cropped_masks, residual)
            # Sized against this frame's own masks: "something subject-sized was missed"
            # travels between a close-up and a distant subject, an absolute pixel count
            # does not. The floor stops a frame of tiny masks flagging on canopy speckle.
            reference_area = float(np.median([m.sum() for m in cropped_masks])) if cropped_masks else 0.0
            exhaustivity_flag = biggest_gap >= max(
                MIN_UNMASKED_BLOB_PX, UNMASKED_BLOB_AREA_RATIO * reference_area
            )
        else:
            unexplained = biggest_gap = None
            exhaustivity_flag = False

        frame_rows.append({
            "video_name": video_name,
            "frame_idx": frame_idx,
            "station": station,
            "site": site_of(video_name),
            "photometric_mode": mode,
            "banner_top": banner_top,
            "n_instances": len(instances),
            "background_scope": background_scope,
            "residual_area_px": int(residual.sum()) if residual is not None else None,
            "unexplained_residual_frac": unexplained,
            "largest_unmasked_blob_px": biggest_gap,
            "exhaustivity_flag": bool(exhaustivity_flag),
        })

    return mask_rows, frame_rows


def collect_entries(export_dir: Path) -> dict[str, list[tuple[str, int]]]:
    """Every (video_name, frame_idx) that has a frame or a mask, grouped by station.

    The union rather than the intersection: frames without masks are the empty-mask cases
    that matter for exhaustivity, and masks without frames are the QC-filtered export's
    leftovers, which must be reported rather than dropped.
    """
    entries: set[tuple[str, int]] = set()
    for name in (export_dir / "frames").iterdir():
        parsed = parse_frame_name(name.name)
        if parsed:
            entries.add(parsed)
    masks_dir = export_dir / "masks"
    if masks_dir.exists():
        for name in masks_dir.iterdir():
            parsed = parse_mask_name(name.name)
            if parsed:
                entries.add(parsed)

    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for video_name, frame_idx in sorted(entries):
        grouped[station_of(video_name)].append((video_name, frame_idx))
    return grouped


MASK_COLUMNS = (
    ["video_name", "frame_idx", "instance_idx", "station", "site", "photometric_mode",
     "status", "background_scope", "banner_top", "center_x", "center_y"]
    + SIGNAL_COLUMNS + RESIDUAL_COLUMNS
    + ["area_log_z", "flags", "bucket", "triage_score"]
)
FRAME_COLUMNS = [
    "video_name", "frame_idx", "station", "site", "photometric_mode", "banner_top",
    "n_instances", "background_scope", "residual_area_px", "unexplained_residual_frac",
    "largest_unmasked_blob_px",
    "exhaustivity_flag",
]


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export-dir", type=Path, required=True,
                   help="directory containing frames/ and masks/ (e.g. .../export_test)")
    p.add_argument("--out-dir", type=Path, required=True, help="where the score tables are written")
    p.add_argument("--workers", type=int, default=8, help="stations scored in parallel (0 = serial)")
    p.add_argument("--limit-stations", type=int, default=None,
                   help="score only the first N stations (smoke-testing)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    grouped = collect_entries(args.export_dir)
    stations = sorted(grouped)
    if args.limit_stations:
        stations = stations[: args.limit_stations]
    print(f"{len(stations)} stations, {sum(len(grouped[s]) for s in stations)} (video, frame) pairs")

    mask_rows: list[dict] = []
    frame_rows: list[dict] = []

    def record(result):
        masks, frames = result
        mask_rows.extend(masks)
        frame_rows.extend(frames)

    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_station, station, grouped[station], args.export_dir): station
                for station in stations
            }
            for n_done, future in enumerate(as_completed(futures), 1):
                try:
                    record(future.result())
                except Exception as exc:  # noqa: BLE001 - one bad station shouldn't sink the run
                    print(f"  !! station {futures[future]} failed: {exc}")
                if n_done % 50 == 0:
                    print(f"  ... {n_done}/{len(stations)} stations")
    else:
        for n_done, station in enumerate(stations, 1):
            record(process_station(station, grouped[station], args.export_dir))
            if n_done % 50 == 0:
                print(f"  ... {n_done}/{len(stations)} stations")

    mask_rows.sort(key=lambda r: (r["video_name"], r["frame_idx"], r.get("instance_idx") or -1))
    frame_rows.sort(key=lambda r: (r["video_name"], r["frame_idx"]))

    write_csv(args.out_dir / "mask_scores.csv", mask_rows, MASK_COLUMNS)
    write_csv(args.out_dir / "frame_scores.csv", frame_rows, FRAME_COLUMNS)

    queue = sorted(
        (r for r in mask_rows if r.get("bucket") != "auto_accept"),
        key=lambda r: -r["triage_score"],
    )
    write_csv(args.out_dir / "review_queue.csv", queue, MASK_COLUMNS)

    buckets: dict[str, int] = defaultdict(int)
    for row in mask_rows:
        buckets[row.get("bucket", "?")] += 1
    print(f"\n{len(mask_rows)} masks scored over {len(frame_rows)} frames")
    for bucket, count in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {bucket:<14} {count:6d}  ({100 * count / max(len(mask_rows), 1):5.1f}%)")
    print(f"review queue: {len(queue)} masks -> {args.out_dir / 'review_queue.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
