#!/usr/bin/env python
"""Stage 2 of the SAM-3 mask QA pilot: a free, deterministic pre-filter over the stored masks.

Every mask that reaches a vision model costs money, so the leverage in the whole funnel is in
how much can be decided without one. This stage reads the persisted COCO-RLE masks
(`run_calibration_eval.py --save-masks-dir`) plus the results CSV, computes cheap geometric
statistics, and sorts each detected instance into:

  * **pass**   -- a clear singleton with nothing anomalous about it
  * **fail**   -- SAM-3 returned nothing at all (`status=empty_mask`); no vision call needed to
                  know the mask is missing
  * **vision** -- anything else, which is what Stage 3 pays to look at

Its headline output is **`f_v`**, the fraction still needing a vision call. That single number
is what the 1M-frame cost estimate scales by, so it is reported per site, per stratum, and
corpus-weighted, not just as one aggregate.

Two design choices worth stating:

**Area is normalised by distance before it is z-scored.** Within one clip the subject walks
toward and away from the camera -- that *is* the calibration protocol -- so mask area varies by
close to an order of magnitude and a raw per-video z-score flags perfectly good masks. Apparent
area falls as 1/distance^2, so `log(area) + 2*log(distance)` is roughly constant across a clip
and its dispersion is a real anomaly signal. This uses ground-truth `distance` as a free
geometric covariate; it is not a distance-accuracy measurement, and rows without a distance
(the unannotated stratum) simply skip this one flag.

**Dispersion is measured with median/MAD, not mean/std.** Clips here have a median of ~6
annotated frames. With n that small a single grossly-wrong mask drags the mean and inflates the
standard deviation enough to hide itself -- the classic masking effect. MAD is unmoved by it.

Frame disposition comes from the results CSV `status` column, prefix-matched (statuses like
`sam_error: ...` carry a variable suffix), and never from file presence: `empty_mask` frames
deliberately write no mask JSON, so "no file" cannot distinguish *empty* from *never run*.

CPU only, seconds to run -- fine on a login node. numpy + cv2, no torch.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.masks import load_instance_masks
from scripts.sites import site_of

REPO_ROOT = Path(__file__).resolve().parents[2]

# How run_calibration_eval.py's `status` values map onto QA dispositions. Matched on the prefix
# before the first ":" (mirroring that script's own print_summary), because sam_error /
# depth_error / video_error all carry a variable exception suffix.
STATUS_EXAMINE = ("processed", "depth_error")   # a mask exists; depth failing is irrelevant here
STATUS_EMPTY = ("empty_mask",)                  # SAM-3 found nothing: auto-fail, no vision spend
STATUS_PIPELINE_FAILURE = (                     # not a mask defect; excluded from every rate
    "sam_error", "frame_decode_error", "video_error", "video_missing",
)

OUTPUT_FIELDS = [
    "video_name", "frame_idx", "instance_idx", "site", "stratum", "inclusion_weight",
    "status", "disposition", "prefilter_class", "flags",
    "area_px", "area_frac", "fill_ratio", "n_components", "n_instances",
    "border_touch_frac", "area_z", "area_jump", "distance",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-csv", type=Path, default=REPO_ROOT / "outputs" / "qa" / "results.csv",
                   help="run_calibration_eval.py output for the sampled frames")
    p.add_argument("--mask-dir", type=Path, default=REPO_ROOT / "outputs" / "qa" / "masks",
                   help="run_calibration_eval.py --save-masks-dir")
    p.add_argument("--sample-csv", type=Path, default=REPO_ROOT / "outputs" / "qa" / "sample.csv",
                   help="scripts/qa/sample.py output, for stratum and inclusion_weight")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "qa" / "prefilter.csv")

    g = p.add_argument_group("thresholds")
    g.add_argument("--min-area-px", type=int, default=200,
                   help="absolute floor; below this a mask is too small to be a person")
    g.add_argument("--min-area-frac", type=float, default=0.0002,
                   help="floor as a fraction of frame area, applied alongside --min-area-px")
    g.add_argument("--max-area-frac", type=float, default=0.60,
                   help="a mask covering more than this share of the frame is bleeding")
    g.add_argument("--min-fill-ratio", type=float, default=0.18,
                   help="area_px / bbox_area; low means a split or a straggling tendril")
    g.add_argument("--max-border-touch-frac", type=float, default=0.08,
                   help="share of the frame border covered by the mask")
    g.add_argument("--min-component-frac", type=float, default=0.05,
                   help="components smaller than this share of the instance's area are speckle "
                        "and are not counted toward the split test")
    g.add_argument("--min-component-px", type=int, default=50,
                   help="absolute speckle floor, applied alongside --min-component-frac")
    g.add_argument("--max-area-z", type=float, default=3.5,
                   help="robust (MAD) z-score of distance-normalised log area within a clip")
    g.add_argument("--min-area-dispersion", type=float, default=0.05,
                   help="floor on the MAD of distance-normalised log area, in log units "
                        "(0.05 is ~5%% of area). Clips steadier than this have no dispersion "
                        "worth scoring against, and without the floor every trivial wobble in "
                        "them scores as a huge outlier -- see robust_z")
    g.add_argument("--max-area-jump", type=float, default=1.10,
                   help="|log ratio| of area between neighbouring frames of one instance track "
                        "(1.10 is a ~3x jump)")
    g.add_argument("--min-clip-points", type=int, default=4,
                   help="clips with fewer instance points than this skip the z-score test; a "
                        "dispersion estimate from 2-3 points is noise")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# pure geometry helpers (unit-tested in test_prefilter.py)
# --------------------------------------------------------------------------------------

def status_group(status: str | None) -> str:
    """'examine' | 'empty' | 'pipeline_failure' | 'unknown' from a results-CSV status value."""
    if not status:
        return "unknown"
    prefix = status.split(":", 1)[0].strip()
    if prefix in STATUS_EXAMINE:
        return "examine"
    if prefix in STATUS_EMPTY:
        return "empty"
    if prefix in STATUS_PIPELINE_FAILURE:
        return "pipeline_failure"
    return "unknown"


def bbox_of(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(xmin, ymin, xmax, ymax) inclusive, or None for an empty mask."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def fill_ratio(mask: np.ndarray) -> float:
    """Mask area as a share of its bounding box. A clean upright person sits around 0.3-0.5;
    a mask split across two subjects, or one trailing a shadow, drops well below that."""
    box = bbox_of(mask)
    if box is None:
        return 0.0
    xmin, ymin, xmax, ymax = box
    box_area = (xmax - xmin + 1) * (ymax - ymin + 1)
    return float(mask.sum()) / box_area if box_area else 0.0


def border_touch_frac(mask: np.ndarray) -> float:
    """Share of the frame's 1px border ring covered by the mask.

    A subject standing at the edge legitimately clips the frame, so this is not decisive on its
    own -- it routes to vision rather than failing, because telling 'clipped subject' from
    'mask leaked into the background' is exactly the judgement a model is for.
    """
    h, w = mask.shape[:2]
    if h < 2 or w < 2:
        return float(mask.any())
    # the side columns exclude the first and last row, so the four corners are counted once
    border = int(
        mask[0, :].sum() + mask[-1, :].sum() + mask[1:-1, 0].sum() + mask[1:-1, -1].sum()
    )
    perimeter = 2 * h + 2 * w - 4
    return border / perimeter if perimeter else 0.0


def count_components(mask: np.ndarray, min_frac: float = 0.05, min_px: int = 50) -> int:
    """Connected components, ignoring speckle below both thresholds.

    SAM-3 masks routinely carry a few stray pixels; counting those as a 'split' would send
    every mask to vision and destroy the pre-filter's whole point.
    """
    total = int(mask.sum())
    if total == 0:
        return 0
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    floor = max(min_px, int(min_frac * total))
    # label 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    kept = int((areas >= floor).sum())
    return max(kept, 1)  # the largest component always counts, even below the floor


def normalised_log_area(area_px: float, distance: float | None) -> float | None:
    """log(area) + 2*log(distance): flat across a clip as the subject walks toward the camera.

    Apparent area falls as 1/distance^2, so the +2*log(distance) term cancels the walk and what
    is left is (approximately) the subject's true size. Returns None when there is no usable
    distance, so callers can skip the test rather than fabricate a value.
    """
    if area_px <= 0:
        return None
    if distance is None or not math.isfinite(distance) or distance <= 0:
        return None
    return math.log(area_px) + 2.0 * math.log(distance)


def robust_z(values: list[float], min_mad: float = 0.0) -> list[float]:
    """Median/MAD z-scores, scaled to be comparable with the usual mean/std ones.

    Used instead of mean/std because clips here have ~6 points: one grossly wrong mask inflates
    the standard deviation enough to keep its own z-score below any sane threshold (masking),
    whereas the median and MAD barely move.

    `min_mad` is a floor on the scale estimate, and it is load-bearing rather than defensive.
    MAD divides, so a clip whose masks are all nearly identical -- a subject standing still,
    which happens -- has a MAD near zero, and then a one-pixel wobble becomes a twenty-sigma
    outlier. Every frame in the steadiest clips would be flagged. Below the floor there is no
    meaningful dispersion to score against, so nothing is flagged.
    """
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad <= 0 or mad <= min_mad:
        return [0.0] * len(values)
    return [float(0.6745 * (v - median) / mad) for v in arr]


def flag_instance(metrics: dict, args: argparse.Namespace) -> list[str]:
    """Per-instance flags that need no cross-frame context."""
    flags: list[str] = []
    if metrics["area_px"] < args.min_area_px or metrics["area_frac"] < args.min_area_frac:
        flags.append("tiny_area")
    if metrics["area_frac"] > args.max_area_frac:
        flags.append("huge_area")
    if metrics["fill_ratio"] < args.min_fill_ratio:
        flags.append("low_fill")
    if metrics["border_touch_frac"] > args.max_border_touch_frac:
        flags.append("border_contact")
    if metrics["n_components"] > 1:
        flags.append("multi_component")
    if metrics["n_instances"] > 1:
        flags.append("multi_instance")
    return flags


def decide(status_grp: str, flags: list[str]) -> tuple[str, str]:
    """(disposition, prefilter_class) for one instance."""
    if status_grp == "empty":
        return "fail", "empty"
    if status_grp == "pipeline_failure":
        return "excluded", "pipeline_failure"
    if status_grp != "examine":
        return "excluded", "unknown_status"
    if not flags:
        return "pass", "clean_singleton"
    return "vision", "flagged"


# --------------------------------------------------------------------------------------
# I/O and cross-frame passes
# --------------------------------------------------------------------------------------

def load_sample_meta(path: Path) -> dict[tuple[str, int], dict]:
    """(video_name, frame_idx) -> {stratum, inclusion_weight} from scripts/qa/sample.py."""
    if not path.exists():
        return {}
    meta: dict[tuple[str, int], dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (row["video_name"], int(row["frame_idx"]))
            except (KeyError, TypeError, ValueError):
                continue
            meta[key] = {
                "stratum": row.get("stratum", "annotated"),
                "inclusion_weight": row.get("inclusion_weight", ""),
            }
    return meta


def load_results(path: Path) -> list[dict]:
    """Results rows deduped to one per (video_name, frame_idx, instance_idx).

    A single frame can appear under several annotation rows, and run_calibration_eval.py fans
    each of those out across every detected instance -- so the raw CSV carries duplicates that
    would otherwise be counted several times in every rate.
    """
    seen: set[tuple[str, int, int | None]] = set()
    rows: list[dict] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                frame_idx = int(row["frame_idx"])
            except (KeyError, TypeError, ValueError):
                continue
            raw = row.get("instance_idx")
            inst = int(raw) if raw not in (None, "", "None") else None
            key = (row["video_name"], frame_idx, inst)
            if key in seen:
                continue
            seen.add(key)
            row["frame_idx"] = frame_idx
            row["instance_idx"] = inst
            rows.append(row)
    return rows


def _as_float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def add_clip_flags(records: list[dict], args: argparse.Namespace) -> None:
    """Cross-frame flags, computed within each clip. Mutates `records` in place.

    Both tests are per (video_name, instance_idx): SAM-3 orders instances left-to-right by bbox
    centre, so instance 0 of consecutive frames is a usable poor-man's track for a walking
    subject, and mixing instances would manufacture jumps that aren't there.
    """
    tracks: dict[tuple[str, int | None], list[dict]] = defaultdict(list)
    for rec in records:
        if rec["disposition"] == "excluded" or rec["area_px"] <= 0:
            continue
        tracks[(rec["video_name"], rec["instance_idx"])].append(rec)

    for track in tracks.values():
        track.sort(key=lambda r: r["frame_idx"])

        # 1. distance-normalised area outlier within the clip
        scored = [(r, normalised_log_area(r["area_px"], r["distance"])) for r in track]
        usable = [(r, v) for r, v in scored if v is not None]
        if len(usable) >= args.min_clip_points:
            zs = robust_z([v for _, v in usable], min_mad=args.min_area_dispersion)
            for (rec, _), z in zip(usable, zs):
                rec["area_z"] = round(z, 3)
                if abs(z) > args.max_area_z:
                    rec["flags"].append("area_outlier")

        # 2. frame-to-frame area jump along the track
        logs = [math.log(r["area_px"]) for r in track]
        for i, rec in enumerate(track):
            neighbours = []
            if i > 0:
                neighbours.append(abs(logs[i] - logs[i - 1]))
            if i + 1 < len(track):
                neighbours.append(abs(logs[i] - logs[i + 1]))
            if not neighbours:
                continue
            # min() not max(): a genuinely displaced mask disagrees with BOTH neighbours, while
            # a legitimate approach step disagrees with the one it moved away from
            jump = min(neighbours)
            rec["area_jump"] = round(jump, 3)
            if jump > args.max_area_jump:
                rec["flags"].append("area_jump")


def build_records(args: argparse.Namespace) -> list[dict]:
    results = load_results(args.results_csv)
    meta = load_sample_meta(args.sample_csv)

    # instance count per frame, taken from the rows themselves rather than the mask files
    n_instances: dict[tuple[str, int], int] = defaultdict(int)
    for row in results:
        if row["instance_idx"] is not None:
            n_instances[(row["video_name"], row["frame_idx"])] += 1

    mask_cache: dict[tuple[str, int], list[dict] | None] = {}

    def masks_for(video_name: str, frame_idx: int) -> list[dict] | None:
        key = (video_name, frame_idx)
        if key not in mask_cache:
            try:
                mask_cache[key] = load_instance_masks(args.mask_dir, video_name, frame_idx)
            except FileNotFoundError:
                mask_cache[key] = None
        return mask_cache[key]

    records: list[dict] = []
    n_missing_masks = 0
    for row in results:
        video_name, frame_idx, inst_idx = row["video_name"], row["frame_idx"], row["instance_idx"]
        grp = status_group(row.get("status"))
        frame_meta = meta.get((video_name, frame_idx), {})

        rec = {
            "video_name": video_name,
            "frame_idx": frame_idx,
            "instance_idx": inst_idx,
            "site": site_of(video_name),
            "stratum": frame_meta.get("stratum", "annotated"),
            "inclusion_weight": frame_meta.get("inclusion_weight", ""),
            "status": row.get("status", ""),
            "distance": _as_float(row.get("distance_gt")),
            "flags": [],
            "area_px": 0,
            "area_frac": 0.0,
            "fill_ratio": 0.0,
            "n_components": 0,
            "n_instances": n_instances.get((video_name, frame_idx), 0),
            "border_touch_frac": 0.0,
            "area_z": "",
            "area_jump": "",
        }

        if grp == "examine":
            instances = masks_for(video_name, frame_idx)
            inst = None
            if instances:
                for candidate in instances:
                    if candidate["instance_idx"] == inst_idx:
                        inst = candidate
                        break
            if inst is None:
                # a `processed` row whose mask JSON is absent is a real inconsistency, not the
                # documented empty_mask case -- surface it rather than silently passing it
                n_missing_masks += 1
                rec["flags"].append("mask_file_missing")
                rec["disposition"], rec["prefilter_class"] = "vision", "flagged"
                records.append(rec)
                continue

            mask = inst["mask"]
            h, w = mask.shape[:2]
            area_px = int(mask.sum())
            rec.update({
                "area_px": area_px,
                "area_frac": area_px / (h * w) if h * w else 0.0,
                "fill_ratio": round(fill_ratio(mask), 4),
                "n_components": count_components(mask, args.min_component_frac, args.min_component_px),
                "border_touch_frac": round(border_touch_frac(mask), 4),
            })
            rec["flags"] = flag_instance(rec, args)

        rec["disposition"], rec["prefilter_class"] = decide(grp, rec["flags"])
        records.append(rec)

    if n_missing_masks:
        print(f"  !! {n_missing_masks} rows had status=processed but no mask JSON; routed to vision")

    add_clip_flags(records, args)

    # re-decide: clip flags can move a previously-clean instance to vision
    for rec in records:
        grp = status_group(rec["status"])
        rec["disposition"], rec["prefilter_class"] = decide(grp, rec["flags"])
    return records


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------

def summarise(records: list[dict]) -> dict:
    """f_v overall, per stratum and per site, plus the flag histogram."""
    eligible = [r for r in records if r["disposition"] != "excluded"]
    annotated = [r for r in eligible if r["stratum"] != "unannotated"]

    def f_v(rows: list[dict]) -> tuple[int, int, float]:
        n = len(rows)
        n_vision = sum(1 for r in rows if r["disposition"] == "vision")
        return n_vision, n, (n_vision / n if n else 0.0)

    flag_counts: dict[str, int] = defaultdict(int)
    for rec in eligible:
        for flag in rec["flags"]:
            flag_counts[flag] += 1

    by_site: dict[str, tuple[int, int, float]] = {}
    for site in sorted({r["site"] for r in annotated}):
        by_site[site] = f_v([r for r in annotated if r["site"] == site])

    # corpus-weighted f_v: equal-per-site sampling over-represents the small sites, so the
    # unweighted figure is not the number to multiply into the 1M estimate
    num = den = 0.0
    for rec in annotated:
        weight = _as_float(rec["inclusion_weight"])
        if weight is None:
            continue
        den += weight
        if rec["disposition"] == "vision":
            num += weight
    weighted = num / den if den else None

    return {
        "overall": f_v(annotated),
        "unannotated": f_v([r for r in eligible if r["stratum"] == "unannotated"]),
        "weighted_f_v": weighted,
        "by_site": by_site,
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda kv: -kv[1])),
        "dispositions": {
            d: sum(1 for r in records if r["disposition"] == d)
            for d in ("pass", "fail", "vision", "excluded")
        },
        "classes": {
            c: sum(1 for r in records if r["prefilter_class"] == c)
            for c in sorted({r["prefilter_class"] for r in records})
        },
    }


def print_summary(summary: dict) -> None:
    n_vision, n, frac = summary["overall"]
    print("\n--- dispositions (all instances) ---")
    for name, count in summary["dispositions"].items():
        print(f"  {name:<10} {count}")
    print("\n--- prefilter classes ---")
    for name, count in summary["classes"].items():
        print(f"  {name:<20} {count}")

    print("\n--- flags raised ---")
    if summary["flag_counts"]:
        for name, count in summary["flag_counts"].items():
            print(f"  {name:<20} {count}")
    else:
        print("  (none)")

    print("\n--- f_v: fraction needing a vision call ---")
    print(f"  annotated stratum:  {n_vision}/{n} = {frac:.3f}")
    if summary["weighted_f_v"] is not None:
        print(f"  corpus-weighted:    {summary['weighted_f_v']:.3f}   <-- use THIS for the 1M estimate")
    u_vision, u_n, u_frac = summary["unannotated"]
    if u_n:
        print(f"  unannotated stratum: {u_vision}/{u_n} = {u_frac:.3f}  (reported separately; on "
              f"these frames an empty mask is often correct)")
    print("\n  per site (unweighted):")
    for site, (sv, sn, sf) in summary["by_site"].items():
        print(f"    {site:<12} {sv}/{sn} = {sf:.3f}")


def main() -> None:
    args = parse_args()
    if not args.results_csv.exists():
        sys.exit(f"no results CSV at {args.results_csv} -- run Stage 1 first")

    records = build_records(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            row["flags"] = "|".join(rec["flags"])
            row["area_frac"] = round(rec["area_frac"], 6)
            row["distance"] = "" if rec["distance"] is None else rec["distance"]
            writer.writerow(row)

    print(f"wrote {len(records)} instance rows to {args.out}")
    print_summary(summarise(records))


if __name__ == "__main__":
    main()
