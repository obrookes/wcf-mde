#!/usr/bin/env python
"""Stage 0 of the SAM-3 mask QA pilot: draw the fixed 1,000-frame cohort.

Reads the QC'd annotations generation (`annotations_20260709_with_fps_clean.csv`, per
README.md's Step 0) and writes a sample CSV that is itself a valid annotations CSV -- it carries
`video_name, frame_idx, frame_timestamp, distance`, which is exactly what
`run_calibration_eval.py::load_rows` reads -- so Stage 1 needs no new code:

    python scripts/run_calibration_eval.py --annotations-csv outputs/qa/sample.csv ...

Three things this script is careful about:

1. **Which QC flags to honour.** `qc_annotations.py --fix` already DROPS ABSURD_DISTANCE /
   ZERO_DISTANCE rows and REPAIRS FRAME_IDX_FPS_MISMATCH ones when it writes `_clean.csv`.
   Excluding all flagged rows would therefore discard a large, systematically-chosen slice of
   perfectly good data (every row whose frame_idx needed fixing) and bias the cohort. Only
   TIMESTAMP_PAST_END / VIDEO_NOT_ON_DISK still disqualify a row -- `--qc-reasons` overrides.

2. **The flags join.** The flags CSV keys on the PRE-fix frame_idx, so it is resolved through
   `scripts/qc_exclusions.py` on `frame_timestamp` rather than joined on frame_idx directly.

3. **Weighting.** Allocation is equal-per-site so the small sites (beauvois, mbnp) are estimable
   at all, which deliberately over-samples them relative to the corpus. Every row therefore
   carries an `inclusion_weight` (corpus share / sample share) so the report can give both an
   unweighted per-site rate and a corpus-weighted aggregate. The weighted one is what multiplies
   into the 1M extrapolation; using the unweighted rate there would be a real error.

Also draws a small **unannotated stratum** -- random frames from the same videos that have no
annotation row. On those frames the sign-holder is often absent, so an empty mask is frequently
*correct* rather than a failure; they are reported separately and never blended into the
headline dials. Their `distance` is written as `nan` (which `float()` parses) and their
`frame_timestamp` derived from the probed fps, so they survive `load_rows` unchanged.

Note that Stage 1 over this sample decodes a subset of each video's annotated frames, so its
joint multi-frame depth estimates differ from a full run's. That is irrelevant here: this pilot
scores mask quality, and SAM-3 segments each frame independently.

CPU only, seconds to run -- fine on a login node. numpy + stdlib, no torch/cv2.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.qc_exclusions import REASONS_FOR_CLEAN, resolve_exclusions
from scripts.sites import site_of

REPO_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_FIELDS = [
    # the annotations-CSV contract run_calibration_eval.py::load_rows depends on
    "video_name", "frame_idx", "frame_timestamp", "distance",
    # QA bookkeeping; ignored by load_rows' DictReader
    "site", "stratum", "inclusion_weight",
]
COHORT_FIELDS = ["video_name", "frame_idx", "stratum"]

QC_REASON_CHOICES = {
    # only what _clean.csv hasn't already handled (the sane default -- see module docstring)
    "clean": REASONS_FOR_CLEAN,
    # every flagged row, whatever the reason: conservative, and biased against repaired rows
    "all": None,
    "none": frozenset(),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations-csv", type=Path,
                   default=REPO_ROOT / "data" / "annotations_20260709_with_fps_clean.csv",
                   help="QC'd annotations generation to sample from (README.md Step 0 prefers "
                        "the *_clean.csv over annotations_06052026.csv, whose frame_idx values "
                        "are inconsistent with the probed fps)")
    p.add_argument("--qc-flags", type=Path,
                   default=REPO_ROOT / "data" / "qc_flags_annotations_20260709_with_fps.csv",
                   help="scripts/qc_annotations.py flags CSV, resolved by timestamp")
    p.add_argument("--qc-reasons", choices=sorted(QC_REASON_CHOICES), default="clean",
                   help="which flag reasons disqualify a row; see the module docstring")
    p.add_argument("--fps-table", type=Path, default=REPO_ROOT / "data" / "video_fps.csv",
                   help="scripts/probe_video_fps.py output; only needed for the unannotated "
                        "stratum, to know each video's frame count without opening it")
    p.add_argument("--n", type=int, default=1000, help="annotated frames to draw")
    p.add_argument("--unannotated", type=int, default=100,
                   help="extra frames with no annotation row, drawn from the sampled videos "
                        "(0 to skip)")
    p.add_argument("--seed", type=int, default=20260731)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "qa" / "sample.csv",
                   help="working sample; a valid --annotations-csv for run_calibration_eval.py")
    p.add_argument("--cohort-out", type=Path, default=REPO_ROOT / "docs" / "qa_cohort_1000.csv",
                   help="two-column cohort record, small enough to commit (data/ and outputs/ "
                        "are both untracked, so this is what pins the cohort in git)")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# allocation
# --------------------------------------------------------------------------------------

def allocate_equally(available: dict[str, int], n_target: int) -> dict[str, int]:
    """Split `n_target` as evenly as possible across sites, capped by what each site has.

    Water-filling: hand out an equal share to every site that still has rows left, then
    redistribute whatever the exhausted sites couldn't absorb, until the target is met or every
    site is drained. Deterministic (sites are visited in sorted order), so the cohort is
    reproducible from the seed alone.
    """
    alloc = {site: 0 for site in available}
    remaining = n_target
    active = {site for site, n in available.items() if n > 0}

    while remaining > 0 and active:
        share = remaining // len(active)
        if share == 0:
            # fewer slots left than sites: one each, largest sites first so the leftovers land
            # where they are least likely to distort a small site's rate
            for site in sorted(active, key=lambda s: (-available[s], s))[:remaining]:
                alloc[site] += 1
                remaining -= 1
            break
        took = 0
        for site in sorted(active):
            take = min(share, available[site] - alloc[site])
            alloc[site] += take
            took += take
        remaining -= took
        active = {site for site in active if available[site] - alloc[site] > 0}
        if took == 0:  # every remaining site is saturated
            break

    return alloc


def inclusion_weights(corpus: dict[str, int], sample: dict[str, int]) -> dict[str, float]:
    """Per-site (corpus share / sample share), normalised so the weights sum to the sample size.

    A site drawn in exact proportion to the corpus gets 1.0; over-sampled small sites get < 1.
    """
    n_corpus = sum(corpus.values())
    n_sample = sum(sample.values())
    if n_corpus == 0 or n_sample == 0:
        return {site: 0.0 for site in sample}
    return {
        site: (corpus.get(site, 0) / n_corpus) / (n / n_sample) if n else 0.0
        for site, n in sample.items()
    }


# --------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------

def load_annotations(path: Path) -> list[dict]:
    """Annotation rows with a usable integer frame_idx.

    `qc_annotations.py::apply_fixes` blanks frame_idx for videos missing from disk (its default
    --fix-mode). Those rows have no frame to segment, and would additionally crash
    run_calibration_eval.py::load_rows' unguarded int() cast, so they are dropped here.
    """
    rows: list[dict] = []
    n_blank = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            raw_idx = row.get("frame_idx")
            if raw_idx in (None, "", "None", "nan", "NaN", "<NA>"):
                n_blank += 1
                continue
            try:
                row["frame_idx"] = int(float(raw_idx))
                row["frame_timestamp"] = float(row["frame_timestamp"])
                row["distance"] = float(row["distance"])
            except (TypeError, ValueError, KeyError):
                n_blank += 1
                continue
            rows.append(row)
    if n_blank:
        print(f"  dropped {n_blank} rows with a blank/unparseable frame_idx (videos missing from disk)")
    return rows


def load_fps_table(path: Path) -> dict[str, tuple[float, float]]:
    """video_name -> (fps, duration_s), from scripts/probe_video_fps.py."""
    table: dict[str, tuple[float, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                table[row["video_name"]] = (float(row["fps"]), float(row["duration_s"]))
            except (TypeError, ValueError, KeyError):
                continue
    return table


# --------------------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------------------

def draw_annotated(rows: list[dict], n_target: int, rng: np.random.Generator) -> list[dict]:
    """Equal-per-site draw without replacement, tagged with site and inclusion_weight."""
    by_site: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_site[site_of(row["video_name"])].append(row)

    corpus = {site: len(rs) for site, rs in by_site.items()}
    alloc = allocate_equally(corpus, n_target)
    weights = inclusion_weights(corpus, alloc)

    drawn: list[dict] = []
    for site in sorted(by_site):
        take = alloc[site]
        if take <= 0:
            continue
        pool = by_site[site]
        picks = rng.choice(len(pool), size=take, replace=False)
        for i in sorted(int(p) for p in picks):
            row = pool[i]
            drawn.append({
                "video_name": row["video_name"],
                "frame_idx": row["frame_idx"],
                "frame_timestamp": row["frame_timestamp"],
                "distance": row["distance"],
                "site": site,
                "stratum": "annotated",
                "inclusion_weight": round(weights[site], 6),
            })

    print(f"\n--- per-site allocation (target {n_target}) ---")
    for site in sorted(corpus):
        print(f"  {site:<12} corpus={corpus[site]:<6} sampled={alloc[site]:<5} "
              f"weight={weights[site]:.3f}")
    if sum(alloc.values()) < n_target:
        print(f"  (only {sum(alloc.values())} available across all sites, short of {n_target})")
    return drawn


def draw_unannotated(
    annotated: list[dict],
    all_rows: list[dict],
    fps_table: dict[str, tuple[float, float]],
    n_target: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Random frames with no annotation row, drawn from the videos already in the sample.

    Restricting to already-sampled videos keeps Stage 1 cheap: `iter_frames_at_indices` decodes
    a video sequentially once for all of its requested indices, so an extra index on a video
    already being decoded is nearly free.

    Frame counts come from the probed fps table rather than by opening each video -- these rows
    exist to measure how the failure mix shifts off the annotated frames, and are not worth a
    decode pass to select.
    """
    if n_target <= 0:
        return []
    if not fps_table:
        print("  !! no fps table; skipping the unannotated stratum")
        return []

    annotated_idx: dict[str, set[int]] = defaultdict(set)
    for row in all_rows:
        annotated_idx[row["video_name"]].add(int(row["frame_idx"]))

    videos = sorted({row["video_name"] for row in annotated})
    usable = [v for v in videos if v in fps_table and fps_table[v][0] > 0]
    if not usable:
        print("  !! no sampled video appears in the fps table; skipping the unannotated stratum")
        return []

    drawn: list[dict] = []
    attempts = 0
    max_attempts = n_target * 50  # generous; only bites if videos are pathologically short
    seen: set[tuple[str, int]] = set()
    while len(drawn) < n_target and attempts < max_attempts:
        attempts += 1
        video = usable[int(rng.integers(len(usable)))]
        fps, duration_s = fps_table[video]
        n_frames = int(math.floor(duration_s * fps))
        if n_frames <= 1:
            continue
        # keep clear of the last frame: iter_frames_at_indices yields None past the end, and a
        # decode that lands exactly on the boundary is a coin flip on container metadata
        frame_idx = int(rng.integers(0, max(1, n_frames - 1)))
        if frame_idx in annotated_idx.get(video, ()) or (video, frame_idx) in seen:
            continue
        seen.add((video, frame_idx))
        drawn.append({
            "video_name": video,
            "frame_idx": frame_idx,
            "frame_timestamp": round(frame_idx / fps, 6),
            "distance": "nan",  # no ground truth; the pre-filter skips these rows' z-score
            "site": site_of(video),
            "stratum": "unannotated",
            "inclusion_weight": "",  # excluded from the headline dials, so never weighted in
        })

    if len(drawn) < n_target:
        print(f"  !! only drew {len(drawn)}/{n_target} unannotated frames after {attempts} attempts")
    return drawn


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"reading {args.annotations_csv}")
    rows = load_annotations(args.annotations_csv)
    print(f"  {len(rows)} annotation rows with a usable frame_idx")

    reasons = QC_REASON_CHOICES[args.qc_reasons]
    if args.qc_reasons != "none" and args.qc_flags.exists():
        exclude = resolve_exclusions(args.qc_flags, args.annotations_csv, reasons)
        before = len(rows)
        rows = [r for r in rows if (r["video_name"], r["frame_idx"]) not in exclude]
        print(f"  QC (--qc-reasons {args.qc_reasons}): {len(exclude)} flagged keys resolved by "
              f"timestamp, {before - len(rows)} rows dropped")
    elif args.qc_reasons != "none":
        print(f"  !! no flags CSV at {args.qc_flags}; sampling without QC exclusions")

    if not rows:
        sys.exit("no annotation rows survived filtering; nothing to sample")

    annotated = draw_annotated(rows, args.n, rng)

    fps_table = load_fps_table(args.fps_table) if args.fps_table.exists() else {}
    if args.unannotated > 0 and not fps_table:
        print(f"  !! no fps table at {args.fps_table}")
    unannotated = draw_unannotated(annotated, rows, fps_table, args.unannotated, rng)

    sample = annotated + unannotated

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(sample)

    args.cohort_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.cohort_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COHORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sample)

    n_videos = len({r["video_name"] for r in sample})
    print(f"\nwrote {len(annotated)} annotated + {len(unannotated)} unannotated rows "
          f"across {n_videos} videos")
    print(f"  sample (feed to run_calibration_eval.py --annotations-csv): {args.out}")
    print(f"  cohort record (commit this):                               {args.cohort_out}")
    print(f"  seed={args.seed} -- re-running with the same seed and inputs reproduces it exactly")


if __name__ == "__main__":
    main()
