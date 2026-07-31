#!/usr/bin/env python3
"""Probe fps/duration of every reference video under data/ with ffprobe.

Writes data/video_fps.csv with one row per video:
    video_name  flattened path (/ -> _, extension dropped), matches the `anno`
                convention of list_reference_videos.xlsx and the `video_name`
                column of the annotation CSVs
    path        relative path under data/
    fps         r_frame_rate as float (declared frame rate)
    avg_fps     avg_frame_rate as float (actual average; differs from fps for VFR)
    duration_s  stream duration in seconds
    nb_frames   frame count reported by the container (may be empty)
    vfr         True where |fps - avg_fps| > 0.01

This table is the single source of truth for fps in downstream checks — never
infer fps from annotation frame_idx / frame_timestamp ratios.
"""

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

VIDEO_EXTS = {".avi", ".mp4", ".mov"}


def parse_rate(s: str) -> float | None:
    if not s or s in ("0/0", "N/A"):
        return None
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        return None


def probe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate,duration,nb_frames",
        "-of", "csv=p=0", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        fields = out.stdout.strip().split(",") if out.returncode == 0 else []
    except subprocess.TimeoutExpired:
        fields = []
    r_rate = parse_rate(fields[0]) if len(fields) > 0 else None
    avg_rate = parse_rate(fields[1]) if len(fields) > 1 else None
    duration = None
    if len(fields) > 2 and fields[2] not in ("", "N/A"):
        try:
            duration = float(fields[2])
        except ValueError:
            pass
    nb_frames = fields[3] if len(fields) > 3 and fields[3] not in ("", "N/A") else ""
    return {
        "fps": r_rate,
        "avg_fps": avg_rate,
        "duration_s": duration,
        "nb_frames": nb_frames,
        "vfr": (r_rate is not None and avg_rate is not None
                and abs(r_rate - avg_rate) > 0.01),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data")
    ap.add_argument("--out", type=Path, default=None,
                    help="output CSV (default: <data-root>/video_fps.csv)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out_path = args.out or args.data_root / "video_fps.csv"

    videos = sorted(
        p for p in args.data_root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    print(f"probing {len(videos)} videos under {args.data_root} ...")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(probe, videos))

    n_fail = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "path", "fps", "avg_fps",
                    "duration_s", "nb_frames", "vfr"])
        for path, r in zip(videos, results):
            rel = path.relative_to(args.data_root)
            video_name = str(rel.with_suffix("")).replace("/", "_")
            if r["fps"] is None:
                n_fail += 1
                print(f"  WARNING: probe failed for {rel}", file=sys.stderr)
            w.writerow([
                video_name, str(rel),
                "" if r["fps"] is None else f"{r['fps']:g}",
                "" if r["avg_fps"] is None else f"{r['avg_fps']:g}",
                "" if r["duration_s"] is None else f"{r['duration_s']:g}",
                r["nb_frames"],
                r["vfr"],
            ])

    n_vfr = sum(r["vfr"] for r in results)
    print(f"wrote {out_path}: {len(videos)} rows, "
          f"{n_fail} probe failures, {n_vfr} VFR videos")
    fps_counts: dict[float, int] = {}
    for r in results:
        if r["fps"] is not None:
            fps_counts[r["fps"]] = fps_counts.get(r["fps"], 0) + 1
    for fps, n in sorted(fps_counts.items(), key=lambda kv: -kv[1]):
        print(f"  fps {fps:g}: {n} videos")


if __name__ == "__main__":
    main()
