#!/usr/bin/env python
"""Export QC-valid calibrated depth maps and their source frames as a paired dataset.

Consumes the *_calib.npy maps written by calibrate_depth.py --out-dir and produces:

  <out-dir>/depth_maps/{video_name}_frame{idx:06d}_calib.npy   (copied as-is, fp16)
  <out-dir>/frames/{video_name}_frame{idx:06d}.png             (decoded from the source video)

With --masks-dir, also copies each exported frame's mask (if any) into <out-dir>/masks/,
completing the frames/ + masks/ layout scripts/score_masks.py expects.

Only QC-valid frames are exported: any (video_name, frame_idx) present in the
scripts/qc_annotations.py flags CSV is dropped. calibrate_depth.py's --qc-flags is optional,
so a calib dir may well contain maps for flagged frames -- the filter is applied here
regardless of whether Stage 2 applied it.

Frames aren't persisted anywhere by the pipeline (they exist only in memory during Stage 1),
so this script re-decodes them from the source videos via list_reference_videos.xlsx. The two
output dirs are kept strictly 1:1 -- if a frame can't be decoded (or its video can't be
resolved), its depth map is not exported either.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.calibrate_depth import build_exclusions
from scripts.frame_source import iter_frames_at_indices
from scripts.masks import mask_path
from scripts.video_lookup import load_anno_to_path

REPO_ROOT = Path(__file__).resolve().parent.parent

CALIB_NAME_RE = re.compile(r"^(?P<video_name>.+)_frame(?P<frame_idx>\d{6})_calib\.npy$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib-dir", type=Path, default=REPO_ROOT / "outputs" / "depth_calib",
                   help="dir of *_calib.npy maps from calibrate_depth.py --out-dir")
    p.add_argument("--qc-flags", type=Path,
                   default=REPO_ROOT / "data" / "qc_flags_annotations_20260709_with_fps.csv",
                   help="scripts/qc_annotations.py flags CSV; any (video_name, frame_idx) "
                        "present there is excluded from the export")
    p.add_argument("--annotations-csv", type=Path,
                   default=REPO_ROOT / "data" / "annotations_20260709_with_fps_clean.csv",
                   help="the annotations CSV Stage 1 was run against, used to resolve --qc-flags "
                        "into that file's frame_idx space (the *_calib.npy names key on it). "
                        "See scripts/qc_exclusions.py for why a direct frame_idx join under-matches")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="export root; depth_maps/ and frames/ are created underneath")
    p.add_argument("--video-list", type=Path, default=REPO_ROOT / "data" / "list_reference_videos.xlsx",
                   help="ori<->anno mapping used to resolve video_name -> source video path")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                   help="root the video-list `ori` paths are relative to")
    p.add_argument("--frame-format", choices=["png", "jpg"], default="png",
                   help="image format for exported frames (png is lossless)")
    p.add_argument("--masks-dir", type=Path, default=None,
                   help="dir of *_masks.json from run_calibration_eval.py --save-masks-dir; if "
                        "given, each successfully exported frame's mask (if any) is also copied "
                        "into <out-dir>/masks/, completing the frames/ + masks/ layout "
                        "scripts/score_masks.py expects. Omit to skip mask export entirely "
                        "(no masks/ dir is created)")
    return p.parse_args()


def copy_mask_for_frame(masks_dir: Path, out_masks_dir: Path, video_name: str, frame_idx: int) -> bool:
    """If `<masks_dir>/{video_name}_frame{idx:06d}_masks.json` exists, copy it (shutil.copy2,
    preserving mtime) into `out_masks_dir` and return True; otherwise return False without
    touching the filesystem -- a frame with no detected instance has no mask JSON at all
    (see run_calibration_eval.py), which is a valid, expected state, not an error.

    `out_masks_dir` is created lazily (only on an actual copy), so passing `--masks-dir` on a
    run where nothing gets exported never leaves behind an empty masks/ dir.
    """
    src = mask_path(masks_dir, video_name, frame_idx)
    if not src.exists():
        return False
    out_masks_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_masks_dir / src.name)
    return True


def main() -> None:
    args = parse_args()

    calib_paths = sorted(args.calib_dir.glob("*_calib.npy"))
    if not calib_paths:
        sys.exit(f"no *_calib.npy files found in {args.calib_dir}")

    exclude = build_exclusions(args.qc_flags, args.annotations_csv)

    # (video_name, frame_idx) -> calib map path, QC-flagged maps dropped up front
    by_video: dict[str, dict[int, Path]] = defaultdict(dict)
    n_flagged = 0
    for path in calib_paths:
        m = CALIB_NAME_RE.match(path.name)
        if m is None:
            print(f"  !! unrecognised filename {path.name}; skipping")
            continue
        video_name, frame_idx = m["video_name"], int(m["frame_idx"])
        if (video_name, frame_idx) in exclude:
            n_flagged += 1
            continue
        by_video[video_name][frame_idx] = path
    n_valid = sum(len(frames) for frames in by_video.values())
    print(f"{len(calib_paths)} maps in {args.calib_dir}: {n_flagged} QC-flagged (dropped), "
          f"{n_valid} valid across {len(by_video)} videos")

    depth_dir = args.out_dir / "depth_maps"
    frames_dir = args.out_dir / "frames"
    depth_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    anno_to_path = load_anno_to_path(args.video_list, args.data_dir)

    masks_out_dir = args.out_dir / "masks"

    n_exported = 0
    n_decode_failed = 0
    n_masks_copied = 0
    n_masks_absent = 0
    unresolved: list[str] = []
    for video_name in sorted(by_video):
        frames = by_video[video_name]
        video_path = anno_to_path.get(video_name)
        if video_path is None or not video_path.exists():
            print(f"  !! cannot resolve {video_name} to a video on disk; "
                  f"skipping its {len(frames)} frames")
            unresolved.append(video_name)
            continue
        for frame_idx, frame in iter_frames_at_indices(video_path, list(frames)):
            if frame is None:
                print(f"  !! {video_name} frame {frame_idx}: decode failed; skipping pair")
                n_decode_failed += 1
                continue
            stem = f"{video_name}_frame{frame_idx:06d}"
            if not cv2.imwrite(str(frames_dir / f"{stem}.{args.frame_format}"), frame):
                print(f"  !! {video_name} frame {frame_idx}: imwrite failed; skipping pair")
                n_decode_failed += 1
                continue
            shutil.copy2(frames[frame_idx], depth_dir / frames[frame_idx].name)
            n_exported += 1
            if args.masks_dir is not None:
                if copy_mask_for_frame(args.masks_dir, masks_out_dir, video_name, frame_idx):
                    n_masks_copied += 1
                else:
                    n_masks_absent += 1

    print(f"\nexported {n_exported} frame/depth-map pairs to {args.out_dir}")
    if unresolved:
        n_skipped = sum(len(by_video[v]) for v in unresolved)
        print(f"skipped {n_skipped} frames from {len(unresolved)} unresolvable videos: "
              f"{', '.join(unresolved)}")
    if n_decode_failed:
        print(f"skipped {n_decode_failed} frames that failed to decode/write")
    if args.masks_dir is not None:
        print(f"copied {n_masks_copied} masks to {masks_out_dir} ({n_masks_absent} frames had no mask)")


if __name__ == "__main__":
    main()
