#!/usr/bin/env python
"""Stage 3a of the SAM-3 mask QA pilot: render one review panel per instance needing vision.

Reads the pre-filter's verdicts, re-decodes the source frames, and writes a PNG per instance:

    [ full frame: translucent fill + hard contour ] | [ zoomed crop of the instance bbox ]

Both halves matter. The full frame is what shows *wrong-subject* (the mask is on a tree, or on
the wrong person) and *multiple* (other detections outlined in a second colour). The crop is
what shows *bleed* and *split* -- those are pixel-boundary judgements, and a vision model reads
them poorly from a full-frame translucent overlay where the subject is 40px tall.

Two things this gets right that the eval script's own `--overlay-dir` does not:

* **Filenames key on `video_name`, not `video_path.stem`.** Camera folders reuse filenames like
  `DSCF0005` across sites, so stem-keyed overlays silently overwrite each other -- the same
  collision `scripts/masks.py` documents for its RLE files.
* **The burned-in frame index is the one this script decoded**, not one re-read from a CSV, so
  comparing it against the visible content is a real check that frame extraction is correct.
  That check is the reason the pilot samples from the QC'd annotations generation at all.

Panel size is capped (`--panel-height`) because it sets the per-frame image-token bill: image
tokens are roughly `w*h/750`, so this is the main cost lever in the whole funnel. The default
~1050x384 panel is ~540 image tokens.

No depth panel: this stage scores mask quality, and a depth colourmap would double the token
cost per frame while adding nothing a reviewer or a model can use to judge a mask boundary.

CPU only, no network -- runs as an offline SLURM job (see slurm/stage3a_render.sbatch).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.frame_source import iter_frames_at_indices
from scripts.masks import load_instance_masks
from scripts.video_lookup import load_anno_to_path, resolve_video_path

REPO_ROOT = Path(__file__).resolve().parents[2]

SUBJECT_BGR = (0, 220, 60)      # translucent fill for the instance under review
SUBJECT_EDGE_BGR = (255, 255, 0)  # its contour, in a contrasting hue -- `bleed` is a boundary
                                  # judgement, and an edge drawn in the fill colour disappears
                                  # into the fill exactly where it needs to be legible
OTHER_BGR = (0, 165, 255)       # every other detection in the same frame, outline only
MANIFEST_FIELDS = [
    "video_name", "frame_idx", "instance_idx", "site", "stratum",
    "prefilter_class", "flags", "overlay_path",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefilter-csv", type=Path,
                   default=REPO_ROOT / "outputs" / "qa" / "prefilter.csv")
    p.add_argument("--mask-dir", type=Path, default=REPO_ROOT / "outputs" / "qa" / "masks")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "qa" / "overlays")
    p.add_argument("--manifest", type=Path,
                   default=REPO_ROOT / "outputs" / "qa" / "overlay_manifest.csv",
                   help="overlay path -> (video_name, frame_idx, instance_idx); Stage 3's "
                        "custom_id map is built from this")
    p.add_argument("--video-list-xlsx", type=Path,
                   default=REPO_ROOT / "data" / "list_reference_videos.xlsx")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--frames-dir", type=Path, default=None,
                   help="read frames from exported PNGs (<video_name>_frame%%06d.png, as written "
                        "by export_calibrated.py) instead of decoding the source videos; use when "
                        "the videos are not on this machine")
    p.add_argument("--dispositions", nargs="+", default=["vision"],
                   help="which pre-filter dispositions to render. Defaults to the ones that go "
                        "to a model; add 'pass'/'fail' to render the gold-set slices too")
    p.add_argument("--panel-height", type=int, default=384,
                   help="output panel height in px; sets the image-token bill (~w*h/750)")
    p.add_argument("--crop-pad-frac", type=float, default=0.35,
                   help="padding around the instance bbox in the zoom panel, as a fraction of "
                        "the bbox's larger side -- context enough to judge a boundary")
    p.add_argument("--alpha", type=float, default=0.45, help="mask fill opacity")
    p.add_argument("--limit", type=int, default=None, help="render at most N panels (smoke test)")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------------------

def draw_mask(frame_bgr: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int],
              alpha: float, contour_px: int = 2,
              edge_colour: tuple[int, int, int] | None = None) -> np.ndarray:
    """Translucent fill plus a hard contour, so both the extent and the exact edge are legible."""
    out = frame_bgr.copy()
    if mask.shape[:2] != out.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (out.shape[1], out.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    if mask.any():
        layer = np.zeros_like(out)
        layer[mask] = colour
        out = cv2.addWeighted(out, 1.0, layer, alpha, 0.0)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, edge_colour or colour, contour_px, cv2.LINE_AA)
    return out


def draw_outline(frame_bgr: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int],
                 thickness: int = 2) -> np.ndarray:
    """Outline only -- used for the frame's *other* detections, so `multiple` is visible without
    two filled masks competing for attention."""
    out = frame_bgr
    if mask.shape[:2] != out.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (out.shape[1], out.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, colour, thickness, cv2.LINE_AA)
    return out


def crop_box(mask: np.ndarray, shape_hw: tuple[int, int], pad_frac: float) -> tuple[int, int, int, int]:
    """Padded, in-bounds (x0, y0, x1, y1) around the mask, or the whole frame if it's empty."""
    h, w = shape_hw
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0, 0, w, h
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    pad = int(round(pad_frac * max(xmax - xmin + 1, ymax - ymin + 1))) + 4
    return (max(0, xmin - pad), max(0, ymin - pad),
            min(w, xmax + pad + 1), min(h, ymax + pad + 1))


def fit_height(img: np.ndarray, height: int) -> np.ndarray:
    """Scale to a fixed height so panels can be hconcat'd; never upscale past 3x (past that the
    crop is interpolation noise, and the extra pixels are billed as image tokens for nothing)."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((height, height, 3), dtype=np.uint8)
    scale = min(height / h, 3.0)
    target = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    out = cv2.resize(img, target, interpolation=interp)
    if out.shape[0] != height:  # capped by the 3x limit -- pad rather than stretch
        pad = np.zeros((height, out.shape[1], 3), dtype=out.dtype)
        top = (height - out.shape[0]) // 2
        pad[top:top + out.shape[0]] = out
        out = pad
    return out


def label(panel: np.ndarray, lines: list[str]) -> np.ndarray:
    """Burn captions into a dark band across the top of the panel.

    A band rather than the per-glyph halo used elsewhere in the repo: the burned-in frame index
    is a verification signal (does it match the visible content?) and the flags tell a human
    reviewer what the pre-filter saw, so both need to be unambiguously readable over arbitrary
    night-time camera-trap footage rather than merely visible.
    """
    if not lines:
        return panel
    band_h = 12 + 18 * len(lines)
    band = panel[:band_h].astype(np.float32) * 0.25
    panel[:band_h] = band.astype(np.uint8)
    for n, line in enumerate(lines):
        org = (8, 20 + 18 * n)
        cv2.putText(panel, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def render_panel(frame_bgr: np.ndarray, instances: list[dict], instance_idx: int,
                 meta: dict, args: argparse.Namespace) -> np.ndarray | None:
    """The full-frame + zoom panel for one instance, or None if that instance isn't present."""
    subject = next((i for i in instances if i["instance_idx"] == instance_idx), None)
    if subject is None:
        return None
    mask = subject["mask"]

    full = draw_mask(frame_bgr, mask, SUBJECT_BGR, args.alpha, edge_colour=SUBJECT_EDGE_BGR)
    for other in instances:
        if other["instance_idx"] != instance_idx:
            full = draw_outline(full, other["mask"], OTHER_BGR)

    x0, y0, x1, y1 = crop_box(mask, frame_bgr.shape[:2], args.crop_pad_frac)
    zoom = full[y0:y1, x0:x1]
    if zoom.size == 0:
        zoom = full

    left = fit_height(full, args.panel_height)
    right = fit_height(zoom, args.panel_height)
    # a divider so the model doesn't read the two panels as one continuous scene
    divider = np.full((args.panel_height, 3, 3), 255, dtype=np.uint8)
    panel = cv2.hconcat([left, divider, right])

    label(panel, [
        f"{meta['video_name']}",
        f"frame {meta['frame_idx']} (decoded)  instance {instance_idx} of {len(instances)}",
        f"flags: {meta.get('flags') or 'none'}",
    ])
    for colour, weight in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(panel, "ZOOM", (left.shape[1] + 12, args.panel_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, weight, cv2.LINE_AA)
    return panel


def overlay_name(video_name: str, frame_idx: int, instance_idx: int) -> str:
    """Keyed on video_name, never video_path.stem -- camera folders reuse clip filenames."""
    return f"{video_name}_frame{frame_idx:06d}_inst{instance_idx}.png"


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def load_targets(path: Path, dispositions: list[str], limit: int | None) -> list[dict]:
    targets: list[dict] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("disposition") not in dispositions:
                continue
            raw = row.get("instance_idx")
            if raw in (None, "", "None"):
                continue  # frame-level rows (empty_mask, pipeline failures) have nothing to draw
            try:
                targets.append({
                    "video_name": row["video_name"],
                    "frame_idx": int(row["frame_idx"]),
                    "instance_idx": int(raw),
                    "site": row.get("site", ""),
                    "stratum": row.get("stratum", ""),
                    "prefilter_class": row.get("prefilter_class", ""),
                    "flags": row.get("flags", ""),
                })
            except (TypeError, ValueError):
                continue
    targets.sort(key=lambda t: (t["video_name"], t["frame_idx"], t["instance_idx"]))
    return targets[:limit] if limit else targets


def main() -> None:
    args = parse_args()
    if not args.prefilter_csv.exists():
        sys.exit(f"no pre-filter CSV at {args.prefilter_csv} -- run Stage 2 first")

    targets = load_targets(args.prefilter_csv, args.dispositions, args.limit)
    if not targets:
        sys.exit(f"no rows with disposition in {args.dispositions}; nothing to render")
    print(f"{len(targets)} instances to render (dispositions: {', '.join(args.dispositions)})")

    anno_to_path = None if args.frames_dir else load_anno_to_path(args.video_list_xlsx, args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # group by video so each is decoded once for all of its requested frames -- decoding is
    # sequential (iter_frames_at_indices), so per-frame reopening would be quadratic
    by_video: dict[str, list[dict]] = defaultdict(list)
    for target in targets:
        by_video[target["video_name"]].append(target)

    manifest: list[dict] = []
    n_written = n_failed = 0
    for video_name in sorted(by_video):
        group = by_video[video_name]
        by_frame: dict[int, list[dict]] = defaultdict(list)
        for target in group:
            by_frame[target["frame_idx"]].append(target)

        if args.frames_dir is not None:
            frames = [(idx, cv2.imread(str(args.frames_dir / f"{video_name}_frame{idx:06d}.png")))
                      for idx in sorted(by_frame)]
        else:
            try:
                video_path = resolve_video_path(video_name, anno_to_path)
            except (KeyError, FileNotFoundError) as exc:
                print(f"  !! {video_name}: {exc}")
                n_failed += len(group)
                continue
            try:
                frames = list(iter_frames_at_indices(video_path, sorted(by_frame)))
            except OSError as exc:
                print(f"  !! {video_name}: {exc}")
                n_failed += len(group)
                continue

        for frame_idx, frame_bgr in frames:
            wanted = by_frame.get(frame_idx, [])
            if frame_bgr is None:
                print(f"  !! {video_name} frame {frame_idx}: decode returned no frame")
                n_failed += len(wanted)
                continue
            try:
                instances = load_instance_masks(args.mask_dir, video_name, frame_idx)
            except FileNotFoundError:
                print(f"  !! {video_name} frame {frame_idx}: no mask JSON")
                n_failed += len(wanted)
                continue

            for target in wanted:
                panel = render_panel(frame_bgr, instances, target["instance_idx"], target, args)
                if panel is None:
                    print(f"  !! {video_name} frame {frame_idx}: no instance "
                          f"{target['instance_idx']} in the mask file")
                    n_failed += 1
                    continue
                out_path = args.out_dir / overlay_name(video_name, frame_idx, target["instance_idx"])
                cv2.imwrite(str(out_path), panel)
                manifest.append({**target, "overlay_path": str(out_path)})
                n_written += 1
                if n_written % 100 == 0:
                    print(f"  ... {n_written} panels written")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\nwrote {n_written} panels to {args.out_dir}")
    print(f"  manifest: {args.manifest}")
    if n_failed:
        print(f"  !! {n_failed} instances could not be rendered (see messages above)")
    if manifest:
        sample_panel = cv2.imread(manifest[0]["overlay_path"])
        if sample_panel is not None:
            h, w = sample_panel.shape[:2]
            print(f"  panel size {w}x{h} -> ~{w * h / 750:.0f} image tokens per frame")


if __name__ == "__main__":
    main()
