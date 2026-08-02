#!/usr/bin/env python3
"""Render contact sheets of triaged masks so the ranking can be checked by eye.

A score table is only trustworthy if the masks it ranks worst actually look worse than the
ones it accepts, and that is a question no summary statistic answers. This renders a grid
of frames with the mask outlined and its flags printed on it, drawn from whichever slice of
scripts/score_masks.py's output you ask for:

    # the worst of the review queue
    python scripts/triage_contact_sheet.py --scores outputs/triage/mask_scores.csv \\
        --export-dir /scratch/.../export_test --out outputs/triage/worst.png --top 24

    # a random sample of what would be auto-accepted -- the audit that catches the
    # failure mode where accepted masks quietly carry errors into the training set
    python scripts/triage_contact_sheet.py --scores outputs/triage/mask_scores.csv \\
        --export-dir /scratch/.../export_test --out outputs/triage/accept_audit.png \\
        --bucket auto_accept --sample 24 --seed 0

Masks are drawn on the full uncropped frame -- the banner is cropped for *scoring*, but
hiding it here would hide the thing a banner_overlap flag is complaining about.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.masks import decode_rle

TILE_WIDTH = 480
COLUMNS = 4
CONTOUR_BGR = (0, 220, 255)
LABEL_BGR = (255, 255, 255)


def load_rows(scores_csv: Path) -> list[dict]:
    with open(scores_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select(rows: list[dict], bucket: str | None, top: int | None,
           sample: int | None, seed: int) -> list[dict]:
    rows = [r for r in rows if r.get("status") == "scored"]
    if bucket:
        rows = [r for r in rows if r.get("bucket") == bucket]
    if sample:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(rows), size=min(sample, len(rows)), replace=False)
        return [rows[i] for i in sorted(indices)]
    rows.sort(key=lambda r: -float(r["triage_score"] or 0))
    return rows[: top or len(rows)]


def render_tile(row: dict, export_dir: Path) -> np.ndarray | None:
    video_name, frame_idx = row["video_name"], int(row["frame_idx"])
    frame_path = export_dir / "frames" / f"{video_name}_frame{frame_idx:06d}.png"
    mask_path = export_dir / "masks" / f"{video_name}_frame{frame_idx:06d}_masks.json"
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None or not mask_path.exists():
        return None

    with open(mask_path) as f:
        instances = json.load(f)
    instance_idx = int(row["instance_idx"])
    match = next((i for i in instances if i["instance_idx"] == instance_idx), None)
    if match is None:
        return None

    mask = decode_rle(match["rle"]).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, CONTOUR_BGR, 2)

    scale = TILE_WIDTH / image.shape[1]
    tile = cv2.resize(image, (TILE_WIDTH, int(round(image.shape[0] * scale))))
    banner = np.zeros((44, TILE_WIDTH, 3), np.uint8)
    flags = row.get("flags") or "-"
    cv2.putText(banner, f'{row["bucket"]} {float(row["triage_score"]):.1f}  {flags}'[:64],
                (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, LABEL_BGR, 1, cv2.LINE_AA)
    cv2.putText(banner, f'{video_name[-40:]} f{frame_idx} i{instance_idx}',
                (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
    return np.vstack([tile, banner])


def contact_sheet(tiles: list[np.ndarray], columns: int = COLUMNS) -> np.ndarray:
    height = max(t.shape[0] for t in tiles)
    padded = [
        np.vstack([t, np.zeros((height - t.shape[0], t.shape[1], 3), np.uint8)])
        if t.shape[0] < height else t
        for t in tiles
    ]
    rows = []
    for start in range(0, len(padded), columns):
        chunk = padded[start:start + columns]
        while len(chunk) < columns:
            chunk.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(chunk))
    return np.vstack(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", type=Path, required=True, help="mask_scores.csv from score_masks.py")
    p.add_argument("--export-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bucket", type=str, default=None,
                   choices=["auto_accept", "needs_review", "reject"])
    p.add_argument("--top", type=int, default=24, help="highest triage_score first")
    p.add_argument("--sample", type=int, default=None,
                   help="random sample instead of the top-N (use for accept-bucket audits)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--columns", type=int, default=COLUMNS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = select(load_rows(args.scores), args.bucket, args.top, args.sample, args.seed)
    tiles = [t for t in (render_tile(r, args.export_dir) for r in rows) if t is not None]
    if not tiles:
        print("no renderable masks matched that selection")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), contact_sheet(tiles, args.columns))
    print(f"{len(tiles)} masks -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
