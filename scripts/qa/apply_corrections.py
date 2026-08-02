#!/usr/bin/env python
"""Apply the decisions recorded by review_server.py to the stored masks.

Reads the corrections CSV (append-only; the LAST decision per key wins), writes corrected mask
JSONs to a parallel output directory in the exact scripts/masks.py schema — the original mask
JSONs are never touched — and logs every non-skip decision to an applied.csv:

  disposition      meaning
  ---------------  -------------------------------------------------------------
  corrected        autofix applied, corrected mask JSON written to --out-masks-dir
  accepted_as_is   reviewer overruled the Haiku flag; use the original mask
  excluded         reviewer discarded the frame; drop it downstream
  pending_sam3     a re-prompt box is recorded, waiting on the SAM3 weights
  errored          mask JSON / instance missing, or the fix degenerated

Subcommands:
  morph   CPU, runnable now: largest-component + hole-fill for `autofix` decisions
          (the exact autofix_mask the reviewer previewed in the UI).
  sam3    GPU, blocked on the SAM3 weights: box-prompt re-segmentation for `box`
          decisions. Currently a stub that reports how many boxes are waiting.

Usage:
    python scripts/qa/apply_corrections.py morph \
        --corrections .../corrections.csv --masks-dir .../masks \
        --out-masks-dir .../masks_corrected
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.masks import load_instance_masks, mask_path, save_instance_masks  # noqa: E402
from scripts.qa.review_server import autofix_mask, load_decisions  # noqa: E402

APPLIED_FIELDS = [
    "key", "video_name", "frame_idx", "instance_idx", "haiku_verdict", "action",
    "disposition", "area_before", "area_after", "out_path", "error",
]


def apply_morph(decisions: dict[str, dict], masks_dir: Path,
                out_masks_dir: Path) -> list[dict]:
    """Turn last-wins decisions into applied.csv rows, writing corrected mask JSONs for the
    `autofix` ones. Non-autofix actions are passed through as their disposition only."""
    rows = []
    for key in sorted(decisions):
        d = decisions[key]
        base = {"key": key, "video_name": d["video_name"], "frame_idx": d["frame_idx"],
                "instance_idx": d["instance_idx"], "haiku_verdict": d["haiku_verdict"],
                "action": d["action"], "area_before": "", "area_after": "",
                "out_path": "", "error": ""}
        action = d["action"]
        if action == "skip":
            continue
        if action == "accept":
            rows.append({**base, "disposition": "accepted_as_is"})
            continue
        if action == "discard":
            rows.append({**base, "disposition": "excluded"})
            continue
        if action == "box":
            rows.append({**base, "disposition": "pending_sam3"})
            continue

        video_name, frame_idx = d["video_name"], int(d["frame_idx"])
        try:
            instances = load_instance_masks(masks_dir, video_name, frame_idx)
        except (FileNotFoundError, OSError) as exc:
            rows.append({**base, "disposition": "errored", "error": str(exc)})
            continue
        target = int(d["instance_idx"])
        inst = next((i for i in instances if i["instance_idx"] == target), None)
        if inst is None:
            rows.append({**base, "disposition": "errored",
                         "error": f"instance {target} not in mask JSON"})
            continue
        before = inst["mask"]
        after = autofix_mask(before)
        ys, xs = np.nonzero(after)
        if len(xs) == 0:
            rows.append({**base, "disposition": "errored",
                         "error": "autofix produced an empty mask"})
            continue
        inst["mask"] = after
        inst["area_px"] = int(after.sum())
        inst["center_xy"] = (int((xs.min() + xs.max()) // 2),
                             int((ys.min() + ys.max()) // 2))
        # instances were saved (and load in) position-ordered, so re-saving the full list
        # preserves every instance_idx, including untouched siblings
        save_instance_masks(out_masks_dir, video_name, frame_idx, instances)
        rows.append({**base, "disposition": "corrected",
                     "area_before": int(before.sum()), "area_after": int(after.sum()),
                     "out_path": str(mask_path(out_masks_dir, video_name, frame_idx))})
    return rows


def write_applied(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=APPLIED_FIELDS)
        w.writeheader()
        w.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_morph = sub.add_parser("morph", help="apply morphology auto-fixes (CPU)")
    sp_morph.add_argument("--corrections", type=Path, required=True)
    sp_morph.add_argument("--masks-dir", type=Path, required=True)
    sp_morph.add_argument("--out-masks-dir", type=Path, required=True)
    sp_morph.add_argument("--applied-csv", type=Path, default=None,
                          help="log CSV (default: applied.csv inside --out-masks-dir)")

    sp_sam3 = sub.add_parser("sam3", help="box-prompt SAM3 re-segmentation (GPU; needs weights)")
    sp_sam3.add_argument("--corrections", type=Path, required=True)
    sp_sam3.add_argument("--frames-dir", type=Path)
    sp_sam3.add_argument("--out-masks-dir", type=Path)
    sp_sam3.add_argument("--weights", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    decisions = load_decisions(args.corrections)
    if args.cmd == "sam3":
        n_box = sum(1 for d in decisions.values() if d["action"] == "box")
        sys.exit(f"sam3: not implemented yet — blocked on the SAM3 weights. {n_box} recorded "
                 f"box decision(s) are waiting in {args.corrections}; they stay "
                 f"disposition=pending_sam3 in applied.csv until this lands.")

    rows = apply_morph(decisions, args.masks_dir, args.out_masks_dir)
    applied_csv = args.applied_csv or args.out_masks_dir / "applied.csv"
    write_applied(rows, applied_csv)
    mix = Counter(r["disposition"] for r in rows)
    print(f"{len(decisions)} decision(s) -> {len(rows)} applied.csv row(s) at {applied_csv}")
    print("dispositions:", dict(mix.most_common()))
    for r in rows:
        if r["disposition"] == "errored":
            print(f"  !! {r['key']}: {r['error']}", file=sys.stderr)


if __name__ == "__main__":
    main()
