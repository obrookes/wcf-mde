#!/usr/bin/env python
"""Pack everything the bad-mask review UI needs into one self-contained tarball, so the
review runs on a laptop instead of through an SSH tunnel to the login node.

Collects, for every flagged-bad verdict row (same queue rule as review_server.py): the raw
frame PNG and the mask JSON (the UI renders all its views from those two); plus the verdicts
CSV (bad rows only), the server code itself, a `run_review.py` launcher, a requirements.txt
and a README. The result extracts to `review_bundle/` and runs with:

    pip install -r requirements.txt
    python run_review.py            # then open http://localhost:8765

Decisions land in corrections.csv inside the bundle dir; copy that one small file back to the
cluster when done and feed it to apply_corrections.py.

Usage (on the cluster):
    python scripts/qa/make_review_bundle.py \
        --verdicts  .../qa_pilot/verdicts_haiku.csv \
        --frames-dir .../export_test/frames \
        --masks-dir  .../export_test/masks \
        --out       .../qa_pilot/review_bundle.tar.gz
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.masks import mask_path  # noqa: E402
from scripts.qa.review_server import BAD_CLASSES, build_queue, frame_filename  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_CODE = ["scripts/masks.py", "scripts/qa/review_server.py"]

REQUIREMENTS = """\
numpy
opencv-python-headless
pycocotools
"""

RUN_REVIEW = '''\
#!/usr/bin/env python
"""Launch the mask review UI from this bundle.

    pip install -r requirements.txt
    python run_review.py [--port 8765] [--only-class split ...]

Then open http://localhost:8765 in a browser. Decisions append to corrections.csv here.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)  # overlay paths in verdicts.csv are bundle-relative
sys.path.insert(0, ROOT)
# defaults first, so any explicitly passed flags win (argparse: last occurrence wins)
sys.argv[1:1] = ["--verdicts", "verdicts.csv", "--frames-dir", "frames",
                 "--masks-dir", "masks", "--out", "corrections.csv"]

from scripts.qa.review_server import main

main()
'''

README = """\
Bad-mask review bundle
======================

Everything needed to review the Haiku-flagged masks, offline, on your own machine.

1.  pip install -r requirements.txt      (numpy, opencv headless, pycocotools)
2.  python run_review.py                 (add --port N if 8765 is taken)
3.  open http://localhost:8765

Tabs show the original SAM-3 mask on the frame (default), an auto-fix preview, a zoom
crop, and the raw frame (keys 1-4). Actions are buttons (with shortcuts): Mask is fine
(a), Accept auto-fix (f), Draw box (b: drag on the image, then Enter to save), Discard
(d), Skip (s); arrows navigate, Esc cancels drawing.

Every keypress appends to corrections.csv in this directory immediately — you can stop
and restart run_review.py at any time and it resumes; re-deciding a key is fine (the
last decision wins downstream).

When you are done, copy corrections.csv (it is tiny) back to the cluster and apply it:

    scp corrections.csv <cluster>:/scratch/b6cn/obrookes.b6cn/qa_pilot/corrections.csv
    # on the cluster, in the wcf-mde repo:
    python scripts/qa/apply_corrections.py morph \\
        --corrections /scratch/b6cn/obrookes.b6cn/qa_pilot/corrections.csv \\
        --masks-dir /scratch/b6cn/obrookes.b6cn/export_test/masks \\
        --out-masks-dir /scratch/b6cn/obrookes.b6cn/export_test/masks_corrected

Note: this tool displays the model verdict, so it must never be used to label the
blind gold set.
"""


def stage_bundle(rows: list[dict], frames_dir: Path, masks_dir: Path, stage: Path,
                 only_classes: list[str] | None = None,
                 repo_root: Path = REPO_ROOT) -> dict:
    """Copy the minimal file set for the review queue into `stage`. Returns stats incl. a
    `missing` list of (kind, path) for anything a queue row references that isn't on disk."""
    queue = build_queue(rows, only_classes)
    if not queue:
        raise SystemExit("queue is empty — nothing to bundle")

    for sub in ("frames", "masks"):
        (stage / sub).mkdir(parents=True, exist_ok=True)
    missing: list[tuple[str, str]] = []
    n_frames = n_masks = 0

    def copy(kind: str, src: Path, dst: Path) -> bool:
        nonlocal missing
        if dst.exists():
            return False
        if not src.exists():
            missing.append((kind, str(src)))
            return False
        shutil.copy2(src, dst)
        return True

    out_rows = []
    for item in queue:
        video, frame = item["video_name"], int(item["frame_idx"])
        fname = frame_filename(video, frame)
        n_frames += copy("frame", frames_dir / fname, stage / "frames" / fname)
        msrc = mask_path(masks_dir, video, frame)
        n_masks += copy("mask", msrc, stage / "masks" / msrc.name)
        # overlay panels are not bundled: the UI renders every view from frame + mask JSON
        out_rows.append({**item, "overlay_path": ""})

    with open(stage / "verdicts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    for rel_code in BUNDLED_CODE:
        dst = stage / rel_code
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / rel_code, dst)
    (stage / "run_review.py").write_text(RUN_REVIEW)
    (stage / "requirements.txt").write_text(REQUIREMENTS)
    (stage / "README.txt").write_text(README)

    return {"queue": len(queue), "frames": n_frames, "masks": n_masks,
            "missing": missing}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verdicts", type=Path, required=True)
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--masks-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="output .tar.gz path")
    p.add_argument("--only-class", action="append", choices=BAD_CLASSES, default=None,
                   help="restrict to these verdict classes (repeatable)")
    p.add_argument("--allow-missing", action="store_true",
                   help="tar anyway when some referenced files are absent")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.verdicts, newline="") as f:
        rows = list(csv.DictReader(f))

    with tempfile.TemporaryDirectory(dir=args.out.parent) as td:
        stage = Path(td) / "review_bundle"
        stats = stage_bundle(rows, args.frames_dir, args.masks_dir, stage, args.only_class)
        print(f"queue {stats['queue']}: staged {stats['frames']} frames, "
              f"{stats['masks']} mask JSONs")
        if stats["missing"]:
            for kind, path in stats["missing"]:
                print(f"  !! missing {kind}: {path}", file=sys.stderr)
            if not args.allow_missing:
                sys.exit(f"{len(stats['missing'])} referenced file(s) missing "
                         "(use --allow-missing to bundle anyway)")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.out, "w:gz") as tar:
            tar.add(stage, arcname="review_bundle")

    mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out}  ({mb:.1f} MB)")
    print("download it, then:  tar xzf", args.out.name,
          "&& cd review_bundle && pip install -r requirements.txt && python run_review.py")


if __name__ == "__main__":
    main()
