"""Merge in-session subagent verdict chunk JSONs into a triage.py-shaped verdicts CSV.

An alternative Stage 3b path: instead of `triage.py submit/poll/fetch` (Batch API), the overlay
manifest is split into chunks and each chunk is graded by a Claude Code subagent, which writes a
`chunk_NN_verdicts.json` of `{file, verdict, confidence, rationale}` entries. This script joins
those back onto the manifest and emits a CSV with the exact `triage.py` VERDICT_FIELDS shape, so
`report.py goldset` and `report.py summarise` run unchanged.

Caveat for interpretation: subagent verdicts come from an agentic multi-image context, not the
single-image Batch requests `triage.py` sends, so triage dials measured this way are indicative
rather than identical to the production path.

Usage:
    python scripts/qa/merge_verdicts.py --manifest <overlay_manifest.csv> \
        --chunks '<dir>/chunk_*_verdicts.json' --out <verdicts.csv>
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter

VERDICT_CLASSES = {"ok", "empty", "wrong-subject", "bleed", "split", "multiple"}
VERDICT_FIELDS = [
    "custom_id", "video_name", "frame_idx", "instance_idx", "site", "stratum",
    "prefilter_class", "flags", "model", "verdict", "confidence", "rationale",
    "result_type", "error", "overlay_path",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="overlay manifest CSV from render_overlays.py")
    p.add_argument("--chunks", required=True,
                   help="glob for chunk verdict JSONs, e.g. 'verdict_chunks/chunk_*_verdicts.json'")
    p.add_argument("--out", required=True, help="output verdicts CSV")
    p.add_argument("--model-label", default="claude-haiku-4-5 (in-session subagent)",
                   help="value recorded in the model column")
    args = p.parse_args()

    manifest = {os.path.basename(r["overlay_path"]): r
                for r in csv.DictReader(open(args.manifest))}

    verdicts: dict[str, dict] = {}
    n_bad = 0
    for path in sorted(glob.glob(args.chunks)):
        for item in json.load(open(path)):
            fname = os.path.basename(item.get("file", ""))
            if fname not in manifest:
                print(f"  !! {os.path.basename(path)}: unknown file {fname!r}", file=sys.stderr)
                n_bad += 1
                continue
            if fname in verdicts:
                print(f"  !! duplicate verdict for {fname} (keeping first)", file=sys.stderr)
                continue
            verdict = item.get("verdict")
            if verdict not in VERDICT_CLASSES:
                verdicts[fname] = {"verdict": "", "confidence": "", "rationale": "",
                                   "result_type": "errored",
                                   "error": item.get("error") or f"bad verdict {verdict!r}"}
                n_bad += 1
                continue
            try:
                conf = min(1.0, max(0.0, float(item.get("confidence"))))
            except (TypeError, ValueError):
                conf = ""
            verdicts[fname] = {"verdict": verdict, "confidence": conf,
                               "rationale": item.get("rationale", ""),
                               "result_type": "succeeded", "error": ""}

    rows = []
    n_missing = 0
    for i, (fname, meta) in enumerate(sorted(manifest.items())):
        v = verdicts.get(fname)
        if v is None:
            v = {"verdict": "", "confidence": "", "rationale": "",
                 "result_type": "errored", "error": "no verdict returned"}
            n_missing += 1
        rows.append({
            "custom_id": f"insession-{i:05d}",
            "video_name": meta["video_name"], "frame_idx": meta["frame_idx"],
            "instance_idx": meta["instance_idx"], "site": meta.get("site", ""),
            "stratum": meta.get("stratum", ""), "prefilter_class": meta.get("prefilter_class", ""),
            "flags": meta.get("flags", ""), "model": args.model_label,
            "overlay_path": meta["overlay_path"], **v,
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=VERDICT_FIELDS)
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["result_type"] == "succeeded")
    print(f"wrote {len(rows)} rows to {args.out}: {ok} succeeded, "
          f"{n_missing} missing, {n_bad} malformed")
    print("verdicts:", dict(Counter(r["verdict"] for r in rows if r["verdict"])))


if __name__ == "__main__":
    main()
