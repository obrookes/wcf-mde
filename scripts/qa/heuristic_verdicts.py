#!/usr/bin/env python
"""Project scripts/score_masks.py's mask_scores.csv into the verdicts-CSV contract.

Backend B (CPU heuristics) and Backend A (the VLM in triage.py) both need to drive the same
downstream tools -- review_server.py, make_review_bundle.py, apply_corrections.py, report.py --
so they must agree on the verdicts-CSV shape documented in scripts/qa/verdicts_schema.py. This
script is the adapter: it reads every row `score_masks.py` wrote (not `review_queue.csv`, which
already dropped the auto-accepted rows -- the server's `--include-ok` needs those rows present
here too) and re-expresses each one as a VERDICT_FIELDS row, deriving a `verdict` from the
`flags` column that `score_masks.py` fused from its own hard/soft thresholds.

Like merge_verdicts.py, this is a non-VLM producer of the verdicts CSV: no model call, nothing
graded by a human or an LLM, just a deterministic re-labelling of signals score_masks.py already
computed. Read that module (and scripts/mask_signals.py, for what each signal column means)
before touching the tables below.

**The heuristic backend never emits "multiple".** Nothing in score_masks.py's flag set implies
"this one mask covers two distinct subjects" as opposed to "this mask bled into extra pixels" --
that distinction needs the VLM to actually look at the panel. So `verdict` here is always one of
`{"ok", "empty", "wrong-subject", "bleed", "split"}`, never `"multiple"`.

Flag -> verdict projection
--------------------------
A mask's `flags` column (";"-joined in mask_scores.csv) is partitioned by
`score_masks.py::_flag_mask`'s own hard/soft split, reproduced here as `HARD_FLAG_VERDICTS` and
`SOFT_FLAG_VERDICTS`. Hard flags always win over soft flags (a hard flag means the mask is
degenerate; a soft flag just means something looks off). Within one tier, more than one flag can
fire at once (e.g. "fragmented;holes"), so ties are broken by the rubric's own mislead-ranking
from triage.py's RUBRIC: "wrong-subject" beats "multiple" beats "bleed" beats "split" -- read as
"beats split beats empty" here, since "empty" is the least misleading verdict a downstream
distance estimate could be handed (see `VERDICT_PRECEDENCE`).

"anomalous_area" is the one flag whose verdict depends on more than its own firing: an
unusually *small* log-area (area_log_z < 0) reads as a mask that shrank onto a fragment --
"empty" -- while an unusually *large* one (area_log_z > 0) reads as a mask that spilled outward
-- "bleed".

A flag name outside both tables (a future signal this script doesn't know about yet) is ignored
for verdict selection but is still preserved in the row's `flags`/`rationale` columns -- silently
dropping evidence would be worse than showing a reviewer a flag with no verdict attached. If
*every* fired flag on a row is unrecognised, there is no known-good verdict to fall back to, so
the row is conservatively routed to review as `"split"` (never silently "ok"), and the rationale
notes `unknown-flags` so a human can tell the difference between "genuinely split" and "this
script didn't know what to do with you".

A `bucket == "auto_accept"` row has no flags at all (score_masks.py only reaches that bucket
when both hard and soft lists are empty) and maps straight to `"ok"`.

Non-scored rows -- `status != "scored"` -- carry no signals to project, so they become
`result_type = "errored"` with a blank `verdict`: `status == "frame_missing"` (the QC-filtered
export dropped the frame; blank `instance_idx`) gets `error = "frame_missing: exported frame
absent"`. `build_queue()` in review_server.py filters on `result_type == "succeeded"` before its
`int(instance_idx)` sort, so these rows can never reach that sort and crash it -- they are kept
in the CSV purely for accounting. Any other status value this script has not seen before is
handled the same way, with a generic `"unrecognised status {status!r}"` error, and reported to
stderr so a genuinely new status doesn't disappear silently.

Confidence is **not** a calibrated probability. For a scored row it is either a fixed value tied
to the bucket (reject -> 0.95, auto_accept -> 0.8) or, for needs_review rows,
`round(1 - exp(-triage_score / 1.5), 3)` -- a monotone-increasing squash of the fused
`triage_score` into (0, 1). It answers "how strongly did the heuristics fire", not "P(verdict
correct)", and is on a different scale to Backend A's VLM-reported confidence: comparing the two
numbers across backends is meaningless.

Usage:
    python scripts/qa/heuristic_verdicts.py --scores outputs/triage/mask_scores.csv \
        --out outputs/qa/verdicts_heuristic.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.qa.verdicts_schema import VERDICT_CLASSES, VERDICT_FIELDS, blank_verdict_row, key_of  # noqa: E402
from scripts.sites import site_of  # noqa: E402

# score_masks.py::_flag_mask's own hard/soft partition (see MIN_AREA_PX etc. there), each flag
# mapped to the verdict it implies. Hard flags are degenerate masks: nothing an annotator could
# salvage by nudging a boundary.
HARD_FLAG_VERDICTS = {
    "tiny": "empty",
    "is_banner": "wrong-subject",
    "engulfs_frame": "bleed",
}
# Soft flags are "look at this" -- a nameable thing looks off, short of degenerate.
# "anomalous_area" is intentionally absent here: its verdict depends on the sign of
# area_log_z, resolved in `_verdict_for_soft_flags` rather than a fixed table lookup.
SOFT_FLAG_VERDICTS = {
    "no_residual_support": "wrong-subject",
    "fragmented": "split",
    "weak_boundary": "bleed",
    "holes": "split",
    "sliver": "split",
    "thin": "split",
    "truncated": "split",
}

# The rubric's own mislead-ranking (triage.py RUBRIC), extended with "empty" at the tail since
# it is the least misleading verdict a downstream distance estimate could be handed. Used only
# to break ties when more than one flag in the same tier maps to a different verdict.
VERDICT_PRECEDENCE = ["wrong-subject", "multiple", "bleed", "split", "empty"]

# One most-relevant signal column per flag, for the compact rationale. Values pulled from
# scripts/mask_signals.py (SIGNAL_COLUMNS) and scripts/background.py (RESIDUAL_COLUMNS), plus
# score_masks.py's own derived area_log_z.
FLAG_SIGNAL_COLUMN = {
    "tiny": "area_px",
    "engulfs_frame": "area_frac",
    "is_banner": "banner_overlap_frac",
    "no_residual_support": "residual_recall",
    "fragmented": "second_component_frac",
    "weak_boundary": "boundary_gradient_ratio",
    "holes": "hole_area_frac",
    "sliver": "compactness",
    "thin": "erosion_survival",
    "truncated": "border_contact_frac",
    "anomalous_area": "area_log_z",
}

STATUS_SCORED = "scored"
STATUS_FRAME_MISSING = "frame_missing"


def _pick_by_precedence(candidates: set[str]) -> str:
    for verdict in VERDICT_PRECEDENCE:
        if verdict in candidates:
            return verdict
    # Unreachable given the tables above (every mapped verdict is in VERDICT_PRECEDENCE), but
    # fail safe rather than crash the whole run over a future table edit.
    return "split"


def _resolve_anomalous_area(row: dict) -> str:
    try:
        z = float(row.get("area_log_z") or 0.0)
    except (TypeError, ValueError):
        z = 0.0
    return "bleed" if z > 0 else "empty"


def verdict_for_flags(fired: list[str], row: dict) -> tuple[str, list[str]]:
    """(verdict, unknown_flags) for one scored row's fired flag list.

    Hard flags beat soft flags outright; within a tier the rubric's mislead-precedence breaks
    ties. Flags this script doesn't recognise are returned separately so the caller can still
    surface them in the rationale even though they didn't drive the verdict.
    """
    hard = [f for f in fired if f in HARD_FLAG_VERDICTS]
    soft = [f for f in fired if f in SOFT_FLAG_VERDICTS or f == "anomalous_area"]
    unknown = [f for f in fired if f not in HARD_FLAG_VERDICTS
               and f not in SOFT_FLAG_VERDICTS and f != "anomalous_area"]

    if hard:
        candidates = {HARD_FLAG_VERDICTS[f] for f in hard}
        return _pick_by_precedence(candidates), unknown
    if soft:
        candidates = {
            _resolve_anomalous_area(row) if f == "anomalous_area" else SOFT_FLAG_VERDICTS[f]
            for f in soft
        }
        return _pick_by_precedence(candidates), unknown
    if unknown:
        # Nothing recognised fired at all -- conservatively route to review rather than let an
        # unmodelled flag silently pass as "ok".
        return "split", unknown
    return "ok", []


def build_rationale(fired: list[str], row: dict, unknown: list[str]) -> str:
    """`fragmented;holes -- second_component_frac=0.41 hole_area_frac=0.13`-style one-liner,
    one signal value per *recognised* flag that actually fired; unknown flags are still named
    in the flag list but contribute no signal reading (there's no table entry for one)."""
    if not fired:
        return ""
    parts = []
    for flag in fired:
        column = FLAG_SIGNAL_COLUMN.get(flag)
        if column is None:
            continue
        value = row.get(column)
        if value in (None, ""):
            continue
        try:
            value = round(float(value), 3)
        except (TypeError, ValueError):
            pass
        parts.append(f"{column}={value}")
    rationale = ";".join(fired)
    if parts:
        rationale += " -- " + " ".join(parts)
    if unknown:
        rationale += " (unknown-flags: " + ";".join(unknown) + ")"
    return rationale


def confidence_for(row: dict) -> float:
    """Ordinal, monotone-in-score confidence -- see module docstring: not a calibrated
    probability, and not comparable across backends."""
    bucket = row.get("bucket")
    if bucket == "reject":
        return 0.95
    if bucket == "auto_accept":
        return 0.8
    try:
        score = float(row.get("triage_score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return round(1 - math.exp(-score / 1.5), 3)


def project_row(row: dict, model_label: str) -> dict:
    out = blank_verdict_row()
    out["video_name"] = row.get("video_name", "")
    out["frame_idx"] = row.get("frame_idx", "")
    out["instance_idx"] = row.get("instance_idx", "")
    out["site"] = row.get("site") or (site_of(out["video_name"]) if out["video_name"] else "")
    out["model"] = model_label
    out["custom_id"] = key_of(out)

    status = row.get("status", STATUS_SCORED)
    if status != STATUS_SCORED:
        out["result_type"] = "errored"
        out["verdict"] = ""
        out["flags"] = (row.get("flags") or "").replace(";", "|")
        if status == STATUS_FRAME_MISSING:
            out["error"] = "frame_missing: exported frame absent"
        else:
            out["error"] = f"unrecognised status {status!r}"
        return out

    fired = [f for f in (row.get("flags") or "").split(";") if f]
    verdict, unknown = verdict_for_flags(fired, row)

    out["flags"] = "|".join(fired)
    out["verdict"] = verdict
    out["confidence"] = confidence_for(row)
    out["rationale"] = build_rationale(fired, row, unknown)
    out["result_type"] = "succeeded"
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", required=True, type=Path,
                   help="score_masks.py's mask_scores.csv (not review_queue.csv -- the "
                        "auto_accept rows are needed here too, for --include-ok parity)")
    p.add_argument("--out", required=True, type=Path, help="output verdicts CSV")
    p.add_argument("--model-label", default="heuristic-v1",
                   help="value recorded in the model column")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    with open(args.scores, newline="") as f:
        scored_rows = list(csv.DictReader(f))

    rows = [project_row(row, args.model_label) for row in scored_rows]

    seen_statuses = {row.get("status", STATUS_SCORED) for row in scored_rows}
    unrecognised = seen_statuses - {STATUS_SCORED, STATUS_FRAME_MISSING}
    for status in sorted(unrecognised):
        print(f"  !! unrecognised status {status!r} in {args.scores} -- rows recorded as "
              f"errored", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VERDICT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_errored = sum(1 for r in rows if r["result_type"] == "errored")
    print(f"wrote {len(rows)} verdicts to {args.out}")
    print("\n--- verdict mix ---")
    counts = Counter(r["verdict"] or f"<{r['result_type']}>" for r in rows)
    for verdict, count in counts.most_common():
        print(f"  {verdict:<16} {count:>5}  ({count / len(rows):.1%})" if rows else
              f"  {verdict:<16} {count:>5}")
    print(f"\n  errored: {n_errored}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
