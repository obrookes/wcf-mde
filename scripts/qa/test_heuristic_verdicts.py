#!/usr/bin/env python
"""Synthetic unit tests for scripts/qa/heuristic_verdicts.py -- the adapter that projects
score_masks.py's mask_scores.csv into the verdicts-CSV contract (scripts/qa/verdicts_schema.py)
-- and its round-trip into review_server.py / apply_corrections.py.
Run directly:  python scripts/qa/test_heuristic_verdicts.py  (also pytest-compatible).
No GPU / data / torch / network needed -- masks are drawn with numpy, files live in tmp dirs."""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.masks import load_instance_masks, save_instance_masks
from scripts.qa.apply_corrections import apply_morph
from scripts.qa.heuristic_verdicts import (
    HARD_FLAG_VERDICTS,
    SOFT_FLAG_VERDICTS,
    build_rationale,
    confidence_for,
    main,
    project_row,
    verdict_for_flags,
)
from scripts.qa.review_server import CORRECTION_FIELDS, build_queue
from scripts.qa.verdicts_schema import VERDICT_FIELDS, key_of
from scripts.score_masks import MASK_COLUMNS

H, W = 120, 160


def _scored_row(video, frame, inst, flags="", bucket="auto_accept", triage_score=0.0,
                area_log_z=""):
    """A MASK_COLUMNS-shaped row (score_masks.py's real header) for a scored mask -- every
    column present, most left blank since the adapter only reads a handful of them."""
    row = {c: "" for c in MASK_COLUMNS}
    row.update({
        "video_name": video, "frame_idx": frame, "instance_idx": inst,
        "station": "st1", "status": "scored", "bucket": bucket,
        "triage_score": triage_score, "flags": flags, "area_log_z": area_log_z,
    })
    return row


def _missing_row(video, frame):
    """A MASK_COLUMNS-shaped row for a frame the QC-filtered export dropped."""
    row = {c: "" for c in MASK_COLUMNS}
    row.update({
        "video_name": video, "frame_idx": frame, "instance_idx": "",
        "status": "frame_missing", "bucket": "needs_review",
        "triage_score": 1.0, "flags": "frame_missing",
    })
    return row


# --------------------------------------------------------------------------------------
# per-flag projection
# --------------------------------------------------------------------------------------


def test_every_hard_and_soft_flag_projects_alone():
    for flag, verdict in {**HARD_FLAG_VERDICTS, **SOFT_FLAG_VERDICTS}.items():
        got, unknown = verdict_for_flags([flag], {})
        assert got == verdict, f"{flag} -> {got}, expected {verdict}"
        assert unknown == []


# --------------------------------------------------------------------------------------
# precedence
# --------------------------------------------------------------------------------------


def test_hard_beats_soft():
    verdict, unknown = verdict_for_flags(["tiny", "fragmented"], {})
    assert verdict == "empty"
    assert unknown == []


def test_within_tier_tie_break_follows_rubric_precedence():
    # both soft: weak_boundary (bleed) beats fragmented (split)
    assert verdict_for_flags(["fragmented", "weak_boundary"], {})[0] == "bleed"
    # both soft: no_residual_support (wrong-subject) beats fragmented (split)
    assert verdict_for_flags(["no_residual_support", "fragmented"], {})[0] == "wrong-subject"


# --------------------------------------------------------------------------------------
# anomalous_area sign dependence
# --------------------------------------------------------------------------------------


def test_anomalous_area_sign_dependence():
    assert verdict_for_flags(["anomalous_area"], {"area_log_z": "2.0"})[0] == "bleed"
    assert verdict_for_flags(["anomalous_area"], {"area_log_z": "-2.0"})[0] == "empty"
    assert verdict_for_flags(["anomalous_area"], {"area_log_z": "not-a-number"})[0] == "empty"
    assert verdict_for_flags(["anomalous_area"], {"area_log_z": ""})[0] == "empty"
    assert verdict_for_flags(["anomalous_area"], {})[0] == "empty"


# --------------------------------------------------------------------------------------
# unknown flags
# --------------------------------------------------------------------------------------


def test_unknown_flag_with_known_flag_lets_known_drive_verdict():
    fired = ["mystery_flag", "tiny"]
    verdict, unknown = verdict_for_flags(fired, {})
    assert verdict == "empty"  # known hard flag drives the verdict
    assert unknown == ["mystery_flag"]
    rationale = build_rationale(fired, {}, unknown)
    assert "unknown-flags: mystery_flag" in rationale


def test_unknown_flag_only_routes_to_split():
    fired = ["mystery_flag"]
    verdict, unknown = verdict_for_flags(fired, {})
    assert verdict == "split"
    assert unknown == ["mystery_flag"]
    rationale = build_rationale(fired, {}, unknown)
    assert "unknown-flags" in rationale


# --------------------------------------------------------------------------------------
# confidence
# --------------------------------------------------------------------------------------


def test_confidence_fixed_for_reject_and_auto_accept_buckets():
    assert confidence_for({"bucket": "reject"}) == 0.95
    assert confidence_for({"bucket": "auto_accept"}) == 0.8


def test_confidence_needs_review_monotone_and_bounded():
    lo = confidence_for({"bucket": "needs_review", "triage_score": 0.1})
    mid = confidence_for({"bucket": "needs_review", "triage_score": 1.0})
    hi = confidence_for({"bucket": "needs_review", "triage_score": 5.0})
    assert 0 < lo < mid < hi < 1
    for c in (lo, mid, hi):
        assert c == round(c, 3)


# --------------------------------------------------------------------------------------
# auto_accept / no flags
# --------------------------------------------------------------------------------------


def test_auto_accept_bucket_with_no_flags_is_ok():
    verdict, unknown = verdict_for_flags([], {})
    assert verdict == "ok"
    assert unknown == []
    out = project_row(_scored_row("v", 1, 0, flags="", bucket="auto_accept"), "heuristic-v1")
    assert out["verdict"] == "ok"
    assert out["confidence"] == 0.8
    assert out["result_type"] == "succeeded"


# --------------------------------------------------------------------------------------
# full-CSV run via the CLI (main(argv))
# --------------------------------------------------------------------------------------


def test_full_csv_run_via_main():
    with tempfile.TemporaryDirectory() as td:
        scores_csv = Path(td) / "mask_scores.csv"
        out_csv = Path(td) / "verdicts.csv"
        rows = [
            _scored_row("pnt_v1", 1, 0, flags="tiny;fragmented", bucket="reject",
                        triage_score=11.9),
            _scored_row("pnt_v1", 2, 0, flags="", bucket="auto_accept", triage_score=0.0),
            _missing_row("pnt_v1", 3),
        ]
        with open(scores_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MASK_COLUMNS)
            w.writeheader()
            w.writerows(rows)

        rc = main(["--scores", str(scores_csv), "--out", str(out_csv)])
        assert rc == 0

        with open(out_csv, newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == VERDICT_FIELDS
            got = list(reader)

        by_frame = {r["frame_idx"]: r for r in got}

        hard_row = by_frame["1"]
        assert hard_row["verdict"] == "empty"
        assert hard_row["flags"] == "tiny|fragmented"  # re-joined with "|"
        assert hard_row["result_type"] == "succeeded"

        ok_row = by_frame["2"]
        assert ok_row["verdict"] == "ok"
        assert ok_row["result_type"] == "succeeded"

        missing_row = by_frame["3"]
        assert missing_row["result_type"] == "errored"
        assert missing_row["verdict"] == ""
        assert missing_row["error"].startswith("frame_missing")


# --------------------------------------------------------------------------------------
# round-trip: adapter -> review queue -> corrections -> apply_corrections morph
# --------------------------------------------------------------------------------------


def test_roundtrip_adapter_to_queue_to_apply_morph():
    rows = [
        project_row(_scored_row("pnt_v1", 1, 0, flags="tiny"), "heuristic-v1"),          # empty
        project_row(_scored_row("pnt_v1", 2, 0, flags="weak_boundary"), "heuristic-v1"),  # bleed
        project_row(_scored_row("pnt_v1", 3, 0, flags="", bucket="auto_accept"),
                    "heuristic-v1"),                                                     # ok
        project_row(_missing_row("pnt_v1", 4), "heuristic-v1"),                          # errored
    ]

    # errored row excluded, and its blank instance_idx never reaches build_queue's int() sort
    queue_default = build_queue(rows)
    assert [r["verdict"] for r in queue_default] == ["empty", "bleed"]

    # include_ok=True: ok rows surface after every flagged (BAD_CLASSES) class
    queue = build_queue(rows, include_ok=True)
    assert [r["verdict"] for r in queue] == ["empty", "bleed", "ok"]

    empty_item = next(r for r in queue if r["verdict"] == "empty")
    bleed_item = next(r for r in queue if r["verdict"] == "bleed")

    with tempfile.TemporaryDirectory() as td:
        masks_dir, out_dir = Path(td) / "masks", Path(td) / "masks_fixed"

        broken = np.zeros((H, W), bool)
        broken[10:60, 10:60] = True
        broken[25:35, 25:35] = False       # hole
        broken[100:110, 140:150] = True    # straggler component
        clean = np.zeros((H, W), bool)
        clean[10:30, 10:30] = True

        save_instance_masks(masks_dir, "pnt_v1", int(empty_item["frame_idx"]),
                            [{"mask": broken, "center_xy": (35, 35),
                              "area_px": int(broken.sum())}])
        save_instance_masks(masks_dir, "pnt_v1", int(bleed_item["frame_idx"]),
                            [{"mask": clean, "center_xy": (20, 20),
                              "area_px": int(clean.sum())}])

        def _decision(item, action):
            return {f: "" for f in CORRECTION_FIELDS} | {
                "key": key_of(item), "video_name": item["video_name"],
                "frame_idx": item["frame_idx"], "instance_idx": item["instance_idx"],
                "source_verdict": item["verdict"], "action": action,
                "decided_at": "2026-08-03T00:00:00+00:00",
            }

        decisions = {
            key_of(empty_item): _decision(empty_item, "autofix"),
            key_of(bleed_item): _decision(bleed_item, "accept"),
        }
        # decisions are shaped exactly like review_server.py's CORRECTION_FIELDS rows
        for d in decisions.values():
            assert set(d) == set(CORRECTION_FIELDS)

        applied_rows = apply_morph(decisions, masks_dir, out_dir)
        by_key = {r["key"]: r for r in applied_rows}

        assert by_key[key_of(empty_item)]["source_verdict"] == "empty"
        assert by_key[key_of(empty_item)]["disposition"] == "corrected"
        assert Path(by_key[key_of(empty_item)]["out_path"]).exists()

        assert by_key[key_of(bleed_item)]["source_verdict"] == "bleed"
        assert by_key[key_of(bleed_item)]["disposition"] == "accepted_as_is"

        fixed = load_instance_masks(out_dir, "pnt_v1", int(empty_item["frame_idx"]))[0]["mask"]
        assert fixed[30, 30], "hole should be filled"
        assert not fixed[105, 145], "straggler should be dropped"


# --------------------------------------------------------------------------------------


def _run_all():
    mod = sys.modules[__name__]
    tests = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
