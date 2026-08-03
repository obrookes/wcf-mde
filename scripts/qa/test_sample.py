#!/usr/bin/env python
"""Synthetic unit tests for scripts/qa/sample.py.
Run directly:  python scripts/qa/test_sample.py  (also pytest-compatible).
No GPU / data / torch / network needed."""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.qa.sample import (
    allocate_equally,
    draw_annotated,
    draw_unannotated,
    inclusion_weights,
    load_annotations,
)


# --------------------------------------------------------------------------------------
# allocation
# --------------------------------------------------------------------------------------

def test_allocate_equally_splits_evenly_when_all_sites_are_large():
    alloc = allocate_equally({"a": 500, "b": 500, "c": 500, "d": 500}, 1000)
    assert sum(alloc.values()) == 1000
    assert set(alloc.values()) == {250}


def test_allocate_equally_redistributes_from_exhausted_sites():
    # 'small' can only supply 10; the other three must absorb the remaining 90
    alloc = allocate_equally({"small": 10, "a": 500, "b": 500, "c": 500}, 400)
    assert sum(alloc.values()) == 400
    assert alloc["small"] == 10
    assert alloc["a"] == alloc["b"] == alloc["c"] == 130


def test_allocate_equally_never_exceeds_availability():
    available = {"a": 3, "b": 7, "c": 1}
    alloc = allocate_equally(available, 1000)
    assert alloc == available  # capped: only 11 rows exist in total
    assert sum(alloc.values()) == 11


def test_allocate_equally_handles_more_sites_than_slots():
    alloc = allocate_equally({"a": 100, "b": 100, "c": 100, "d": 100}, 3)
    assert sum(alloc.values()) == 3
    assert all(v <= 1 for v in alloc.values())


def test_allocate_equally_is_deterministic():
    available = {"pnt": 6017, "pss": 2353, "fello": 1453, "kora": 934,
                 "mafou": 710, "mbnp": 701, "beauvois": 290}
    first = allocate_equally(available, 1000)
    assert first == allocate_equally(available, 1000)
    assert sum(first.values()) == 1000
    # beauvois (290) and every other site can supply 1000/7 = 142, so this is a clean split
    assert set(first.values()) <= {142, 143}


def test_allocate_equally_zero_target():
    assert allocate_equally({"a": 10}, 0) == {"a": 0}


def test_allocate_equally_empty_sites():
    assert allocate_equally({}, 100) == {}


# --------------------------------------------------------------------------------------
# weighting
# --------------------------------------------------------------------------------------

def test_inclusion_weights_are_one_under_proportional_sampling():
    corpus = {"a": 900, "b": 100}
    sample = {"a": 90, "b": 10}  # exactly proportional
    weights = inclusion_weights(corpus, sample)
    assert abs(weights["a"] - 1.0) < 1e-9
    assert abs(weights["b"] - 1.0) < 1e-9


def test_inclusion_weights_downweight_oversampled_small_sites():
    corpus = {"big": 9000, "small": 1000}
    sample = {"big": 50, "small": 50}  # equal-per-site: 'small' is 9x over-represented
    weights = inclusion_weights(corpus, sample)
    assert weights["big"] > 1.0 > weights["small"]
    assert abs(weights["big"] / weights["small"] - 9.0) < 1e-9


def test_inclusion_weights_sum_to_the_sample_size():
    """So a weighted mean over the sample uses the same denominator as an unweighted one."""
    corpus = {"pnt": 6017, "pss": 2353, "fello": 1453, "kora": 934,
              "mafou": 710, "mbnp": 701, "beauvois": 290}
    sample = allocate_equally(corpus, 1000)
    weights = inclusion_weights(corpus, sample)
    total = sum(weights[site] * n for site, n in sample.items())
    assert abs(total - sum(sample.values())) < 1e-6


def test_weighted_rate_recovers_the_corpus_rate():
    """The property the 1M extrapolation depends on: equal-per-site sampling plus these weights
    reproduces the corpus-wide rate, where the unweighted sample mean would not."""
    corpus = {"big": 9000, "small": 1000}
    bad_rate = {"big": 0.10, "small": 0.50}
    sample = {"big": 100, "small": 100}
    weights = inclusion_weights(corpus, sample)

    unweighted = sum(bad_rate[s] * n for s, n in sample.items()) / sum(sample.values())
    weighted = (sum(bad_rate[s] * n * weights[s] for s, n in sample.items())
                / sum(n * weights[s] for s, n in sample.items()))
    true_rate = sum(bad_rate[s] * n for s, n in corpus.items()) / sum(corpus.values())

    assert abs(unweighted - 0.30) < 1e-9      # naive mean: badly wrong
    assert abs(true_rate - 0.14) < 1e-9
    assert abs(weighted - true_rate) < 1e-9   # weighted mean: recovers the corpus rate


# --------------------------------------------------------------------------------------
# loading and drawing
# --------------------------------------------------------------------------------------

def _write_annotations(path: Path, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "frame_idx", "frame_timestamp", "distance"])
        w.writerows(rows)


def test_load_annotations_drops_blank_frame_idx():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "anno.csv"
        _write_annotations(path, [
            ("mafou_e1_c_v", 50, 2.0, 4.0),
            ("mafou_e1_c_v", "", 3.0, 5.0),     # video missing from disk
            ("mafou_e1_c_v", "<NA>", 4.0, 6.0),  # pandas Int64 NA spelling
        ])
        rows = load_annotations(path)
        assert len(rows) == 1
        assert rows[0]["frame_idx"] == 50 and isinstance(rows[0]["frame_idx"], int)
        assert rows[0]["distance"] == 4.0


def test_draw_annotated_is_reproducible_and_carries_weights():
    rows = [
        {"video_name": f"{site}_e1_cam_v{i // 5}", "frame_idx": i,
         "frame_timestamp": i / 25.0, "distance": 5.0}
        for site in ("mafou", "pnt") for i in range(200)
    ]
    first = draw_annotated(rows, 40, np.random.default_rng(7))
    second = draw_annotated(rows, 40, np.random.default_rng(7))
    assert [(r["video_name"], r["frame_idx"]) for r in first] == \
           [(r["video_name"], r["frame_idx"]) for r in second]
    assert len(first) == 40
    assert {r["site"] for r in first} == {"mafou", "pnt"}
    assert all(r["stratum"] == "annotated" for r in first)
    assert all(r["inclusion_weight"] > 0 for r in first)


def test_draw_annotated_never_repeats_a_frame():
    rows = [{"video_name": "mafou_e1_cam_v", "frame_idx": i,
             "frame_timestamp": i / 25.0, "distance": 5.0} for i in range(50)]
    drawn = draw_annotated(rows, 50, np.random.default_rng(0))
    keys = [(r["video_name"], r["frame_idx"]) for r in drawn]
    assert len(keys) == len(set(keys)) == 50


def test_draw_unannotated_avoids_annotated_indices_and_marks_nan_distance():
    annotated = [{"video_name": "mafou_e1_cam_v", "frame_idx": i} for i in range(0, 100, 10)]
    all_rows = [{"video_name": "mafou_e1_cam_v", "frame_idx": i} for i in range(0, 100, 10)]
    fps_table = {"mafou_e1_cam_v": (25.0, 20.0)}  # 500 frames
    drawn = draw_unannotated(annotated, all_rows, fps_table, 30, np.random.default_rng(3))

    assert len(drawn) == 30
    taken = {r["frame_idx"] for r in drawn}
    assert taken.isdisjoint({i for i in range(0, 100, 10)})
    assert len(taken) == 30  # no duplicates
    assert all(r["distance"] == "nan" for r in drawn)
    assert all(r["inclusion_weight"] == "" for r in drawn)
    assert all(r["stratum"] == "unannotated" for r in drawn)
    # timestamps must be derivable back to the frame index, since load_rows reads both
    assert all(abs(r["frame_timestamp"] * 25.0 - r["frame_idx"]) < 1e-3 for r in drawn)


def test_draw_unannotated_stays_within_video_length():
    annotated = [{"video_name": "v", "frame_idx": 0}]
    fps_table = {"v": (25.0, 4.0)}  # 100 frames
    drawn = draw_unannotated(annotated, annotated, fps_table, 20, np.random.default_rng(1))
    assert drawn, "expected some draws from a 100-frame video"
    assert all(0 <= r["frame_idx"] < 99 for r in drawn)


def test_draw_unannotated_skips_cleanly_without_an_fps_table():
    annotated = [{"video_name": "v", "frame_idx": 0}]
    assert draw_unannotated(annotated, annotated, {}, 10, np.random.default_rng(0)) == []
    assert draw_unannotated(annotated, annotated, {"v": (25.0, 10.0)}, 0,
                            np.random.default_rng(0)) == []


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nall {len(fns)} sample tests passed")


if __name__ == "__main__":
    _run_all()
