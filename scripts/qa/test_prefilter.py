#!/usr/bin/env python
"""Synthetic unit tests for scripts/qa/prefilter.py.
Run directly:  python scripts/qa/test_prefilter.py  (also pytest-compatible).
No GPU / data / torch / network needed -- every mask here is drawn with numpy."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.qa.prefilter import (
    add_clip_flags,
    bbox_of,
    border_touch_frac,
    count_components,
    decide,
    fill_ratio,
    flag_instance,
    normalised_log_area,
    robust_z,
    status_group,
    summarise,
)

H, W = 200, 300


def _blank() -> np.ndarray:
    return np.zeros((H, W), dtype=bool)


def _person(cx: int = 150, cy: int = 100, half_w: int = 12, half_h: int = 45) -> np.ndarray:
    """A plausible upright subject: tall, narrow, solid, away from the frame edge."""
    m = _blank()
    m[cy - half_h:cy + half_h, cx - half_w:cx + half_w] = True
    return m


def _thresholds(**overrides) -> argparse.Namespace:
    """The prefilter's CLI defaults, so the tests exercise what actually ships."""
    defaults = dict(
        min_area_px=200, min_area_frac=0.0002, max_area_frac=0.60,
        min_fill_ratio=0.18, max_border_touch_frac=0.08,
        min_component_frac=0.05, min_component_px=50,
        max_area_z=3.5, max_area_jump=1.10, min_clip_points=4,
        min_area_dispersion=0.05,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _metrics(mask: np.ndarray, n_instances: int = 1, args=None) -> dict:
    args = args or _thresholds()
    area = int(mask.sum())
    return {
        "area_px": area,
        "area_frac": area / (H * W),
        "fill_ratio": fill_ratio(mask),
        "n_components": count_components(mask, args.min_component_frac, args.min_component_px),
        "n_instances": n_instances,
        "border_touch_frac": border_touch_frac(mask),
    }


# --------------------------------------------------------------------------------------
# status prefix matching
# --------------------------------------------------------------------------------------

def test_status_group_matches_on_prefix_not_equality():
    """sam_error / depth_error / video_error all carry a variable exception suffix."""
    assert status_group("processed") == "examine"
    assert status_group("depth_error: no depth map (inference failed for this video)") == "examine"
    assert status_group("empty_mask") == "empty"
    assert status_group("sam_error: CUDA out of memory") == "pipeline_failure"
    assert status_group("video_error: cannot open video: /data/x.AVI") == "pipeline_failure"
    assert status_group("frame_decode_error") == "pipeline_failure"
    assert status_group("video_missing") == "pipeline_failure"


def test_status_group_treats_depth_error_as_examinable():
    """A frame with masks but failed depth still has a mask worth QA-ing -- depth is irrelevant
    to mask quality, and dropping these would silently shrink the denominator."""
    assert status_group("depth_error: something") == "examine"


def test_status_group_handles_missing_and_unknown():
    assert status_group(None) == "unknown"
    assert status_group("") == "unknown"
    assert status_group("brand_new_status") == "unknown"


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------

def test_bbox_of_empty_mask_is_none():
    assert bbox_of(_blank()) is None


def test_bbox_of_rectangle():
    m = _blank()
    m[10:20, 30:45] = True
    assert bbox_of(m) == (30, 10, 44, 19)


def test_fill_ratio_of_a_solid_rectangle_is_one():
    assert abs(fill_ratio(_person()) - 1.0) < 1e-9


def _two_subjects(gap: str = "wide") -> np.ndarray:
    """One mask spanning two separated subjects -- the signature of a bleed or a split."""
    m = _blank()
    if gap == "wide":
        m[60:140, 20:38] = True    # subject A, near the left edge
        m[60:140, 275:293] = True  # subject B, near the right edge
    else:
        m[60:140, 40:64] = True
        m[60:140, 240:264] = True
    return m


def test_fill_ratio_drops_when_a_mask_spans_two_subjects():
    """A bbox stretched across a mostly-empty gap has a far lower fill than one subject."""
    assert fill_ratio(_two_subjects()) < 0.18
    assert fill_ratio(_two_subjects()) < 0.3 * fill_ratio(_person())


def test_fill_ratio_of_empty_mask_is_zero():
    assert fill_ratio(_blank()) == 0.0


def test_border_touch_frac_is_zero_for_an_interior_mask():
    assert border_touch_frac(_person()) == 0.0


def test_border_touch_frac_detects_edge_contact():
    m = _blank()
    m[:, 0:20] = True  # full-height strip flush against the left edge
    assert border_touch_frac(m) > 0.08


def test_border_touch_frac_of_full_frame_is_one():
    assert abs(border_touch_frac(np.ones((H, W), dtype=bool)) - 1.0) < 1e-9


def test_count_components_single_blob():
    assert count_components(_person()) == 1


def test_count_components_two_real_blobs():
    assert count_components(_two_subjects()) == 2


def test_count_components_ignores_speckle():
    """SAM-3 masks routinely carry stray pixels; counting them as splits would send everything
    to vision and destroy the pre-filter's entire purpose."""
    m = _person()
    m[5, 5] = True          # 1px speck
    m[8:10, 8:10] = True    # 4px speck
    assert count_components(m) == 1


def test_count_components_of_empty_mask_is_zero():
    assert count_components(_blank()) == 0


def test_count_components_never_reports_zero_for_a_nonempty_mask():
    tiny = _blank()
    tiny[0:3, 0:3] = True  # smaller than min_component_px
    assert count_components(tiny) == 1


# --------------------------------------------------------------------------------------
# distance normalisation and robust dispersion
# --------------------------------------------------------------------------------------

def test_normalised_log_area_is_flat_as_the_subject_walks():
    """The property the whole z-score test rests on: a subject of fixed real size at 4m and 8m
    subtends 4x the pixel area, and the normalisation cancels exactly that."""
    near = normalised_log_area(area_px=4000, distance=4.0)
    far = normalised_log_area(area_px=1000, distance=8.0)
    assert abs(near - far) < 1e-9


def test_normalised_log_area_returns_none_without_a_usable_distance():
    assert normalised_log_area(1000, None) is None
    assert normalised_log_area(1000, float("nan")) is None
    assert normalised_log_area(1000, 0.0) is None
    assert normalised_log_area(1000, -3.0) is None
    assert normalised_log_area(0, 5.0) is None


def test_robust_z_flags_a_single_outlier():
    values = [10.0, 10.1, 9.9, 10.05, 25.0]
    zs = robust_z(values)
    assert abs(zs[-1]) > 3.5
    assert all(abs(z) < 3.5 for z in zs[:-1])


def test_robust_z_beats_mean_std_masking():
    """Why MAD: with n~6 a single gross outlier inflates the standard deviation enough to keep
    its own conventional z-score under 3.5, hiding the very mask we are looking for."""
    values = [10.0, 10.1, 9.9, 10.05, 10.0, 60.0]
    arr = np.asarray(values)
    classic = abs((arr[-1] - arr.mean()) / arr.std(ddof=0))
    assert classic < 3.5, "precondition: mean/std should fail to flag this"
    assert abs(robust_z(values)[-1]) > 3.5


def test_robust_z_of_identical_values_is_all_zero():
    assert robust_z([5.0] * 6) == [0.0] * 6


def test_robust_z_of_empty_list():
    assert robust_z([]) == []


def test_robust_z_min_mad_floor_suppresses_a_degenerate_scale():
    """Without the floor, a clip whose masks barely vary (a subject standing still) divides by a
    near-zero MAD and every trivial wobble scores as a huge outlier -- flagging the steadiest,
    healthiest clips in the corpus."""
    values = [10.000, 10.001, 9.999, 10.0005, 10.0002, 10.02]
    assert max(abs(z) for z in robust_z(values)) > 3.5, "precondition: unfloored MAD explodes"
    assert robust_z(values, min_mad=0.05) == [0.0] * len(values)


def test_robust_z_min_mad_floor_still_scores_real_dispersion():
    values = [10.0, 10.4, 9.6, 10.2, 9.8, 25.0]
    zs = robust_z(values, min_mad=0.05)
    assert abs(zs[-1]) > 3.5
    assert all(abs(z) < 3.5 for z in zs[:-1])


# --------------------------------------------------------------------------------------
# flagging and dispositions
# --------------------------------------------------------------------------------------

def test_clean_singleton_raises_no_flags_and_passes():
    flags = flag_instance(_metrics(_person()), _thresholds())
    assert flags == []
    assert decide("examine", flags) == ("pass", "clean_singleton")


def test_tiny_mask_is_flagged():
    m = _blank()
    m[100:104, 150:154] = True  # 16 px
    assert "tiny_area" in flag_instance(_metrics(m), _thresholds())


def test_huge_mask_is_flagged():
    m = _blank()
    m[0:180, 0:280] = True  # 84% of the frame
    assert "huge_area" in flag_instance(_metrics(m), _thresholds())


def test_split_mask_raises_low_fill_and_multi_component():
    flags = flag_instance(_metrics(_two_subjects()), _thresholds())
    assert "multi_component" in flags
    assert "low_fill" in flags
    assert decide("examine", flags) == ("vision", "flagged")


def test_a_narrowly_split_mask_still_reaches_vision_via_multi_component():
    """The flags are complementary on purpose: a split whose bbox fill stays above the
    threshold is still caught, so tuning min_fill_ratio conservatively costs no recall here."""
    flags = flag_instance(_metrics(_two_subjects(gap="narrow")), _thresholds())
    assert "low_fill" not in flags, "precondition: this geometry clears the fill threshold"
    assert "multi_component" in flags
    assert decide("examine", flags) == ("vision", "flagged")


def test_multiple_instances_routes_to_vision():
    flags = flag_instance(_metrics(_person(), n_instances=2), _thresholds())
    assert "multi_instance" in flags
    assert decide("examine", flags)[0] == "vision"


def test_empty_mask_auto_fails_without_a_vision_call():
    """The single most valuable QA category, and it costs nothing to detect."""
    assert decide("empty", []) == ("fail", "empty")


def test_pipeline_failures_are_excluded_not_counted_as_defects():
    assert decide("pipeline_failure", []) == ("excluded", "pipeline_failure")
    assert decide("unknown", []) == ("excluded", "unknown_status")


# --------------------------------------------------------------------------------------
# cross-frame clip flags
# --------------------------------------------------------------------------------------

def _rec(frame_idx: int, area_px: int, distance: float, instance_idx: int = 0) -> dict:
    return {
        "video_name": "mafou_e1_cam_v", "frame_idx": frame_idx, "instance_idx": instance_idx,
        "area_px": area_px, "distance": distance, "flags": [], "disposition": "pass",
        "area_z": "", "area_jump": "",
    }


# A subject walking 4m -> 9m, with realistic per-frame segmentation noise on the mask area so
# the clip has genuine dispersion to score against (an exactly-1/d^2 clip would sit under the
# min_area_dispersion floor and trivially flag nothing, which would not test the normalisation).
_WALK = [(4.0, 1.00), (5.0, 0.82), (6.0, 1.20), (7.0, 0.90), (8.0, 1.15), (9.0, 0.86)]


def _walk_records() -> list[dict]:
    return [_rec(i, int(64000 / (d ** 2) * noise), d)
            for i, (d, noise) in enumerate(_WALK)]


def test_add_clip_flags_leaves_a_consistent_walk_alone():
    """The regression this normalisation exists to prevent: a subject walking 4m -> 9m changes
    mask area by ~5x, and a naive per-video z-score on raw area would flag the whole clip."""
    records = _walk_records()
    add_clip_flags(records, _thresholds())
    assert all(r["flags"] == [] for r in records), [r["flags"] for r in records]


def test_the_walk_fixture_really_does_have_dispersion_to_score():
    """Guards the test above from passing for the wrong reason (dispersion under the floor)."""
    values = [normalised_log_area(r["area_px"], r["distance"]) for r in _walk_records()]
    zs = robust_z(values, min_mad=0.05)
    assert any(z != 0.0 for z in zs), "fixture is degenerate; the walk test would be vacuous"


def test_add_clip_flags_catches_a_mask_that_does_not_fit_the_walk():
    records = _walk_records()
    records.append(_rec(6, 90000, 9.0))  # far away but huge: a bleed onto the background
    add_clip_flags(records, _thresholds())
    assert "area_outlier" in records[-1]["flags"]
    assert all("area_outlier" not in r["flags"] for r in records[:-1])


def test_add_clip_flags_does_not_flag_a_steady_clip():
    """A subject standing still: near-identical areas, no real dispersion. Every frame here
    would be an 'outlier' without the MAD floor."""
    records = [_rec(i, a, 5.0) for i, a in enumerate([3000, 3001, 2999, 3000, 3002, 2998])]
    add_clip_flags(records, _thresholds())
    assert all("area_outlier" not in r["flags"] for r in records)


def test_add_clip_flags_skips_the_z_test_on_short_clips():
    """A dispersion estimate from three points is noise, not a signal."""
    records = [_rec(i, a, 5.0) for i, a in enumerate([3000, 3100, 30000])]
    add_clip_flags(records, _thresholds())
    assert all("area_outlier" not in r["flags"] for r in records)


def test_add_clip_flags_skips_rows_without_a_distance():
    """The unannotated stratum has no ground-truth distance; it must not crash or be flagged."""
    records = [_rec(i, 3000, float("nan")) for i in range(6)]
    add_clip_flags(records, _thresholds())
    assert all("area_outlier" not in r["flags"] for r in records)


def test_add_clip_flags_detects_an_isolated_area_jump():
    records = [_rec(i, a, 5.0) for i, a in enumerate([3000, 3050, 40000, 3020, 2990])]
    add_clip_flags(records, _thresholds())
    assert "area_jump" in records[2]["flags"]


def test_add_clip_flags_does_not_fire_on_a_monotonic_approach():
    """min() of the two neighbour gaps, not max(): a legitimate step toward the camera
    disagrees with the frame it moved away from but agrees with the one it moved toward."""
    records = [_rec(i, a, 5.0) for i, a in enumerate([1000, 2000, 4000, 8000, 16000])]
    add_clip_flags(records, _thresholds())
    assert all("area_jump" not in r["flags"] for r in records)


def test_add_clip_flags_keeps_instance_tracks_separate():
    """Interleaving a big instance 0 and a small instance 1 must not manufacture jumps."""
    records = []
    for i in range(5):
        records.append(_rec(i, 8000, 5.0, instance_idx=0))
        records.append(_rec(i, 800, 5.0, instance_idx=1))
    add_clip_flags(records, _thresholds())
    assert all(r["flags"] == [] for r in records)


def test_add_clip_flags_ignores_excluded_rows():
    records = [_rec(i, 3000, 5.0) for i in range(5)]
    records.append({**_rec(9, 0, 5.0), "disposition": "excluded"})
    add_clip_flags(records, _thresholds())
    assert records[-1]["flags"] == []


# --------------------------------------------------------------------------------------
# summarise / f_v
# --------------------------------------------------------------------------------------

def _summary_rec(disposition: str, site: str, weight: str, stratum: str = "annotated") -> dict:
    return {
        "disposition": disposition, "site": site, "inclusion_weight": weight,
        "stratum": stratum, "flags": [], "prefilter_class": "x",
    }


def test_summarise_excludes_pipeline_failures_from_f_v():
    records = [
        _summary_rec("vision", "pnt", "1.0"),
        _summary_rec("pass", "pnt", "1.0"),
        _summary_rec("excluded", "pnt", "1.0"),  # must not enter the denominator
    ]
    n_vision, n, frac = summarise(records)["overall"]
    assert (n_vision, n) == (1, 2)
    assert abs(frac - 0.5) < 1e-9


def test_summarise_separates_the_unannotated_stratum():
    records = [
        _summary_rec("vision", "pnt", "1.0"),
        _summary_rec("pass", "pnt", "1.0"),
        _summary_rec("vision", "pnt", "", stratum="unannotated"),
        _summary_rec("vision", "pnt", "", stratum="unannotated"),
    ]
    summary = summarise(records)
    assert summary["overall"][:2] == (1, 2)          # headline: annotated only
    assert summary["unannotated"][:2] == (2, 2)      # reported, never blended in


def test_summarise_weighted_f_v_differs_from_unweighted():
    """Equal-per-site sampling over-represents small sites, so the corpus-weighted f_v is the
    one that may legitimately be multiplied into the 1M estimate."""
    records = (
        [_summary_rec("vision", "big", "3.0") for _ in range(1)]
        + [_summary_rec("pass", "big", "3.0") for _ in range(3)]
        + [_summary_rec("vision", "small", "0.2") for _ in range(3)]
        + [_summary_rec("pass", "small", "0.2") for _ in range(1)]
    )
    summary = summarise(records)
    _, _, unweighted = summary["overall"]
    weighted = summary["weighted_f_v"]
    assert abs(unweighted - 0.5) < 1e-9
    # big site (weight 3.0) is mostly clean, so weighting pulls f_v down
    assert weighted < unweighted
    assert abs(weighted - (3.0 * 1 + 0.2 * 3) / (3.0 * 4 + 0.2 * 4)) < 1e-9


def test_summarise_handles_an_all_excluded_input():
    summary = summarise([_summary_rec("excluded", "pnt", "1.0")])
    assert summary["overall"] == (0, 0, 0.0)
    assert summary["weighted_f_v"] is None


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nall {len(fns)} prefilter tests passed")


if __name__ == "__main__":
    _run_all()
