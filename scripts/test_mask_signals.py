#!/usr/bin/env python
"""Synthetic unit tests for the CPU mask-triage tier:
scripts/stations.py, scripts/banner.py, scripts/mask_signals.py, scripts/background.py.
Run directly:  python scripts/test_mask_signals.py  (also pytest-compatible).
No GPU / data / torch needed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.background import (
    build_background,
    foreground_residual,
    iou,
    largest_unmasked_blob_px,
    photometric_mode,
    residual_signals,
    unexplained_residual_frac,
)
from scripts.banner import DEFAULT_BANNER_ROWS, banner_top_or_default, crop_banner, detect_banner_top
from scripts.mask_signals import (
    border_contact_frac,
    boundary_gradient_ratio,
    connected_component_stats,
    hole_stats,
    mask_signals,
    shape_stats,
)
from scripts.stations import parse_frame_name, parse_mask_name, site_of, station_of
import scripts.sites
import scripts.stations

H, W = 120, 200


def _blank(value=0, dtype=bool):
    return np.full((H, W), value, dtype=dtype)


# --- stations --------------------------------------------------------------------------

def test_station_of_strips_only_the_clip_stem():
    # the camera folder itself contains underscores, so only the last token may go
    assert station_of("beauvois_T_16_16_vid_ref_Cam_184_DSCF0005") == "beauvois_T_16_16_vid_ref_Cam_184"
    assert station_of("mbnp_ds_nw_1_NW215_06200009") == "mbnp_ds_nw_1_NW215"


def test_station_of_survives_spaces_and_non_ascii():
    name = "pnt_pnt_p1_video_reference_Secteur_Taï_Djou 04_06060007"
    assert station_of(name) == "pnt_pnt_p1_video_reference_Secteur_Taï_Djou 04"


def test_station_of_without_underscore_is_identity():
    assert station_of("solo") == "solo"


def test_site_of():
    assert site_of("pnt_pnt_p1_video_reference_Secteur_Taï_Tai20_08240007") == "pnt"


def test_stations_site_of_is_sites_site_of():
    # scripts/stations.py re-exports scripts/sites.py's site_of rather than redefining it, so
    # existing callers (qc_annotations.py included) all resolve to the one implementation.
    assert scripts.stations.site_of is scripts.sites.site_of


def test_parse_names_roundtrip_and_reject():
    assert parse_frame_name("a_b_frame000048.png") == ("a_b", 48)
    assert parse_mask_name("a_b_frame000048_masks.json") == ("a_b", 48)
    # a mask filename is not a frame filename, and vice versa
    assert parse_frame_name("a_b_frame000048_masks.json") is None
    assert parse_mask_name("a_b_frame000048.png") is None
    # frame index must be the zero-padded 6-digit form the export writes
    assert parse_frame_name("a_b_frame48.png") is None


# --- banner ----------------------------------------------------------------------------

def _station_frames(n=10, banner_rows=20, clock_columns=6, seed=0):
    """A static scene with a constant bottom strip whose last few columns tick like a clock."""
    rng = np.random.default_rng(seed)
    scene = rng.integers(0, 255, size=(H, W), dtype=np.uint8)
    frames = []
    for i in range(n):
        frame = scene.copy()
        frame[H - banner_rows:] = 255                      # constant banner
        frame[H - banner_rows:, -clock_columns:] = i * 20  # the ticking clock
        frame[:H - banner_rows] = np.clip(
            scene[:H - banner_rows].astype(int) + rng.integers(-40, 40, size=(H - banner_rows, W)), 0, 255
        ).astype(np.uint8)
        frames.append(frame)
    return np.stack(frames)


def test_detect_banner_top_finds_the_constant_strip():
    assert detect_banner_top(_station_frames(banner_rows=20)) == H - 20


def test_detect_banner_top_ignores_the_ticking_clock():
    # a mean-over-columns statistic would be dragged above threshold by the changing digits
    assert detect_banner_top(_station_frames(banner_rows=20, clock_columns=25)) == H - 20


def test_detect_banner_top_needs_enough_frames():
    assert detect_banner_top(_station_frames(n=3)) is None


def test_detect_banner_top_rejects_implausible_strip():
    # a wholly static station has no scene variation, so "banner" would swallow the frame
    static = np.stack([np.full((H, W), 100, np.uint8)] * 10)
    assert detect_banner_top(static) is None


def test_detect_banner_top_accepts_colour_frames():
    frames = np.repeat(_station_frames(banner_rows=20)[..., None], 3, axis=-1)
    assert detect_banner_top(frames) == H - 20


def test_banner_top_or_default_falls_back():
    assert banner_top_or_default(None, H) == H - DEFAULT_BANNER_ROWS
    assert banner_top_or_default(_station_frames(n=2), H) == H - DEFAULT_BANNER_ROWS


def test_crop_banner_shape():
    assert crop_banner(np.zeros((H, W, 3)), H - 20).shape == (H - 20, W, 3)
    assert crop_banner(_blank(), H - 20).shape == (H - 20, W)


# --- geometry --------------------------------------------------------------------------

def test_connected_components_single_blob():
    mask = _blank()
    mask[10:40, 10:40] = True
    stats = connected_component_stats(mask)
    assert stats["n_components"] == 1
    assert stats["second_component_frac"] == 0.0


def test_connected_components_flags_a_comparable_second_blob():
    mask = _blank()
    mask[10:40, 10:40] = True   # 900 px
    mask[60:80, 60:80] = True   # 400 px -> 0.44 of the largest
    stats = connected_component_stats(mask)
    assert stats["n_components"] == 2
    assert 0.4 < stats["second_component_frac"] < 0.5


def test_hole_stats_counts_only_enclosed_background():
    mask = _blank()
    mask[10:50, 10:50] = True
    mask[20:30, 20:30] = False   # 100 px hole in a 1600 - 100 px mask
    stats = hole_stats(mask)
    assert stats["n_holes"] == 1
    assert abs(stats["hole_area_frac"] - 100 / 1500) < 1e-6


def test_hole_stats_ignores_the_surrounding_scene():
    mask = _blank()
    mask[10:50, 10:50] = True
    assert hole_stats(mask)["n_holes"] == 0


def test_shape_stats_square_is_compact_and_survives_erosion():
    mask = _blank()
    mask[10:50, 10:50] = True
    stats = shape_stats(mask)
    assert stats["compactness"] > 0.7
    assert stats["erosion_survival"] > 0.85


def test_shape_stats_sliver_is_not():
    mask = _blank()
    mask[60, 10:190] = True      # a 1 px filament
    stats = shape_stats(mask)
    assert stats["compactness"] < 0.05
    assert stats["erosion_survival"] == 0.0


def test_shape_stats_empty_mask_is_zero_not_nan():
    stats = shape_stats(_blank())
    assert stats == {"perimeter_px": 0.0, "compactness": 0.0, "erosion_survival": 0.0}


def test_border_contact_zero_when_interior():
    mask = _blank()
    mask[10:50, 10:50] = True
    assert border_contact_frac(mask) == 0.0


def test_border_contact_high_when_truncated():
    mask = _blank()
    mask[:40, :40] = True        # jammed into the top-left corner
    assert border_contact_frac(mask) > 0.4


def test_boundary_gradient_ratio_rewards_a_real_edge():
    gray = np.zeros((H, W), np.float32)
    gray[10:50, 10:50] = 255.0   # a genuine intensity step
    on_edge = _blank()
    on_edge[10:50, 10:50] = True
    off_edge = _blank()
    off_edge[60:100, 60:100] = True   # same shape, drawn across flat background
    assert boundary_gradient_ratio(on_edge, gray) > boundary_gradient_ratio(off_edge, gray)
    assert boundary_gradient_ratio(off_edge, gray) < 1.0


def test_mask_signals_returns_every_declared_column():
    from scripts.mask_signals import SIGNAL_COLUMNS
    mask = _blank()
    mask[10:50, 10:50] = True
    signals = mask_signals(mask, np.zeros((H, W), np.uint8), banner_overlap_frac=0.25)
    assert set(SIGNAL_COLUMNS) == set(signals)
    assert signals["area_px"] == 1600
    assert abs(signals["area_frac"] - 1600 / (H * W)) < 1e-9
    assert signals["banner_overlap_frac"] == 0.25


# --- background ------------------------------------------------------------------------

def test_photometric_mode_day_vs_mono_ir():
    rng = np.random.default_rng(1)
    grey = rng.integers(60, 180, size=(H, W), dtype=np.uint8)
    mono = np.repeat(grey[..., None], 3, axis=-1)          # equal channels -> no saturation
    assert photometric_mode(mono) == "ir_mono"

    day = mono.copy()
    day[..., 1] = np.clip(day[..., 1].astype(int) + 60, 0, 255)   # green-dominant foliage
    assert photometric_mode(day) == "day"


def test_photometric_mode_magenta_ir():
    magenta = np.zeros((H, W, 3), np.uint8)
    magenta[..., 0], magenta[..., 1], magenta[..., 2] = 150, 30, 190   # BGR: R and B high, G low
    assert photometric_mode(magenta) == "ir_magenta"


def test_build_background_recovers_the_static_scene():
    rng = np.random.default_rng(2)
    scene = rng.integers(40, 200, size=(H, W)).astype(np.float32)
    frames = []
    for i in range(11):
        frame = scene.copy()
        # a subject that crosses the scene, never lingering over any pixel for more than
        # two frames -- which is exactly the condition a temporal median needs
        frame[10 + 8 * i:30 + 8 * i, 20:40] = 255.0
        frames.append(frame)
    background = build_background(np.stack(frames))
    # the median sees background at every pixel more often than it sees the subject
    assert np.abs(background - scene).mean() < 1.0


def test_build_background_absorbs_exposure_drift():
    rng = np.random.default_rng(3)
    scene = rng.integers(40, 200, size=(H, W)).astype(np.float32)
    frames = np.stack([scene * gain for gain in np.linspace(0.7, 1.3, 11)])
    background = build_background(frames)
    # gain normalisation puts every frame on a common level before the median
    assert np.corrcoef(background.ravel(), scene.ravel())[0, 1] > 0.99


def test_build_background_needs_enough_frames():
    assert build_background(np.zeros((3, H, W), np.float32)) is None


def test_foreground_residual_finds_the_subject_only():
    rng = np.random.default_rng(4)
    scene = rng.integers(40, 200, size=(H, W)).astype(np.float32)
    frame = scene.copy()
    frame[30:70, 30:70] = 255.0
    residual = foreground_residual(frame, scene)
    truth = np.zeros((H, W), bool)
    truth[30:70, 30:70] = True
    assert iou(residual, truth) > 0.8


def test_residual_signals_separate_recall_from_precision():
    residual = _blank()
    residual[20:60, 20:60] = True           # 1600 px of real change
    half = _blank()
    half[20:40, 20:60] = True               # mask covers half the change, all of it real
    signals = residual_signals(half, residual)
    assert abs(signals["residual_recall"] - 1.0) < 1e-9
    assert abs(signals["residual_precision"] - 0.5) < 1e-9

    spilled = _blank()
    spilled[20:100, 20:60] = True           # mask spills onto ground that never moved
    signals = residual_signals(spilled, residual)
    assert abs(signals["residual_recall"] - 0.5) < 1e-9
    assert abs(signals["residual_precision"] - 1.0) < 1e-9


def test_largest_unmasked_blob_ignores_the_halo_but_catches_a_missed_subject():
    # a correct mask sits inside a slightly larger residual: the leftover ring is an
    # artefact of the threshold, not a missed subject, and must not register
    residual = _blank()
    residual[20:62, 20:62] = True
    mask = _blank()
    mask[22:60, 22:60] = True
    assert largest_unmasked_blob_px([mask], residual) == 0

    # a second subject of comparable size is a coherent blob and must register
    residual[70:100, 70:100] = True
    blob = largest_unmasked_blob_px([mask], residual)
    assert 600 < blob < 900


def test_unexplained_residual_frac_is_the_exhaustivity_signal():
    residual = _blank()
    residual[20:40, 20:40] = True           # subject A
    residual[60:80, 60:80] = True           # subject B, same size
    only_a = _blank()
    only_a[20:40, 20:40] = True
    assert abs(unexplained_residual_frac([only_a], residual) - 0.5) < 1e-9
    assert unexplained_residual_frac([residual], residual) == 0.0
    assert unexplained_residual_frac([], _blank()) == 0.0


def test_iou_of_two_empty_masks_is_zero_not_nan():
    assert iou(_blank(), _blank()) == 0.0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
