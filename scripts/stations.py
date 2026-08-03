"""Parsing and grouping keys for the exported frame/mask dataset.

The export lays artifacts out flat, keyed on the flattened annotation `video_name`:

    frames/{video_name}_frame{idx:06d}.png
    masks/{video_name}_frame{idx:06d}_masks.json      (scripts/masks.py)
    depth_maps/{video_name}_frame{idx:06d}_calib.npy

`video_name` is the video's path under data/ with "/" replaced by "_" and the extension
dropped, so the last underscore-separated token is always the clip's own stem
(DSCF0007, 06200009, ...) and everything before it is the camera-reference folder --
i.e. one physical, static camera deployment. That prefix is the **station**, and it is
the grouping key for every signal that needs several views of the same fixed scene
(banner detection, background modelling, per-scene area priors).

Splitting on "_" is only safe from the right: the site/trial/camera components contain
underscores themselves ("beauvois_T_16_16_vid_ref_Cam_184") and some contain spaces and
non-ASCII ("...Secteur_Taï_Tai20", "...Djou 04"), so no left-to-right parse is possible
without data/list_reference_videos.xlsx.
"""
from __future__ import annotations

import re

FRAME_RE = re.compile(r"^(?P<video_name>.+)_frame(?P<frame_idx>\d{6})\.(?:png|jpg|jpeg)$")
MASK_RE = re.compile(r"^(?P<video_name>.+)_frame(?P<frame_idx>\d{6})_masks\.json$")
DEPTH_RE = re.compile(r"^(?P<video_name>.+)_frame(?P<frame_idx>\d{6})_calib\.npy$")


def _parse(pattern: re.Pattern, name: str) -> tuple[str, int] | None:
    m = pattern.match(name)
    if m is None:
        return None
    return m["video_name"], int(m["frame_idx"])


def parse_frame_name(name: str) -> tuple[str, int] | None:
    """"foo_frame000048.png" -> ("foo", 48); None if it isn't a frame filename."""
    return _parse(FRAME_RE, name)


def parse_mask_name(name: str) -> tuple[str, int] | None:
    return _parse(MASK_RE, name)


def parse_depth_name(name: str) -> tuple[str, int] | None:
    return _parse(DEPTH_RE, name)


def site_of(video_name: str) -> str:
    """Leading path component: mafou, pss, pnt, kora, beauvois, mbnp, fello.

    Mirrors scripts/qc_annotations.py::site_of; duplicated rather than imported so this
    module stays free of the pandas dependency.
    """
    return video_name.split("_", 1)[0]


def station_of(video_name: str) -> str:
    """Camera-reference folder -- one static camera deployment -- by dropping the clip stem.

    "beauvois_T_16_16_vid_ref_Cam_184_DSCF0005" -> "beauvois_T_16_16_vid_ref_Cam_184"
    "mbnp_ds_nw_1_NW215_06200009"               -> "mbnp_ds_nw_1_NW215"
    """
    head, sep, _tail = video_name.rpartition("_")
    return head if sep else video_name
