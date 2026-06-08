"""Resolve annotation video_name values (e.g. "beauvois_T_16_16_vid_ref_Cam_184_DSCF0005")
to actual video file paths under data/, via the ori<->anno mapping in list_reference_videos.xlsx.
"""
from __future__ import annotations

from pathlib import Path

import openpyxl


def load_anno_to_path(xlsx_path: Path | str, data_dir: Path | str) -> dict[str, Path]:
    """Build a {anno_name: resolved_video_path} mapping from list_reference_videos.xlsx.

    `anno` matches the CSV's `video_name`; `ori` is the path relative to `data_dir`.
    """
    data_dir = Path(data_dir)
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb["Sheet1"]

    mapping: dict[str, Path] = {}
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    assert header[:2] == ("ori", "anno"), f"unexpected xlsx header: {header}"

    for ori, anno, *_ in rows:
        if anno is None:
            continue
        mapping[anno] = data_dir / ori

    return mapping


def resolve_video_path(video_name: str, anno_to_path: dict[str, Path]) -> Path:
    """Look up `video_name` (CSV column) and confirm the resolved file exists on disk."""
    try:
        path = anno_to_path[video_name]
    except KeyError:
        raise KeyError(f"video_name {video_name!r} not found in list_reference_videos.xlsx") from None

    if not path.exists():
        raise FileNotFoundError(f"resolved path for {video_name!r} does not exist: {path}")

    return path
