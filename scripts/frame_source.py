"""Extract specific frames from a video by sequential decode.

Mirrors the pattern used in Unmarked-Anything-export-distances'
apps/camera_trap/cli/dap3_cli.py (sequential cap.read() with a running counter rather
than CAP_PROP_POS_FRAMES seeking, which is unreliable for variable-frame-rate AVI/MP4 files).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def iter_frames_at_indices(video_path: Path | str, frame_indices: list[int]) -> Iterator[tuple[int, np.ndarray | None]]:
    """Sequentially decode `video_path`, yielding (frame_idx, frame_bgr) for each requested index.

    `frame_indices` may contain duplicates; each is yielded once, in the order given to
    `cv2.VideoCapture` decode (i.e. ascending). Indices beyond the video's length yield
    (frame_idx, None).
    """
    wanted = sorted(set(frame_indices))
    if not wanted:
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {video_path}")

    try:
        target_iter = iter(wanted)
        target = next(target_iter)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx == target:
                yield target, frame
                try:
                    target = next(target_iter)
                except StopIteration:
                    return
            idx += 1

        # ran out of frames before satisfying all requested indices
        yield target, None
        for target in target_iter:
            yield target, None
    finally:
        cap.release()
