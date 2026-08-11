"""COCO run-length-encoded persistence for per-instance SAM-3 masks.

One JSON file per frame (not one giant pickle), keyed the same way as run_calibration_eval.py's
`*_orig.npy` depth maps -- by `video_name` (the flattened annotation name), not the video file's
own stem, so files don't collide across camera folders that reuse filenames like DSCF0005.

RLE (via pycocotools, already a dependency of this repo's dap-3_py3-11 env) keeps a frame's
mask(s) to a few hundred bytes each instead of a full HxW boolean array, which matters once
every annotated frame's mask -- not just a scalar reduction -- is being persisted for Stage-1
alignment (scripts/alignment.py).

On-disk shape:

    {"distance_gt": 12.5, "instances": [{"instance_idx", "center_xy", "area_px", "rle"}, ...]}

`distance_gt` is the frame's ground-truth subject-to-camera distance, carried here so an
exported frames/ + masks/ bundle (scripts/export_calibrated.py) is self-describing instead of
requiring a join back to the annotations CSV. It is *frame*-level, not per-instance: the
annotations only ever give one distance per frame, so when a frame yields several instances
they all share it (the same assumption calibrate_depth.py already makes). It is None when
unknown -- nothing in the pipeline requires it, so a mask file written without one stays valid.

Files written before `distance_gt` existed are a bare JSON list of instances; the loaders below
accept that older shape and report distance_gt=None for it, so a mask dir from an earlier run
does not need re-generating.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


def encode_rle(mask: np.ndarray) -> dict:
    """bool/uint8 HxW mask -> COCO RLE dict with JSON-safe (str) 'counts'."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def decode_rle(rle: dict) -> np.ndarray:
    """Inverse of encode_rle: COCO RLE dict -> bool HxW mask."""
    rle = {**rle, "counts": rle["counts"].encode("ascii")}
    return mask_utils.decode(rle).astype(bool)


def mask_path(out_dir: Path, video_name: str, frame_idx: int) -> Path:
    return out_dir / f"{video_name}_frame{frame_idx:06d}_masks.json"


def save_instance_masks(
    out_dir: Path,
    video_name: str,
    frame_idx: int,
    instances: list[dict],
    distance_gt: float | None = None,
) -> None:
    """Persist one frame's detected-instance masks. `instances` is the list produced by
    run_calibration_eval.py::extract_instance_masks (each a {"mask", "center_xy", "area_px"}
    dict); saved in the same left-to-right order so instance_idx here matches the CSV's.

    `distance_gt` is that frame's ground-truth distance (see module docstring); pass None when
    it isn't known. Re-savers that only mean to rewrite masks (scripts/qa/apply_corrections.py)
    should read the existing value with load_mask_record and pass it back through, or the
    rewritten file silently loses it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "distance_gt": None if distance_gt is None else float(distance_gt),
        "instances": [
            {
                "instance_idx": idx,
                "center_xy": list(inst["center_xy"]),
                "area_px": inst["area_px"],
                "rle": encode_rle(inst["mask"]),
            }
            for idx, inst in enumerate(instances)
        ],
    }
    with open(mask_path(out_dir, video_name, frame_idx), "w") as f:
        json.dump(payload, f)


def load_mask_record(out_dir: Path, video_name: str, frame_idx: int) -> dict:
    """Inverse of save_instance_masks, as {"distance_gt": float|None, "instances": [...]} where
    each instance is {"instance_idx", "center_xy", "area_px", "mask"}. A legacy bare-list file
    (written before distance_gt existed) loads with distance_gt None."""
    with open(mask_path(out_dir, video_name, frame_idx)) as f:
        payload = json.load(f)
    if isinstance(payload, list):  # legacy shape: instances only
        payload = {"distance_gt": None, "instances": payload}
    return {
        "distance_gt": payload.get("distance_gt"),
        "instances": [
            {
                "instance_idx": item["instance_idx"],
                "center_xy": tuple(item["center_xy"]),
                "area_px": item["area_px"],
                "mask": decode_rle(item["rle"]),
            }
            for item in payload["instances"]
        ],
    }


def load_instance_masks(out_dir: Path, video_name: str, frame_idx: int) -> list[dict]:
    """load_mask_record's instance list alone, for the callers that don't need the distance."""
    return load_mask_record(out_dir, video_name, frame_idx)["instances"]
