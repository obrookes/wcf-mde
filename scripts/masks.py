"""COCO run-length-encoded persistence for per-instance SAM-3 masks.

One JSON file per frame (not one giant pickle), keyed the same way as run_calibration_eval.py's
`*_orig.npy` depth maps -- by `video_name` (the flattened annotation name), not the video file's
own stem, so files don't collide across camera folders that reuse filenames like DSCF0005.

RLE (via pycocotools, already a dependency of this repo's dap-3_py3-11 env) keeps a frame's
mask(s) to a few hundred bytes each instead of a full HxW boolean array, which matters once
every annotated frame's mask -- not just a scalar reduction -- is being persisted for Stage-1
alignment (scripts/alignment.py).
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


def save_instance_masks(out_dir: Path, video_name: str, frame_idx: int, instances: list[dict]) -> None:
    """Persist one frame's detected-instance masks. `instances` is the list produced by
    run_calibration_eval.py::extract_instance_masks (each a {"mask", "center_xy", "area_px"}
    dict); saved in the same left-to-right order so instance_idx here matches the CSV's."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "instance_idx": idx,
            "center_xy": list(inst["center_xy"]),
            "area_px": inst["area_px"],
            "rle": encode_rle(inst["mask"]),
        }
        for idx, inst in enumerate(instances)
    ]
    with open(mask_path(out_dir, video_name, frame_idx), "w") as f:
        json.dump(payload, f)


def load_instance_masks(out_dir: Path, video_name: str, frame_idx: int) -> list[dict]:
    """Inverse of save_instance_masks: [{"instance_idx", "center_xy", "area_px", "mask"}, ...]."""
    with open(mask_path(out_dir, video_name, frame_idx)) as f:
        payload = json.load(f)
    return [
        {
            "instance_idx": item["instance_idx"],
            "center_xy": tuple(item["center_xy"]),
            "area_px": item["area_px"],
            "mask": decode_rle(item["rle"]),
        }
        for item in payload
    ]
