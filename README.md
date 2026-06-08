# wcf-mde

Wild Chimpanzee Foundation camera-trap reference videos and ground-truth
distance annotations, plus a pipeline that evaluates monocular distance
estimation against that ground truth. See `CLAUDE.md` for the dataset layout.

## Pipeline

`scripts/run_calibration_eval.py` walks `data/annotations_06052026.csv`, and
for each annotated row:

1. resolves `video_name` to an actual video file under `data/` (via
   `data/list_reference_videos.xlsx`, see `scripts/video_lookup.py`)
2. extracts the annotated frame by sequential decode (`scripts/frame_source.py`)
3. segments it with **SAM-3** using a text prompt (default `"person holding sign"`)
4. estimates per-pixel metric depth with **Pi3X**
5. reduces mask + depth to a single predicted distance (mean-in-mask, centroid)

and writes `video_name, frame_idx, frame_timestamp, distance_gt, status,
mask_area_px, depth_mask_mean, depth_centroid` to an output CSV, plus a
summary (status counts, MAE/RMSE/bias vs. ground truth) at the end.

## Environment setup

Requires a CUDA-capable GPU for any non-trivial run (CPU works for smoke-testing
only). Tested with conda env `dap-3_py3-11` (Python 3.10/3.11, CUDA 12.6 torch).

```bash
conda create -n dap-3_py3-11 "python>=3.7,<3.11"
conda activate dap-3_py3-11

# torch + ultralytics (SAM-3)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics openpyxl

# SAM-3's promptable text segmentation needs Ultralytics' CLIP fork
pip install "git+https://github.com/ultralytics/CLIP.git"

# Pi3 (metric monocular depth) — clone and install in editable mode
git clone https://github.com/yyfz/Pi3 third_party/Pi3
pip install -e third_party/Pi3
```

A SAM-3 checkpoint is required. Vanilla `sam3.pt` is gated on Hugging Face;
the default `--sam3-checkpoint` points at the SA-FARI wildlife checkpoint —
adjust the path to wherever you have a checkpoint locally:

```
/home/dl18206/projs/Unmarked-Anything/weights/sam3/safari_checkpoint_hf.pt
```

`Pi3X.from_pretrained("yyfz233/Pi3X")` downloads weights from Hugging Face on
first run (only `Pi3X`, not plain `Pi3`, gives metric-scale depth).

## Running

```bash
python scripts/run_calibration_eval.py \
  --device cuda \
  --output-csv outputs/calibration_results.csv
```

Useful flags (all have sensible defaults pointing at `data/`):

| Flag | Purpose |
|---|---|
| `--limit N` | process only the first N annotation rows (smoke-testing) |
| `--device {auto,cpu,cuda}` | compute device (default `auto`) |
| `--sam3-checkpoint PATH` | SAM-3 checkpoint to load |
| `--sam3-prompt TEXT` | text prompt for SAM-3 segmentation |
| `--pi3-model-id ID` | Pi3X model id/path (default `yyfz233/Pi3X`) |
| `--conf FLOAT` | SAM-3 confidence threshold (default `0.25`) |
| `--output-csv PATH` | where to write per-row results |

Run `python scripts/run_calibration_eval.py --help` for the full list.
