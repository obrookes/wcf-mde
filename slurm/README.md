# Running the mask QA pilot on SLURM

The funnel splits three ways, because **compute nodes on this cluster have no outbound
internet**. Nothing that needs the network ever runs inside a job.

| Where | What | Why there |
|---|---|---|
| **Login node** | `preflight.sh`, `sample.py`, `prefilter.py`, `triage.py`, `report.py`, the tests | Needs internet (model weights, Batch API) or is seconds of CPU |
| **GPU job** | `run_calibration_eval.py` (Stage 1) | Needs a GPU; runs offline against a pre-warmed cache |
| **CPU job** | `render_overlays.py` (Stage 3a) | Video decode, no GPU, no network |

## One-time setup

```bash
# 1. Point outputs/ at scratch. It is gitignored, and the artifacts are far too big for a
#    home quota (see the sizes below).
ln -s "$SCRATCH/wcf-mde-outputs" outputs
mkdir -p slurm/logs

# 2. Fill in the TODOs in slurm/env.sh -- partition, account, GPU gres, wall clock, and how
#    this cluster activates the dap-3_py3-11 env. It is the only file you should need to edit.
$EDITOR slurm/env.sh

# 3. Warm the model cache while you still have internet. Skipping this makes the GPU job fail.
bash slurm/preflight.sh
```

Artifact sizes for the ~1,100-frame cohort, so you can size the scratch allocation:

| Directory | Size | Notes |
|---|---|---|
| `outputs/qa/masks/` | ~5 MB | COCO RLE JSON, a few hundred bytes per instance |
| `outputs/qa/depth/` | ~600 MB | fp16 `.npy`, written for every decoded frame |
| `outputs/qa/overlays/` | ~1 GB | lossless PNG review panels |

`depth/` is not used by the QA funnel at all — it is persisted because Stage 1 writes it for
free and it saves re-running inference if you later want to calibrate this cohort. Drop
`--save-depth-dir` from `stage1_eval.sbatch` if scratch is tight.

## Run order

`bash slurm/pipeline.sh all --with-cohort` (see [The pipeline driver](#the-pipeline-driver-slurmpipelinesh)
below) now drives this whole funnel — plus the main fps/qc/infer/calibrate/export/score funnel
from README.md, in parallel on GPU — non-interactively up to the same `verdicts`/`report`
hand-off documented below. The by-hand sequence in this section remains valid stage-by-stage; it
is exactly what each `pipeline.sh` subcommand wraps.

```bash
# --- login node ------------------------------------------------------------------
python scripts/qa/test_prefilter.py          # and the other test_*.py; all offline
python scripts/qa/sample.py                  # -> outputs/qa/sample.csv + docs/qa_cohort_1000.csv

# --- GPU job ---------------------------------------------------------------------
bash slurm/submit.sh stage1_eval             # -> results.csv, masks/, depth/
                                             #    watch: tail -f slurm/logs/wcf-qa-stage1-*.out

# --- login node ------------------------------------------------------------------
python scripts/qa/prefilter.py               # -> prefilter.csv, and f_v

# --- CPU job ---------------------------------------------------------------------
bash slurm/submit.sh stage3a_render          # -> overlays/, overlay_manifest.csv

# --- login node (needs internet + ANTHROPIC_API_KEY) -----------------------------
python scripts/qa/triage.py estimate         # free; check the bill before spending
tmux new -s triage                           # a batch can take up to 24h
python scripts/qa/triage.py submit --name haiku
python scripts/qa/triage.py poll   --name haiku    # re-runnable; safe to disconnect
python scripts/qa/triage.py fetch  --name haiku    # -> verdicts_haiku.csv

# --- laptop: review the flagged masks (Stage 4; no SLURM job, no tunnel) ---------
# On the login node, pack everything the review needs into one self-contained tarball:
python scripts/qa/make_review_bundle.py \
    --verdicts outputs/qa/verdicts_haiku.csv --frames-dir outputs/export/frames \
    --masks-dir outputs/qa/masks --out outputs/qa/review_bundle.tar.gz --include-ok
#   scp it down, tar xzf, python run_review.py; scp corrections.csv back, then:
python scripts/qa/apply_corrections.py morph \
    --corrections outputs/qa/corrections.csv --masks-dir outputs/qa/masks \
    --out-masks-dir outputs/qa/masks_corrected

# --- login node: gold set, then the report ---------------------------------------
python scripts/qa/report.py goldset          # -> gold_template.csv (blind: no verdicts in it)
#   ... label gold_verdict by hand, then:
python scripts/qa/report.py summarise --gold outputs/qa/gold_labelled.csv
```

Escalation, once the threshold is fitted on the gold set's tune half, uses the same four
subcommands against the stronger model:

```bash
python scripts/qa/triage.py submit --name opus --model claude-opus-5 \
    --escalate-from outputs/qa/verdicts_haiku.csv --confidence-below 0.7
python scripts/qa/triage.py poll  --name opus
python scripts/qa/triage.py fetch --name opus
python scripts/qa/report.py summarise --gold outputs/qa/gold_labelled.csv \
    --escalated-verdicts outputs/qa/verdicts_opus.csv
```

## The pipeline driver (`slurm/pipeline.sh`)

`slurm/pipeline.sh` is a resumable driver over both funnels above — the main README.md
fps/qc/infer/calibrate/export/score sequence and the QA-pilot cohort funnel on this page — not a
replacement job script. It reuses the same two job shapes already on this page under new names:

- **`pipeline_infer.sbatch`** — the GPU job, wraps `run_calibration_eval.py` exactly like
  `stage1_eval.sbatch` does. It is submitted for both the `infer` and `cohort-infer`
  subcommands; which annotations/output paths it runs against comes entirely from
  `PIPE_ANNOTATIONS_CSV`, `PIPE_RESULTS_CSV`, `PIPE_MASKS_DIR`, `PIPE_DEPTH_DIR` (and optional
  `PIPE_EXTRA` for `--limit`), which `pipeline.sh` exports before calling
  `slurm/submit.sh pipeline_infer` — the job script hard-fails if any of them is unset, so it can
  never silently run against a leftover value from a previous stage.
- **`pipeline_export.sbatch`** — the CPU job, wraps `export_calibrated.py` the same way, via
  `PIPE_CALIB_DIR`, `PIPE_OUT_DIR`, `PIPE_QC_FLAGS`, `PIPE_ANNOTATIONS_CSV`, `PIPE_MASKS_DIR`.

`slurm/submit.sh` picks scheduler flags for both exactly as it does for `stage1_eval`/
`stage3a_render`: `pipeline_infer` gets `SLURM_PARTITION_GPU`/`GPU_GRES`/`GPU_CPUS`/`GPU_MEM`/
`TIME_STAGE1`, `pipeline_export` gets `SLURM_PARTITION_CPU`/`RENDER_CPUS`/`RENDER_MEM`/
`TIME_RENDER` — nothing GPU/CPU-shape-specific to configure beyond `slurm/env.sh`'s existing
scheduler block.

**Resuming.** Every stage writes a sentinel to `$RUN_ROOT/.done/<stage>.ok` once it finishes;
re-running `bash slurm/pipeline.sh all` (or any individual subcommand) skips stages whose
sentinel already exists, so a wall-clock kill or a failed `check` only costs you the stage it
interrupted. `--force` clears one *named* stage's sentinel and reruns just that stage — it does
not cascade through `all`. `bash slurm/pipeline.sh status` prints a done/pending table for every
stage under the current `RUN_NAME`.

| `pipeline.sh` subcommand | wraps | legacy stage (this page / README.md's numbering) |
|---|---|---|
| `fps`, `qc` | `probe_video_fps.py`, `qc_annotations.py --fix` | Step 0 (README.md calibration walkthrough) |
| `infer` | `pipeline_infer.sbatch` → `run_calibration_eval.py` | Stage 1, full corpus |
| `calibrate` | `calibrate_depth.py` | Step/Stage 2 |
| `export` | `pipeline_export.sbatch` → `export_calibrated.py` | -- (not stage-numbered) |
| `score` | `score_masks.py` + `heuristic_verdicts.py` | Mask triage, Backend B |
| `cohort` | `sample.py` | Stage 0 sample (Run order, above) |
| `cohort-infer` | `pipeline_infer.sbatch` over the cohort | Stage 1 masks (`stage1_eval`) |
| `prefilter` | `prefilter.py` | Stage 2 pre-filter |
| `overlays` | `stage3a_render` → `render_overlays.py` | Stage 3a render |
| `verdicts` | in-session subagent grading (stop point) | Stage 3b triage (replaces `triage.py`) |
| `report` | `report.py goldset` / `summarise` (stop point) | Stage 5 report |
| `review` | `make_review_bundle.py` (+ `apply_corrections.py`) | Stage 4 review + correct |

## Things that will bite you

**Submit from the repo root.** SLURM copies the batch script to the node's spool, so `$0` is
not a usable path inside the job; the scripts use `$SLURM_SUBMIT_DIR` instead. `submit.sh`
`cd`s there for you — just don't call `sbatch` on the `.sbatch` files directly, or you will
also lose every scheduler flag from `env.sh`.

**Stage 1 has no checkpointing.** `run_calibration_eval.py` flushes its results CSV once per
video, so a wall-clock kill leaves usable partial results (and the masks and depth maps already
written) rather than nothing — but the run still has to be repeated to complete the cohort. Ask
for generous `TIME_STAGE1`; queue time is cheaper than a re-run.

**Check the GPU job's log in the first minute.** With `HF_HUB_OFFLINE=1` a missing cache entry
fails immediately and loudly. That is deliberate: without it, an offline node would hang on a
connect timeout and burn the entire allocation before segmenting anything.

**`empty_mask` frames write no mask JSON.** The mask and depth directories will not have
matching file counts, and Stage 2 reads dispositions from the results CSV `status` column
rather than from file presence for exactly this reason. Not a bug.

**Triage cannot run in a job.** It needs the Batch API. Run it from the login node under `tmux`
or `nohup`; the batch id and the `custom_id` map are persisted to `outputs/qa/batches/`, so
`poll` and `fetch` survive any disconnect.
