#!/usr/bin/env bash
#SBATCH --job-name=wcf-pipe-export
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Partition, account, CPUs, memory and wall clock come from slurm/env.sh via slurm/submit.sh --
# submit with `bash slurm/submit.sh pipeline_export`.
#
# CPU only: this decodes video and copies files. No GPU is requested, and no network is touched,
# so it runs happily on an offline compute node.
#
# CPU driver stage, wrapping scripts/export_calibrated.py, part of slurm/pipeline.sh's `export`
# subcommand. Every input/output path comes from the PIPE_* env vars pipeline.sh exports before
# submitting -- never from a flag or a default baked in here.
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
source "$REPO_ROOT/slurm/env.sh"
activate_env

for v in PIPE_CALIB_DIR PIPE_OUT_DIR PIPE_QC_FLAGS PIPE_ANNOTATIONS_CSV PIPE_MASKS_DIR; do
    if [[ -z "${!v:-}" ]]; then
        echo "!! $v is unset -- this job must be submitted by slurm/pipeline.sh, which sets" >&2
        echo "   every PIPE_* var before calling slurm/submit.sh pipeline_export." >&2
        exit 1
    fi
done

echo "host        : $(hostname)"
echo "job         : ${SLURM_JOB_ID:-none}"
echo "repo        : $REPO_ROOT"
echo "calib dir   : $PIPE_CALIB_DIR"
echo "out dir     : $PIPE_OUT_DIR"
echo "qc flags    : $PIPE_QC_FLAGS"
echo "annotations : $PIPE_ANNOTATIONS_CSV"
echo "masks       : $PIPE_MASKS_DIR"
echo

mkdir -p "$PIPE_OUT_DIR"

srun python scripts/export_calibrated.py \
    --calib-dir "$PIPE_CALIB_DIR" \
    --out-dir "$PIPE_OUT_DIR" \
    --qc-flags "$PIPE_QC_FLAGS" \
    --annotations-csv "$PIPE_ANNOTATIONS_CSV" \
    --video-list "$DATA_ROOT/list_reference_videos.xlsx" \
    --data-dir "$DATA_ROOT" \
    --masks-dir "$PIPE_MASKS_DIR"

echo
echo "wrote $PIPE_OUT_DIR/depth_maps, $PIPE_OUT_DIR/frames, $PIPE_OUT_DIR/masks"
