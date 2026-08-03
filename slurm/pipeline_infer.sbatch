#!/usr/bin/env bash
#SBATCH --job-name=wcf-pipe-infer
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Partition, account, GPU, CPUs, memory and wall clock all come from slurm/env.sh via
# slurm/submit.sh -- submit with `bash slurm/submit.sh pipeline_infer`, not `sbatch` directly.
#
# GPU driver stage, wrapping scripts/run_calibration_eval.py. Reused by two of
# slurm/pipeline.sh's subcommands (`infer` and `cohort-infer`) against different inputs; which
# is which comes entirely from the PIPE_* env vars pipeline.sh exports before submitting, never
# from a flag or a default baked in here.
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
source "$REPO_ROOT/slurm/env.sh"
activate_env

# Compute nodes here have no outbound internet. Fail fast on a cache miss instead of hanging on
# a connect timeout until the wall clock expires -- slurm/preflight.sh populates HF_HOME on the
# login node precisely so this is safe.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

for v in PIPE_ANNOTATIONS_CSV PIPE_RESULTS_CSV PIPE_MASKS_DIR PIPE_DEPTH_DIR; do
    if [[ -z "${!v:-}" ]]; then
        echo "!! $v is unset -- this job must be submitted by slurm/pipeline.sh, which sets" >&2
        echo "   every PIPE_* var before calling slurm/submit.sh pipeline_infer." >&2
        exit 1
    fi
done

echo "host        : $(hostname)"
echo "job         : ${SLURM_JOB_ID:-none}"
echo "repo        : $REPO_ROOT"
echo "HF_HOME     : $HF_HOME"
echo "checkpoint  : $SAM3_CHECKPOINT"
echo "depth model : $DEPTH_MODEL"
echo "annotations : $PIPE_ANNOTATIONS_CSV"
echo "results     : $PIPE_RESULTS_CSV"
echo "masks       : $PIPE_MASKS_DIR"
echo "depth       : $PIPE_DEPTH_DIR"
echo "extra       : ${PIPE_EXTRA:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "  (no nvidia-smi)"
echo

mkdir -p "$(dirname "$PIPE_RESULTS_CSV")" "$PIPE_MASKS_DIR" "$PIPE_DEPTH_DIR"

# $PIPE_EXTRA is deliberately word-split (e.g. "--limit 40" from pipeline.sh's --limit flag) --
# leave it unquoted.
srun python scripts/run_calibration_eval.py \
    --annotations-csv "$PIPE_ANNOTATIONS_CSV" \
    --results-csv "$PIPE_RESULTS_CSV" \
    --save-masks-dir "$PIPE_MASKS_DIR" \
    --save-depth-dir "$PIPE_DEPTH_DIR" \
    --sam3-checkpoint "$SAM3_CHECKPOINT" \
    --depth-model "$DEPTH_MODEL" \
    --data-dir "$DATA_ROOT" \
    --video-list "$DATA_ROOT/list_reference_videos.xlsx" \
    --device cuda \
    ${PIPE_EXTRA:-}

echo
echo "--- status histogram ---"
python - "$PIPE_RESULTS_CSV" <<'PY'
import csv, sys
from collections import Counter
with open(sys.argv[1], newline="") as f:
    counts = Counter(r["status"].split(":", 1)[0] for r in csv.DictReader(f))
for status, n in counts.most_common():
    print(f"  {status:<22} {n}")
PY
