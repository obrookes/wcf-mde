#!/usr/bin/env bash
# Submit a QA pilot job, taking every scheduler value from slurm/env.sh.
#
#   bash slurm/submit.sh stage1_eval
#   bash slurm/submit.sh stage3a_render
#   bash slurm/submit.sh pipeline_infer
#   bash slurm/submit.sh pipeline_export
#
# Anything after the job name is forwarded to sbatch, so one-offs don't need an env.sh edit:
#
#   bash slurm/submit.sh stage1_eval --time=12:00:00
#
# pipeline_infer/pipeline_export are slurm/pipeline.sh's driver stages -- one GPU job script,
# one CPU job script, both reused across several driver subcommands (infer AND cohort-infer both
# submit pipeline_infer, for instance). This script only ever picks scheduler flags (GPU vs CPU);
# which input/output paths the job runs against comes from PIPE_* env vars pipeline.sh exports
# before calling this -- see slurm/pipeline_infer.sbatch / slurm/pipeline_export.sbatch.
#
# Run from the repo root: SLURM sets SLURM_SUBMIT_DIR from the submitting directory, and the
# job scripts use it to find the repo (the batch script itself is copied to the node's spool,
# so $0 inside the job is not a usable path).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/slurm/env.sh"

JOB="${1:-}"
if [[ -z "$JOB" ]]; then
    echo "usage: bash slurm/submit.sh {stage1_eval|stage3a_render|pipeline_infer|pipeline_export}" \
         "[extra sbatch flags]" >&2
    exit 2
fi
shift

SCRIPT="$REPO_ROOT/slurm/${JOB}.sbatch"
[[ -f "$SCRIPT" ]] || { echo "no such job script: $SCRIPT" >&2; exit 2; }

case "$JOB" in
    stage1_eval)
        FLAGS=(--partition="$SLURM_PARTITION_GPU" --gres="$GPU_GRES"
               --cpus-per-task="$GPU_CPUS" --mem="$GPU_MEM" --time="$TIME_STAGE1")
        ;;
    stage3a_render)
        FLAGS=(--partition="$SLURM_PARTITION_CPU"
               --cpus-per-task="$RENDER_CPUS" --mem="$RENDER_MEM" --time="$TIME_RENDER")
        ;;
    pipeline_infer)
        # GPU driver stage (slurm/pipeline.sh's `infer` and `cohort-infer`): same flag shape as
        # stage1_eval, since it wraps the same GPU-bound run_calibration_eval.py.
        FLAGS=(--partition="$SLURM_PARTITION_GPU" --gres="$GPU_GRES"
               --cpus-per-task="$GPU_CPUS" --mem="$GPU_MEM" --time="$TIME_STAGE1")
        ;;
    pipeline_export)
        # CPU driver stage (slurm/pipeline.sh's `export`): same flag shape as stage3a_render.
        FLAGS=(--partition="$SLURM_PARTITION_CPU"
               --cpus-per-task="$RENDER_CPUS" --mem="$RENDER_MEM" --time="$TIME_RENDER")
        ;;
    *)
        echo "unknown job: $JOB" >&2; exit 2 ;;
esac

[[ -n "$SLURM_ACCOUNT" ]] && FLAGS+=(--account="$SLURM_ACCOUNT")
[[ -n "$SLURM_QOS" ]] && FLAGS+=(--qos="$SLURM_QOS")

# Fail early and locally rather than 6 hours into a queue wait.
if [[ "$JOB" == "stage1_eval" ]]; then
    [[ -f "$REPO_ROOT/$QA_SAMPLE" ]] || {
        echo "missing $QA_SAMPLE -- run scripts/qa/sample.py on the login node first" >&2; exit 1; }
fi
if [[ "$JOB" == "stage1_eval" || "$JOB" == "pipeline_infer" ]]; then
    [[ -d "$HF_HOME" && -n "$(ls -A "$HF_HOME" 2>/dev/null)" ]] || {
        echo "HF_HOME ($HF_HOME) is empty -- run 'bash slurm/preflight.sh' on the LOGIN node" >&2
        echo "first, or the job will fail offline on the compute node." >&2; exit 1; }
fi

echo "sbatch ${FLAGS[*]} $* $SCRIPT"
cd "$REPO_ROOT"
exec sbatch "${FLAGS[@]}" "$@" "$SCRIPT"
