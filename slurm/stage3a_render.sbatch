#!/usr/bin/env bash
#SBATCH --job-name=wcf-qa-render
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Partition, account, CPUs, memory and wall clock come from slurm/env.sh via slurm/submit.sh --
# submit with `bash slurm/submit.sh stage3a_render`.
#
# CPU only: this decodes video and composites panels. No GPU is requested, and no network is
# touched, so it runs happily on an offline compute node.
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
source "$REPO_ROOT/slurm/env.sh"
activate_env

echo "host : $(hostname)"
echo "job  : ${SLURM_JOB_ID:-none}"
echo

srun python scripts/qa/render_overlays.py \
    --prefilter-csv "$QA_DIR/prefilter.csv" \
    --mask-dir "$QA_MASKS" \
    --out-dir "$QA_OVERLAYS" \
    --manifest "$QA_DIR/overlay_manifest.csv"

echo
echo "next (LOGIN node -- the Batch API needs outbound internet):"
echo "  python scripts/qa/triage.py estimate   # free, check the bill first"
echo "  python scripts/qa/triage.py submit"
