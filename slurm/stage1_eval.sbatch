#!/usr/bin/env bash
#SBATCH --job-name=wcf-qa-stage1
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Partition, account, GPU, CPUs, memory and wall clock all come from slurm/env.sh via
# slurm/submit.sh -- submit with `bash slurm/submit.sh stage1_eval`, not `sbatch` directly.
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

echo "host        : $(hostname)"
echo "job         : ${SLURM_JOB_ID:-none}"
echo "repo        : $REPO_ROOT"
echo "HF_HOME     : $HF_HOME"
echo "checkpoint  : $SAM3_CHECKPOINT"
echo "depth model : $DEPTH_MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "  (no nvidia-smi)"
echo

mkdir -p "$(dirname "$QA_RESULTS")" "$QA_MASKS" "$QA_DEPTH"

# The sample CSV *is* a valid annotations CSV (video_name, frame_idx, frame_timestamp,
# distance), which is why this stage needs no bespoke code -- see scripts/qa/sample.py.
srun python scripts/run_calibration_eval.py \
    --device cuda \
    --annotations-csv "$QA_SAMPLE" \
    --output-csv "$QA_RESULTS" \
    --save-mask-dir "$QA_MASKS" \
    --save-depth-dir "$QA_DEPTH" \
    --sam3-checkpoint "$SAM3_CHECKPOINT" \
    --depth-model "$DEPTH_MODEL"

echo
echo "--- status histogram (the source of truth for Stage 2, not file presence) ---"
python - "$QA_RESULTS" <<'PY'
import csv, sys
from collections import Counter
with open(sys.argv[1], newline="") as f:
    counts = Counter(r["status"].split(":", 1)[0] for r in csv.DictReader(f))
for status, n in counts.most_common():
    print(f"  {status:<22} {n}")
print("\n  A non-zero empty_mask count with no matching mask JSON is correct, not a bug:")
print("  run_calibration_eval.py only persists masks when SAM-3 detects something.")
PY

echo
echo "next (login node): python scripts/qa/prefilter.py"
