#!/usr/bin/env bash
# Warm the model cache on the LOGIN node, where there is outbound internet.
#
#   bash slurm/preflight.sh
#
# Compute nodes on this cluster have no egress, so slurm/stage1_eval.sbatch runs with
# HF_HUB_OFFLINE=1. Without this step, the GPU job's first from_pretrained() call would fail --
# and if it were left online it would instead hang on a DNS/connect timeout, burning the whole
# wall-clock allocation before anything is segmented.
#
# Everything here loads on CPU: no GPU allocation is needed or requested.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/slurm/env.sh"
activate_env

mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"
cd "$REPO_ROOT"

python - "$SAM3_CHECKPOINT" "$DEPTH_MODEL" <<'PY'
import sys
from pathlib import Path

checkpoint, depth_model = sys.argv[1], sys.argv[2]
failures = []

# 1. The depth backend. Weights come from Hugging Face on first use.
try:
    if depth_model == "pi3x":
        from pi3.models.pi3x import Pi3X
        Pi3X.from_pretrained("yyfz233/Pi3X")
    else:
        from depth_anything_3.api import DepthAnything3
        DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    print(f"  ok  {depth_model} weights cached")
except Exception as exc:
    failures.append(f"{depth_model}: {exc}")

# 2. SAM-3. The checkpoint is a local file, but instantiating the predictor once here also
#    pulls whatever first-run assets the Ultralytics CLIP fork fetches for text prompting --
#    which would otherwise be attempted from an offline compute node.
try:
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    from ultralytics.models.sam import SAM3SemanticPredictor
    SAM3SemanticPredictor(overrides=dict(
        conf=0.25, task="segment", mode="predict", model=checkpoint,
        half=False, save=False, verbose=False, device="cpu",
    ))
    print("  ok  SAM-3 predictor constructed, prompt assets cached")
except Exception as exc:
    failures.append(f"sam3: {exc}")

if failures:
    print("\nPRE-FLIGHT FAILED -- fix these before submitting the GPU job:")
    for failure in failures:
        print(f"  !! {failure}")
    sys.exit(1)
PY

echo
echo "cache size: $(du -sh "$HF_HOME" 2>/dev/null | cut -f1)"
echo "pre-flight OK -- slurm/stage1_eval.sbatch can now run offline."
