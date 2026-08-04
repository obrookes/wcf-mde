# Site configuration for the SAM-3 mask QA pilot on SLURM.
#
# THIS IS THE ONLY FILE YOU SHOULD NEED TO EDIT. Every job script sources it, and
# slurm/submit.sh passes the scheduler values through on the sbatch command line, so nothing
# site-specific is duplicated in the .sbatch files.
#
# Sourced, not executed -- no shebang, no `set -e` (that would kill your login shell).

# ---------------------------------------------------------------------------------------
# scheduler  -- TODO: fill these in for your cluster (`sinfo -s`, `sacctmgr show assoc user=$USER`)
# ---------------------------------------------------------------------------------------
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"            # TODO e.g. "phys-chimp-2026"; leave empty to omit
SLURM_PARTITION_GPU="${SLURM_PARTITION_GPU:-workq}"  # single-partition cluster: GPU and CPU jobs
SLURM_PARTITION_CPU="${SLURM_PARTITION_CPU:-workq}"  # both land on workq
SLURM_QOS="${SLURM_QOS:-}"                    # TODO optional QOS; leave empty to omit

GPU_GRES="${GPU_GRES:-gpu:1}"                 # TODO e.g. "gpu:a100:1" if your site needs a type
GPU_CPUS="${GPU_CPUS:-8}"                     # cores for decode + dataloading alongside the GPU
GPU_MEM="${GPU_MEM:-64G}"
RENDER_CPUS="${RENDER_CPUS:-8}"
RENDER_MEM="${RENDER_MEM:-16G}"

# Stage 1 has no checkpointing: run_calibration_eval.py flushes its results CSV per video, so a
# kill leaves partial results rather than nothing, but the run still has to be repeated. Ask for
# more wall clock than you think you need -- it is cheaper than re-running the whole pass.
TIME_STAGE1="${TIME_STAGE1:-08:00:00}"
TIME_RENDER="${TIME_RENDER:-02:00:00}"

# ---------------------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------------------
activate_env() {
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate wcf-pipe
    # Static ffmpeg/ffprobe build (probe_video_fps.py needs ffprobe; the cluster has none and
    # conda-forge ffmpeg is not installed in the env). Harmless no-op if the dir is absent.
    if [ -d "$SCRATCH_ROOT/tools/ffmpeg-7.0.2-arm64-static" ]; then
        export PATH="$SCRATCH_ROOT/tools/ffmpeg-7.0.2-arm64-static:$PATH"
    fi
}

# ---------------------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------------------
# Big artifacts belong on scratch, not in your home quota. `outputs/` is gitignored, so the
# recommended setup is a symlink (see slurm/README.md):
#     ln -s "$SCRATCH_ROOT/wcf-mde-outputs" outputs
SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCH:-$HOME/scratch}}"

# Hugging Face cache. slurm/preflight.sh populates this on the LOGIN node; the GPU job then
# runs with HF_HUB_OFFLINE=1 so a cache miss fails in seconds instead of hanging on a network
# timeout until the wall clock expires. Compute nodes here have no outbound internet.
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/hf}"

SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-/scratch/b6cn/obrookes.b6cn/safari_checkpoint_hf.pt}"
DEPTH_MODEL="${DEPTH_MODEL:-da3}"

# QA artifact locations, all relative to the repo root
QA_DIR="${QA_DIR:-outputs/qa}"
QA_SAMPLE="${QA_SAMPLE:-$QA_DIR/sample.csv}"
QA_RESULTS="${QA_RESULTS:-$QA_DIR/results.csv}"
QA_MASKS="${QA_MASKS:-$QA_DIR/masks}"
QA_DEPTH="${QA_DEPTH:-$QA_DIR/depth}"
QA_OVERLAYS="${QA_OVERLAYS:-$QA_DIR/overlays}"

# ---------------------------------------------------------------------------------------
# pipeline  -- slurm/pipeline.sh, the end-to-end driver (fps -> qc -> infer -> calibrate ->
# export -> score -> [cohort funnel] -> review). Everything below is one run's identity and
# is overridable per invocation, e.g. `RUN_NAME=full bash slurm/pipeline.sh status`.
# ---------------------------------------------------------------------------------------
RUN_NAME="${RUN_NAME:-smoke}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH_ROOT/data}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH_ROOT/runs/$RUN_NAME}"
ANNOTATIONS_CSV="${ANNOTATIONS_CSV:-$DATA_ROOT/annotations_20260709_with_fps.csv}"
MODEL_LABEL="${MODEL_LABEL:-heuristic-v1}"
