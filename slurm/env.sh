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
SLURM_PARTITION_GPU="${SLURM_PARTITION_GPU:-gpu}"   # TODO partition with GPUs
SLURM_PARTITION_CPU="${SLURM_PARTITION_CPU:-compute}"  # TODO CPU-only partition
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
# environment  -- TODO: replace with however this cluster activates the dap-3_py3-11 env
# ---------------------------------------------------------------------------------------
activate_env() {
    # TODO uncomment / adjust the lines your site needs, e.g.:
    # module purge
    # module load cuda/12.6
    # source "$(conda info --base)/etc/profile.d/conda.sh"
    # conda activate dap-3_py3-11
    :
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

SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-/home/dl18206/projs/Unmarked-Anything/weights/sam3/safari_checkpoint_hf.pt}"
DEPTH_MODEL="${DEPTH_MODEL:-pi3x}"

# QA artifact locations, all relative to the repo root
QA_DIR="${QA_DIR:-outputs/qa}"
QA_SAMPLE="${QA_SAMPLE:-$QA_DIR/sample.csv}"
QA_RESULTS="${QA_RESULTS:-$QA_DIR/results.csv}"
QA_MASKS="${QA_MASKS:-$QA_DIR/masks}"
QA_DEPTH="${QA_DEPTH:-$QA_DIR/depth}"
QA_OVERLAYS="${QA_OVERLAYS:-$QA_DIR/overlays}"
