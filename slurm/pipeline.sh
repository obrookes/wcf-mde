#!/usr/bin/env bash
# The end-to-end wcf-mde pipeline driver: fps -> qc -> infer -> calibrate -> export -> score,
# plus the optional cohort/QA funnel (cohort -> cohort-infer -> prefilter -> overlays ->
# verdicts -> report) and the laptop mask-review loop (review).
#
#   bash slurm/pipeline.sh <subcommand> [--force] [--limit N] [--with-cohort]
#
# Every path this script reads or writes is derived from slurm/env.sh (RUN_NAME, DATA_ROOT,
# RUN_ROOT, ANNOTATIONS_CSV, MODEL_LABEL, SAM3_CHECKPOINT, DEPTH_MODEL, ...) -- override any of
# them as an env var, e.g. `RUN_NAME=full bash slurm/pipeline.sh status`.
#
# HARD RULE (enforced by construction, not just documented): every stage below writes only
# under $RUN_ROOT (or slurm/logs/, for the sbatch scheduler's own stdout/stderr). Nothing here
# ever touches $SCRATCH_ROOT/export_test, $SCRATCH_ROOT/qa_pilot, or this repo's outputs/ --
# those belong to the earlier ad-hoc pilot and other pipelines, not this driver. If you are
# editing this file: every `--out`/`--out-dir`/`--*-csv` you add must resolve under $RUN_ROOT.
#
# Subcommands (legacy stage numbers from slurm/README.md / README.md in brackets):
#   status         table of every stage: done / pending, plus RUN_ROOT/DATA_ROOT/RUN_NAME
#   check          diagnostic preflight (no sentinel, re-runnable, safe to run anytime)
#   setup          one-time env completion + slurm/preflight.sh (idempotent; never in `all`)
#   fps            [Step 0]   probe_video_fps.py            -> qc/video_fps.csv
#   qc             [Step 0]   qc_annotations.py --fix        -> qc/*_clean.csv, qc/qc_flags_*.csv
#   infer          [Stage 1]  run_calibration_eval.py (GPU)  -> infer/{calibration_results.csv,masks,depth_orig}
#   benchmark      (optional, not in `all`) benchmark_calibration.py -> calib/calibration_benchmark.csv
#   calibrate      [Stage 2]  calibrate_depth.py             -> calib/{depth_calib,calibration_fits.csv}
#   export         export_calibrated.py (CPU)                -> export/{frames,depth_maps,masks}
#   score          score_masks.py + heuristic_verdicts.py    -> triage_b/{mask_scores.csv,verdicts_heuristic.csv}
#   cohort         [Stage 0 pilot] qa/sample.py               -> qa/{sample.csv,cohort.csv}
#   cohort-infer   [Stage 1 pilot] run_calibration_eval.py (GPU) over the cohort -> qa/stage1/*
#   prefilter      [Stage 2 pilot] qa/prefilter.py            -> qa/prefilter.csv
#   overlays       [Stage 3a pilot] qa/render_overlays.py     -> qa/{overlays,overlay_manifest.csv}
#   verdicts       [Stage 3b pilot] STOP POINT: prints the in-session subagent grading workflow
#   report         [Stage 5 pilot] qa/report.py goldset + hand-labelling stop-point instructions
#   review         [Stage 4 pilot] qa/make_review_bundle.py (+ apply_corrections.py if ready)
#   all            fps -> qc -> infer -> calibrate -> export -> score
#                  (--with-cohort additionally weaves in cohort/cohort-infer/prefilter/overlays;
#                   stops before verdicts either way -- that stage can never be non-interactive)
#
# Flags:
#   --force        redo the NAMED stage even if its sentinel says done (only applies to a single
#                  named stage -- `all --force` does NOT cascade and force every sub-stage)
#   --limit N      forwarded as `--limit N` to run_calibration_eval.py (infer/cohort-infer only)
#   --with-cohort  `all` also runs the cohort/QA-pilot funnel alongside the full-corpus one
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/slurm/env.sh"
cd "$REPO_ROOT"

DONE_DIR="$RUN_ROOT/.done"
LOG_DIR="$RUN_ROOT/logs"

# --------------------------------------------------------------------------------------
# small helpers shared by every stage
# --------------------------------------------------------------------------------------

usage() {
    sed -n '2,44p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# qc_annotations.py writes its outputs NEXT TO its input (qc_flags_<stem>.csv, <stem>_clean.csv
# in the input's own directory) -- this is the one place that derives those names from the
# copied-into-$RUN_ROOT/qc stem, so `qc` and every later stage agree on the same paths.
qc_paths() {
    local basename stem
    basename="$(basename "$ANNOTATIONS_CSV")"
    stem="${basename%.csv}"
    QC_COPY="$RUN_ROOT/qc/$basename"
    QC_CLEAN="$RUN_ROOT/qc/${stem}_clean.csv"
    QC_FLAGS_CSV="$RUN_ROOT/qc/qc_flags_${stem}.csv"
}

sentinel_path() { echo "$DONE_DIR/$1.ok"; }

# Returns 0 (proceed) if the stage should run, 1 (skip) if its sentinel already says done.
# --force (per-stage, never cascaded through `all`) clears the sentinel first.
stage_guard() {
    local stage="$1" force="$2" sentinel
    sentinel="$(sentinel_path "$stage")"
    if [[ "$force" == "1" ]]; then
        rm -f "$sentinel"
    fi
    if [[ -f "$sentinel" ]]; then
        echo "[$stage] already done ($sentinel) -- skipping. Use --force to redo."
        return 1
    fi
    mkdir -p "$DONE_DIR" "$LOG_DIR"
    return 0
}

mark_done() {
    mkdir -p "$DONE_DIR"
    touch "$(sentinel_path "$1")"
}

require_file() {
    local path="$1" msg="$2"
    [[ -f "$path" ]] || { echo "!! missing $path -- $msg" >&2; exit 1; }
}

require_dir() {
    local path="$1" msg="$2"
    [[ -d "$path" ]] || { echo "!! missing $path -- $msg" >&2; exit 1; }
}

# --------------------------------------------------------------------------------------
# status / check / setup
# --------------------------------------------------------------------------------------

cmd_status() {
    mkdir -p "$DONE_DIR" "$LOG_DIR"
    echo "RUN_NAME  : $RUN_NAME"
    echo "DATA_ROOT : $DATA_ROOT"
    echo "RUN_ROOT  : $RUN_ROOT"
    echo
    printf '%-14s %-8s %s\n' "STAGE" "STATUS" "SENTINEL / ARTIFACT"
    local stage sentinel
    for stage in fps qc infer benchmark calibrate export score \
                 cohort cohort-infer prefilter overlays report review; do
        sentinel="$(sentinel_path "$stage")"
        if [[ -f "$sentinel" ]]; then
            printf '%-14s %-8s %s\n' "$stage" "done" "$sentinel"
        else
            printf '%-14s %-8s %s\n' "$stage" "pending" "$sentinel"
        fi
    done
    # verdicts is a manual stop-point: pipeline.sh never writes its own sentinel for it (see
    # cmd_verdicts), so status reports on the merged output file instead.
    if [[ -f "$RUN_ROOT/qa/verdicts_subagent.csv" ]]; then
        printf '%-14s %-8s %s\n' "verdicts" "done" "$RUN_ROOT/qa/verdicts_subagent.csv"
    else
        printf '%-14s %-8s %s\n' "verdicts" "pending" \
            "$RUN_ROOT/qa/verdicts_subagent.csv (manual: bash slurm/pipeline.sh verdicts)"
    fi
}

cmd_check() {
    local hard_fail=0
    echo "=== check: diagnostic preflight (no sentinel; safe to re-run anytime) ==="
    activate_env || echo "  !! activate_env failed -- imports in (3) will likely fail too" >&2

    echo
    echo "--- (1) video coverage: $DATA_ROOT/list_reference_videos.xlsx vs disk (REPORT ONLY -- a transfer may be in flight) ---"
    if ! python - "$REPO_ROOT" "$DATA_ROOT" <<'PY'
import sys
from pathlib import Path

repo_root, data_dir = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo_root))
from scripts.video_lookup import load_anno_to_path

xlsx = data_dir / "list_reference_videos.xlsx"
mapping = load_anno_to_path(xlsx, data_dir)
missing = [(anno, path) for anno, path in mapping.items() if not path.exists()]
print(f"  {len(mapping)} videos listed in {xlsx}")
print(f"  {len(missing)} missing on disk")
for anno, path in missing[:10]:
    print(f"    missing: {anno} -> {path}")
if len(missing) > 10:
    print(f"    ... and {len(missing) - 10} more")
PY
    then
        echo "  !! video coverage check errored (not counted as a hard failure -- report only)"
    fi

    echo
    echo "--- (2) SAM-3 checkpoint ---"
    if [[ ! -e "$SAM3_CHECKPOINT" ]]; then
        echo "  FAIL  checkpoint not found: $SAM3_CHECKPOINT"
        hard_fail=1
    elif [[ "$(basename "$SAM3_CHECKPOINT")" == .* ]]; then
        echo "  FAIL  checkpoint basename starts with '.' ($(basename "$SAM3_CHECKPOINT")) -- looks like an in-progress rsync/scp temp file"
        hard_fail=1
    else
        local size1 size2
        size1="$(stat -c%s "$SAM3_CHECKPOINT")"
        echo "  ..    checkpoint present ($size1 bytes); waiting 15s to confirm its size is stable ..."
        sleep 15
        size2="$(stat -c%s "$SAM3_CHECKPOINT")"
        if [[ "$size1" == "$size2" ]]; then
            echo "  PASS  checkpoint size stable at $size2 bytes"
        else
            echo "  FAIL  checkpoint size changed ($size1 -> $size2 bytes) -- still being written"
            hard_fail=1
        fi
    fi

    echo
    echo "--- (3) python imports ---"
    if python -c "import ultralytics" 2>&1; then
        echo "  PASS  ultralytics imports"
    else
        echo "  FAIL  ultralytics import failed (see traceback above)"
        hard_fail=1
    fi
    local depth_label depth_import
    if [[ "$DEPTH_MODEL" == "da3" ]]; then
        depth_label="depth_anything_3"
        depth_import="from depth_anything_3.api import DepthAnything3"
    else
        depth_label="pi3 (pi3.models.pi3x)"
        depth_import="from pi3.models.pi3x import Pi3X"
    fi
    if python -c "$depth_import" 2>&1; then
        echo "  PASS  $depth_label imports"
    else
        echo "  FAIL  $depth_label import failed (see traceback above)"
        hard_fail=1
    fi

    echo
    echo "--- (4) Hugging Face cache (warn-only) ---"
    if [[ -d "$HF_HOME" && -n "$(ls -A "$HF_HOME" 2>/dev/null)" ]]; then
        echo "  PASS  HF_HOME ($HF_HOME) exists and is non-empty"
    else
        echo "  WARN  HF_HOME ($HF_HOME) missing or empty -- run 'bash slurm/pipeline.sh setup' or 'bash slurm/preflight.sh' before infer/cohort-infer"
    fi

    echo
    echo "--- (5) verdict grading ---"
    echo "  NOTE  'verdicts'/'report' grade masks IN-SESSION via a Claude Code subagent -- no ANTHROPIC_API_KEY or network needed"

    echo
    if [[ "$hard_fail" -ne 0 ]]; then
        echo "=== check: FAIL (see FAIL lines above) ==="
        return 1
    fi
    echo "=== check: PASS ==="
}

cmd_setup() {
    echo "=== setup: idempotent one-time environment completion (never run as part of 'all') ==="
    if ! activate_env; then
        echo "!! conda env 'wcf-pipe' not found/activatable -- create it first, then re-run setup" >&2
        exit 1
    fi

    check_and_install() {
        local module="$1" pip_spec="$2"
        if python -c "import $module" >/dev/null 2>&1; then
            echo "  ok    $module already importable"
        else
            echo "  ..    installing (missing $module): pip install $pip_spec"
            pip install "$pip_spec"
        fi
    }

    check_and_install ultralytics ultralytics
    # SAM-3's promptable text segmentation needs Ultralytics' CLIP fork (see README.md)
    check_and_install clip "git+https://github.com/ultralytics/CLIP.git"

    if [[ "$DEPTH_MODEL" == "da3" ]]; then
        if python -c "import depth_anything_3" >/dev/null 2>&1; then
            echo "  ok    depth_anything_3 already importable"
        else
            echo "  ..    depth_anything_3 missing -- cloning + installing into third_party/"
            [[ -d "$REPO_ROOT/third_party/depth_anything_3" ]] || \
                git clone https://github.com/bytedance-seed/depth-anything-3 \
                    "$REPO_ROOT/third_party/depth_anything_3"
            pip install -e "$REPO_ROOT/third_party/depth_anything_3"
        fi
    else
        if python -c "from pi3.models.pi3x import Pi3X" >/dev/null 2>&1; then
            echo "  ok    pi3x already importable"
        else
            echo "  ..    pi3x missing -- installing from third_party/Pi3 (clone it first if absent)"
            require_dir "$REPO_ROOT/third_party/Pi3" \
                "clone Pi3 into third_party/Pi3 first (see README.md), setup cannot fetch it"
            pip install -e "$REPO_ROOT/third_party/Pi3"
        fi
    fi

    echo
    echo "running slurm/preflight.sh (warms the HF cache on this login node) ..."
    bash "$REPO_ROOT/slurm/preflight.sh"
}

# --------------------------------------------------------------------------------------
# main funnel: fps -> qc -> infer -> calibrate -> export -> score
# --------------------------------------------------------------------------------------

cmd_fps() {
    local force="$1"
    stage_guard fps "$force" || return 0
    mkdir -p "$RUN_ROOT/qc"
    activate_env || true
    (
        set -euo pipefail
        python "$REPO_ROOT/scripts/probe_video_fps.py" \
            --data-dir "$DATA_ROOT" \
            --out "$RUN_ROOT/qc/video_fps.csv"
    ) 2>&1 | tee "$LOG_DIR/fps.log"
    mark_done fps
}

cmd_qc() {
    local force="$1"
    stage_guard qc "$force" || return 0
    require_file "$RUN_ROOT/qc/video_fps.csv" "run 'bash slurm/pipeline.sh fps' first"
    mkdir -p "$RUN_ROOT/qc"
    cp -f "$ANNOTATIONS_CSV" "$QC_COPY"
    activate_env || true
    (
        set -euo pipefail
        # --compare-export points at a path that deliberately never exists: passing '' hits a
        # pathlib quirk (Path("") == Path(".")) that makes qc_annotations.py try to read the
        # cwd as a CSV instead of skipping the (optional, informational-only) comparison.
        python "$REPO_ROOT/scripts/qc_annotations.py" \
            "$QC_COPY" \
            --fps-table "$RUN_ROOT/qc/video_fps.csv" \
            --compare-export "$RUN_ROOT/qc/.no-compare-export.csv" \
            --fix
    ) 2>&1 | tee "$LOG_DIR/qc.log"
    mark_done qc
}

cmd_infer() {
    local force="$1"
    stage_guard infer "$force" || return 0
    require_file "$QC_CLEAN" "run 'bash slurm/pipeline.sh qc' first"
    mkdir -p "$RUN_ROOT/infer"
    export PIPE_ANNOTATIONS_CSV="$QC_CLEAN"
    export PIPE_RESULTS_CSV="$RUN_ROOT/infer/calibration_results.csv"
    export PIPE_MASKS_DIR="$RUN_ROOT/infer/masks"
    export PIPE_DEPTH_DIR="$RUN_ROOT/infer/depth_orig"
    export PIPE_EXTRA="${LIMIT:+--limit $LIMIT}"
    (
        set -euo pipefail
        cd "$REPO_ROOT"
        bash slurm/submit.sh pipeline_infer --wait
    ) 2>&1 | tee "$LOG_DIR/infer.log"
    mark_done infer
}

cmd_benchmark() {
    local force="$1"
    stage_guard benchmark "$force" || return 0
    require_file "$RUN_ROOT/infer/calibration_results.csv" "run 'bash slurm/pipeline.sh infer' first"
    mkdir -p "$RUN_ROOT/calib"
    activate_env || true
    (
        set -euo pipefail
        python "$REPO_ROOT/scripts/benchmark_calibration.py" \
            --results-csv "$RUN_ROOT/infer/calibration_results.csv" \
            --qc-flags "$QC_FLAGS_CSV" \
            --annotations-csv "$QC_CLEAN" \
            --video-list-xlsx "$DATA_ROOT/list_reference_videos.xlsx" \
            --data-dir "$DATA_ROOT" \
            --depth-dir "$RUN_ROOT/infer/depth_orig" \
            --masks-dir "$RUN_ROOT/infer/masks" \
            --out "$RUN_ROOT/calib/calibration_benchmark.csv"
    ) 2>&1 | tee "$LOG_DIR/benchmark.log"
    mark_done benchmark
}

cmd_calibrate() {
    local force="$1"
    stage_guard calibrate "$force" || return 0
    require_file "$RUN_ROOT/infer/calibration_results.csv" "run 'bash slurm/pipeline.sh infer' first"
    mkdir -p "$RUN_ROOT/calib/depth_calib" "$RUN_ROOT/calib/depth_calib_viz"
    activate_env || true
    (
        set -euo pipefail
        python "$REPO_ROOT/scripts/calibrate_depth.py" \
            --results-csv "$RUN_ROOT/infer/calibration_results.csv" \
            --depth-dir "$RUN_ROOT/infer/depth_orig" \
            --out-dir "$RUN_ROOT/calib/depth_calib" \
            --fits-csv "$RUN_ROOT/calib/calibration_fits.csv" \
            --viz-dir "$RUN_ROOT/calib/depth_calib_viz" \
            --masks-dir "$RUN_ROOT/infer/masks" \
            --qc-flags "$QC_FLAGS_CSV" \
            --annotations-csv "$QC_CLEAN" \
            --video-list-xlsx "$DATA_ROOT/list_reference_videos.xlsx" \
            --data-dir "$DATA_ROOT"
    ) 2>&1 | tee "$LOG_DIR/calibrate.log"
    mark_done calibrate
}

cmd_export() {
    local force="$1"
    stage_guard export "$force" || return 0
    require_dir "$RUN_ROOT/calib/depth_calib" "run 'bash slurm/pipeline.sh calibrate' first"
    mkdir -p "$RUN_ROOT/export"
    export PIPE_CALIB_DIR="$RUN_ROOT/calib/depth_calib"
    export PIPE_OUT_DIR="$RUN_ROOT/export"
    export PIPE_QC_FLAGS="$QC_FLAGS_CSV"
    export PIPE_ANNOTATIONS_CSV="$QC_CLEAN"
    export PIPE_MASKS_DIR="$RUN_ROOT/infer/masks"
    (
        set -euo pipefail
        cd "$REPO_ROOT"
        bash slurm/submit.sh pipeline_export --wait
    ) 2>&1 | tee "$LOG_DIR/export.log"
    mark_done export
}

cmd_score() {
    local force="$1"
    stage_guard score "$force" || return 0
    require_dir "$RUN_ROOT/export" "run 'bash slurm/pipeline.sh export' first"
    mkdir -p "$RUN_ROOT/triage_b"
    activate_env || true
    (
        set -euo pipefail
        python "$REPO_ROOT/scripts/score_masks.py" \
            --export-dir "$RUN_ROOT/export" \
            --out-dir "$RUN_ROOT/triage_b" \
            --workers "${SCORE_WORKERS:-12}"
        python "$REPO_ROOT/scripts/qa/heuristic_verdicts.py" \
            --scores "$RUN_ROOT/triage_b/mask_scores.csv" \
            --out "$RUN_ROOT/triage_b/verdicts_heuristic.csv" \
            --model-label "$MODEL_LABEL"
    ) 2>&1 | tee "$LOG_DIR/score.log"
    mark_done score
}

# --------------------------------------------------------------------------------------
# cohort / QA-pilot funnel
# --------------------------------------------------------------------------------------

cmd_cohort() {
    local force="$1"
    stage_guard cohort "$force" || return 0
    require_file "$QC_CLEAN" "run 'bash slurm/pipeline.sh qc' first"
    mkdir -p "$RUN_ROOT/qa"
    activate_env || true
    (
        set -euo pipefail
        python "$REPO_ROOT/scripts/qa/sample.py" \
            --annotations-csv "$QC_CLEAN" \
            --qc-flags "$QC_FLAGS_CSV" \
            --fps-table "$RUN_ROOT/qc/video_fps.csv" \
            --out "$RUN_ROOT/qa/sample.csv" \
            --cohort-out "$RUN_ROOT/qa/cohort.csv"
    ) 2>&1 | tee "$LOG_DIR/cohort.log"
    mark_done cohort
}

cmd_cohort_infer() {
    local force="$1"
    stage_guard cohort-infer "$force" || return 0
    require_file "$RUN_ROOT/qa/sample.csv" "run 'bash slurm/pipeline.sh cohort' first"
    mkdir -p "$RUN_ROOT/qa/stage1"
    export PIPE_ANNOTATIONS_CSV="$RUN_ROOT/qa/sample.csv"
    export PIPE_RESULTS_CSV="$RUN_ROOT/qa/stage1/results.csv"
    export PIPE_MASKS_DIR="$RUN_ROOT/qa/stage1/masks"
    export PIPE_DEPTH_DIR="$RUN_ROOT/qa/stage1/depth"
    export PIPE_EXTRA="${LIMIT:+--limit $LIMIT}"
    (
        set -euo pipefail
        cd "$REPO_ROOT"
        bash slurm/submit.sh pipeline_infer --wait
    ) 2>&1 | tee "$LOG_DIR/cohort-infer.log"
    mark_done cohort-infer
}

cmd_prefilter() {
    local force="$1"
    stage_guard prefilter "$force" || return 0
    require_file "$RUN_ROOT/qa/stage1/results.csv" "run 'bash slurm/pipeline.sh cohort-infer' first"
    mkdir -p "$RUN_ROOT/qa"
    activate_env || true
    (
        set -euo pipefail
        python "$REPO_ROOT/scripts/qa/prefilter.py" \
            --results-csv "$RUN_ROOT/qa/stage1/results.csv" \
            --mask-dir "$RUN_ROOT/qa/stage1/masks" \
            --sample-csv "$RUN_ROOT/qa/sample.csv" \
            --out "$RUN_ROOT/qa/prefilter.csv"
    ) 2>&1 | tee "$LOG_DIR/prefilter.log"
    mark_done prefilter
}

cmd_overlays() {
    local force="$1"
    stage_guard overlays "$force" || return 0
    require_file "$RUN_ROOT/qa/prefilter.csv" "run 'bash slurm/pipeline.sh prefilter' first"
    mkdir -p "$RUN_ROOT/qa/overlays"
    activate_env || true
    # CPU-bound over a few hundred frames -- cheap enough to run inline on the login node
    # rather than round-tripping through another sbatch script (judgment call; see the driver
    # report for the reasoning).
    (
        set -euo pipefail
        python "$REPO_ROOT/scripts/qa/render_overlays.py" \
            --prefilter-csv "$RUN_ROOT/qa/prefilter.csv" \
            --mask-dir "$RUN_ROOT/qa/stage1/masks" \
            --out-dir "$RUN_ROOT/qa/overlays" \
            --manifest "$RUN_ROOT/qa/overlay_manifest.csv" \
            --video-list "$DATA_ROOT/list_reference_videos.xlsx" \
            --data-dir "$DATA_ROOT"
    ) 2>&1 | tee "$LOG_DIR/overlays.log"
    mark_done overlays
}

cmd_verdicts() {
    require_file "$RUN_ROOT/qa/overlay_manifest.csv" "run 'bash slurm/pipeline.sh overlays' first"
    cat <<EOF
=== verdicts: STOP POINT -- this step runs IN-SESSION, not as a script ===

scripts/qa/merge_verdicts.py expects verdict-chunk JSONs written by a Claude Code subagent
grading the overlay manifest; slurm/pipeline.sh cannot do that grading itself. Do this by hand
(or hand it to a subagent), then re-run 'bash slurm/pipeline.sh report'.

1. Chunk $RUN_ROOT/qa/overlay_manifest.csv into groups of ~20-40 rows (small enough for one
   subagent call to look at every referenced overlay PNG and grade it consistently).

2. For each chunk, have a subagent look at every overlay image the chunk's rows reference and
   write $RUN_ROOT/qa/verdict_chunks/chunk_NN_verdicts.json as a JSON array of:
       {"file": <overlay_path basename>, "verdict": <ok|empty|wrong-subject|bleed|split|multiple>,
        "confidence": <0-1 float>, "rationale": <string>}
   one entry per graded row. The verdict classes must match scripts/qa/verdicts_schema.py's
   VERDICT_CLASSES; see scripts/qa/merge_verdicts.py's module docstring for the exact contract.

3. Once every chunk file exists, merge them into a verdicts CSV:

     python scripts/qa/merge_verdicts.py \\
         --manifest "$RUN_ROOT/qa/overlay_manifest.csv" \\
         --chunks '$RUN_ROOT/qa/verdict_chunks/chunk_*_verdicts.json' \\
         --out "$RUN_ROOT/qa/verdicts_subagent.csv"

This driver does not run step 3 for you -- do it once every chunk exists, then run
'bash slurm/pipeline.sh report' to fold the merged verdicts into the funnel summary.
EOF
}

cmd_report() {
    local force="$1"
    stage_guard report "$force" || return 0
    require_file "$RUN_ROOT/qa/verdicts_subagent.csv" \
        "run 'bash slurm/pipeline.sh verdicts' and merge_verdicts.py first"
    mkdir -p "$RUN_ROOT/qa"
    activate_env || true
    (
        set -euo pipefail
        python "$REPO_ROOT/scripts/qa/report.py" goldset \
            --prefilter-csv "$RUN_ROOT/qa/prefilter.csv" \
            --verdicts "$RUN_ROOT/qa/verdicts_subagent.csv" \
            --manifest "$RUN_ROOT/qa/overlay_manifest.csv" \
            --out "$RUN_ROOT/qa/gold_template.csv"
    ) 2>&1 | tee "$LOG_DIR/report.log"
    mark_done report

    cat <<EOF

=== report: gold_template.csv written -- hand-labelling stop point ===

$RUN_ROOT/qa/gold_template.csv is BLIND by design (no model verdict shown -- see
scripts/qa/report.py's module docstring, the whole point is an independent trust anchor).
Fill in its gold_verdict column by hand, then run:

    python scripts/qa/report.py summarise \\
        --prefilter-csv "$RUN_ROOT/qa/prefilter.csv" \\
        --verdicts "$RUN_ROOT/qa/verdicts_subagent.csv" \\
        --gold "$RUN_ROOT/qa/gold_labelled.csv" \\
        --out "$RUN_ROOT/qa/report.md"

This driver does not run 'summarise' for you -- it needs the hand-labelled file to exist first.
EOF
}

cmd_review() {
    local force="$1"
    stage_guard review "$force" || return 0
    require_file "$RUN_ROOT/triage_b/verdicts_heuristic.csv" "run 'bash slurm/pipeline.sh score' first"
    require_dir "$RUN_ROOT/export" "run 'bash slurm/pipeline.sh export' first"
    mkdir -p "$RUN_ROOT/review"
    activate_env || true
    (
        set -euo pipefail
        python "$REPO_ROOT/scripts/qa/make_review_bundle.py" \
            --verdicts "$RUN_ROOT/triage_b/verdicts_heuristic.csv" \
            --frames-dir "$RUN_ROOT/export/frames" \
            --masks-dir "$RUN_ROOT/export/masks" \
            --out "$RUN_ROOT/review/review_bundle.tar.gz" \
            --include-ok
    ) 2>&1 | tee "$LOG_DIR/review.log"

    cat <<EOF

=== review: bundle written to $RUN_ROOT/review/review_bundle.tar.gz ===

Copy it to your laptop and review offline (no SLURM job, no tunnel):
    scp <cluster>:$RUN_ROOT/review/review_bundle.tar.gz .
    tar xzf review_bundle.tar.gz && cd review_bundle
    pip install -r requirements.txt
    python run_review.py            # then open http://localhost:8765

When done, copy corrections.csv back and re-run 'bash slurm/pipeline.sh review --force':
    scp corrections.csv <cluster>:$RUN_ROOT/review/corrections.csv
EOF

    if [[ -f "$RUN_ROOT/review/corrections.csv" ]]; then
        echo
        echo "found $RUN_ROOT/review/corrections.csv -- applying morphology auto-fixes ..."
        (
            set -euo pipefail
            python "$REPO_ROOT/scripts/qa/apply_corrections.py" morph \
                --corrections "$RUN_ROOT/review/corrections.csv" \
                --masks-dir "$RUN_ROOT/export/masks" \
                --out-masks-dir "$RUN_ROOT/review/masks_corrected" \
                --applied-csv "$RUN_ROOT/review/applied.csv"
        ) 2>&1 | tee -a "$LOG_DIR/review.log"
    fi
    mark_done review
}

# --------------------------------------------------------------------------------------
# all
# --------------------------------------------------------------------------------------

cmd_all() {
    if [[ -n "$FORCE" ]]; then
        echo "note: --force has no effect on 'all' -- force an individual stage by name instead"
    fi

    cmd_fps ""
    cmd_qc ""

    if [[ -n "$WITH_COHORT" ]]; then
        cmd_cohort ""
        echo "--- launching infer and cohort-infer in parallel (two backgrounded GPU submissions) ---"
        ( cmd_cohort_infer "" ) & local cohort_pid=$!
        ( cmd_infer "" )        & local infer_pid=$!
        local rc=0
        wait "$cohort_pid" || rc=1
        wait "$infer_pid"  || rc=1
        if [[ "$rc" -ne 0 ]]; then
            echo "!! infer and/or cohort-infer failed -- stopping before calibrate" >&2
            exit 1
        fi
    else
        cmd_infer ""
    fi

    cmd_calibrate ""
    cmd_export ""
    cmd_score ""

    if [[ -n "$WITH_COHORT" ]]; then
        cmd_prefilter ""
        cmd_overlays ""
        echo
        echo "=== all (--with-cohort) done. Next stop point: bash slurm/pipeline.sh verdicts ==="
    else
        echo
        echo "=== all done. ==="
        echo "    Cohort/QA-pilot funnel (cohort/prefilter/overlays/verdicts/report) needs --with-cohort."
        echo "    Full-corpus next step: bash slurm/pipeline.sh review"
    fi
}

# --------------------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------------------

SUBCOMMAND="${1:-}"
if [[ -z "$SUBCOMMAND" ]]; then
    usage
    exit 2
fi
shift

FORCE=""
LIMIT=""
WITH_COHORT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --limit) LIMIT="${2:?--limit needs a value}"; shift 2 ;;
        --with-cohort) WITH_COHORT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown flag: $1" >&2; usage; exit 2 ;;
    esac
done

qc_paths

case "$SUBCOMMAND" in
    status)       cmd_status ;;
    check)        cmd_check ;;
    setup)        cmd_setup ;;
    fps)          cmd_fps "$FORCE" ;;
    qc)           cmd_qc "$FORCE" ;;
    infer)        cmd_infer "$FORCE" ;;
    benchmark)    cmd_benchmark "$FORCE" ;;
    calibrate)    cmd_calibrate "$FORCE" ;;
    export)       cmd_export "$FORCE" ;;
    score)        cmd_score "$FORCE" ;;
    cohort)       cmd_cohort "$FORCE" ;;
    cohort-infer) cmd_cohort_infer "$FORCE" ;;
    prefilter)    cmd_prefilter "$FORCE" ;;
    overlays)     cmd_overlays "$FORCE" ;;
    verdicts)     cmd_verdicts ;;
    report)       cmd_report "$FORCE" ;;
    review)       cmd_review "$FORCE" ;;
    all)          cmd_all ;;
    *)
        echo "unknown subcommand: $SUBCOMMAND" >&2
        usage
        exit 2
        ;;
esac
