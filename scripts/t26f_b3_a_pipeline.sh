#!/usr/bin/env bash
# T26F-B3-A remaining-pipeline driver (orchestration only).
#
# This wrapper does NOT implement or alter any scientific step: it invokes the
# already-frozen, hash-registered stage scripts in their contracted order with
# the same --run-dir argument, and stops immediately if any stage fails. Each
# stage is itself resume-safe, so re-running this script after an interruption
# continues from the last verified artifact.
#
# Contracted order (each gated by the previous stage's sentinel file):
#   D  rollouts   -> STAGE_D_ROLLOUTS_COMPLETE
#   E  posteval   -> STAGE_E_TARGETS_LOCKED      (invokes extract_targets)
#   F1 join       -> STAGE_F_TARGET_JOIN_COMPLETE
#   F2 analyze    -> STAGE_F_ANALYSIS_COMPLETE
#   F3 finalize   -> final manifest + reports

set -uo pipefail

RUN_DIR=/storage/alpasafe/safeworld-alpamayo/artifacts/safeworld_t26f_b3_a_prospective_untouched_scene_replication/20260718T204000Z
PY=/home/qiren/alpasafe/as_venv/bin/python
SCRIPTS=/home/qiren/alpasafe/scripts
STATUS="$RUN_DIR/logs/pipeline_status.txt"

export HF_HOME=/storage/hf_cache

log() { printf '%s | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$STATUS"; }

run_stage() {
  local name="$1" script="$2" logfile="$3" sentinel="$4"
  if [ -n "$sentinel" ] && [ -f "$RUN_DIR/$sentinel" ]; then
    log "SKIP  $name (sentinel $sentinel already present)"
    return 0
  fi
  log "START $name -> logs/$(basename "$logfile")"
  "$PY" "$SCRIPTS/$script" --run-dir "$RUN_DIR" >> "$logfile" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "FAIL  $name exit=$rc (see logs/$(basename "$logfile"))"
    log "PIPELINE_HALTED"
    exit $rc
  fi
  if [ -n "$sentinel" ] && [ ! -f "$RUN_DIR/$sentinel" ]; then
    log "FAIL  $name exited 0 but sentinel $sentinel is missing"
    log "PIPELINE_HALTED"
    exit 90
  fi
  log "DONE  $name"
}

log "PIPELINE_START host=$(hostname) run_dir=$RUN_DIR"

run_stage "D  rollouts (2400)" t26f_b3_a_rollouts.py        "$RUN_DIR/logs/stageD_master.log"   STAGE_D_ROLLOUTS_COMPLETE
run_stage "E  official posteval" t26f_b3_a_posteval.py      "$RUN_DIR/logs/stageE_master.log"   STAGE_E_TARGETS_LOCKED
run_stage "F1 target join"      t26f_b3_a_join.py           "$RUN_DIR/logs/stageF_join.log"     STAGE_F_TARGET_JOIN_COMPLETE
run_stage "F2 analysis"         t26f_b3_a_analyze.py        "$RUN_DIR/logs/stageF_analyze.log"  STAGE_F_ANALYSIS_COMPLETE
run_stage "F3 finalize"         t26f_b3_a_finalize.py       "$RUN_DIR/logs/stageF_finalize.log" ""

log "PIPELINE_COMPLETE"
