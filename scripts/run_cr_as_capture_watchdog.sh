#!/bin/bash
# CR-AS Step 1 overnight watchdog (pattern: run_cr_ai_stage2_watchdog.sh).
# Runs cr_as_capture_0dte_condor_legs.py under an 8-hour budget and restarts on
# failure (up to MAX_ATTEMPTS). The script checkpoints per date and measures the
# budget from the first start recorded in the checkpoint, so restarts resume
# where they left off and never extend the budget.
# Launch with:  nohup bash scripts/run_cr_as_capture_watchdog.sh >/dev/null 2>&1 & disown

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO_ROOT/scripts/logs/cr_as_capture_watchdog.log"
SCRIPT="$REPO_ROOT/scripts/cr_as_capture_0dte_condor_legs.py"
PYTHON="$REPO_ROOT/apps/web/.venv/bin/python"
MAX_HOURS="${MAX_HOURS:-8}"

MAX_ATTEMPTS=8
WAIT_BETWEEN=300   # 5 min between retries

mkdir -p "$REPO_ROOT/scripts/logs"
cd "$REPO_ROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== CR-AS capture watchdog started (pid=$$, host=$(hostname -s)) ==="
log "MAX_ATTEMPTS=$MAX_ATTEMPTS  WAIT_BETWEEN=${WAIT_BETWEEN}s  MAX_HOURS=$MAX_HOURS  extra args: $*"

for ATTEMPT in $(seq 1 $MAX_ATTEMPTS); do
    log "--- Attempt $ATTEMPT / $MAX_ATTEMPTS ---"
    PYTHONUNBUFFERED=1 "$PYTHON" -u "$SCRIPT" --max-hours "$MAX_HOURS" "$@" >> "$LOG" 2>&1
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        log "SUCCESS on attempt $ATTEMPT. Watchdog exiting."
        exit 0
    fi
    log "FAILED (exit=$EXIT_CODE) on attempt $ATTEMPT."
    if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then
        log "Sleeping ${WAIT_BETWEEN}s before retry..."
        sleep $WAIT_BETWEEN
    fi
done

log "GAVE UP after $MAX_ATTEMPTS attempts."
exit 1
