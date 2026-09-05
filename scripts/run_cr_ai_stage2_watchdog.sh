#!/bin/bash
# CR-AI Stage 2 overnight watchdog.
# Runs cr_ai_stage2_backfill.py and restarts on failure (up to MAX_ATTEMPTS times).
# Launch with:  nohup bash scripts/run_cr_ai_stage2_watchdog.sh >/dev/null 2>&1 & disown

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO_ROOT/scripts/logs/cr_ai_stage2_watchdog.log"
BACKFILL_SCRIPT="$REPO_ROOT/scripts/cr_ai_stage2_backfill.py"

MAX_ATTEMPTS=8
WAIT_BETWEEN=300   # 5 min between retries

mkdir -p "$REPO_ROOT/scripts/logs"
cd "$REPO_ROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Watchdog started (pid=$$, host=$(hostname -s)) ==="
log "MAX_ATTEMPTS=$MAX_ATTEMPTS  WAIT_BETWEEN=${WAIT_BETWEEN}s"
log "Script: $BACKFILL_SCRIPT"

for ATTEMPT in $(seq 1 $MAX_ATTEMPTS); do
    log "--- Attempt $ATTEMPT / $MAX_ATTEMPTS ---"

    python3 "$BACKFILL_SCRIPT" >> "$LOG" 2>&1
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
