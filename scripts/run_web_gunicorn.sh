#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8060}"

# Kill anything currently listening on PORT (macOS)
if lsof -ti "tcp:${PORT}" >/dev/null 2>&1; then
  echo "[run_web_gunicorn] freeing port ${PORT}..."
  lsof -ti "tcp:${PORT}" | xargs kill -9 || true
fi

cd "$(dirname "$0")/../apps/web"

exec ./.venv/bin/gunicorn app:server \
  --bind "0.0.0.0:${PORT}" \
  --workers 4 \
  --threads 2 \
  --timeout 120 \
  --reload
