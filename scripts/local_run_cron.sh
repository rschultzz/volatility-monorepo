
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../apps/cron"
python -m venv .venv || true
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=../../packages
python job_orats_eod.py --date $(date +%F)
