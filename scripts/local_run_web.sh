
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../apps/web"
python -m venv .venv || true
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=../../packages
python app.py
