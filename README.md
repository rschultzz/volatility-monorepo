
# Volatility Monorepo (Skeleton)

This is a clean monorepo layout for your trading platform. Copy your working code into the appropriate folders, commit, and point Render services to the matching `rootDir`.

## Structure
```
apps/
  web/       # Dash web app
  cron/      # Nightly/cron jobs (ORATS EOD, backfills, etc.)
packages/
  shared/    # Reusable code (utils, ids, data_io, db helpers)
infra/
  sql/       # SQL migrations, seed scripts
scripts/     # Local helper scripts
render.yaml  # (optional) Declarative Render config for services
```

## Quick Start (local)

### Web
```bash
cd apps/web
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Copy your existing Dash app file into this folder (rename to app.py if needed)
# Ensure `server = app.server` exists if you want to run via gunicorn on Render
python app.py
```

### Cron
```bash
cd apps/cron
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Copy your job_orats_eod.py into this folder (or rename the placeholder)
python job_orats_eod.py --date 2025-10-31
```

> Tip: Put any shared modules into `packages/shared` and add `PYTHONPATH=packages` (Render env var) in each service so `from shared.utils import data_io` works everywhere.

## Render (two services)

- **Web Service**: rootDir `apps/web`, build `pip install -r requirements.txt`, start `gunicorn app:server` (or `python app.py` for dev).
- **Worker/Cron**: rootDir `apps/cron`, build `pip install -r requirements.txt`, start `python job_orats_eod.py --token $ORATS_API_KEY`. Add a Render cron schedule (e.g., weekdays 20:00 PT).

Keep your *old* Render services running for rollback. Stand up **new** services pointed at this repo, verify, then switch DNS/CNAME to the new web service and disable the old cron.

## Environment Variables (examples)

- `DATABASE_URL` — Render Postgres connection string
- `ORATS_API_KEY` — your ORATS token
- `PYTHONPATH=packages` — so both apps can import from `packages/shared`
- (Optional) `APP_ENV=production|staging` to guard side effects
