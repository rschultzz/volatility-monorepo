"""CR-DC (from CR-CH): the full Dash app (apps/web/app.py) locally, read-only, auth disabled. Port 8072."""
import os, sys, runpy
REPO = os.environ.get("CRCH_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for line in open(os.path.join(REPO, ".env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("="); os.environ.setdefault(k.strip(), v.strip())
os.environ["DATABASE_URL"] = os.environ["BACKFILL_DATABASE_URL"]   # every connection on the backfill role
os.environ["PGOPTIONS"] = "-c default_transaction_read_only=on"     # ...and read-only
import dash_auth
dash_auth.BasicAuth = lambda *a, **k: None                          # no login locally
sys.path.insert(0, os.path.join(REPO, "apps", "web"))
os.chdir(os.path.join(REPO, "apps", "web"))
mod = runpy.run_path(os.path.join(REPO, "apps", "web", "app.py"), run_name="crch_app")
mod["app"].run(host="127.0.0.1", port=int(os.environ.get("CRCH_DASH_PORT", "8072")), debug=False, threaded=True)
