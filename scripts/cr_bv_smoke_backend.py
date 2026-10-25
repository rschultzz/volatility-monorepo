"""CR-BV Step 0 smoke backend: auth-free, read-only (PGOPTIONS default_transaction_read_only=on, backfill role).
Registers only the Ironbeam (bars + gex-landscape) and Today Setup routes. Port 8060."""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the checkout this script lives in
for line in open(os.path.join(REPO, ".env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("="); os.environ.setdefault(k.strip(), v.strip())
os.environ["DATABASE_URL"] = os.environ["BACKFILL_DATABASE_URL"]
os.environ["PGOPTIONS"] = "-c default_transaction_read_only=on"
for p in (REPO, os.path.join(REPO, "apps", "web")):
    sys.path.insert(0, p)
import dash
import modules.Ironbeam.callbacks as cb
from modules.Analogues.routes import register_analogues_routes
from modules.TodaySetup.routes import register_today_setup_routes
from modules.Bars.routes import register_bars_routes
from modules.AuditFlags.routes import register_audit_flags_routes
from modules.DayBrowser.routes import register_day_browser_routes
from modules.Proposals.routes import register_proposals_routes
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = dash.html.Div()
cb.register_ironbeam_callbacks(app)
for f in (register_analogues_routes, register_today_setup_routes, register_bars_routes,
          register_audit_flags_routes, register_day_browser_routes, register_proposals_routes):
    f(app.server)
from flask import request
@app.server.after_request
def _smoke_cors(resp):
    o = request.headers.get("Origin")
    if o and o.startswith("http://localhost:517") and "Access-Control-Allow-Origin" not in resp.headers:
        resp.headers["Access-Control-Allow-Origin"] = o
        resp.headers["Access-Control-Allow-Credentials"] = "true"
    return resp
if __name__ == "__main__":
    app.server.run(host="127.0.0.1", port=8060, debug=False)
