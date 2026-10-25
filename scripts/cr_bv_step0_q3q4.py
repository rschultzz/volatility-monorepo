"""CR-BV Step 0 Q3/Q4 prototype (scratch, read-only): Corrected landscape = the CR-010 high path with each row's
kernel centre replaced by CR-BT corrected L (P_target = ES 15:45 ET close on A), rows/filters from CR-BT's
_fetch_gex_rows_for_placement. No repo code changed."""
import os, sys, time, json
REPO = "/Users/ryan/code/volatility-monorepo"
for line in open(os.path.join(REPO, ".env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("="); os.environ.setdefault(k.strip(), v.strip())
os.environ["DATABASE_URL"] = os.environ["BACKFILL_DATABASE_URL"]
os.environ["PGOPTIONS"] = "-c default_transaction_read_only=on"
for p in (REPO, os.path.join(REPO, "apps", "web")):
    sys.path.insert(0, p)
import datetime as dt
from unittest.mock import patch
import numpy as np, pandas as pd
import modules.Ironbeam.callbacks as cb
import packages.shared.gex_landscape_api as api
from packages.shared.level_placement import place_levels
from packages.shared.trading_calendar import prev_trading_day

D = dt.date(2026, 10, 23); SPOT = float(sys.argv[1]) if len(sys.argv) > 1 else 7660.75
IV = float(sys.argv[2]) if len(sys.argv) > 2 else None

def peaks(land: pd.DataFrame, col="gex_total", k=3):
    y = land[col].to_numpy(); p = land["price"].to_numpy()
    i = np.arange(1, len(y) - 1)
    pos = i[(y[i] > y[i - 1]) & (y[i] >= y[i + 1]) & (y[i] > 0)]
    neg = i[(y[i] < y[i - 1]) & (y[i] <= y[i + 1]) & (y[i] < 0)]
    top = lambda idx, s: [(float(p[j]), round(float(y[j]) / 1e9, 2)) for j in sorted(idx, key=lambda j: s * y[j])[:k]]
    return {"positive": top(pos, -1), "negative": top(neg, 1)}

with cb.engine.connect() as conn:
    ro = conn.exec_driver_sql("SHOW default_transaction_read_only").scalar()
    t0 = time.perf_counter()
    cur, st = api.build_gex_landscape_response(conn, "SPX", D.isoformat(), SPOT, iv=IV)
    t_cur = time.perf_counter() - t0
    t0 = time.perf_counter()
    curh, sth = api.build_gex_landscape_response(conn, "SPX", D.isoformat(), SPOT, iv=IV, accuracy="high")
    t_curh = time.perf_counter() - t0

# corrected: timed end to end (ES snapshot + rows + placement + high-path builder)
t0 = time.perf_counter()
p_es = cb._fetch_es_snapshot_close(prev_trading_day(D))
rows = cb._fetch_gex_rows_for_placement(D)
rows = place_levels(rows, p_es)
corr_df = rows.assign(discounted_level=rows["level_corrected"])[["discounted_level", "dte", "gex_call", "gex_put", "stock_price"]]
with cb.engine.connect() as conn, patch.object(api, "_fetch_oi_gamma_strikes", lambda c, t, d: corr_df):
    corr, stc = api.build_gex_landscape_response(conn, "SPX", D.isoformat(), SPOT, iv=IV, accuracy="high")
t_corr = time.perf_counter() - t0

out = {"read_only": ro, "spot": SPOT, "iv": IV, "es_snapshot_A": p_es, "rows": len(rows),
       "status": [st, sth, stc], "seconds": {"current_low": round(t_cur, 2), "current_high": round(t_curh, 2), "corrected_high_e2e": round(t_corr, 2)},
       "current_low_peaks": peaks(pd.DataFrame(cur["landscape"])), "current_high_peaks": peaks(pd.DataFrame(curh["landscape"])),
       "corrected_peaks": peaks(pd.DataFrame(corr["landscape"])),
       "regime": {"current": cur["regime"].get("regime"), "corrected": corr["regime"].get("regime")},
       "confluences": {"current": [round(c["center_price"], 1) for c in cur["confluences"]][:6], "corrected": [round(c["center_price"], 1) for c in corr["confluences"]][:6]},
       "neg_zones": {"current": [round(z["price"], 1) for z in cur["neg_zones"]], "corrected": [round(z["price"], 1) for z in corr["neg_zones"]]},
       "table_spot": {"current": cur["table_spot"], "corrected": corr["table_spot"]},
       "grid": {"current": [cur["landscape"][0]["price"], cur["landscape"][-1]["price"]], "corrected": [corr["landscape"][0]["price"], corr["landscape"][-1]["price"]]}}
print(json.dumps(out, indent=1, default=str))
