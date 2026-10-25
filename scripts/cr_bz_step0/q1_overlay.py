"""CR-BZ Q1: replay /api/ironbeam/live-trades-overlay for 2026-10-23 under each env.
Simulates 'now' = a live minute on 10-23 and truncates base bars to < now's bucket
(as the table would be live). Read-only (SELECTs via the module's own engine)."""
import os, sys, json, datetime as dt
WT = sys.argv[1]
sys.path[:0] = [WT, WT + "/apps/web"]
from dotenv import load_dotenv
load_dotenv()
for k in ("IRONBEAM_SYMBOL", "IRONBEAM_TRADES_SYMBOL", "IRONBEAM_FLOW_SYMBOL"):
    os.environ.pop(k, None)
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1); os.environ[k] = v
from zoneinfo import ZoneInfo
import pandas as pd
import modules.Ironbeam.callbacks as cb
PT = ZoneInfo("America/Los_Angeles")
NOW = dt.datetime.fromisoformat(os.environ.get("CRBZ_NOW", "2026-10-23T11:15:25")).replace(tzinfo=PT)
class _DT(dt.datetime):
    @classmethod
    def now(cls, tz=None): return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)
class _dtmod:  # shim for module-level `dt`
    def __getattr__(self, n): return _DT if n == "datetime" else getattr(dt, n)
cb.dt = _dtmod()
cb._current_session_trade_date = lambda tz: dt.date(2026, 10, 23)
_orig = cb._fetch_bars_pt
def _fb(s, e, interval, tz):
    df = _orig(s, e, interval, tz)
    return df[df["datetime_pt"] < pd.Timestamp(NOW).floor("1min")].reset_index(drop=True)
cb._fetch_bars_pt = _fb
p, st = cb._build_react_live_trades_overlay_payload("2026-10-23", "1min")
print(json.dumps({"env": sys.argv[2:] or "none", "TRADES_SYMBOL": cb.TRADES_SYMBOL, "FLOW_SYMBOL": cb.FLOW_SYMBOL,
                  "status": st, "n_bars": len(p["bars"]), "symbol": p.get("symbol"),
                  "bars": [{k: b.get(k) for k in ("time", "open", "high", "low", "close")} for b in p["bars"]]}, default=str))
