#!/usr/bin/env python3
"""CR-BK Step 0 (read-only, train only) — see specs/CR-BK-corrected-gamma-profile-test.md. Run: apps/web/.venv/bin/python scripts/cr_bk_step0_q7_counts.py"""
import sys, datetime as dt; from pathlib import Path; ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
from packages.shared.backfill_safety import get_backfill_db_conn
from packages.shared.trading_calendar import prev_trading_day, is_trading_day
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
c = get_backfill_db_conn(); c.execute("SET default_transaction_read_only = on"); c.execute("SET statement_timeout = '300s'")
q = lambda s, a=None: c.execute(s, a).fetchall()
SPLIT = dt.date(2026, 6, 5)
store = [r[0] for r in q("select distinct trade_date from orats_oi_gamma where ticker='SPX' and trade_date between '2023-05-02' and %s order by 1", (SPLIT,))]
td = [d for d in store if is_trading_day(d)]
print("store dates", len(store), "trading days", len(td), "non-trading:", [str(d) for d in store if not is_trading_day(d)][:12])
mon = {r[0] for r in q("select distinct trade_date from orats_monies_minute where trade_date between '2023-05-01' and %s and snapshot_pt::time between '12:40' and '12:50'", (SPLIT.isoformat(),))}
ok = [d for d in td if prev_trading_day(d).isoformat() in mon]
print("with prior-session 12:40–12:50 monies snapshot:", len(ok), "missing:", [str(d) for d in td if prev_trading_day(d).isoformat() not in mon][:15])
print("canonical version:", CANONICAL_FEATURE_VERSION)
im = {r[0] for r in q("select trade_date from bt_daily_features where ticker='SPX' and active and feature_version=%s and trade_date<=%s and (feature_vector->>'implied_move_1d')::float > 0", (CANONICAL_FEATURE_VERSION, SPLIT))}
print("… and implied_move_1d > 0:", len([d for d in ok if d in im]))
am = q("""select count(distinct (trade_date, expir_date)) from orats_monies_minute where trade_date between '2023-05-01' and %s and expiry_tod='am' and snapshot_pt::time='12:46'""", (SPLIT.isoformat(),))
print("AM-tagged (date, expiry) pairs at 12:46:", am)
