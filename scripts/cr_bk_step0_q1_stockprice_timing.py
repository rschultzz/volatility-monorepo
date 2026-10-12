#!/usr/bin/env python3
"""CR-BK Step 0 (read-only, train only) — see specs/CR-BK-corrected-gamma-profile-test.md. Run: apps/web/.venv/bin/python scripts/cr_bk_step0_q1_stockprice_timing.py"""
import sys, datetime as dt; from pathlib import Path; ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
from packages.shared.backfill_safety import get_backfill_db_conn
from packages.shared.trading_calendar import prev_trading_day
c = get_backfill_db_conn(); c.execute("SET default_transaction_read_only = on"); c.execute("SET statement_timeout = '120s'")
q = lambda s, a=None: c.execute(s, a).fetchall()
for store in ["2024-02-14","2024-08-06","2025-03-12","2025-11-19","2026-04-08","2026-05-20"]:
    sd = dt.date.fromisoformat(store); api = prev_trading_day(sd)
    sp = q("select stock_price from orats_oi_gamma where ticker='SPX' and trade_date=%s and expir_date=%s limit 1", (sd, api))
    sp = float(sp[0][0]) if sp else None
    # 0DTE monies on api date (expir_date = trade_date), minute series of stock_price
    rows = q("""select snapshot_pt, stock_price, spot_price from orats_monies_minute
                where trade_date=%s and expir_date=%s and ticker='SPX' and snapshot_pt::time >= '12:30' order by 1""", (api.isoformat(), api.isoformat()))
    best = min(rows, key=lambda r: abs(r[1]-sp)) if rows and sp else None
    close = rows[-1] if rows else None
    print(store, "api", api, "oi_gamma S(expiring)", sp, "| 13:00 stock", close and round(close[1],2), "| closest minute", best and (best[0].time(), round(best[1],2)))
    for r in rows:
        if r[0].time().minute in (40,44,45,46,50,55,0) : print("    ", r[0].time(), round(r[1],2), round(r[2],2))
