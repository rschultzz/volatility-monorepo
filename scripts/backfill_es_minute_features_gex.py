#!/usr/bin/env python
"""
Backfill daily GEX into es_minute_features from orats_oi_gamma.

For each trade_date where es_minute_features.net_gex IS NULL:
- Compute daily call_gex, put_gex, net_gex from orats_oi_gamma
- Write those values into all rows for that trade_date

Assumptions:
- GEX table: orats_oi_gamma
- Columns:
    trade_date   DATE
    ticker       TEXT  (we'll use 'SPX')
    gex_call     DOUBLE PRECISION
    gex_put      DOUBLE PRECISION
"""

import os
import datetime as dt

import pandas as pd
from sqlalchemy import create_engine, text

# Load .env so DATABASE_URL is available
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Please set DATABASE_URL in your environment or .env")

ENGINE = create_engine(DATABASE_URL)

# Table names (override via env vars if desired)
FEATURES_TABLE = os.environ.get("ES_FEATURES_TABLE", "es_minute_features")
GEX_TABLE = os.environ.get("ORATS_GEX_TABLE", "orats_oi_gamma")


def get_dates_needing_gex(engine):
    """
    Find all trade_date values in es_minute_features that have net_gex IS NULL.
    """
    q = text(f"""
        SELECT DISTINCT trade_date
        FROM {FEATURES_TABLE}
        WHERE net_gex IS NULL
        ORDER BY trade_date;
    """)
    df = pd.read_sql_query(q, engine)
    dates = [r["trade_date"] for _, r in df.iterrows()]
    print(f"[get_dates_needing_gex] Found {len(dates)} trade_dates needing GEX")
    return dates


def get_daily_gex(engine, trade_date: dt.date):
    """
    Compute daily call_gex, put_gex, net_gex for a given trade_date
    from the ORATS OI gamma table.

    net_gex = SUM(gex_call + gex_put)
    """
    q = text(f"""
        SELECT
            SUM(gex_call)               AS call_gex,
            SUM(gex_put)                AS put_gex,
            SUM(gex_call + gex_put)     AS net_gex
        FROM {GEX_TABLE}
        WHERE trade_date = :trade_date
          AND ticker = 'SPX';
    """)

    with engine.begin() as conn:
        row = conn.execute(q, {"trade_date": trade_date}).one_or_none()

    if not row or row.net_gex is None:
        print(f"[get_daily_gex] {trade_date}: no GEX data found")
        return None, None, None

    call_gex = row.call_gex
    put_gex = row.put_gex
    net_gex = row.net_gex

    print(f"[get_daily_gex] {trade_date}: call_gex={call_gex}, "
          f"put_gex={put_gex}, net_gex={net_gex}")
    return call_gex, put_gex, net_gex


def update_features_for_date(engine, trade_date: dt.date,
                             call_gex: float, put_gex: float, net_gex: float):
    """
    Write call_gex, put_gex, net_gex into es_minute_features for the given trade_date.
    """
    q = text(f"""
        UPDATE {FEATURES_TABLE}
        SET
            call_gex = :call_gex,
            put_gex  = :put_gex,
            net_gex  = :net_gex
        WHERE trade_date = :trade_date;
    """)

    with engine.begin() as conn:
        res = conn.execute(
            q,
            {
                "call_gex": call_gex,
                "put_gex": put_gex,
                "net_gex": net_gex,
                "trade_date": trade_date,
            },
        )
        print(f"[update_features_for_date] {trade_date}: updated {res.rowcount} rows")


def main():
    dates = get_dates_needing_gex(ENGINE)
    if not dates:
        print("[main] No dates need GEX; nothing to do.")
        return

    for d in dates:
        call_gex, put_gex, net_gex = get_daily_gex(ENGINE, d)
        if net_gex is None:
            print(f"[main] WARNING: skipping {d} (no GEX data).")
            continue
        update_features_for_date(ENGINE, d, call_gex, put_gex, net_gex)

    print("[main] Done populating call_gex, put_gex, net_gex.")


if __name__ == "__main__":
    main()
