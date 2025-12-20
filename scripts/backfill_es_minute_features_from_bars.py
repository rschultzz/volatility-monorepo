#!/usr/bin/env python
"""
Backfill es_minute_features from ironbeam_es_1m_bars starting at a given date.

Assumptions:
- ironbeam_es_1m_bars has columns: datetime (timestamp, stored as UTC),
  open, high, low, close, volume.
- es_minute_features table already exists (see CREATE TABLE we defined earlier).
- DATABASE_URL env var points to your Postgres DB.

This script:
- Loads all bars from START_DATE onward.
- Treats `datetime` as UTC, converts to PT.
- Computes trade_date (PT), is_rth, bar_index, ret_1m, range_1m.
- Inserts rows into es_minute_features.
"""

import os
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import create_engine, text
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

# ---------- Config ----------
START_DATE = dt.date(2025, 11, 24)  # <-- you can change this if needed

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Please set DATABASE_URL env var to your Postgres URL")

BARS_TABLE = os.environ.get("IRONBEAM_BARS_TABLE", "ironbeam_es_1m_bars")
FEATURES_TABLE = os.environ.get("ES_FEATURES_TABLE", "es_minute_features")

TZ_UTC = ZoneInfo("UTC")
TZ_PT = ZoneInfo("America/Los_Angeles")


def load_bars(engine):
    """Load all bars from START_DATE onward into a pandas DataFrame."""
    print(f"[load_bars] Loading bars from {BARS_TABLE} starting {START_DATE} ...")
    start_ts = dt.datetime.combine(START_DATE, dt.time(0, 0, 0))

    query = text(
        f"""
        SELECT datetime, open, high, low, close, volume
        FROM {BARS_TABLE}
        WHERE datetime >= :start_ts
        ORDER BY datetime
        """
    )

    df = pd.read_sql_query(query, engine, params={"start_ts": start_ts})
    if df.empty:
        raise RuntimeError(f"No bars found in {BARS_TABLE} from {START_DATE} onward")
    print(f"[load_bars] Loaded {len(df)} rows")
    return df


def add_time_columns(df):
    """Add ts_utc, ts_pt (PT local), trade_date, is_rth, bar_index."""
    print("[add_time_columns] Adding PT time and session columns...")

    # Treat `datetime` as UTC
    df["ts_utc"] = pd.to_datetime(df["datetime"]).dt.tz_localize(TZ_UTC)

    # PT local clock time (no tz info, matches TIMESTAMP WITHOUT TIME ZONE)
    df["ts_pt"] = df["ts_utc"].dt.tz_convert(TZ_PT).dt.tz_localize(None)

    # PT trade date
    df["trade_date"] = df["ts_pt"].dt.date

    # RTH session: 6:30 to 13:00 PT
    mins = df["ts_pt"].dt.hour * 60 + df["ts_pt"].dt.minute
    rth_start = 6 * 60 + 30   # 06:30
    rth_end = 13 * 60         # 13:00 (exclusive)
    df["is_rth"] = (mins >= rth_start) & (mins < rth_end)

    # Bar index: 0-based within RTH, -1 outside RTH
    df = df.sort_values("ts_utc").reset_index(drop=True)
    df["bar_index"] = -1
    mask_rth = df["is_rth"]
    df.loc[mask_rth, "bar_index"] = (
        df.loc[mask_rth].groupby("trade_date").cumcount()
    )

    print("[add_time_columns] Done.")
    return df



def add_price_features(df):
    """Add simple derived price features: ret_1m, range_1m."""
    print("[add_price_features] Adding ret_1m and range_1m...")

    df = df.sort_values("ts_utc").reset_index(drop=True)

    # 1-minute return based on close; fills first row with NaN
    df["ret_1m"] = df["close"].pct_change()

    # Intrabar range
    df["range_1m"] = df["high"] - df["low"]

    print("[add_price_features] Done.")
    return df


def prepare_for_insert(df):
    """
    Keep only the columns that exist in es_minute_features we want to populate now.
    GEX and smile columns will remain NULL (not included in insert).
    """
    cols = [
        "ts_utc",
        "ts_pt",
        "trade_date",
        "bar_index",
        "is_rth",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ret_1m",
        "range_1m",
    ]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns in DataFrame: {missing}")

    out = df[cols].copy()
    return out


def insert_features(engine, df_features):
    """Insert rows into es_minute_features."""
    print(f"[insert_features] Inserting {len(df_features)} rows into {FEATURES_TABLE}...")

    # Use to_sql for convenience; assumes es_minute_features already exists.
    # If ts_utc primary key collisions occur, the insert will fail; we can
    # add ON CONFLICT logic later if needed.
    df_features.to_sql(
        FEATURES_TABLE,
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )

    print("[insert_features] Insert complete.")


def main():
    engine = create_engine(DATABASE_URL)

    df_bars = load_bars(engine)
    df_bars = add_time_columns(df_bars)
    df_bars = add_price_features(df_bars)
    df_features = prepare_for_insert(df_bars)

    print(df_features.head())
    insert_features(engine, df_features)
    print("[main] Backfill completed successfully.")


if __name__ == "__main__":
    main()
