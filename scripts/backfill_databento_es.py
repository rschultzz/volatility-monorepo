#!/usr/bin/env python3
"""
Databento ES 1-min Backfill
===========================
Backfills 1 year of clean ES 1-min OHLCV data (2025-01-01 through 2025-12-31)
from Databento into ironbeam_es_1m_bars, replacing damaged Ironbeam rows.
Also populates the downstream es_minutes table (which feeds the dashboard
view es_minutes_with_features) so the dashboard shows the 2025 range.

Data source   : Databento GLBX.MDP3, schema ohlcv-1m, symbol ES.c.0 (continuous front-month)
Target tables : ironbeam_es_1m_bars  (adds source='databento' column)
                es_minutes           (adds source='databento' column)

Both tables are tagged with a 'source' column so rows from different data
sources can be distinguished.  Existing rows default to 'ironbeam'.

Workflow:
  1. Preflight: count existing rows in the backfill range
  2. Cost preview from Databento
  3. Confirm with user
  4. Add 'source' column to both tables (if missing)
  5. DELETE existing 2025 rows from both tables
  6. Disable trigger trg_es_minutes_from_ironbeam
  7. Fetch + insert from Databento, month-by-month
  8. Populate es_minutes in one SQL statement using window functions
  9. Re-enable trigger
 10. Summary

Requirements:
    pip install databento python-dotenv sqlalchemy psycopg pandas

Usage:
    python backfill_databento_es.py                # interactive, default 2025 range
    python backfill_databento_es.py --dry-run      # show plan, don't execute
    python backfill_databento_es.py --yes          # skip confirmation (careful!)
    python backfill_databento_es.py --start-date 2024-01-01 --end-date 2025-01-01
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

try:
    import databento as db
except ImportError:
    sys.exit("ERROR: databento package not installed.  Run: pip install databento")


# --- Config ---------------------------------------------------------------
DATASET  = "GLBX.MDP3"
SCHEMA   = "ohlcv-1m"
SYMBOL   = "ES.c.0"
STYPE_IN = "continuous"

BACKFILL_START = datetime(2025, 1, 1)     # inclusive (UTC)
BACKFILL_END   = datetime(2026, 1, 1)     # exclusive (UTC)

BARS_TABLE    = "ironbeam_es_1m_bars"
MINUTES_TABLE = "es_minutes"
TRIGGER_NAME  = "trg_es_minutes_from_ironbeam"

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


# --- Helpers --------------------------------------------------------------
def normalize_db_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def confirm(prompt: str) -> bool:
    r = input(f"\n{prompt} [type 'yes' to proceed]: ").strip().lower()
    return r == "yes"


def column_exists(engine, table: str, column: str) -> bool:
    q = text("""
        SELECT 1 FROM information_schema.columns
        WHERE table_name = :t AND column_name = :c
    """)
    with engine.begin() as conn:
        return conn.execute(q, {"t": table, "c": column}).fetchone() is not None


def add_source_column(engine, table: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"""
            ALTER TABLE {table}
            ADD COLUMN source TEXT NOT NULL DEFAULT 'ironbeam'
        """))


def preflight_counts(engine):
    s, e = BACKFILL_START, BACKFILL_END
    s_tz = s.replace(tzinfo=timezone.utc)
    e_tz = e.replace(tzinfo=timezone.utc)
    with engine.begin() as conn:
        bars = conn.execute(text(f"""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE datetime >= :s AND datetime < :e) AS in_range,
                COUNT(*) FILTER (WHERE datetime >= :e) AS after_range
            FROM {BARS_TABLE}
        """), {"s": s, "e": e}).fetchone()
        minutes = conn.execute(text(f"""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE ts_utc >= :s AND ts_utc < :e) AS in_range,
                COUNT(*) FILTER (WHERE ts_utc >= :e) AS after_range
            FROM {MINUTES_TABLE}
        """), {"s": s_tz, "e": e_tz}).fetchone()
    return bars, minutes


def drop_in_range(engine):
    s, e = BACKFILL_START, BACKFILL_END
    s_tz = s.replace(tzinfo=timezone.utc)
    e_tz = e.replace(tzinfo=timezone.utc)
    with engine.begin() as conn:
        bars_del = conn.execute(text(f"""
            DELETE FROM {BARS_TABLE}
            WHERE datetime >= :s AND datetime < :e
        """), {"s": s, "e": e}).rowcount
        minutes_del = conn.execute(text(f"""
            DELETE FROM {MINUTES_TABLE}
            WHERE ts_utc >= :s AND ts_utc < :e
        """), {"s": s_tz, "e": e_tz}).rowcount
    return bars_del, minutes_del


def set_trigger(engine, enabled: bool) -> None:
    action = "ENABLE" if enabled else "DISABLE"
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {BARS_TABLE} {action} TRIGGER {TRIGGER_NAME}"))


def month_chunks(start: datetime, end: datetime):
    """Yield (chunk_start, chunk_end) pairs covering [start, end) by month."""
    cur = start
    while cur < end:
        if cur.month == 12:
            nxt = cur.replace(year=cur.year + 1, month=1)
        else:
            nxt = cur.replace(month=cur.month + 1)
        yield cur, min(nxt, end)
        cur = nxt


def fetch_databento_month(client, start: datetime, end: datetime) -> pd.DataFrame:
    data = client.timeseries.get_range(
        dataset=DATASET, schema=SCHEMA, symbols=[SYMBOL], stype_in=STYPE_IN,
        start=start.strftime("%Y-%m-%dT%H:%M:%S"),
        end=end.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    df = data.to_df().reset_index()
    if "ts_event" in df.columns and "datetime" not in df.columns:
        df = df.rename(columns={"ts_event": "datetime"})
    # Normalize to tz-naive UTC to match ironbeam_es_1m_bars.datetime column type
    if isinstance(df["datetime"].dtype, pd.DatetimeTZDtype):
        df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df[["datetime"] + OHLCV_COLS].copy()
    for c in OHLCV_COLS:
        df[c] = df[c].astype(float)
    df["source"] = "databento"
    return df


def insert_bars(engine, df: pd.DataFrame) -> int:
    with engine.begin() as conn:
        df.to_sql(
            BARS_TABLE, conn, if_exists="append", index=False,
            method="multi", chunksize=1000,
        )
    return len(df)


def populate_minutes(engine) -> int:
    """Replicate the trigger logic for the backfill range in a single SQL.
    Also tags the inserted rows with source='databento' so es_minutes rows
    can be distinguished by source."""
    sql = text(f"""
        INSERT INTO {MINUTES_TABLE} (
            ts_utc, ts_pt, trade_date, bar_index, is_rth,
            open, high, low, close, volume, source
        )
        SELECT
            ts_utc,
            ts_pt,
            trade_date,
            CASE
                WHEN is_rth THEN
                    (ROW_NUMBER() OVER (PARTITION BY trade_date, is_rth ORDER BY ts_utc) - 1)::int
                ELSE -1
            END AS bar_index,
            is_rth,
            open, high, low, close, volume,
            'databento'::text AS source
        FROM (
            SELECT
                (datetime AT TIME ZONE 'UTC') AS ts_utc,
                ((datetime AT TIME ZONE 'UTC') AT TIME ZONE 'America/Los_Angeles') AS ts_pt,
                (((datetime AT TIME ZONE 'UTC') AT TIME ZONE 'America/Los_Angeles')::date) AS trade_date,
                (
                    (EXTRACT(HOUR FROM ((datetime AT TIME ZONE 'UTC') AT TIME ZONE 'America/Los_Angeles'))::int * 60
                     + EXTRACT(MINUTE FROM ((datetime AT TIME ZONE 'UTC') AT TIME ZONE 'America/Los_Angeles'))::int
                    ) >= 390
                    AND
                    (EXTRACT(HOUR FROM ((datetime AT TIME ZONE 'UTC') AT TIME ZONE 'America/Los_Angeles'))::int * 60
                     + EXTRACT(MINUTE FROM ((datetime AT TIME ZONE 'UTC') AT TIME ZONE 'America/Los_Angeles'))::int
                    ) < 780
                ) AS is_rth,
                open, high, low, close, volume
            FROM {BARS_TABLE}
            WHERE datetime >= :s AND datetime < :e
              AND source = 'databento'
        ) sub
        ON CONFLICT (ts_utc) DO NOTHING
    """)
    with engine.begin() as conn:
        res = conn.execute(sql, {"s": BACKFILL_START, "e": BACKFILL_END})
        return res.rowcount


# --- Main -----------------------------------------------------------------
def main() -> None:
    global BACKFILL_START, BACKFILL_END

    p = argparse.ArgumentParser(description="Backfill ES 1-min OHLCV from Databento.")
    p.add_argument("--env", default=".env", help="Path to .env file")
    p.add_argument("--dry-run", action="store_true", help="Show plan, don't execute")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    p.add_argument("--start-date", default=None,
                   help=f"Override BACKFILL_START (YYYY-MM-DD, inclusive). "
                        f"Default: {BACKFILL_START.date()}")
    p.add_argument("--end-date", default=None,
                   help=f"Override BACKFILL_END (YYYY-MM-DD, exclusive). "
                        f"Default: {BACKFILL_END.date()}")
    args = p.parse_args()

    # Apply CLI overrides for the backfill range
    if args.start_date:
        BACKFILL_START = datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.end_date:
        BACKFILL_END = datetime.strptime(args.end_date, "%Y-%m-%d")
    if BACKFILL_START >= BACKFILL_END:
        sys.exit("ERROR: --start-date must be earlier than --end-date.")

    env_path = Path(args.env)
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    db_url = os.getenv("DATABASE_URL")
    db_key = os.getenv("DATABENTO_API_KEY")
    if not db_url:
        sys.exit("ERROR: DATABASE_URL not found.")
    if not db_key:
        sys.exit("ERROR: DATABENTO_API_KEY not found.")

    engine = create_engine(normalize_db_url(db_url), pool_pre_ping=True)
    client = db.Historical(db_key)

    print("=" * 92)
    print("Databento ES 1-min Backfill")
    print("=" * 92)
    print(f"Range          : {BACKFILL_START.date()}  to  {BACKFILL_END.date()}  (end exclusive)")
    print(f"Source         : {DATASET} / {SCHEMA} / {SYMBOL}")
    print(f"Target tables  : {BARS_TABLE}, {MINUTES_TABLE}")

    # --- Preflight ---
    print("\n--- Preflight ---")
    bars_has_source    = column_exists(engine, BARS_TABLE, "source")
    minutes_has_source = column_exists(engine, MINUTES_TABLE, "source")
    print(f"  source column on {BARS_TABLE:22}: "
          f"{'exists' if bars_has_source else 'MISSING (will add)'}")
    print(f"  source column on {MINUTES_TABLE:22}: "
          f"{'exists' if minutes_has_source else 'MISSING (will add)'}")

    bars, minutes = preflight_counts(engine)
    print(f"  {BARS_TABLE:30} total={bars[0]:>8,}  in_range={bars[1]:>8,}  after_range={bars[2]:>8,}")
    print(f"  {MINUTES_TABLE:30} total={minutes[0]:>8,}  in_range={minutes[1]:>8,}  after_range={minutes[2]:>8,}")

    # --- Cost preview ---
    print("\n--- Cost preview ---")
    try:
        cost = client.metadata.get_cost(
            dataset=DATASET, schema=SCHEMA, symbols=[SYMBOL], stype_in=STYPE_IN,
            start=BACKFILL_START.strftime("%Y-%m-%dT%H:%M:%S"),
            end=BACKFILL_END.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        print(f"  Estimated cost: ${cost:.4f} USD")
    except Exception as e:
        print(f"  ERROR getting cost: {e}")
        sys.exit(1)

    # --- Plan summary ---
    print("\n--- Plan ---")
    print(f"  1. source column: "
          f"{'add to ' + BARS_TABLE if not bars_has_source else 'ok on ' + BARS_TABLE}, "
          f"{'add to ' + MINUTES_TABLE if not minutes_has_source else 'ok on ' + MINUTES_TABLE}")
    print(f"  2. DELETE {bars[1]:,} rows from {BARS_TABLE}")
    print(f"  3. DELETE {minutes[1]:,} rows from {MINUTES_TABLE}")
    print(f"  4. Disable trigger {TRIGGER_NAME}")
    print(f"  5. Fetch + insert ~350k rows from Databento (monthly chunks)")
    print(f"  6. Populate {MINUTES_TABLE} via window-function SQL (source='databento')")
    print(f"  7. Re-enable trigger {TRIGGER_NAME}")
    print(f"  Cost: ${cost:.4f}")

    if args.dry_run:
        print("\n[dry-run] No changes made.  Exiting.")
        return

    if not args.yes:
        if not confirm("Proceed with the full backfill?"):
            print("Aborted.")
            return

    # --- Execute ---
    print(f"\n[1/7] Ensuring source column on both tables...")
    if not bars_has_source:
        add_source_column(engine, BARS_TABLE)
        print(f"      Added to {BARS_TABLE}.")
    else:
        print(f"      {BARS_TABLE}: already present, skipping.")
    if not minutes_has_source:
        add_source_column(engine, MINUTES_TABLE)
        print(f"      Added to {MINUTES_TABLE}.")
    else:
        print(f"      {MINUTES_TABLE}: already present, skipping.")

    print(f"\n[2-3/7] Deleting existing rows in {BACKFILL_START.date()}..{BACKFILL_END.date()}...")
    bars_del, minutes_del = drop_in_range(engine)
    print(f"      {BARS_TABLE}:    -{bars_del:,}")
    print(f"      {MINUTES_TABLE}: -{minutes_del:,}")

    print(f"\n[4/7] Disabling trigger {TRIGGER_NAME}...")
    set_trigger(engine, enabled=False)

    total_inserted = 0
    try:
        print("\n[5/7] Backfilling from Databento (by month)...")
        for chunk_start, chunk_end in month_chunks(BACKFILL_START, BACKFILL_END):
            label = chunk_start.strftime("%Y-%m")
            print(f"      {label}: fetching...", end=" ", flush=True)
            df = fetch_databento_month(client, chunk_start, chunk_end)
            print(f"got {len(df):>7,} bars, inserting...", end=" ", flush=True)
            n = insert_bars(engine, df)
            total_inserted += n
            print(f"done.  (running total: {total_inserted:,})")

        print(f"\n[6/7] Populating {MINUTES_TABLE} via window function...")
        n_min = populate_minutes(engine)
        print(f"      Inserted {n_min:,} rows into {MINUTES_TABLE} (source='databento')")

    finally:
        print(f"\n[7/7] Re-enabling trigger {TRIGGER_NAME}...")
        set_trigger(engine, enabled=True)
        print("      Done.")

    # --- Final summary ---
    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)
    bars, minutes = preflight_counts(engine)
    range_label = f"{BACKFILL_START.date()}..{BACKFILL_END.date()}"
    print(f"  {BARS_TABLE:30} total={bars[0]:>8,}  "
          f"in_range[{range_label}]={bars[1]:>8,}  after_range={bars[2]:>8,}")
    print(f"  {MINUTES_TABLE:30} total={minutes[0]:>8,}  "
          f"in_range[{range_label}]={minutes[1]:>8,}  after_range={minutes[2]:>8,}")
    print(f"\n  Databento rows inserted: {total_inserted:,}")
    print(f"  Backfill complete.\n")


if __name__ == "__main__":
    main()
