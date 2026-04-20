#!/usr/bin/env python3
"""
ORATS monies/implied 1-min Backfill for orats_monies_minute
===========================================================
Backfills 10 months of 1-min SPX smile/skew snapshots (2025-01-02 through
2025-11-02 API dates) into orats_monies_minute.  Runs reverse-chronologically
(November first, working back to January) so the most useful recent data
lands first.

Logic is byte-identical to orats_monies_today_ingest_2.py:
  - Same endpoint: /datav2/hist/live/one-minute/monies/implied.csv
  - Same params: ticker, tradeDate=YYYYMMDDHHMM, token
  - Same transform: camelCase -> snake_case, snapshot_pt from UTC -> PT
  - Same upsert: INSERT ... ON CONFLICT DO NOTHING on PK
    (ticker, expir_date, quote_date)

Window: 9:30 AM ET through 4:00 PM ET inclusive (390 snapshots/trading day).
Matches cron's RTH filter exactly.

Cost: $0 (all calls included in your ORATS subscription).

Requirements:
    pip install requests python-dotenv sqlalchemy psycopg pandas pandas-market-calendars

Usage:
    python backfill_orats_monies_minute.py              # interactive
    python backfill_orats_monies_minute.py --dry-run    # show plan only
    python backfill_orats_monies_minute.py --yes        # skip confirmation
    python backfill_orats_monies_minute.py --start-date 2025-09-01
    python backfill_orats_monies_minute.py --end-date   2025-07-15
"""

import argparse
import datetime as dt
import io
import os
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import requests
from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert


# --- Config ---------------------------------------------------------------
TICKER   = "SPX"
BASE_URL = "https://api.orats.io"
ENDPOINT = "/datav2/hist/live/one-minute/monies/implied.csv"
TABLE    = "orats_monies_minute"

ET = ZoneInfo("America/New_York")

# Backfill window (API dates, inclusive)
BACKFILL_START = dt.date(2025, 1, 2)    # oldest
BACKFILL_END   = dt.date(2025, 11, 2)   # newest — stops before existing data (2025-11-03)

# Per-day RTH window, inclusive on both ends (matches cron's 09:30 <= t <= 16:00)
RTH_OPEN_ET  = dt.time(9, 30)
RTH_CLOSE_ET = dt.time(16, 0)

# Delays
PER_CALL_SLEEP_SEC = 0.1   # between snapshot calls (~2.25hr total)
PER_DAY_SLEEP_SEC  = 1.0   # between trading days
BACKOFF_INITIAL    = 2.0   # seconds
BACKOFF_MAX        = 60.0

HTTP_TIMEOUT = 30


# --- Helpers (identical to cron) ------------------------------------------
def camel_to_snake(name: str) -> str:
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def normalize_db_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def transform_orats_df(df: pd.DataFrame) -> pd.DataFrame:
    """Adds snapshot_pt, renames columns camelCase -> snake_case.
    Matches transform_orats_df() in the live cron exactly."""
    if df.empty:
        return df
    utc_ts = pd.to_datetime(df["snapShotDate"], errors="coerce", utc=True)
    df["snapshot_pt"] = (
        utc_ts.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
    )
    df.columns = [camel_to_snake(col) for col in df.columns]
    return df


# --- Trading day / snapshot generation ------------------------------------
def trading_days_between(d_from: dt.date, d_to: dt.date) -> list:
    nyse = mcal.get_calendar("XNYS")
    sched = nyse.schedule(start_date=d_from, end_date=d_to)
    return [d.date() for d in sched.index]


def snapshots_for_day(api_date: dt.date) -> list:
    """Return every 09:30 -> 16:00 ET minute as a naive ET datetime.
    Cron runs _previous_minute_et() and checks 09:30 <= t <= 16:00,
    so the set of minutes we pull is inclusive on both ends.
    """
    out = []
    cur = dt.datetime.combine(api_date, RTH_OPEN_ET)
    end = dt.datetime.combine(api_date, RTH_CLOSE_ET)
    while cur <= end:
        out.append(cur)
        cur += dt.timedelta(minutes=1)
    return out


def tradeDate_param(et_naive: dt.datetime) -> str:
    """ORATS param YYYYMMDDHHMM, interpreted as ET wall-clock."""
    return et_naive.strftime("%Y%m%d%H%M")


# --- ORATS fetch with backoff ---------------------------------------------
def fetch_snapshot(session: requests.Session, token: str,
                   ticker: str, et_naive: dt.datetime) -> pd.DataFrame:
    """Fetch one minute snapshot.  Returns empty DataFrame if no data.
    Raises RuntimeError on persistent failure after backoff."""
    url = f"{BASE_URL}{ENDPOINT}"
    params = {
        "ticker":    ticker,
        "tradeDate": tradeDate_param(et_naive),
        "token":     token,
    }

    backoff = BACKOFF_INITIAL
    for attempt in range(1, 7):
        try:
            r = session.get(url, params=params, timeout=HTTP_TIMEOUT)
        except requests.RequestException as e:
            if attempt == 6:
                raise RuntimeError(f"network error after retries: {e}") from e
            time.sleep(min(backoff, BACKOFF_MAX))
            backoff *= 2
            continue

        if r.status_code == 200:
            text_body = (r.text or "").strip()
            if not text_body or text_body.startswith("<"):
                return pd.DataFrame()
            try:
                return pd.read_csv(io.StringIO(text_body))
            except Exception:
                return pd.DataFrame()

        if r.status_code in (404,):
            return pd.DataFrame()

        if r.status_code in (429, 500, 502, 503, 504):
            if attempt == 6:
                raise RuntimeError(f"HTTP {r.status_code} after retries")
            time.sleep(min(backoff, BACKOFF_MAX))
            backoff *= 2
            continue

        # Any other status (401, 403, etc.) is fatal
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

    return pd.DataFrame()


# --- DB upsert (identical to cron) ----------------------------------------
def upsert_dataframe(df: pd.DataFrame, engine, table: Table) -> int:
    if df.empty:
        return 0
    records = df.to_dict(orient="records")
    chunk_size = 500
    with engine.begin() as conn:
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            stmt = pg_insert(table).values(chunk).on_conflict_do_nothing()
            conn.execute(stmt)
    return len(records)


# --- Resume: find quote_dates already stored for a given API day ----------
def existing_quote_dates_for_day(engine, ticker: str,
                                  api_date: dt.date) -> set:
    """Return set of quote_date (text) values already present for this day.
    Used to skip already-fetched snapshots on resume.

    Note: orats_monies_minute stores ~54 rows per snapshot (one per expir),
    but they all share the same quote_date text.  We just grab the distinct
    set of quote_dates for the trade_date and skip snapshots whose fetch
    would write to an already-present quote_date.
    """
    q = text(f"""
        SELECT DISTINCT quote_date
        FROM {TABLE}
        WHERE ticker = :t AND trade_date = :td
    """)
    with engine.begin() as conn:
        rows = conn.execute(q, {"t": ticker, "td": api_date.isoformat()}).fetchall()
    return {row[0] for row in rows}


# --- Main day-processor ---------------------------------------------------
def process_day(session, token, engine, table: Table, ticker: str,
                api_date: dt.date, per_call_sleep: float, resume: bool) -> dict:
    """Process all snapshots for one trading day.
    Returns stats dict: {processed, skipped_resume, skipped_empty, rows_added, failures}
    """
    stats = {"processed": 0, "skipped_resume": 0, "skipped_empty": 0,
             "rows_added": 0, "failures": []}

    snapshots = snapshots_for_day(api_date)

    # Build resume filter — map existing quote_dates to approx minute they cover.
    # quote_date looks like '2025-12-01T14:25:03Z' — we extract date+HH:MM.
    already = set()
    if resume:
        qdates = existing_quote_dates_for_day(engine, ticker, api_date)
        for qd in qdates:
            if not qd:
                continue
            # Extract YYYY-MM-DDTHH:MM (strip seconds and Z)
            # quote_date is ET-ish UTC from ORATS; minute granularity is what we need
            already.add(qd[:16])  # '2025-12-01T14:25'

    for et_naive in snapshots:
        # Cheap resume check: transform this target minute to the same truncated
        # quote_date format ORATS would return for it.  ORATS quote_date is UTC.
        # We convert the ET minute to UTC and truncate to minute precision.
        et_aware = et_naive.replace(tzinfo=ET)
        utc_dt   = et_aware.astimezone(dt.timezone.utc)
        # ORATS quote_date format uses 'T' and a 'Z', e.g., '2025-12-01T14:25'
        approx_qd = utc_dt.strftime("%Y-%m-%dT%H:%M")

        # This is approximate — ORATS's actual quote_date may be a few seconds
        # different from our minute boundary.  Check both this minute and the
        # minute before (common case: quote taken at XX:XX:25 for an ET
        # minute boundary).
        prev_qd = (utc_dt - dt.timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M")
        if approx_qd in already or prev_qd in already:
            stats["skipped_resume"] += 1
            continue

        try:
            df = fetch_snapshot(session, token, ticker, et_naive)
        except Exception as e:
            stats["failures"].append((et_naive.isoformat(), str(e)[:200]))
            time.sleep(per_call_sleep)
            continue

        if df.empty:
            stats["skipped_empty"] += 1
        else:
            df = transform_orats_df(df)
            try:
                n = upsert_dataframe(df, engine, table)
                stats["rows_added"] += n
                stats["processed"]  += 1
            except Exception as e:
                stats["failures"].append((et_naive.isoformat(), f"db: {str(e)[:200]}"))

        time.sleep(per_call_sleep)

    return stats


# --- Main ------------------------------------------------------------------
def confirm(prompt: str) -> bool:
    r = input(f"\n{prompt} [type 'yes' to proceed]: ").strip().lower()
    return r == "yes"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=".env", help="Path to .env file")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true", help="Skip confirmation")
    p.add_argument("--no-resume", action="store_true",
                   help="Don't skip snapshots already in DB")
    p.add_argument("--start-date", default=None,
                   help="Override BACKFILL_START (YYYY-MM-DD)")
    p.add_argument("--end-date", default=None,
                   help="Override BACKFILL_END (YYYY-MM-DD)")
    p.add_argument("--per-call-sleep", type=float, default=PER_CALL_SLEEP_SEC,
                   help=f"Seconds between API calls (default {PER_CALL_SLEEP_SEC})")
    p.add_argument("--per-day-sleep", type=float, default=PER_DAY_SLEEP_SEC,
                   help=f"Seconds between trading days (default {PER_DAY_SLEEP_SEC})")
    args = p.parse_args()

    env_path = Path(args.env)
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    db_url = os.getenv("CURVE_DB_URL") or os.getenv("DATABASE_URL")
    token  = os.getenv("ORATS_API_KEY") or os.getenv("ORATS_TOKEN")
    if not db_url:
        sys.exit("ERROR: DATABASE_URL (or CURVE_DB_URL) not found.")
    if not token:
        sys.exit("ERROR: ORATS_API_KEY (or ORATS_TOKEN) not found.")

    start_date = (dt.date.fromisoformat(args.start_date)
                  if args.start_date else BACKFILL_START)
    end_date   = (dt.date.fromisoformat(args.end_date)
                  if args.end_date else BACKFILL_END)
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    engine = create_engine(normalize_db_url(db_url), future=True,
                           pool_pre_ping=True)
    metadata = MetaData()
    table = Table(TABLE, metadata, autoload_with=engine)

    all_days = trading_days_between(start_date, end_date)
    # Reverse chronological — most recent first
    all_days = sorted(all_days, reverse=True)

    snaps_per_day = len(snapshots_for_day(start_date))
    total_snaps   = len(all_days) * snaps_per_day
    est_sec = total_snaps * (args.per_call_sleep + 0.5) \
              + len(all_days) * args.per_day_sleep

    print("=" * 92)
    print("ORATS monies/implied 1-min Backfill for orats_monies_minute")
    print("=" * 92)
    print(f"Ticker              : {TICKER}")
    print(f"Token (...{token[-4:]})")
    print(f"API date range      : {start_date} to {end_date}  "
          f"(reverse chronological)")
    print(f"Trading days        : {len(all_days)}")
    print(f"Window per day      : {RTH_OPEN_ET.strftime('%H:%M')} ET to "
          f"{RTH_CLOSE_ET.strftime('%H:%M')} ET "
          f"({snaps_per_day} snapshots/day)")
    print(f"Total snapshots     : {total_snaps:,}")
    print(f"Per-call sleep      : {args.per_call_sleep}s")
    print(f"Per-day sleep       : {args.per_day_sleep}s")
    print(f"Est. runtime        : ~{est_sec / 3600:.1f} hours "
          f"({est_sec / 60:.0f} min)")
    print(f"Resume              : {'ENABLED' if not args.no_resume else 'DISABLED'}")
    print(f"Target table        : {TABLE}")
    print(f"Cost                : $0 (included in ORATS subscription)")

    if args.dry_run:
        print("\n[dry-run] No API calls or DB writes. Exiting.")
        return

    if not args.yes:
        if not confirm("Proceed?"):
            print("Aborted.")
            return

    session = requests.Session()
    totals = {"days_processed": 0, "snapshots_processed": 0,
              "snapshots_skipped_resume": 0, "snapshots_skipped_empty": 0,
              "rows_added": 0, "failures": 0}

    t_start = time.time()

    try:
        for i, api_date in enumerate(all_days, start=1):
            t_day_start = time.time()
            day_stats = process_day(
                session, token, engine, table, TICKER,
                api_date, args.per_call_sleep, resume=not args.no_resume,
            )

            totals["days_processed"]           += 1
            totals["snapshots_processed"]      += day_stats["processed"]
            totals["snapshots_skipped_resume"] += day_stats["skipped_resume"]
            totals["snapshots_skipped_empty"]  += day_stats["skipped_empty"]
            totals["rows_added"]               += day_stats["rows_added"]
            totals["failures"]                 += len(day_stats["failures"])

            t_day = time.time() - t_day_start
            elapsed_total = time.time() - t_start
            avg_per_day = elapsed_total / i
            eta = avg_per_day * (len(all_days) - i)

            print(
                f"[{i:>3}/{len(all_days)}] {api_date}  "
                f"processed={day_stats['processed']:>3}  "
                f"resume_skipped={day_stats['skipped_resume']:>3}  "
                f"empty={day_stats['skipped_empty']:>3}  "
                f"rows+={day_stats['rows_added']:>6,}  "
                f"fails={len(day_stats['failures']):>2}  "
                f"[{t_day:>3.0f}s  total={elapsed_total/60:.1f}min  "
                f"eta={eta/60:.0f}min]"
            )

            if day_stats["failures"][:3]:
                for ts, err in day_stats["failures"][:3]:
                    print(f"    FAIL {ts}: {err[:120]}")

            if i < len(all_days):
                time.sleep(args.per_day_sleep)

    finally:
        session.close()

    elapsed = time.time() - t_start
    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)
    print(f"  days_processed          : {totals['days_processed']}")
    print(f"  snapshots_processed     : {totals['snapshots_processed']:,}")
    print(f"  snapshots_skipped_resume: {totals['snapshots_skipped_resume']:,}")
    print(f"  snapshots_skipped_empty : {totals['snapshots_skipped_empty']:,}")
    print(f"  rows_added              : {totals['rows_added']:,}")
    print(f"  failures                : {totals['failures']}")
    print(f"  elapsed                 : {elapsed/60:.1f} min")

    with engine.begin() as conn:
        cnt = conn.execute(text(
            f"SELECT COUNT(*) FROM {TABLE} WHERE ticker = :t"
        ), {"t": TICKER}).scalar()
        earliest = conn.execute(text(
            f"SELECT MIN(trade_date) FROM {TABLE} WHERE ticker = :t"
        ), {"t": TICKER}).scalar()
        latest = conn.execute(text(
            f"SELECT MAX(trade_date) FROM {TABLE} WHERE ticker = :t"
        ), {"t": TICKER}).scalar()

    print(f"\n  {TABLE} (ticker={TICKER}) post-run:")
    print(f"    total rows : {cnt:,}")
    print(f"    date range : {earliest} to {latest}")
    print()


if __name__ == "__main__":
    main()
