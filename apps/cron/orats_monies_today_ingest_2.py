#!/usr/bin/env python
"""
ORATS one-minute intraday ingest (improved DB handling).

- Runs once per minute as a Render cron job.
- Uses the SAME ORATS endpoint & params as your old script:

    https://api.orats.io/datav2/hist/live/one-minute/monies/implied.csv
      params: ticker=..., tradeDate=YYYYMMDDHHMM, token=...

- Targets the **previous** ET minute (to avoid partial bars) and only runs in RTH:

    RTH: 09:30–16:00 ET, Monday–Friday

- Transforms the CSV exactly like your gap-fill script:
    * adds snapshot_pt (PT-local timestamp)
    * converts camelCase columns to snake_case
      (snapShotDate -> snap_shot_date, quoteDate -> quote_date, expirDate -> expir_date, etc.)

- Writes into Postgres using PostgreSQL `INSERT ... ON CONFLICT DO NOTHING`,
  so:
    * duplicates by PRIMARY KEY (ticker, expir_date, quote_date) are ignored
    * missing rows for that minute are inserted

Environment variables:

    ORATS_API_KEY      (required)
    DATABASE_URL       (required)  - or CURVE_DB_URL as an alternative name
    ORATS_TICKER       (optional)  - default "SPX"
    ORATS_MONIES_TABLE (optional)  - default "orats_monies_minute"
"""

import os
import re
import io
import datetime as dt
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.dialects.postgresql import insert as pg_insert

# ----- ORATS config (same as old script) -----

BASE_URL = "https://api.orats.io"
ENDPOINT = "/datav2/hist/live/one-minute/monies/implied.csv"

TICKER = os.environ.get("ORATS_TICKER", "SPX")
DB_TABLE_NAME = os.environ.get("ORATS_MONIES_TABLE", "orats_monies_minute")

ET = ZoneInfo("America/New_York")


# ---------- Helpers ----------

def camel_to_snake(name: str) -> str:
    """Converts a camelCase string to snake_case."""
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _normalize_db_url(url: str) -> str:
    """
    Ensure SQLAlchemy-friendly driver.
    Examples:
        postgres://...     -> postgresql+psycopg://...
        postgresql://...   -> postgresql+psycopg://...
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _get_env() -> tuple[str, str]:
    """Load ORATS_API_KEY and DB URL from env."""
    api_key = os.environ.get("ORATS_API_KEY")
    db_url = os.environ.get("CURVE_DB_URL") or os.environ.get("DATABASE_URL")

    if not api_key:
        raise RuntimeError("ORATS_API_KEY is not set in the environment")
    if not db_url:
        raise RuntimeError("CURVE_DB_URL or DATABASE_URL is not set in the environment")

    return api_key, _normalize_db_url(db_url)


def _previous_minute_et(now_utc: dt.datetime | None = None) -> dt.datetime:
    """Return previous minute in America/New_York (handles DST)."""
    if now_utc is None:
        now_utc = dt.datetime.now(dt.timezone.utc)
    now_et = now_utc.astimezone(ET)
    return now_et - dt.timedelta(minutes=1)


def _is_rth(et_dt: dt.datetime) -> bool:
    """RTH: 09:30–16:00 ET, Mon–Fri."""
    if et_dt.weekday() >= 5:  # 5 = Sat, 6 = Sun
        return False
    t = et_dt.time()
    return dt.time(9, 30) <= t <= dt.time(16, 0)


def transform_orats_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform ORATS dataframe:

    - Add snapshot_pt (PT-local naive timestamp).
    - Convert all columns to snake_case to match DB schema
      (snapShotDate -> snap_shot_date, quoteDate -> quote_date, etc.).
    """
    if df.empty:
        return df

    # ORATS snapShotDate is a UTC ISO string
    utc_ts = pd.to_datetime(df["snapShotDate"], errors="coerce", utc=True)
    df["snapshot_pt"] = (
        utc_ts.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
    )

    # Rename columns to snake_case
    df.columns = [camel_to_snake(col) for col in df.columns]
    return df


def upsert_dataframe(df: pd.DataFrame, engine, table: Table) -> None:
    """
    Upsert dataframe rows into Postgres with ON CONFLICT DO NOTHING.

    This means:
      - If a row with the same PRIMARY KEY (ticker, expir_date, quote_date) already exists,
        it is ignored.
      - Missing rows are inserted.

    We don't try to track an exact 'rows inserted' count because rowcount is
    unreliable with ON CONFLICT; we just log how many rows we attempted.
    """
    if df.empty:
        print("[cron] transformed dataframe is empty, nothing to upsert.")
        return

    records = df.to_dict(orient="records")
    chunk_size = 500  # keep parameter count under Postgres limit

    attempted = len(records)

    with engine.begin() as conn:
        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            stmt = pg_insert(table).values(chunk).on_conflict_do_nothing()
            conn.execute(stmt)

    print(
        f"[cron] upserted {attempted} rows "
        f"(duplicates, if any, were ignored by primary key)."
    )


# ---------- Main cron entrypoint ----------

def run_ingest_for_previous_minute() -> None:
    try:
        api_key, db_url = _get_env()
    except RuntimeError as e:
        print(f"[cron] env error: {e}")
        return

    target_et = _previous_minute_et()
    if not _is_rth(target_et):
        print(f"[cron] {target_et:%Y-%m-%d %H:%M} ET is outside RTH, skipping.")
        return

    trade_datetime_str = target_et.strftime("%Y%m%d%H%M")
    print(
        f"[cron] Fetching {TICKER} for {target_et:%Y-%m-%d %H:%M} ET "
        f"(tradeDate={trade_datetime_str})"
    )

    params = {
        "ticker": TICKER,
        "tradeDate": trade_datetime_str,
        "token": api_key,
    }
    url = f"{BASE_URL}{ENDPOINT}"

    # --- Fetch from ORATS (same endpoint as old script) ---
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        csv_text = r.text.strip()
        if not csv_text or csv_text.startswith("<"):
            print("[cron] No CSV data returned from API.")
            return
        df = pd.read_csv(io.StringIO(csv_text))
        if df.empty:
            print("[cron] Empty CSV for this minute.")
            return
        print(f"[cron] fetched {len(df)} rows from ORATS.")
    except Exception as e:
        print(f"[cron] fetch error: {e}")
        return

    # --- Transform (same as gap-fill) ---
    try:
        df = transform_orats_df(df)
        print(f"[cron] {len(df)} rows after transform.")
    except Exception as e:
        print(f"[cron] transform error: {e}")
        return

    # --- Upsert into DB with ON CONFLICT DO NOTHING ---
    try:
        engine = create_engine(db_url, future=True)
        metadata = MetaData()
        table = Table(DB_TABLE_NAME, metadata, autoload_with=engine)
        upsert_dataframe(df, engine, table)
    except Exception as e:
        print(f"[cron] DB upsert error: {e}")


if __name__ == "__main__":
    run_ingest_for_previous_minute()
