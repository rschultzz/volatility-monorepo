# apps/cron/orats_monies_today_ingest.py
import os
import re
import io
import datetime as dt
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from sqlalchemy import create_engine

BASE_URL = "https://api.orats.io"
ENDPOINT = "/datav2/hist/live/one-minute/monies/implied.csv"
TICKER = os.environ.get("ORATS_TICKER", "SPX")

DB_TABLE_NAME = os.environ.get("ORATS_MONIES_TABLE", "orats_monies_minute")


def camel_to_snake(name: str) -> str:
    """Converts a camelCase string to snake_case."""
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _get_env() -> tuple[str, str]:
    api_key = os.environ.get("ORATS_API_KEY")
    db_url = os.environ.get("DATABASE_URL")

    if not api_key:
        raise RuntimeError("ORATS_API_KEY is not set in the environment")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in the environment")

    return api_key, db_url


def _previous_minute_et(now_utc: dt.datetime | None = None) -> dt.datetime:
    """Return previous minute in America/New_York (handles DST)."""
    if now_utc is None:
        now_utc = dt.datetime.now(dt.timezone.utc)
    et = ZoneInfo("America/New_York")
    now_et = now_utc.astimezone(et)
    return now_et - dt.timedelta(minutes=1)


def _is_rth(et_dt: dt.datetime) -> bool:
    """RTH: 09:30–16:00 ET, Mon–Fri."""
    if et_dt.weekday() >= 5:  # 5 = Sat, 6 = Sun
        return False
    t = et_dt.time()
    return dt.time(9, 30) <= t <= dt.time(16, 0)


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
    print(f"[cron] Fetching {TICKER} for {target_et:%Y-%m-%d %H:%M} ET "
          f"(tradeDate={trade_datetime_str})")

    params = {
        "ticker": TICKER,
        "tradeDate": trade_datetime_str,
        "token": api_key,
    }
    url = f"{BASE_URL}{ENDPOINT}"

    # --- Fetch ---
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        csv_text = r.text.strip()
        if not csv_text or csv_text.startswith("<"):
            print("[cron] No data returned from API.")
            return
        df = pd.read_csv(io.StringIO(csv_text))
        if df.empty:
            print("[cron] Empty CSV for this minute.")
            return
        print(f"[cron] fetched {len(df)} rows.")
    except Exception as e:
        print(f"[cron] fetch error: {e}")
        return

    # --- Transform (match your backfill script) ---
    try:
        utc_ts = pd.to_datetime(df["snapShotDate"], errors="coerce", utc=True)
        df["snapshot_pt"] = (
            utc_ts.dt.tz_convert("America/Los_Angeles")
                  .dt.tz_localize(None)
        )
        df.columns = [camel_to_snake(c) for c in df.columns]
    except Exception as e:
        print(f"[cron] transform error: {e}")
        return

    # --- Write to DB ---
    try:
        engine = create_engine(db_url)
        df.to_sql(DB_TABLE_NAME, engine, if_exists="append", index=False)
        print(f"[cron] wrote {len(df)} rows to {DB_TABLE_NAME}.")
    except Exception as e:
        msg = str(e)
        if "violates unique constraint" in msg or "duplicate key value" in msg:
            print("[cron] duplicates for this minute, nothing to do.")
        else:
            print(f"[cron] DB write error: {e}")


if __name__ == "__main__":
    run_ingest_for_previous_minute()
