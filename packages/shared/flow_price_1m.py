"""es_flow_price_1m — exact aggressor volume per minute per 0.25-pt price (CR-DD).

One definition for both writers:
  * live (scripts/cron_flow_price_1m.py): SQL over ironbeam_es_trades, source 'ironbeam';
  * backfill (scripts/cr_dd_backfill.py): pandas over CR-CA Databento DBN files, source 'databento'.

Cell = (minute_utc = trade time floored to the minute, symbol, price floored to the 0.25 grid).
buy_vol / sell_vol = contracts by aggressor side; trade_count = all prints, unknown side included (DD3).
Insert-only: a minute is written once, whole (DD4). Readers (CR-DC A1.3) never pool the two sources.

Completed minute (DD4): minute M is written once now >= M + 3 min, except M in [16:45, 17:00) America/New_York
(the last 15 min before the daily close), which wait until 18:00 ET that day (the reopen), so the closing
print that arrives during the maintenance break is caught (Step 0 S3: 43 min late on 2026-10-27).
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

TABLE = "es_flow_price_1m"
TRADES = "ironbeam_es_trades"
ET = ZoneInfo("America/New_York")
GRACE = timedelta(minutes=3)
HOLD_FROM, CLOSE, REOPEN = time(16, 45), time(17, 0), time(18, 0)
LOOKBACK = timedelta(hours=47)   # inside the 2-day prune (rows older than now - 48 h are deleted at 01:00 UTC)
COLS = ["minute_utc", "symbol", "price", "buy_vol", "sell_vol", "trade_count", "source"]


def floor_tick(price):
    """Floor to the 0.25 grid (works on floats and pandas Series)."""
    return (price * 4) // 1 / 4


def is_held(minute: datetime, now: datetime) -> bool:
    """True while minute M (UTC, tz-aware) is not yet complete under DD4."""
    if now < minute + GRACE:
        return True
    m_et = minute.astimezone(ET)
    if HOLD_FROM <= m_et.time() < CLOSE:
        return now < datetime.combine(m_et.date(), REOPEN, ET)
    return False


def window(now: datetime) -> tuple[datetime, datetime]:
    """[since, end) of trade times the live pass reads; both on minute boundaries."""
    now = now.astimezone(timezone.utc)
    since = (now - LOOKBACK).replace(second=0, microsecond=0)
    end = (now - GRACE).replace(second=0, microsecond=0) + timedelta(minutes=1)
    return since, end


# Minutes (per symbol) present in the trades window, not yet stored for 'ironbeam', not held.
_PENDING = f"""
  WITH t AS (
    SELECT date_trunc('minute', ts_utc) AS m, symbol, price, size, aggressor_side
    FROM {TRADES} WHERE ts_utc >= %(since)s AND ts_utc < %(end)s
  ),
  stored AS (
    SELECT DISTINCT minute_utc, symbol FROM {TABLE}
    WHERE source = 'ironbeam' AND minute_utc >= %(since)s AND minute_utc < %(end)s
  ),
  pending AS (
    SELECT t.* FROM t
    LEFT JOIN stored s ON s.minute_utc = t.m AND s.symbol = t.symbol
    WHERE s.minute_utc IS NULL
      AND NOT ((t.m AT TIME ZONE 'America/New_York')::time >= TIME '16:45'
               AND (t.m AT TIME ZONE 'America/New_York')::time < TIME '17:00'
               AND %(now)s < (((t.m AT TIME ZONE 'America/New_York')::date + TIME '18:00')
                              AT TIME ZONE 'America/New_York'))
  )
"""

LIVE_COUNT_SQL = _PENDING + "SELECT count(DISTINCT (m, symbol)), count(*) FROM pending"

LIVE_INSERT_SQL = f"""
  INSERT INTO {TABLE} ({", ".join(COLS)})
  {_PENDING}
  SELECT m, symbol, floor(price * 4) / 4,
         coalesce(sum(size) FILTER (WHERE aggressor_side = 'BUY'), 0),
         coalesce(sum(size) FILTER (WHERE aggressor_side = 'SELL'), 0),
         count(*), 'ironbeam'
  FROM pending GROUP BY 1, 2, 3
  ON CONFLICT DO NOTHING
  RETURNING minute_utc, symbol
"""


def aggregate_trades(tr: pd.DataFrame, symbol: str, ts_col: str, side_col: str,
                     buy: str, sell: str, source: str) -> pd.DataFrame:
    """pandas version of the cell rule: trades (ts tz-aware UTC, price, size, side) -> es_flow_price_1m rows."""
    df = pd.DataFrame({
        "minute_utc": pd.to_datetime(tr[ts_col], utc=True).dt.floor("min"),
        "price": floor_tick(tr["price"].astype(float)),
        "size": tr["size"].astype("int64"),
        "side": tr[side_col].astype(str),
    })
    df["buy_vol"] = df["size"].where(df["side"] == buy, 0)
    df["sell_vol"] = df["size"].where(df["side"] == sell, 0)
    g = df.groupby(["minute_utc", "price"], sort=True)
    out = pd.DataFrame({
        "buy_vol": g["buy_vol"].sum().astype("int64"),
        "sell_vol": g["sell_vol"].sum().astype("int64"),
        "trade_count": g.size().astype("int64"),
    }).reset_index()
    out.insert(1, "symbol", symbol)
    out["source"] = source
    return out[COLS]
