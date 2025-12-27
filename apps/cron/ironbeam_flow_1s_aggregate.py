#!/usr/bin/env python3
"""
Aggregate Ironbeam raw trades into 1-second "flow" buckets for CVD / orderflow indicators.

Design goals:
- Keep websocket ingest focused on writing raw trades into ironbeam_es_trades.
- This script independently tails the raw trades table and upserts 1-second aggregates
  into ironbeam_es_flow_1s (or env IRONBEAM_FLOW_TABLE).
- Safe on restarts: it uses a per-symbol watermark table (ironbeam_flow_agg_state).
- Safe with late trades: it reprocesses a small overlap window and does "replace"
  UPSERTs (not additive), so double-processing does NOT double-count.

Expected raw trades schema (matches your screenshot):
  ironbeam_es_trades:
    ts_utc (timestamptz), symbol (text), price (double), size (bigint),
    aggressor_side (text: 'BUY'/'SELL'), trade_date (text yyyymmdd),
    trade_id (bigint), sequence_number (bigint), trade_key (text), ...

Expected flow table schema (recommended):
  ironbeam_es_flow_1s:
    ts_utc (timestamptz), symbol (text), trade_date (date),
    volume (int), notional (double), trade_count (int),
    buy_vol/sell_vol/unknown_vol (int), buy_count/sell_count/unknown_count (int),
    open/high/low/close (double), first_trade_ms/last_trade_ms (bigint),
    PRIMARY KEY (symbol, ts_utc)

Environment variables:
  DATABASE_URL                    (required) SQLAlchemy URL
  IRONBEAM_TRADES_TABLE           default: ironbeam_es_trades
  IRONBEAM_FLOW_TABLE             default: ironbeam_es_flow_1s
  IRONBEAM_FLOW_STATE_TABLE       default: ironbeam_flow_agg_state
  IRONBEAM_SYMBOLS                comma-separated symbols (optional)
  IRONBEAM_SYMBOL                 single symbol fallback (optional)

  IRONBEAM_FLOW_INITIAL_LOOKBACK_MIN  default: 15   (first run backfill window)
  IRONBEAM_FLOW_MAX_LATENESS_SEC      default: 2    (don't finalize last N seconds)
  IRONBEAM_FLOW_OVERLAP_SEC           default: 10   (reprocess last N seconds each loop)
  IRONBEAM_FLOW_MAX_CATCHUP_SEC       default: 600  (process at most N seconds per loop)
  IRONBEAM_FLOW_LOOP_SLEEP_SEC        default: 1.0  (sleep between loops)

Run (local):
  python apps/cron/ironbeam_flow_1s_aggregate.py

Run (Render background worker):
  python apps/cron/ironbeam_flow_1s_aggregate.py
"""

from __future__ import annotations

import os
import time
import datetime as dt
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text

# Optional: load .env locally if python-dotenv is installed
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _floor_to_second(ts: dt.datetime) -> dt.datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc).replace(microsecond=0)


def _parse_symbols() -> List[str]:
    s = os.getenv("IRONBEAM_SYMBOLS", "").strip()
    if s:
        return [x.strip() for x in s.split(",") if x.strip()]
    s1 = os.getenv("IRONBEAM_SYMBOL", "").strip()
    if s1:
        return [s1]
    return ["XCME:ES.H26"]


def _parse_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _parse_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def ensure_flow_table(engine, flow_table: str) -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {flow_table} (
      ts_utc         timestamptz NOT NULL,
      symbol         text        NOT NULL,
      trade_date     date        NULL,

      volume         integer     NOT NULL DEFAULT 0,
      notional       double precision NOT NULL DEFAULT 0,
      trade_count    integer     NOT NULL DEFAULT 0,

      buy_vol        integer     NOT NULL DEFAULT 0,
      sell_vol       integer     NOT NULL DEFAULT 0,
      unknown_vol    integer     NOT NULL DEFAULT 0,

      buy_count      integer     NOT NULL DEFAULT 0,
      sell_count     integer     NOT NULL DEFAULT 0,
      unknown_count  integer     NOT NULL DEFAULT 0,

      open           double precision NULL,
      high           double precision NULL,
      low            double precision NULL,
      close          double precision NULL,

      first_trade_ms bigint NULL,
      last_trade_ms  bigint NULL,

      PRIMARY KEY (symbol, ts_utc)
    );
    """
    idx1 = f"CREATE INDEX IF NOT EXISTS idx_{flow_table}_ts ON {flow_table} (ts_utc);"
    idx2 = f"CREATE INDEX IF NOT EXISTS idx_{flow_table}_trade_date ON {flow_table} (trade_date);"

    with engine.begin() as conn:
        conn.execute(text(ddl))
        conn.execute(text(idx1))
        conn.execute(text(idx2))


def ensure_state_table(engine, state_table: str) -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {state_table} (
      symbol      text PRIMARY KEY,
      last_ts_utc timestamptz NOT NULL,
      updated_at  timestamptz NOT NULL DEFAULT now()
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def read_state(engine, state_table: str, symbols: List[str]) -> Dict[str, dt.datetime]:
    q = text(f"SELECT symbol, last_ts_utc FROM {state_table} WHERE symbol = ANY(:symbols);")
    out: Dict[str, dt.datetime] = {}
    with engine.begin() as conn:
        rows = conn.execute(q, {"symbols": symbols}).fetchall()
    for sym, last_ts in rows:
        if isinstance(last_ts, dt.datetime):
            out[sym] = last_ts.astimezone(dt.timezone.utc)
    return out


def write_state(engine, state_table: str, symbol: str, last_ts_utc: dt.datetime) -> None:
    q = text(f"""
      INSERT INTO {state_table}(symbol, last_ts_utc, updated_at)
      VALUES (:symbol, :last_ts_utc, now())
      ON CONFLICT (symbol)
      DO UPDATE SET last_ts_utc = EXCLUDED.last_ts_utc, updated_at = now();
    """)
    with engine.begin() as conn:
        conn.execute(q, {"symbol": symbol, "last_ts_utc": last_ts_utc})


def initialize_state_if_missing(engine, state_table: str, symbol: str, initial_ts: dt.datetime) -> None:
    q = text(f"""
      INSERT INTO {state_table}(symbol, last_ts_utc, updated_at)
      VALUES (:symbol, :last_ts_utc, now())
      ON CONFLICT (symbol) DO NOTHING;
    """)
    with engine.begin() as conn:
        conn.execute(q, {"symbol": symbol, "last_ts_utc": initial_ts})


def fetch_trades(engine, trades_table: str, symbols: List[str], start_ts: dt.datetime, end_ts: dt.datetime) -> pd.DataFrame:
    q = text(f"""
      SELECT
        ts_utc, symbol, price, size, aggressor_side, trade_date,
        trade_id, sequence_number, trade_key
      FROM {trades_table}
      WHERE symbol = ANY(:symbols)
        AND ts_utc >= :start_ts
        AND ts_utc < :end_ts
      ORDER BY ts_utc ASC, trade_id ASC NULLS LAST, sequence_number ASC NULLS LAST, trade_key ASC NULLS LAST;
    """)
    with engine.begin() as conn:
        rows = conn.execute(q, {"symbols": symbols, "start_ts": start_ts, "end_ts": end_ts}).fetchall()

    if not rows:
        return pd.DataFrame(columns=[
            "ts_utc","symbol","price","size","aggressor_side","trade_date",
            "trade_id","sequence_number","trade_key"
        ])

    df = pd.DataFrame(rows, columns=[
        "ts_utc","symbol","price","size","aggressor_side","trade_date",
        "trade_id","sequence_number","trade_key"
    ])

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_utc", "symbol", "price", "size"])
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0).astype("int64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["aggressor_side"] = df["aggressor_side"].astype("string")
    df["trade_date"] = df["trade_date"].astype("string")
    return df


def aggregate_1s(df_trades: pd.DataFrame) -> pd.DataFrame:
    if df_trades.empty:
        return pd.DataFrame()

    df = df_trades.copy()
    df["ts_sec"] = df["ts_utc"].dt.floor("s")

    is_buy = df["aggressor_side"].str.upper() == "BUY"
    is_sell = df["aggressor_side"].str.upper() == "SELL"

    df["buy_vol"] = df["size"].where(is_buy, 0)
    df["sell_vol"] = df["size"].where(is_sell, 0)
    df["unknown_vol"] = df["size"].where(~(is_buy | is_sell), 0)

    df["buy_count"] = is_buy.astype("int64")
    df["sell_count"] = is_sell.astype("int64")
    df["unknown_count"] = (~(is_buy | is_sell)).astype("int64")

    df["notional"] = df["price"] * df["size"]
    df["ts_ms"] = (df["ts_utc"].view("int64") // 1_000_000).astype("int64")

    gb = df.groupby(["symbol", "ts_sec"], sort=False)

    out = pd.DataFrame({
        "symbol": gb["symbol"].first(),
        "ts_utc": gb["ts_sec"].first(),
        "volume": gb["size"].sum().astype("int64"),
        "notional": gb["notional"].sum().astype("float64"),
        "trade_count": gb["size"].count().astype("int64"),
        "buy_vol": gb["buy_vol"].sum().astype("int64"),
        "sell_vol": gb["sell_vol"].sum().astype("int64"),
        "unknown_vol": gb["unknown_vol"].sum().astype("int64"),
        "buy_count": gb["buy_count"].sum().astype("int64"),
        "sell_count": gb["sell_count"].sum().astype("int64"),
        "unknown_count": gb["unknown_count"].sum().astype("int64"),
        "open": gb["price"].first().astype("float64"),
        "close": gb["price"].last().astype("float64"),
        "high": gb["price"].max().astype("float64"),
        "low": gb["price"].min().astype("float64"),
        "first_trade_ms": gb["ts_ms"].min().astype("int64"),
        "last_trade_ms": gb["ts_ms"].max().astype("int64"),
        "trade_date_raw": gb["trade_date"].first(),
    }).reset_index(drop=True)

    def _to_date(s: Optional[str]) -> Optional[dt.date]:
        if s is None:
            return None
        s = str(s)
        if not s or s.lower() == "nan":
            return None
        try:
            return dt.datetime.strptime(s, "%Y%m%d").date()
        except Exception:
            return None

    out["trade_date"] = out["trade_date_raw"].apply(_to_date)
    out = out.drop(columns=["trade_date_raw"], errors="ignore")

    out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts_utc", "symbol"])
    return out


def upsert_flow(engine, flow_table: str, df_flow: pd.DataFrame) -> int:
    if df_flow.empty:
        return 0

    q = text(f"""
      INSERT INTO {flow_table} (
        ts_utc, symbol, trade_date,
        volume, notional, trade_count,
        buy_vol, sell_vol, unknown_vol,
        buy_count, sell_count, unknown_count,
        open, high, low, close,
        first_trade_ms, last_trade_ms
      )
      VALUES (
        :ts_utc, :symbol, :trade_date,
        :volume, :notional, :trade_count,
        :buy_vol, :sell_vol, :unknown_vol,
        :buy_count, :sell_count, :unknown_count,
        :open, :high, :low, :close,
        :first_trade_ms, :last_trade_ms
      )
      ON CONFLICT (symbol, ts_utc) DO UPDATE SET
        trade_date     = EXCLUDED.trade_date,
        volume         = EXCLUDED.volume,
        notional       = EXCLUDED.notional,
        trade_count    = EXCLUDED.trade_count,
        buy_vol        = EXCLUDED.buy_vol,
        sell_vol       = EXCLUDED.sell_vol,
        unknown_vol    = EXCLUDED.unknown_vol,
        buy_count      = EXCLUDED.buy_count,
        sell_count     = EXCLUDED.sell_count,
        unknown_count  = EXCLUDED.unknown_count,
        open           = EXCLUDED.open,
        high           = EXCLUDED.high,
        low            = EXCLUDED.low,
        close          = EXCLUDED.close,
        first_trade_ms = EXCLUDED.first_trade_ms,
        last_trade_ms  = EXCLUDED.last_trade_ms;
    """)

    records = df_flow.to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(q, records)
    return len(records)


def main() -> None:
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        raise SystemExit("Missing DATABASE_URL (SQLAlchemy format).")

    trades_table = os.getenv("IRONBEAM_TRADES_TABLE", "ironbeam_es_trades").strip()
    flow_table = os.getenv("IRONBEAM_FLOW_TABLE", "ironbeam_es_flow_1s").strip()
    state_table = os.getenv("IRONBEAM_FLOW_STATE_TABLE", "ironbeam_flow_agg_state").strip()

    symbols = _parse_symbols()

    initial_lookback_min = _parse_int("IRONBEAM_FLOW_INITIAL_LOOKBACK_MIN", 15)
    max_lateness_sec = _parse_int("IRONBEAM_FLOW_MAX_LATENESS_SEC", 2)
    overlap_sec = _parse_int("IRONBEAM_FLOW_OVERLAP_SEC", 10)
    max_catchup_sec = _parse_int("IRONBEAM_FLOW_MAX_CATCHUP_SEC", 600)
    loop_sleep = _parse_float("IRONBEAM_FLOW_LOOP_SLEEP_SEC", 1.0)

    engine = create_engine(db_url, pool_pre_ping=True)

    ensure_flow_table(engine, flow_table)
    ensure_state_table(engine, state_table)

    init_ts = _floor_to_second(_utcnow() - dt.timedelta(minutes=initial_lookback_min))
    for sym in symbols:
        initialize_state_if_missing(engine, state_table, sym, init_ts)

    backoff = 1.0
    while True:
        try:
            state = read_state(engine, state_table, symbols)
            now = _utcnow()
            end = _floor_to_second(now - dt.timedelta(seconds=max_lateness_sec))

            did_work = False
            for sym in symbols:
                last = state.get(sym, init_ts).astimezone(dt.timezone.utc)

                gap = (end - last).total_seconds()
                if gap <= 0:
                    continue

                did_work = True
                chunk_end = end if gap <= max_catchup_sec else _floor_to_second(last + dt.timedelta(seconds=max_catchup_sec))
                q_start = last - dt.timedelta(seconds=overlap_sec)

                df_tr = fetch_trades(engine, trades_table, [sym], q_start, chunk_end)
                df_flow = aggregate_1s(df_tr)
                buckets = upsert_flow(engine, flow_table, df_flow)

                write_state(engine, state_table, sym, chunk_end)

                print(
                    f"[flow_1s] {sym} [{q_start.isoformat()} -> {chunk_end.isoformat()}] "
                    f"trades={len(df_tr)} buckets={buckets} state={chunk_end.isoformat()}"
                )

            backoff = 1.0
            time.sleep(loop_sleep if did_work else max(loop_sleep, 0.5))

        except KeyboardInterrupt:
            print("Exiting...")
            return
        except Exception as e:
            print(f"[flow_1s] loop exception: {e!r}")
            time.sleep(min(60.0, backoff))
            backoff = min(60.0, backoff * 2.0)


if __name__ == "__main__":
    main()
