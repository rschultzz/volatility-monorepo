"""CR-CD A4 — one definition of the 1-minute flow rollup (1-second flow -> 1-minute rows).

Used by the backfill (scripts/cr_cd_a4_flow_1m_backfill.py), the Ironbeam worker
(apps/cron/ironbeam_flow_1s_aggregate.py) and the identity check. The arithmetic is Flow v2's
on-the-fly aggregation (apps/web/modules/Ironbeam/flow_v2.py `fetch_flow_minutes`): minute =
floor(epoch / 60) * 60; open = first 1-second open by ts_utc; close = last 1-second close;
high = max; low = min; volumes and counts summed; first/last trade ms = min/max; n_seconds = the
number of 1-second rows. Pure strings; no imports beyond the standard library (the cron worker
imports it with only apps/cron/requirements.txt installed).
"""
from __future__ import annotations

TABLE_1M = {
    "es_flow_1s_databento": "es_flow_1m_databento",
    "ironbeam_es_flow_1s": "ironbeam_es_flow_1m",
}
SOURCE = {
    "es_flow_1s_databento": "databento",
    "ironbeam_es_flow_1s": "ironbeam",
}

_SUM_COLS = ("volume", "buy_vol", "sell_vol", "unknown_vol", "trade_count")


def _columns(table_1s: str) -> list[str]:
    cols = ["minute_utc", "symbol"]
    if table_1s == "es_flow_1s_databento":
        cols.append("raw_symbol")
    cols += list(_SUM_COLS) + ["open", "high", "low", "close", "first_trade_ms", "last_trade_ms",
                               "n_seconds", "source"]
    return cols


def rollup_select(table_1s: str, where_sql: str) -> str:
    """SELECT producing 1-minute rows (columns in `_columns` order) from `table_1s` WHERE `where_sql`."""
    raw = ", max(raw_symbol) AS raw_symbol" if table_1s == "es_flow_1s_databento" else ""
    sums = ", ".join(f"sum({c})::bigint AS {c}" for c in _SUM_COLS)
    return f"""
        SELECT to_timestamp(floor(extract(epoch FROM ts_utc) / 60) * 60) AS minute_utc, symbol{raw},
               {sums},
               (array_agg(open ORDER BY ts_utc))[1] AS open, max(high) AS high, min(low) AS low,
               (array_agg(close ORDER BY ts_utc DESC))[1] AS close,
               min(first_trade_ms) AS first_trade_ms, max(last_trade_ms) AS last_trade_ms,
               count(*)::integer AS n_seconds, '{SOURCE[table_1s]}'::text AS source
        FROM {table_1s}
        WHERE {where_sql}
        GROUP BY 1, 2"""


def insert_sql(table_1s: str, where_sql: str, on_conflict: str = "nothing") -> str:
    """INSERT the rollup of `table_1s` WHERE `where_sql` into its 1-minute table.

    on_conflict: "nothing" (backfill: insert-only) or "update" (worker: replace the recent minutes).
    """
    cols = _columns(table_1s)
    if on_conflict == "nothing":
        tail = "ON CONFLICT (minute_utc, symbol) DO NOTHING"
    elif on_conflict == "update":
        sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in ("minute_utc", "symbol"))
        tail = f"ON CONFLICT (minute_utc, symbol) DO UPDATE SET {sets}"
    else:
        raise ValueError(on_conflict)
    return (f"INSERT INTO {TABLE_1M[table_1s]} ({', '.join(cols)})"
            f"{rollup_select(table_1s, where_sql)}\n        {tail}")


def v2_select(table_1m: str, where_sql: str) -> str:
    """The 1-minute table read in the shape Flow v2's on-the-fly query returns
    (symbol, t, o, h, l, c, buy, sell, vol, n)."""
    return f"""
        SELECT symbol, extract(epoch FROM minute_utc)::bigint AS t,
               open AS o, high AS h, low AS l, close AS c,
               buy_vol::float8 AS buy, sell_vol::float8 AS sell, volume::float8 AS vol,
               n_seconds AS n
        FROM {table_1m}
        WHERE {where_sql}"""
