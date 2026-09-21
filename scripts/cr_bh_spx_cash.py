"""CR-BH — SPX-cash session OHLC built from 0DTE `orats_monies_minute` spot prints.

SPX cash per minute = avg(spot_price) over the 0DTE SPX/SPXW rows of that minute
(predicate matches partial index idx_omm_front_minute_pt_ticker). `spot_price`, not
`stock_price`. `snapshot_pt` is naive Pacific time.

Session window: 06:30:00–13:00:00 PT inclusive on the snapshot minute — the same bounds
cr_b_backfill_outcomes._RTH_BARS_SQL applies to ES bar-open timestamps (see the spec,
Step 0 Q3, for the one-minute mismatch this implies at the close).

Minute-sampled: one print per minute, so high/low are max/min of samples, not of ticks
(spec Step 0 Q5 quantifies the understatement).

Public API
----------
fetch_spx_minutes(conn, date_from, date_to) -> DataFrame[session_date, minute, spx, n_rows]
filter_bad_prints(minutes) -> (clean, dropped)
daily_ohlc(clean) -> DataFrame indexed by date with open/high/low/close/n_minutes
load_spx_daily_bars(conn, date_from, date_to) -> (daily, dropped, minutes)
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

SESSION_START = dt.time(6, 30)
SESSION_END = dt.time(13, 0)          # inclusive

# Bad-print rule — DRAFT. Step 0 Q6/Q7 rejected this version (drops ES-confirmed real moves on
# 2025-04-07 / 04-09, misses frozen runs and the 2023 15-minute lag). Final rule is pending the
# price-series decision recorded in specs/CR-BH-spx-cash-shadow-outcomes.md (Q7).
NEIGHBOR_WINDOW = 11                  # centered, minutes (±5), within one session
NEIGHBOR_MIN_PERIODS = 4
NEIGHBOR_MAX_DEV = 0.005              # |print − neighbor median| / median
DAY_MAX_DEV = 0.12                    # |print − session median| / median (gross guard for long bad runs)

_SPX_MINUTES_SQL = """
    SELECT date_trunc('minute', snapshot_pt)                 AS minute,
           avg(spot_price) FILTER (WHERE spot_price > 0)     AS spx,
           count(*)                                          AS n_rows,
           count(*) FILTER (WHERE spot_price IS NULL OR spot_price <= 0) AS n_nonpos
    FROM orats_monies_minute
    WHERE expir_date = trade_date
      AND ticker IN ('SPX', 'SPXW')
      AND date_trunc('minute', snapshot_pt) >= %s
      AND date_trunc('minute', snapshot_pt) <  %s
    GROUP BY 1
    ORDER BY 1
"""


def fetch_spx_minutes(conn, date_from: dt.date, date_to: dt.date) -> pd.DataFrame:
    """One row per PT minute inside the session window, [date_from, date_to] inclusive."""
    start = dt.datetime.combine(date_from, dt.time(0, 0))
    end = dt.datetime.combine(date_to + dt.timedelta(days=1), dt.time(0, 0))
    with conn.cursor() as cur:
        cur.execute(_SPX_MINUTES_SQL, (start, end))
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["minute", "spx", "n_rows", "n_nonpos"])
    if df.empty:
        df["session_date"] = []
        return df
    df["minute"] = pd.to_datetime(df["minute"])
    df["spx"] = pd.to_numeric(df["spx"], errors="coerce")
    t = df["minute"].dt.time
    df = df[(t >= SESSION_START) & (t <= SESSION_END)].copy()
    df["session_date"] = df["minute"].dt.date
    return df.reset_index(drop=True)


def filter_bad_prints(minutes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop non-positive / NULL prints, then prints far from their neighbors.

    Returns (clean, dropped); dropped carries `reason`, `ref` (the median compared to).
    A level shift is not dropped: past the step the centered median is the new level.
    Only spikes shorter than half the window, or prints > DAY_MAX_DEV from the
    session median, are removed.
    """
    df = minutes.copy()
    df["reason"] = None
    df["ref"] = float("nan")
    nonpos = df["spx"].isna() | (df["spx"] <= 0)
    df.loc[nonpos, "reason"] = "nonpositive"

    ok = df[~nonpos]
    day_med = ok.groupby("session_date")["spx"].transform("median")
    gross = (ok["spx"] - day_med).abs() / day_med > DAY_MAX_DEV
    df.loc[gross[gross].index, "reason"] = "session_median"
    df.loc[gross[gross].index, "ref"] = day_med[gross]

    ok = df[df["reason"].isna()]
    nb_med = ok.groupby("session_date")["spx"].transform(
        lambda s: s.rolling(NEIGHBOR_WINDOW, center=True, min_periods=NEIGHBOR_MIN_PERIODS).median()
    )
    spike = (ok["spx"] - nb_med).abs() / nb_med > NEIGHBOR_MAX_DEV
    df.loc[spike[spike].index, "reason"] = "neighbor_median"
    df.loc[spike[spike].index, "ref"] = nb_med[spike]

    dropped = df[df["reason"].notna()].copy()
    clean = df[df["reason"].isna()].drop(columns=["reason", "ref"]).copy()
    return clean, dropped


def daily_ohlc(clean: pd.DataFrame) -> pd.DataFrame:
    """Session OHLC per date: open = first print, close = last print, high/low = max/min of prints."""
    if clean.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "n_minutes"])
    g = clean.sort_values("minute").groupby("session_date")["spx"]
    out = pd.DataFrame({
        "open": g.first(), "high": g.max(), "low": g.min(), "close": g.last(), "n_minutes": g.size(),
    })
    out.index = [d for d in out.index]   # keep as date objects (compute_outcome contract)
    return out


def load_spx_daily_bars(conn, date_from: dt.date, date_to: dt.date):
    minutes = fetch_spx_minutes(conn, date_from, date_to)
    clean, dropped = filter_bad_prints(minutes)
    return daily_ohlc(clean), dropped, minutes
