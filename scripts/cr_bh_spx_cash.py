"""CR-BH — SPX-cash session OHLC from 0DTE `orats_monies_minute` rows (per-period hybrid series).

Spec: specs/CR-BH-spx-cash-shadow-outcomes.md, Step 0 Q7 + Amendment A1.

Neither price column is a clean SPX cash series over the whole corpus:
  - `spot_price` is 15 minutes delayed on 2023-05-01 → 2023-09-07 and 2023-10-24 → 2023-11-08
    (measured: cross-correlation lag vs ES 1m returns), and drifts off the index on ~27 % of
    2026 days;
  - `stock_price` is clean from 2023-11-09 but its ES basis wanders 11–24 pts on the lagged days.

Series used (A1):
  trade_date <  CUTOVER  'spot_shifted_2023'  spot_price; on lagged days timestamps −15 min and
                                              the close = median stock_price 12:56–13:00 PT
                                              (true 12:46–13:00 is absent from the lagged column)
  trade_date >= CUTOVER  'stock_price'        stock_price

No ES price enters this module. `snapshot_pt` is naive Pacific time; one row per minute
(the SPX/SPXW avg() is a no-op today). Predicate matches idx_omm_front_minute_pt_ticker.

Session window (after any shift): prints 06:33–13:00 PT; open = first print >= 06:33 (the
06:30–06:32 index prints are largely the prior close). Minute-sampled: high/low are max/min of
one print per minute (Step 0 Q5: understates high 0.75 / low 1.38 pts, median).

Public API
----------
fetch_minutes(conn, date_from, date_to) -> DataFrame[minute, spot, stock]
build_series(raw) -> (clean[session_date, minute, px], dropped[session_date, minute, px, col, reason])
daily_ohlc(clean) -> DataFrame indexed by date: open/high/low/close/n_minutes/first/last
load_spx_daily_bars(conn, date_from, date_to) -> (daily, dropped, clean)
series_segment(d), is_lagged(d)
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

CUTOVER = dt.date(2023, 11, 9)
LAGGED_WINDOWS = (
    (dt.date(2023, 5, 1), dt.date(2023, 9, 7)),
    (dt.date(2023, 10, 24), dt.date(2023, 11, 8)),
)
LAG = pd.Timedelta(minutes=15)

RAW_START = dt.time(6, 30)            # raw prints considered (original stamps)
SESSION_OPEN = dt.time(6, 33)         # first print used, after any shift
SESSION_END = dt.time(13, 0)          # inclusive
CLOSE_MEDIAN_START = dt.time(12, 56)  # lagged days: close = median stock_price 12:56–13:00

GROSS_MAX_DEV = 0.10                  # vs centered 5-session median of session medians
GROSS_WINDOW_SESSIONS = 5
FROZEN_MIN_RUN = 3                    # consecutive identical prints
SPIKE_MAX_DEV = 0.003                 # interior run of <= 2 prints beyond both neighbours (floor)
SPIKE_VOL_MULT = 10.0                 # …or 10 × the session's median |1-min return| if larger (2025-04 days)
SPIKE_MAX_RUN = 2
SPIKE_NEIGHBOR_MAX_GAP = pd.Timedelta(minutes=3)
EDGE_MAX_DEV = 0.005                  # first / last print vs its neighbours

_MINUTES_SQL = """
    SELECT date_trunc('minute', snapshot_pt)                 AS minute,
           avg(spot_price)  FILTER (WHERE spot_price  > 0)   AS spot,
           avg(stock_price) FILTER (WHERE stock_price > 0)   AS stock
    FROM orats_monies_minute
    WHERE expir_date = trade_date
      AND ticker IN ('SPX', 'SPXW')
      AND date_trunc('minute', snapshot_pt) >= %s
      AND date_trunc('minute', snapshot_pt) <  %s
    GROUP BY 1
    ORDER BY 1
"""


def series_segment(d: dt.date) -> str:
    return "spot_shifted_2023" if d < CUTOVER else "stock_price"


def is_lagged(d: dt.date) -> bool:
    return any(a <= d <= b for a, b in LAGGED_WINDOWS)


def fetch_minutes(conn, date_from: dt.date, date_to: dt.date) -> pd.DataFrame:
    start = dt.datetime.combine(date_from, dt.time(0, 0))
    end = dt.datetime.combine(date_to + dt.timedelta(days=1), dt.time(0, 0))
    with conn.cursor() as cur:
        cur.execute(_MINUTES_SQL, (start, end))
        rows = cur.fetchall()
    return normalize_minutes(pd.DataFrame(rows, columns=["minute", "spot", "stock"]))


def normalize_minutes(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[["minute", "spot", "stock"]].copy()
    df["minute"] = pd.to_datetime(df["minute"])
    for c in ("spot", "stock"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    t = df["minute"].dt.time
    df = df[(t >= RAW_START) & (t <= SESSION_END)].sort_values("minute")
    df["session_date"] = df["minute"].dt.date
    return df.reset_index(drop=True)


def _frozen_runs(v: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of runs of >= FROZEN_MIN_RUN identical values."""
    runs, i, n = [], 0, len(v)
    while i < n:
        j = i + 1
        while j < n and v[j] == v[i]:
            j += 1
        if j - i >= FROZEN_MIN_RUN:
            runs.append((i, j))
        i = j
    return runs


def _filter_day(minute: np.ndarray, v: np.ndarray, ref: float) -> np.ndarray:
    """Return an object array of drop reasons (None = keep) for one session's prints."""
    n = len(v)
    reason = np.full(n, None, dtype=object)
    reason[~(v > 0)] = "nonpositive"                      # also catches NaN
    if ref and ref > 0:
        bad = (reason == None) & (np.abs(v - ref) / ref > GROSS_MAX_DEV)  # noqa: E711
        reason[bad] = "gross"

    keep = np.flatnonzero(reason == None)  # noqa: E711
    for a, b in _frozen_runs(v[keep]):
        first = a if a == 0 else a + 1                    # opening run: drop whole
        reason[keep[first:b]] = "frozen_open" if a == 0 else "frozen"

    # isolated spikes: interior runs of <= SPIKE_MAX_RUN prints beyond both neighbours
    kv0 = v[np.flatnonzero(reason == None)]  # noqa: E711
    vol = float(np.median(np.abs(np.diff(kv0)) / kv0[:-1])) if len(kv0) > 10 else 0.0
    spike_dev = max(SPIKE_MAX_DEV, SPIKE_VOL_MULT * vol)
    edge_dev = max(EDGE_MAX_DEV, SPIKE_VOL_MULT * vol)
    changed = True
    while changed:
        changed = False
        keep = np.flatnonzero(reason == None)  # noqa: E711
        kv, kt = v[keep], minute[keep]
        for k in range(1, SPIKE_MAX_RUN + 1):
            for i in range(1, len(keep) - k):
                prev, nxt, run = kv[i - 1], kv[i + k], kv[i:i + k]
                if (kt[i] - kt[i - 1] > SPIKE_NEIGHBOR_MAX_GAP
                        or kt[i + k] - kt[i + k - 1] > SPIKE_NEIGHBOR_MAX_GAP):
                    continue
                hi, lo = max(prev, nxt), min(prev, nxt)
                if (run > hi * (1 + spike_dev)).all() or (run < lo * (1 - spike_dev)).all():
                    reason[keep[i:i + k]] = "spike"
                    changed = True
                    break
            if changed:
                break

    # edges: first print vs median of the next 3, last print vs the previous one
    keep = np.flatnonzero(reason == None)  # noqa: E711
    if len(keep) >= 5:
        kv, kt = v[keep], minute[keep]
        nxt = np.median(kv[1:4])
        if kt[3] - kt[0] <= 2 * SPIKE_NEIGHBOR_MAX_GAP and abs(kv[0] - nxt) / nxt > edge_dev:
            reason[keep[0]] = "edge_first"
        if kt[-1] - kt[-2] <= SPIKE_NEIGHBOR_MAX_GAP and abs(kv[-1] - kv[-2]) / kv[-2] > edge_dev:
            reason[keep[-1]] = "edge_last"
    return reason


def build_series(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply segment choice, filter, lag shift and the lagged-day close rule."""
    raw = raw if "session_date" in raw.columns else normalize_minutes(raw)
    days = sorted(raw["session_date"].unique())
    col_of = {d: ("spot" if d < CUTOVER else "stock") for d in days}

    # gross-guard reference: centered 5-session median of session medians of the chosen column
    chosen = raw["spot"].where(raw["session_date"] < CUTOVER, raw["stock"])
    med = chosen[chosen > 0].groupby(raw["session_date"]).median().reindex(days)
    ref = med.rolling(GROSS_WINDOW_SESSIONS, center=True, min_periods=3).median()

    clean_parts, dropped_parts = [], []
    for d, g in raw.groupby("session_date"):
        col = col_of[d]
        minute, v = g["minute"].to_numpy(), g[col].to_numpy(dtype=float)
        reason = _filter_day(minute, v, float(ref.get(d, np.nan)))
        drop = reason != None  # noqa: E711
        if drop.any():
            dropped_parts.append(pd.DataFrame({
                "session_date": d, "minute": minute[drop], "px": v[drop], "col": col, "reason": reason[drop]}))
        s = pd.DataFrame({"session_date": d, "minute": minute[~drop], "px": v[~drop]})

        if is_lagged(d):
            s["minute"] = s["minute"] - LAG                 # print stamped t shows the index at t − 15
            tt = g["minute"].dt.time
            last5 = g.loc[(tt >= CLOSE_MEDIAN_START) & (tt <= SESSION_END), "stock"]
            r = float(ref.get(d, np.nan))
            last5 = last5[(last5 > 0) & ((last5 - r).abs() / r <= GROSS_MAX_DEV)] if r > 0 else last5[last5 > 0]
            if len(last5):
                close_row = pd.DataFrame({"session_date": [d],
                                          "minute": [pd.Timestamp(dt.datetime.combine(d, SESSION_END))],
                                          "px": [float(last5.median())]})
                s = pd.concat([s, close_row], ignore_index=True)

        t = s["minute"].dt.time
        clean_parts.append(s[(t >= SESSION_OPEN) & (t <= SESSION_END)])

    clean = pd.concat(clean_parts, ignore_index=True) if clean_parts else pd.DataFrame(
        columns=["session_date", "minute", "px"])
    dropped = pd.concat(dropped_parts, ignore_index=True) if dropped_parts else pd.DataFrame(
        columns=["session_date", "minute", "px", "col", "reason"])
    return clean, dropped


def daily_ohlc(clean: pd.DataFrame) -> pd.DataFrame:
    """open = first print, close = last print, high/low = max/min of prints."""
    if clean.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "n_minutes", "first", "last"])
    c = clean.sort_values("minute")
    g = c.groupby("session_date")
    out = pd.DataFrame({
        "open": g["px"].first(), "high": g["px"].max(), "low": g["px"].min(), "close": g["px"].last(),
        "n_minutes": g["px"].size(), "first": g["minute"].min().dt.time, "last": g["minute"].max().dt.time,
    })
    out.index = [d for d in out.index]   # date objects (compute_outcome contract)
    return out


def load_spx_daily_bars(conn, date_from: dt.date, date_to: dt.date):
    clean, dropped = build_series(fetch_minutes(conn, date_from, date_to))
    return daily_ohlc(clean), dropped, clean
