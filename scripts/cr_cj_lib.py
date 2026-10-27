"""CR-CJ shared loaders — 0DTE PM skew snapshots and Databento ES minute bars, train only. READ-ONLY.

Imported by scripts/cr_cj_step0.py and scripts/cr_cj_calibrate.py. Periods (spec C1): R-a 2024-04-03 .. 2024-12-31
and 2026-01-02 .. 2026-06-05. 2025 is never queried; every read is asserted inside a period and before
2026-06-05 21:00 UTC. Early closes are excluded. Loads are cached under scripts/logs/cr_cj/ (git-ignored).

  - skew (C2): orats_monies_minute, SPX, expir_date = trade_date, expiry_tod = 'pm', 06:30 <= snapshot_pt < 13:00 PT.
  - ES (C3): es_flow_1s_databento built into 1-min bars per (session, symbol) in SQL; the session's most-volume
    outright symbol is kept. Bars are indexed by start minute (PT) from 06:30.
  - alignment (C4 as amended by A1): a snapshot pairs with the ES close at its quote time, `quote_date` rounded to
    the nearest minute Q (the close of the bar starting Q - 1). Snapshots whose quotes are > 5 min from the label
    are dropped. R-a quotes lag the label by a median 129 s; 2026 by 10 s (Step 0).
"""
from __future__ import annotations

import os
import sys
import warnings
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
for line in (REPO_ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from packages.shared.trading_calendar import is_trading_day  # noqa: E402

PT, UTC = ZoneInfo("America/Los_Angeles"), timezone.utc
PERIODS = {"ra": (date(2024, 4, 3), date(2024, 12, 31)), "y26": (date(2026, 1, 2), date(2026, 6, 5)),
           "y23": (date(2023, 1, 3), date(2024, 4, 2))}     # CR-CO O1
HOLDOUT_TS = datetime(2026, 6, 5, 21, 0, tzinfo=UTC)
EARLY_CLOSES = {date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24), date(2023, 7, 3), date(2023, 11, 24)}
NB = 390                                       # RTH minutes 06:30 .. 12:59 PT
QUOTE_MAX_AGE_S = 300                          # A1: drop snapshots whose quotes are > 5 min from the label
VOLS = [f"vol{n}" for n in range(5, 100, 5)]
OUT_DIR = REPO_ROOT / "scripts" / "logs" / "cr_cj"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_CONN = None


def conn():
    global _CONN
    if _CONN is None:
        from packages.shared.backfill_safety import get_backfill_db_conn
        _CONN = get_backfill_db_conn()
        _CONN.read_only = True
    return _CONN


def q(sql, params=None) -> pd.DataFrame:
    with conn().cursor() as cur:
        cur.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def pt(d: date, hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime.combine(d, time(hh, mm, ss), tzinfo=PT)


def guard(a: date, b: date) -> None:
    """Every read must sit inside one period, never 2025, never past the holdout cut."""
    assert any(lo <= a <= b <= hi for lo, hi in PERIODS.values()), (a, b)
    assert pt(b, 13, 0) <= HOLDOUT_TS
    assert not (a.year == 2025 or b.year == 2025)


def sessions(period: str) -> list[date]:
    lo, hi = PERIODS[period]
    out, d = [], lo
    while d <= hi:
        if is_trading_day(d) and d not in EARLY_CLOSES:
            out.append(d)
        d += timedelta(days=1)
    return out


def load_skew(period: str) -> pd.DataFrame:
    path = OUT_DIR / f"skew_{period}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    lo, hi = PERIODS[period]
    guard(lo, hi)
    r = q(f"""
        SELECT trade_date, expir_date, expiry_tod, snapshot_pt, snap_shot_date, quote_date, stock_price, atmiv,
               {", ".join(VOLS)}
        FROM orats_monies_minute
        WHERE ticker = 'SPX' AND trade_date >= %s AND trade_date <= %s
          AND expir_date = trade_date AND expiry_tod = 'pm'
          AND snapshot_pt::time >= '06:30' AND snapshot_pt::time < '13:00'
        ORDER BY trade_date, snapshot_pt
    """, (lo.isoformat(), hi.isoformat()))
    r["d"] = pd.to_datetime(r["trade_date"]).dt.date
    assert r["d"].min() >= lo and r["d"].max() <= hi
    for c in ["stock_price", "atmiv"] + VOLS:
        r[c] = r[c].astype(float)
    r.to_parquet(path)
    return r


def load_es(period: str) -> pd.DataFrame:
    """1-min bars per (session, symbol) from es_flow_1s_databento, RTH 06:30-13:00 PT."""
    path = OUT_DIR / f"es_{period}.parquet"
    if path.exists():
        r = pd.read_parquet(path)
        return r[r["symbol"] == "ES.c.0"].reset_index(drop=True) if period == "y23" else r
    lo, hi = PERIODS[period]
    guard(lo, hi)
    a, b = pt(lo, 6, 30), pt(hi, 13, 0)
    if period == "y23":
        # CR-CO O4: es_flow_1s_databento starts 2024-04-03; before it the only Databento ES series is the ES.c.0
        # minute bars in ironbeam_es_1m_bars (naive UTC bar start; no contract column -> pseudo-symbol). Non-Databento rows are kept
        # with their source so Step 0 can list them; front_bars uses only source = 'databento'.
        r = q("""
            SELECT ((datetime AT TIME ZONE 'UTC') AT TIME ZONE 'America/Los_Angeles')::date AS d,
                   CASE WHEN source = 'databento' THEN 'ES.c.0' ELSE 'src:' || coalesce(source, 'null') END AS symbol,
                   ((datetime AT TIME ZONE 'UTC') AT TIME ZONE 'America/Los_Angeles') AS m_pt,
                   open::float AS o, high::float AS h, low::float AS l, close::float AS c, volume::float AS vol
            FROM ironbeam_es_1m_bars
            WHERE datetime >= %s AND datetime < %s
              AND ((datetime AT TIME ZONE 'UTC') AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
              AND ((datetime AT TIME ZONE 'UTC') AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
            ORDER BY datetime
        """, (a.astimezone(UTC).replace(tzinfo=None), b.astimezone(UTC).replace(tzinfo=None)))   # naive UTC column
        assert r["d"].min() >= lo and r["d"].max() <= hi
        r.to_parquet(path)
        return r[r["symbol"] == "ES.c.0"].reset_index(drop=True)
    r = q("""
        WITH s AS (
            SELECT symbol, ts_utc, open, high, low, close, volume,
                   (ts_utc AT TIME ZONE 'America/Los_Angeles') AS ts_pt
            FROM es_flow_1s_databento
            WHERE ts_utc >= %s AND ts_utc < %s
              AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
              AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
        )
        SELECT ts_pt::date AS d, symbol, date_trunc('minute', ts_pt) AS m_pt,
               (array_agg(open ORDER BY ts_utc))[1]::float AS o, max(high)::float AS h, min(low)::float AS l,
               (array_agg(close ORDER BY ts_utc DESC))[1]::float AS c, sum(volume)::float AS vol
        FROM s GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
    """, (a, b))
    assert r["d"].min() >= lo and r["d"].max() <= hi
    r.to_parquet(path)
    return r


def front_bars(es: pd.DataFrame) -> tuple[dict, dict]:
    """Per session: the most-volume symbol and its (NB,) o/h/l/c arrays indexed by minute from 06:30 PT."""
    vol = es.groupby(["d", "symbol"])["vol"].sum().reset_index().sort_values(["d", "vol"], ascending=[True, False])
    sym = {d: g.iloc[0]["symbol"] for d, g in vol.groupby("d")}
    bars = {}
    for d, g in es.groupby("d"):
        g = g[g["symbol"] == sym[d]]
        idx = ((pd.to_datetime(g["m_pt"]) - pd.Timestamp(datetime.combine(d, time(6, 30)))).dt.total_seconds() // 60)
        idx = idx.astype(int).to_numpy()
        ok = (idx >= 0) & (idx < NB)
        arr = {k: np.full(NB, np.nan) for k in "ohlc"}
        for k in "ohlc":
            arr[k][idx[ok]] = g[k].to_numpy(float)[ok]
        bars[d] = arr
    return sym, bars


def skew_minutes(sk: pd.DataFrame) -> dict:
    """Per session: dict of (NB,) arrays by snapshot minute from 06:30 PT, plus the raw rows (for the module)."""
    out = {}
    for d, g in sk.groupby("d"):
        idx = ((pd.to_datetime(g["snapshot_pt"]) - pd.Timestamp(datetime.combine(d, time(6, 30))))
               .dt.total_seconds() // 60).astype(int).to_numpy()
        ok = (idx >= 0) & (idx < NB)
        g = g[ok]
        idx = idx[ok]
        _, first = np.unique(idx, return_index=True)        # one row per minute (first if duplicated)
        g, idx = g.iloc[first], idx[first]
        lab = pd.to_datetime(g["snap_shot_date"], utc=True)
        qts = pd.to_datetime(g["quote_date"], utc=True, errors="coerce")
        fresh = ((lab - qts).dt.total_seconds().abs() <= QUOTE_MAX_AGE_S).to_numpy()
        g, idx, qts = g[fresh], idx[fresh], qts[fresh]
        qidx = np.rint((qts - pd.Timestamp(pt(d, 6, 30))).dt.total_seconds().to_numpy() / 60.0).astype(int)
        m = {k: np.full(NB, np.nan) for k in ("put", "atm", "spx")}
        m["qidx"] = np.full(NB, -1)
        m["qidx"][idx] = qidx
        m["put"][idx] = (g["vol75"].to_numpy() - g["vol50"].to_numpy()) * 100.0
        m["atm"][idx] = g["vol50"].to_numpy() * 100.0
        m["spx"][idx] = g["stock_price"].to_numpy()
        m["row"] = np.full(NB, -1)
        m["row"][idx] = np.arange(len(g))
        m["rows"] = g.reset_index(drop=True)
        out[d] = m
    return out


def es_close_at_snapshot(bars: dict, d: date, mins: dict | None = None) -> np.ndarray:
    """ES price on the snapshot-minute index. With `mins` (A1): the close at the snapshot's quote minute Q, i.e. the
    close of the bar starting Q - 1. Without: the label minute T (the frozen C4; Step 0 comparison only)."""
    c = bars[d]["c"]
    out = np.full(NB, np.nan)
    if mins is None:
        out[1:] = c[:-1]
        return out
    q = mins[d]["qidx"]
    ok = (q >= 1) & (q <= NB)
    out[ok] = c[q[ok] - 1]
    return out


def tod_bucket(i: np.ndarray | int):
    """30-min bucket index by minute-from-06:30 (0 = 06:30-07:00 ... 12 = 12:30-13:00)."""
    return np.asarray(i) // 30


def bucket_label(b: int) -> str:
    m = 6 * 60 + 30 + 30 * int(b)
    return f"{m // 60:02d}:{m % 60:02d}"
