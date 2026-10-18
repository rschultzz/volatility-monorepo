"""CR-BQ shared definitions — CR-BK gamma inputs, ES RTH minute grid, L1 universe.

Imported by scripts/cr_bq_step0.py and the run scripts so Step 0 and the run use one definition of
every exclusion and of the price grid (specs/CR-BQ-gamma-autocorrelation.md, L1–L3).

READ-ONLY. No ORATS monies field at the open is read here.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cr_bo_common as BO  # noqa: E402  (EARLY_CLOSES, roll_week — the CR-BO Step 0 rule)
from packages.shared.backfill_safety import get_backfill_db_conn  # noqa: E402
from packages.shared.trading_calendar import is_trading_day, prev_trading_day  # noqa: E402

SPLIT = dt.date(2026, 6, 5)
ES_FROM = dt.date(2023, 4, 28)          # one session before the first CR-BK date (prev-session statistics)
BK_DIR = REPO_ROOT / "scripts" / "logs" / "cr_bk"
OUT_DIR = REPO_ROOT / "scripts" / "output" / "cr_bq"
N_MIN = 390                             # 06:30 … 12:59 PT bar starts
MIN_BARS = 370                          # L1
SNAP_BAR = dt.time(12, 45)              # bar 12:45–12:46 PT: its close is ES at the CR-BK 12:46 PT snapshot


def connect():
    conn = get_backfill_db_conn()
    conn.execute("SET default_transaction_read_only = on")
    conn.execute("SET statement_timeout = '300s'")
    return conn


def load_bk() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(shape_summary, linear-wing profiles) from scripts/logs/cr_bk/ (CR-BK `cr_bk_profile.py shape`)."""
    summ = pd.read_parquet(BK_DIR / "shape_summary.parquet")
    prof = pd.read_parquet(BK_DIR / "profiles.parquet")
    prof = prof[prof.wing == "linear"]
    assert summ.d.max() <= SPLIT and prof.d.max() <= SPLIT
    return summ, prof


def load_es(conn, date_from: dt.date = ES_FROM, date_to: dt.date = SPLIT) -> pd.DataFrame:
    """ES RTH 1-min bars (es_minutes, is_rth: bar-start 06:30–12:59 PT). Columns trade_date, minute (0…389), open, close."""
    rows = conn.execute(
        "SELECT trade_date, ts_pt, open, close FROM es_minutes WHERE is_rth AND trade_date BETWEEN %s AND %s "
        "ORDER BY trade_date, ts_pt", (date_from, date_to)).fetchall()
    df = pd.DataFrame(rows, columns=["trade_date", "ts_pt", "open", "close"])
    t = df.ts_pt.dt.hour * 60 + df.ts_pt.dt.minute
    df["minute"] = (t - (6 * 60 + 30)).astype(int)
    assert df.minute.between(0, N_MIN - 1).all()
    assert not df.duplicated(["trade_date", "minute"]).any()
    return df


def price_grid(day: pd.DataFrame) -> np.ndarray:
    """391 prices: P[0] = open of the first bar present (06:30 if present), P[k] = close of bar k−1 (k = 1…390).
    Missing bars carry the last close forward (their minute return is 0; the move lands in the next present bar)."""
    close = np.full(N_MIN, np.nan)
    close[day.minute.to_numpy()] = day.close.to_numpy()
    first = day.sort_values("minute").iloc[0]
    p = np.concatenate([[float(first.open)], close])
    return pd.Series(p).ffill().to_numpy()


def bar_counts(es: pd.DataFrame) -> pd.Series:
    return es.groupby("trade_date").size()


def universe(summ: pd.DataFrame, counts: pd.Series) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """L1, applied in the stated order. Returns (kept rows of shape_summary, [(step label, n after step)])."""
    u = summ.copy()
    steps = [("CR-BK parquet rows (effective train dates ≤ 2026-06-05)", len(u))]
    u = u[[is_trading_day(d) for d in u.d]]
    steps.append(("− not an NYSE session (holiday / weekend)", len(u)))
    u = u[u.usable.astype(bool)]
    steps.append(("− dates CR-BK excluded (usable = False: no 12:46 PT prior-session smile)", len(u)))
    u = u[[d not in BO.EARLY_CLOSES for d in u.d]]
    steps.append(("− NYSE early closes", len(u)))
    u = u[[not BO.roll_week(d) for d in u.d]]
    steps.append(("− ES roll weeks, every year (CR-BO/CR-BM rule: Mon of week before 3rd Fri → expiry Fri, Mar/Jun/Sep/Dec)", len(u)))
    u = u[[counts.get(d, 0) >= MIN_BARS for d in u.d]]
    steps.append((f"− ES RTH bars < {MIN_BARS} of {N_MIN}", len(u)))
    return u.sort_values("d").reset_index(drop=True), steps


def prev_session(d: dt.date) -> dt.date:
    return prev_trading_day(d)
