"""CR-BS shared definitions — the 10:30 ET universe, monies/ES loaders, clocks.

Imported by scripts/cr_bs_step0.py and scripts/cr_bs_extract.py so Step 0 and the run use one definition
of every exclusion (specs/CR-BS-morning-surface.md, L1–L3). SELECTs only.

Reuses CR-BO's calendar, event dates, roll weeks and ES loader (scripts/cr_bo_common.py, scripts/cr_bo_step0.py).
"""
from __future__ import annotations

from datetime import date, datetime, time

import numpy as np
import pandas as pd

from packages.shared.trading_calendar import is_trading_day, next_trading_day
from scripts import cr_bo_common as C
from scripts.cr_bo_step0 import year_chunks

TRAIN_START = C.TRAIN_START            # 2023-05-01, first orats_monies_minute date
TRAIN_END = C.TRAIN_END                # 2026-06-05 (t ≤ TRAIN_END; holdout never read)
ERA_CUT = date(2024, 3, 1)             # E0 t < cut, E1 t ≥ cut (CR-BP A1.1)
SNAP_LO, SNAP_HI = time(7, 30), time(7, 33)   # L1(i): first snapshot ≥ 07:30 and ≤ 07:33 PT (10:30 ET)
MIN_RTH_BARS = 370                     # L1(ii)
SESSION_YEAR = 252.0
TAU_REM_SESSION = (5.5 / 6.5) / SESSION_YEAR       # L3 session clock, 10:30 → 16:00
TAU_MORN_SESSION = (1.0 / 6.5) / SESSION_YEAR      # L2(e), 09:30 → 10:30, same clock
TAU_REM_CAL = 5.5 / 24.0 / 365.0                   # L3 calendar alternative (reported as a level shift only)
TAU_MORN_CAL = 1.0 / 24.0 / 365.0

MONIES_SQL = """
SELECT trade_date, expir_date, expiry_tod, dte, dte_pct, snapshot_pt, quote_date, stock_price,
       atmiv, slope, vol5, vol10, vol15, vol20, vol25, vol30, vol35, vol40, vol45, vol50, vol55, vol60, vol65,
       vol70, vol75, vol80, vol85, vol90, vol95
FROM orats_monies_minute
WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s AND dte BETWEEN 0 AND 7
  AND snapshot_pt::time IN ('07:30', '07:31', '07:32', '07:33', '07:50')
"""


def era(t: date) -> str:
    return "E0" if t < ERA_CUT else "E1"


def train_days() -> list[date]:
    return [d for d in C.train_sessions() if d <= TRAIN_END]


def load_monies(conn) -> pd.DataFrame:
    parts = [pd.read_sql(MONIES_SQL, conn, params=(lo.isoformat(), hi.isoformat()))
             for lo, hi in year_chunks(TRAIN_START, TRAIN_END)]
    m = pd.concat(parts, ignore_index=True)
    m["td"] = pd.to_datetime(m.trade_date).dt.date
    m["ex"] = pd.to_datetime(m.expir_date).dt.date
    m["hm"] = m.snapshot_pt.dt.strftime("%H:%M")
    return m


def snapshot_rows(m: pd.DataFrame) -> dict[date, pd.DataFrame]:
    """L1(i): per trade date, the rows of the first snapshot minute in [07:30, 07:33] PT."""
    w = m[m.hm.isin(["07:30", "07:31", "07:32", "07:33"])]
    first = w.groupby("td").snapshot_pt.min()
    w = w.merge(first.rename("first"), left_on="td", right_index=True)
    w = w[w.snapshot_pt == w["first"]]
    return {d: g for d, g in w.groupby("td")}


def pick(rows: pd.DataFrame, expir: date) -> pd.Series | None:
    """The PM-settled row for expir at that snapshot, or None. AM rows never qualify (L1(i))."""
    r = rows[(rows.ex == expir) & (rows.expiry_tod == "pm")]
    return r.iloc[0] if len(r) else None


def smile_ok(r: pd.Series | None) -> bool:
    return r is not None and all(pd.notna(r[c]) and r[c] > 0 for c in ("atmiv", "vol25", "vol50", "vol75"))


def rth_frame(es: pd.DataFrame, t: date) -> pd.DataFrame:
    lo, hi = datetime.combine(t, time(9, 30)), datetime.combine(t, time(16, 0))
    return es[(es.et >= lo) & (es.et < hi)]


def rv(w: pd.DataFrame, lo: datetime, hi: datetime) -> tuple[float, float, float, int]:
    """Σr², Σr²·1{r>0}, Σr²·1{r<0}, n over bars with bar-start in [lo, hi). r = log close / prior close
    (the bar's own minute; the prior bar is the previous stored minute, so a gap's move is kept)."""
    x = w[(w.et >= lo) & (w.et < hi)].r.to_numpy()
    x = x[np.isfinite(x)]
    return float((x ** 2).sum()), float((x[x > 0] ** 2).sum()), float((x[x < 0] ** 2).sum()), len(x)


def next_session(t: date) -> date:
    return next_trading_day(t)


__all__ = ["C", "is_trading_day"]
