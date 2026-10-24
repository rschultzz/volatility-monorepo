"""Corrected GEX level placement (CR-BT).

An option's gamma peaks where *its own expiry's forward* equals the strike. To
draw that peak on another price axis, scale the strike by the ratio of the
target-frame price to the option's forward, both at the same snapshot:

    L = K × (P_target ÷ F_e)

F_e = the row's own `orats_oi_gamma.stock_price` — ORATS' per-expiry
parity-implied price at the prior session's 15:46 ET snapshot. No rates, no
dividend yields, no day counts (CR-BT L2).

Callers (CR-BT L3):
  - ES chart:  P_target = close of the ES 1-min bar starting 15:45 ET on the
               API trade date (`es_snapshot_bar_start_utc`).
  - SPX cash:  P_target = S0 = `stock_price` of the nearest live expiry (CR-BU).

Validity (CR-BT Step 0 / A1.1): `stock_price` behaves as a forward only for
session dates >= CORRECTED_LEVELS_FIRST_SESSION; before that it falls with DTE
(≈ dividend-discounted spot) and the formula is not defined.
"""
from __future__ import annotations

import datetime as dt
import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# First session date on which orats_oi_gamma.stock_price is forward-like on every
# session (CR-BT Step 0 Q1: last non-forward-like session 2024-04-02).
CORRECTED_LEVELS_FIRST_SESSION = dt.date(2024, 4, 3)

_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


def place_levels(
    rows: pd.DataFrame,
    p_target: float,
    *,
    strike_col: str = "strike",
    forward_col: str = "stock_price",
    out_col: str = "level_corrected",
) -> pd.DataFrame:
    """Return a copy of `rows` with `out_col` = strike × p_target / forward.

    Rows whose strike or forward is missing, non-finite or non-positive get NaN,
    as do all rows when p_target is missing, non-finite or non-positive.
    """
    out = rows.copy()
    if out.empty:
        out[out_col] = pd.Series(dtype="float64")
        return out

    strike = pd.to_numeric(out[strike_col], errors="coerce").astype("float64").to_numpy()
    fwd = pd.to_numeric(out[forward_col], errors="coerce").astype("float64").to_numpy()

    try:
        p = float(p_target)
    except (TypeError, ValueError):
        p = math.nan
    if not (math.isfinite(p) and p > 0):
        out[out_col] = np.full(len(out), np.nan)
        return out

    ok = np.isfinite(strike) & np.isfinite(fwd) & (strike > 0) & (fwd > 0)
    level = np.full(len(out), np.nan)
    level[ok] = strike[ok] * (p / fwd[ok])
    out[out_col] = level
    return out


def bucket_levels(levels, bucket: float) -> np.ndarray:
    """ROUND(level / bucket) * bucket with half-away-from-zero rounding.

    Matches Postgres ROUND(double precision) as used by the existing level
    queries (numpy's np.round is half-to-even and would not). NaN stays NaN.
    """
    b = max(float(bucket), 1.0)
    x = np.asarray(levels, dtype="float64") / b
    return np.sign(x) * np.floor(np.abs(x) + 0.5) * b


def es_snapshot_bar_start_utc(api_trade_date: dt.date) -> dt.datetime:
    """Naive-UTC start of the ES 1-min bar whose close is the 15:46 ET snapshot.

    `ironbeam_es_1m_bars.datetime` is the bar start in naive UTC; the bar starting
    15:45 America/New_York closes at the ORATS 15:46 ET snapshot (CR-BT Step 0:
    median |F_matched − ES| 0.45 pts at this bar vs 1.8 one minute either side).
    DST via zoneinfo, never a fixed offset.
    """
    start_et = dt.datetime.combine(api_trade_date, dt.time(15, 45), tzinfo=_ET)
    return start_et.astimezone(_UTC).replace(tzinfo=None)


def corrected_levels_defined(session_date: dt.date) -> bool:
    """True when L2's F_e is a forward for this session (CR-BT A1.1)."""
    return session_date >= CORRECTED_LEVELS_FIRST_SESSION
