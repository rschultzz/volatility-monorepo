"""Vol surface feature computations for CR-D backfill.

Public API — pure compute functions (no DB I/O):
    compute_atm_iv_percentile(trade_date, iv_history)        -> float | None
    compute_skew_percentile(trade_date, skew_history)        -> float | None
    compute_term_structure_slope(trade_date, front_iv,
        back_iv, slope_history) -> tuple[float, float] | tuple[None, None]
    compute_smile_convexity(trade_date, convexity_history)   -> float | None
    compute_vol_risk_premium(trade_date, realized_vol_20d,
        current_atm_iv)         -> float | None
    compute_realized_vol_20d(closes)                         -> float | None

DB helpers — called by runner only; no DB I/O inside compute_* functions:
    fetch_eod_vol_snapshot(conn, trade_date, ticker)         -> dict | None
    fetch_iv_history_for_date(conn, trade_date, ticker,
        lookback_sessions)       -> pd.DataFrame
    fetch_es_closes_before(conn, trade_date, n)              -> list[float]

orats_monies_minute schema used:
    atmiv        — ATM IV (direct column)
    vol25        — 25-delta call IV (OTM call, low delta)
    vol75        — 25-delta put IV (= 75-delta call; put protection demand)
    dte          — days to expiration per row
    snapshot_pt  — timestamp; MAX per (trade_date, ticker) = EOD close
                   (snap_shot_est_time=1600 = 4 PM ET market close)
    trade_date   — TEXT column; pass .isoformat() for comparisons

EOD snapshot: MAX(snapshot_pt) per (trade_date, ticker).
Near/far DTE anchoring: ORDER BY ABS(dte - 30/90) LIMIT 1.
    DTE values in the table are not continuous; no row at exactly 30 or 90.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

_log = logging.getLogger(__name__)

_MIN_HISTORY = 60      # minimum prior sessions for percentile computation
_MAX_DTE_OFFSET = 10   # log warning when nearest DTE is this far from target


# ─── private helpers ─────────────────────────────────────────────────────────

def _pct_of(history: pd.Series, current: float) -> float:
    """Percentile of `current` within `history` using kind='mean'.

    kind='mean' = average of weak (<=) and strict (<) percentiles.
    Consequence: constant history matching `current` → 50.0.
    Result is guaranteed in [0.0, 100.0].
    """
    return float(percentileofscore(history.to_numpy(), current, kind='mean'))


def _prior_and_current(
    df: pd.DataFrame,
    trade_date: date,
    col: str,
) -> tuple[Optional[float], Optional[pd.Series]]:
    """Extract current value and the most recent _MIN_HISTORY prior values.

    `df` must have a 'trade_date' column of Python date objects and a column
    named `col`. Includes current trade_date row.

    Returns (current_value, prior_series) where prior_series holds the
    _MIN_HISTORY most recent rows strictly before trade_date. Returns
    (None, None) if the current row is missing or prior count < _MIN_HISTORY.
    """
    cur_mask = df['trade_date'] == trade_date
    cur_rows = df.loc[cur_mask, col].dropna()
    if cur_rows.empty:
        return None, None

    prior = df.loc[df['trade_date'] < trade_date, col].dropna()
    if len(prior) < _MIN_HISTORY:
        return None, None

    return float(cur_rows.iloc[0]), prior.iloc[-_MIN_HISTORY:]


# ─── compute_atm_iv_percentile ───────────────────────────────────────────────

def compute_atm_iv_percentile(
    trade_date: date,
    iv_history: pd.DataFrame,
) -> Optional[float]:
    """Percentile of trade_date's ATM IV against prior 60 sessions.

    Args:
        trade_date: current trading date.
        iv_history: DataFrame with columns [trade_date (date), atm_iv (float)],
            including the current trade_date row plus prior sessions.
            atm_iv = EOD ATM IV at nearest-to-30-DTE expiration
            (orats_monies_minute.atmiv, MAX(snapshot_pt), ORDER BY ABS(dte-30)).

    Returns:
        Percentile 0–100 (float), or None if fewer than 60 prior sessions.

    Examples:
        Constant history (all 0.20, current 0.20)    → 50.0
        Current above all history                    → 100.0
        Current below all history                    → 0.0
        Fewer than 60 prior rows                     → None
    """
    current_val, prior = _prior_and_current(iv_history, trade_date, 'atm_iv')
    if current_val is None:
        _log.debug("atm_iv_percentile: skip %s (insufficient history or missing row)", trade_date)
        return None
    return _pct_of(prior, current_val)


# ─── compute_skew_percentile ─────────────────────────────────────────────────

def compute_skew_percentile(
    trade_date: date,
    skew_history: pd.DataFrame,
) -> Optional[float]:
    """Percentile of trade_date's raw put-call skew against prior 60 sessions.

    Raw skew = vol75 − vol25 (25-delta put IV minus 25-delta call IV).
    Structurally positive for SPX-like underlyings (put demand > call demand).
    Note: skew_percentile is always 0–100; to verify skew sign, query raw_skew.

    Args:
        skew_history: DataFrame columns [trade_date (date), raw_skew (float)],
            including current trade_date row.

    Returns:
        Percentile 0–100, or None if fewer than 60 prior sessions.
    """
    current_val, prior = _prior_and_current(skew_history, trade_date, 'raw_skew')
    if current_val is None:
        _log.debug("skew_percentile: skip %s (insufficient history or missing row)", trade_date)
        return None
    return _pct_of(prior, current_val)


# ─── compute_term_structure_slope ────────────────────────────────────────────

def compute_term_structure_slope(
    trade_date: date,
    front_iv: Optional[float],
    back_iv: Optional[float],
    slope_history: pd.DataFrame,
) -> tuple[Optional[float], Optional[float]]:
    """Raw term structure slope and its percentile against prior 60 sessions.

    raw_slope = front_iv − back_iv, where:
        front_iv = ATM IV at expiration nearest to 30 DTE (EOD snapshot,
                   orats_monies_minute ORDER BY ABS(dte-30) LIMIT 1).
        back_iv  = ATM IV at expiration nearest to 90 DTE (ORDER BY ABS(dte-90)).
    Positive = near-term IV > far-term IV = backwardation (stress).
    Negative = near-term IV < far-term IV = contango (calm markets).

    Unlike the other features, current-day values are passed as scalars
    (front_iv, back_iv) so the runner can check for missing expirations before
    calling this function.

    Args:
        trade_date:    current trading date (for logging).
        front_iv:      current day's near-term ATM IV; None if no expiration found.
        back_iv:       current day's far-term ATM IV; None if no expiration found.
        slope_history: DataFrame columns [trade_date, slope] — PRIOR sessions only
                       (does NOT include current trade_date row).
                       slope = near_iv − far_iv for each prior session.

    Returns:
        (raw_slope, percentile) both floats, or (None, None) if either IV is
        None or fewer than 60 prior sessions exist.
    """
    if front_iv is None or back_iv is None:
        _log.debug(
            "term_structure_slope: missing IV (front=%s, back=%s) for %s",
            front_iv, back_iv, trade_date,
        )
        return None, None

    prior = slope_history['slope'].dropna()
    if len(prior) < _MIN_HISTORY:
        _log.debug("term_structure_slope: insufficient history for %s", trade_date)
        return None, None

    raw_slope = front_iv - back_iv
    percentile = _pct_of(prior.iloc[-_MIN_HISTORY:], raw_slope)
    return raw_slope, percentile


# ─── compute_smile_convexity ─────────────────────────────────────────────────

def compute_smile_convexity(
    trade_date: date,
    convexity_history: pd.DataFrame,
) -> Optional[float]:
    """Percentile of trade_date's smile convexity against prior 60 sessions.

    Convexity = (vol75 + vol25) / 2 − atmiv at same EOD snapshot and expiration.
    Measures wing premium above ATM. Zero = flat smile. Higher = more tail demand.

    Args:
        convexity_history: DataFrame columns [trade_date (date), convexity (float)],
            including current trade_date row.

    Returns:
        Percentile 0–100, or None if fewer than 60 prior sessions.
    """
    current_val, prior = _prior_and_current(convexity_history, trade_date, 'convexity')
    if current_val is None:
        _log.debug("smile_convexity: skip %s (insufficient history or missing row)", trade_date)
        return None
    return _pct_of(prior, current_val)


# ─── compute_vol_risk_premium ────────────────────────────────────────────────

def compute_vol_risk_premium(
    trade_date: date,
    realized_vol_20d: Optional[float],
    current_atm_iv: Optional[float],
) -> Optional[float]:
    """20-session trailing realized vol minus current ATM IV (raw vol points).

    VRP = realized_vol_20d − current_atm_iv. Not a percentile.
    Positive VRP → realized historically exceeds implied = market under-pricing risk.
    Negative VRP → IV elevated above realized = premium-selling environment.

    Realized vol: std(daily log-returns over 20 sessions) × sqrt(252).
    Source: ironbeam_es_1m_bars, last close per DATE(datetime).

    Args:
        trade_date:       current trading date (for logging).
        realized_vol_20d: annualized 20-session realized vol; None if unavailable.
        current_atm_iv:   EOD ATM IV at nearest-to-30-DTE expiration.
                          None or ≤ 0 → returns None.

    Returns:
        VRP float (may be negative) or None if any input is missing or invalid.
    """
    if realized_vol_20d is None:
        _log.debug("vol_risk_premium: realized_vol_20d is None for %s", trade_date)
        return None
    if current_atm_iv is None or current_atm_iv <= 0:
        _log.debug(
            "vol_risk_premium: current_atm_iv invalid (%s) for %s",
            current_atm_iv, trade_date,
        )
        return None
    return realized_vol_20d - current_atm_iv


# ─── compute_realized_vol_20d ────────────────────────────────────────────────

def compute_realized_vol_20d(closes: list[float]) -> Optional[float]:
    """Annualized 20-session realized vol from a sequence of daily closes.

    Uses the 21 most recent closes (yields 20 log-returns):
        std(log(closes[-20:] / closes[-21:-1]), ddof=1) × sqrt(252)

    Args:
        closes: daily closes in chronological order (oldest first);
                must have ≥ 21 values.

    Returns:
        Annualized realized vol float, or None if fewer than 21 closes.
    """
    if len(closes) < 21:
        return None
    tail = np.array(closes[-21:], dtype=float)
    log_returns = np.log(tail[1:] / tail[:-1])
    return float(log_returns.std(ddof=1) * math.sqrt(252))


# ─── DB helpers ──────────────────────────────────────────────────────────────

_EOD_SNAPSHOT_SQL = """
    WITH eod_snap AS (
        SELECT MAX(snapshot_pt) AS snap
        FROM orats_monies_minute
        WHERE trade_date = %(trade_date)s
          AND ticker = %(ticker)s
    )
    SELECT
        near30.atmiv                                              AS atm_iv,
        near30.vol25,
        near30.vol75,
        near30.dte                                               AS near_dte,
        far90.atmiv                                              AS far_atm_iv,
        far90.dte                                                AS far_dte,
        (near30.atmiv   - far90.atmiv)                          AS slope,
        (near30.vol75   - near30.vol25)                         AS raw_skew,
        ((near30.vol75  + near30.vol25) / 2.0 - near30.atmiv)  AS convexity
    FROM eod_snap
    CROSS JOIN LATERAL (
        SELECT atmiv, vol25, vol75, dte
        FROM orats_monies_minute
        WHERE trade_date = %(trade_date)s
          AND ticker    = %(ticker)s
          AND snapshot_pt = (SELECT snap FROM eod_snap)
          AND atmiv IS NOT NULL
          AND dte > 0
        ORDER BY ABS(dte - 30)
        LIMIT 1
    ) near30
    CROSS JOIN LATERAL (
        SELECT atmiv, dte
        FROM orats_monies_minute
        WHERE trade_date = %(trade_date)s
          AND ticker    = %(ticker)s
          AND snapshot_pt = (SELECT snap FROM eod_snap)
          AND atmiv IS NOT NULL
          AND dte > 0
        ORDER BY ABS(dte - 90)
        LIMIT 1
    ) far90
"""

_EOD_SNAPSHOT_COLS = (
    'atm_iv', 'vol25', 'vol75', 'near_dte', 'far_atm_iv', 'far_dte',
    'slope', 'raw_skew', 'convexity',
)


def fetch_eod_vol_snapshot(
    conn,
    trade_date: date,
    ticker: str = 'SPX',
) -> Optional[dict]:
    """EOD vol surface values for a single trade_date.

    Returns a dict with keys: atm_iv, vol25, vol75, near_dte, far_atm_iv,
    far_dte, slope, raw_skew, convexity. Returns None if no data for trade_date.

    Logs a warning if the nearest expiration to 30 or 90 DTE deviates by more
    than _MAX_DTE_OFFSET days (informational; not a skip condition).
    """
    with conn.cursor() as cur:
        cur.execute(_EOD_SNAPSHOT_SQL, {
            'trade_date': trade_date.isoformat(),
            'ticker': ticker,
        })
        row = cur.fetchone()

    if row is None:
        return None

    snap = dict(zip(_EOD_SNAPSHOT_COLS, row))

    near_dte = snap.get('near_dte')
    far_dte = snap.get('far_dte')
    if near_dte is not None and abs(int(near_dte) - 30) > _MAX_DTE_OFFSET:
        _log.warning(
            "fetch_eod_vol_snapshot: near DTE=%d is >%d days from 30 on %s",
            near_dte, _MAX_DTE_OFFSET, trade_date,
        )
    if far_dte is not None and abs(int(far_dte) - 90) > _MAX_DTE_OFFSET:
        _log.warning(
            "fetch_eod_vol_snapshot: far DTE=%d is >%d days from 90 on %s",
            far_dte, _MAX_DTE_OFFSET, trade_date,
        )

    return snap


_IV_HISTORY_SQL = """
    WITH ranked_dates AS (
        SELECT DISTINCT trade_date
        FROM orats_monies_minute
        WHERE trade_date <= %(trade_date)s
          AND ticker = %(ticker)s
        ORDER BY trade_date DESC
        LIMIT %(limit)s
    ),
    daily_eod AS (
        SELECT rd.trade_date, MAX(o.snapshot_pt) AS snap
        FROM ranked_dates rd
        JOIN orats_monies_minute o
          ON o.trade_date = rd.trade_date
         AND o.ticker = %(ticker)s
        GROUP BY rd.trade_date
    )
    SELECT
        d.trade_date::date                                       AS trade_date,
        near30.atmiv                                             AS atm_iv,
        near30.vol25,
        near30.vol75,
        (near30.vol75   - near30.vol25)                         AS raw_skew,
        ((near30.vol75  + near30.vol25) / 2.0 - near30.atmiv)  AS convexity,
        far90.atmiv                                              AS far_atm_iv,
        (near30.atmiv   - far90.atmiv)                          AS slope
    FROM daily_eod d
    CROSS JOIN LATERAL (
        SELECT atmiv, vol25, vol75, dte
        FROM orats_monies_minute
        WHERE trade_date = d.trade_date
          AND ticker    = %(ticker)s
          AND snapshot_pt = d.snap
          AND atmiv IS NOT NULL
          AND dte > 0
        ORDER BY ABS(dte - 30)
        LIMIT 1
    ) near30
    CROSS JOIN LATERAL (
        SELECT atmiv, dte
        FROM orats_monies_minute
        WHERE trade_date = d.trade_date
          AND ticker    = %(ticker)s
          AND snapshot_pt = d.snap
          AND atmiv IS NOT NULL
          AND dte > 0
        ORDER BY ABS(dte - 90)
        LIMIT 1
    ) far90
    ORDER BY d.trade_date ASC
"""

_IV_HISTORY_COLS = (
    'trade_date', 'atm_iv', 'vol25', 'vol75', 'raw_skew',
    'convexity', 'far_atm_iv', 'slope',
)


def fetch_iv_history_for_date(
    conn,
    trade_date: date,
    ticker: str = 'SPX',
    lookback_sessions: int = _MIN_HISTORY,
) -> pd.DataFrame:
    """EOD vol history up to and including trade_date.

    Returns a DataFrame with columns (all typed per below), sorted ascending:
        trade_date  — Python date
        atm_iv      — ATM IV at nearest-to-30-DTE expiration
        vol25       — 25-delta call IV
        vol75       — 25-delta put IV
        raw_skew    — vol75 - vol25
        convexity   — (vol75 + vol25) / 2 - atm_iv
        far_atm_iv  — ATM IV at nearest-to-90-DTE expiration
        slope       — atm_iv - far_atm_iv

    Includes the current trade_date row so that compute_atm_iv_percentile /
    compute_skew_percentile / compute_smile_convexity can split current vs prior
    from a single DataFrame. Returns up to lookback_sessions+1 rows total.

    trade_date in orats_monies_minute is TEXT (ISO); passed via isoformat() for
    the string-range comparison so the index is usable.
    """
    with conn.cursor() as cur:
        cur.execute(_IV_HISTORY_SQL, {
            'trade_date': trade_date.isoformat(),
            'ticker': ticker,
            'limit': lookback_sessions + 1,
        })
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame(columns=list(_IV_HISTORY_COLS))

    return pd.DataFrame(rows, columns=list(_IV_HISTORY_COLS))


_ES_CLOSES_SQL = """
    SELECT DISTINCT ON (DATE(datetime))
        DATE(datetime) AS trade_date,
        close
    FROM ironbeam_es_1m_bars
    WHERE DATE(datetime) <= %(trade_date)s
      AND DATE(datetime) >= %(start_date)s
    ORDER BY DATE(datetime) DESC, datetime DESC
    LIMIT %(n)s
"""


def fetch_es_closes_before(
    conn,
    trade_date: date,
    n: int = 21,
) -> list[float]:
    """Last n ES daily closes on or before trade_date, chronological order.

    Uses ironbeam_es_1m_bars. Daily close = last bar per DATE(datetime).
    Returns list of close prices oldest-to-newest. n=21 yields 20 log-returns
    for compute_realized_vol_20d.
    """
    from datetime import timedelta
    start_date = trade_date - timedelta(days=n * 2)

    with conn.cursor() as cur:
        cur.execute(_ES_CLOSES_SQL, {
            'trade_date': trade_date,
            'start_date': start_date,
            'n': n,
        })
        rows = cur.fetchall()

    # DISTINCT ON ... ORDER BY DATE DESC returns newest-first; reverse for chronological.
    return [float(row[1]) for row in reversed(rows)]
