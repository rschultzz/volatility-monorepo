"""DayBrowser service — query days by effective regime (CR-016).

Returns all (ticker, date) pairs from bt_daily_features whose effective
regime (auto-classified, unless a promoted override exists) matches the
requested regime label and falls within the date range.

Each returned row includes:
  trade_date, effective_regime, feature_vector,
  landscape_summary (regime, top cluster, dominant bucket from orats_gex_landscape),
  outcomes (from ironbeam_es_1m_bars via _fetch_session_outcomes convention)

Public:
    query_days_by_regime(conn, ticker, regime, date_from, date_to) -> list[dict]
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from packages.shared.audit_overrides import get_effective_regimes


_DAYS_SQL = """
    SELECT
        bdf.trade_date,
        bdf.feature_vector,
        bdf.regime_at_classification,
        ogl.landscape,
        ogl.table_spot
    FROM bt_daily_features_active bdf
    LEFT JOIN orats_gex_landscape ogl
        ON ogl.ticker = bdf.ticker AND ogl.trade_date = bdf.trade_date
    WHERE bdf.ticker = %s
      AND bdf.trade_date BETWEEN %s AND %s
    ORDER BY bdf.trade_date DESC
"""

_OUTCOMES_SQL = """
    SELECT
        EXTRACT(EPOCH FROM MIN(datetime AT TIME ZONE 'UTC'))::bigint AS session_open_epoch,
        (array_agg(open ORDER BY datetime ASC))[1]   AS open_px,
        (array_agg(close ORDER BY datetime DESC))[1]  AS close_px,
        MAX(high)  AS day_high,
        MIN(low)   AS day_low
    FROM ironbeam_es_1m_bars
    WHERE (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date = %s
"""


def _fetch_outcomes(conn, trade_date: dt.date) -> dict:
    with conn.cursor() as cur:
        cur.execute(_OUTCOMES_SQL, (trade_date,))
        row = cur.fetchone()
    if not row or row[1] is None:
        return {}
    session_open_epoch, open_px, close_px, day_high, day_low = row
    out = {
        "open_px": float(open_px),
    }
    if close_px is not None:
        out["eod_return_pts"] = float(close_px) - float(open_px)
    if day_high is not None:
        out["day_high"] = float(day_high)
        out["mfe_above_open_pts"] = float(day_high) - float(open_px)
    if day_low is not None:
        out["day_low"] = float(day_low)
        out["mfe_below_open_pts"] = float(day_low) - float(open_px)
    if day_high is not None and day_low is not None:
        out["intraday_range_pts"] = float(day_high) - float(day_low)
    return out


def query_days_by_regime(
    conn,
    ticker: str,
    regime: str,
    date_from: dt.date,
    date_to: dt.date,
) -> list[dict]:
    """Return all days in [date_from, date_to] where effective regime == regime.

    Effective regime applies promoted overrides from bt_audit_flags.
    """
    with conn.cursor() as cur:
        cur.execute(_DAYS_SQL, (ticker, date_from, date_to))
        rows = cur.fetchall()

    if not rows:
        return []

    trade_dates = [r[0] for r in rows]
    effective_regimes = get_effective_regimes(conn, ticker, trade_dates)

    result = []
    for row in rows:
        trade_date, feature_vector, auto_regime, landscape, table_spot = row
        eff_regime = effective_regimes.get(trade_date) or auto_regime or "untethered"
        if eff_regime != regime:
            continue

        outcomes = _fetch_outcomes(conn, trade_date)

        result.append({
            "trade_date": trade_date.isoformat() if hasattr(trade_date, "isoformat") else str(trade_date),
            "regime": eff_regime,
            "auto_regime": auto_regime,
            "feature_vector": feature_vector,
            "outcomes": outcomes,
        })

    return result
