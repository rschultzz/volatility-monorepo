"""audit_overrides — effective-regime resolution with promotion support (CR-016).

A "promoted" bt_audit_flags row of type regime_wrong overrides the auto-
classified regime for display and template selection. The stored feature
vector is unchanged — overrides are an escape valve, not a recalibration.

Exported:
    get_effective_regime(conn, ticker, trade_date) -> str
    get_effective_regimes(conn, ticker, trade_dates) -> dict[date, str]

Both functions are read-only and do not modify any rows.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION

# ─── SQL ──────────────────────────────────────────────────────────────────────

_PROMOTED_FLAG_SQL = """
    SELECT corrected_regime
    FROM bt_audit_flags
    WHERE ticker = %s
      AND trade_date = %s
      AND flag_type = 'regime_wrong'
      AND promoted = TRUE
    LIMIT 1
"""

_AUTO_REGIME_SQL = """
    SELECT regime_at_classification
    FROM bt_daily_features_active
    WHERE ticker = %s AND trade_date = %s
      AND feature_version = %s
"""

_PROMOTED_BATCH_SQL = """
    SELECT trade_date, corrected_regime
    FROM bt_audit_flags
    WHERE ticker = %s
      AND trade_date = ANY(%s)
      AND flag_type = 'regime_wrong'
      AND promoted = TRUE
"""

_AUTO_REGIME_BATCH_SQL = """
    SELECT trade_date, regime_at_classification
    FROM bt_daily_features_active
    WHERE ticker = %s
      AND trade_date = ANY(%s)
      AND feature_version = %s
      AND regime_at_classification IS NOT NULL
"""

_LANDSCAPE_REGIME_SQL = """
    SELECT table_spot
    FROM orats_gex_landscape
    WHERE ticker = %s AND trade_date = %s
    LIMIT 1
"""


def _rematerialize_regime(conn, ticker: str, trade_date: dt.date) -> Optional[str]:
    """Last-resort fallback: re-classify from landscape when regime_at_classification
    is NULL (rows written before the CR-016 migration). Expensive — only used when
    the stored column is missing."""
    from packages.shared.gex_landscape import classify_regime
    import pandas as pd

    with conn.cursor() as cur:
        cur.execute(
            "SELECT landscape, table_spot FROM orats_gex_landscape "
            "WHERE ticker = %s AND trade_date = %s LIMIT 1",
            (ticker, trade_date),
        )
        row = cur.fetchone()
    if not row or row[1] is None:
        return None
    landscape_rows, table_spot = row
    spot = float(table_spot)
    try:
        df = pd.DataFrame(landscape_rows)
        regime = classify_regime(df, spot, prior_spot=spot, implied_move=0.0)
        return regime.get("regime")
    except Exception:
        return None


def get_effective_regime(conn, ticker: str, trade_date: dt.date) -> Optional[str]:
    """Return the effective regime for (ticker, trade_date).

    Resolution order:
    1. Promoted regime_wrong override from bt_audit_flags.
    2. regime_at_classification from bt_daily_features.
    3. Re-materialized from orats_gex_landscape (NULL column fallback).
    4. None if no landscape data available.
    """
    # 1. Promoted override
    with conn.cursor() as cur:
        cur.execute(_PROMOTED_FLAG_SQL, (ticker, trade_date))
        row = cur.fetchone()
    if row and row[0]:
        return row[0]

    # 2. Stored auto-regime
    with conn.cursor() as cur:
        cur.execute(_AUTO_REGIME_SQL, (ticker, trade_date, CANONICAL_FEATURE_VERSION))
        row = cur.fetchone()
    if row and row[0]:
        return row[0]

    # 3. Re-materialize (pre-migration rows only)
    return _rematerialize_regime(conn, ticker, trade_date)


def get_effective_regimes(
    conn,
    ticker: str,
    trade_dates: list[dt.date],
) -> dict[dt.date, Optional[str]]:
    """Batched version of get_effective_regime.

    Returns a dict mapping each date in trade_dates to its effective regime.
    Dates with no data map to None.
    """
    if not trade_dates:
        return {}

    result: dict[dt.date, Optional[str]] = {d: None for d in trade_dates}

    # 1. Batch-fetch promoted overrides.
    with conn.cursor() as cur:
        cur.execute(_PROMOTED_BATCH_SQL, (ticker, list(trade_dates)))
        for row in cur.fetchall():
            td, corrected = row[0], row[1]
            if corrected:
                result[td] = corrected

    # 2. Fill remaining from stored regime_at_classification.
    missing = [d for d, v in result.items() if v is None]
    if missing:
        with conn.cursor() as cur:
            cur.execute(_AUTO_REGIME_BATCH_SQL, (ticker, missing, CANONICAL_FEATURE_VERSION))
            for row in cur.fetchall():
                td, regime = row[0], row[1]
                if result.get(td) is None and regime:
                    result[td] = regime

    # 3. Per-date fallback for still-missing dates (pre-migration rows).
    for td in [d for d, v in result.items() if v is None]:
        result[td] = _rematerialize_regime(conn, ticker, td)

    return result
