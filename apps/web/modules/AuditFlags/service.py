"""AuditFlags service — per-day and per-pair audit flag CRUD (CR-016).

All functions take a live connection and execute within the caller's
transaction (no COMMIT here). Service is pure functions over a stub-able
conn — no global state, no direct DB URL handling.

Exported:
    create_flag(conn, flag_type, ticker, trade_date, ...) -> dict
    delete_flag(conn, flag_id) -> bool
    promote_flag(conn, flag_id) -> dict
    demote_flag(conn, flag_id) -> dict
    list_flags_for_date(conn, ticker, trade_date) -> list[dict]
"""
from __future__ import annotations

import datetime as dt
from typing import Optional


# ─── SQL ──────────────────────────────────────────────────────────────────────

_INSERT_SQL = """
    INSERT INTO bt_audit_flags
        (flag_type, ticker, trade_date, analogue_date,
         auto_regime, corrected_regime, note)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    RETURNING flag_id, flag_type, ticker, trade_date, analogue_date,
              auto_regime, corrected_regime, promoted, note, created_at
"""

_AUTO_REGIME_SQL = """
    SELECT regime_at_classification
    FROM bt_daily_features
    WHERE ticker = %s AND trade_date = %s
    ORDER BY computed_at DESC
    LIMIT 1
"""

_DELETE_SQL = "DELETE FROM bt_audit_flags WHERE flag_id = %s RETURNING flag_id"

_PROMOTE_SQL = """
    UPDATE bt_audit_flags
    SET promoted = TRUE
    WHERE flag_id = %s AND flag_type = 'regime_wrong'
    RETURNING flag_id, flag_type, ticker, trade_date, analogue_date,
              auto_regime, corrected_regime, promoted, note, created_at
"""

_DEMOTE_SQL = """
    UPDATE bt_audit_flags
    SET promoted = FALSE
    WHERE flag_id = %s AND flag_type = 'regime_wrong'
    RETURNING flag_id, flag_type, ticker, trade_date, analogue_date,
              auto_regime, corrected_regime, promoted, note, created_at
"""

_LIST_SQL = """
    SELECT flag_id, flag_type, ticker, trade_date, analogue_date,
           auto_regime, corrected_regime, promoted, note, created_at
    FROM bt_audit_flags
    WHERE ticker = %s
      AND (
        (flag_type = 'regime_wrong' AND trade_date = %s)
        OR (flag_type = 'not_a_true_analogue'
            AND (trade_date = %s OR analogue_date = %s))
      )
    ORDER BY created_at DESC
"""

_GET_SQL = """
    SELECT flag_id, flag_type, ticker, trade_date, analogue_date,
           auto_regime, corrected_regime, promoted, note, created_at
    FROM bt_audit_flags
    WHERE flag_id = %s
"""


# ─── Row serializer ───────────────────────────────────────────────────────────

def _row_to_dict(row) -> dict:
    flag_id, flag_type, ticker, trade_date, analogue_date, \
        auto_regime, corrected_regime, promoted, note, created_at = row
    return {
        "flag_id": int(flag_id),
        "flag_type": flag_type,
        "ticker": ticker,
        "trade_date": trade_date.isoformat() if hasattr(trade_date, "isoformat") else str(trade_date),
        "analogue_date": analogue_date.isoformat() if analogue_date and hasattr(analogue_date, "isoformat") else (str(analogue_date) if analogue_date else None),
        "auto_regime": auto_regime,
        "corrected_regime": corrected_regime,
        "promoted": bool(promoted),
        "note": note,
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
    }


# ─── Public API ───────────────────────────────────────────────────────────────

def _lookup_auto_regime(conn, ticker: str, trade_date: dt.date) -> Optional[str]:
    """Read regime_at_classification from bt_daily_features, or None."""
    with conn.cursor() as cur:
        cur.execute(_AUTO_REGIME_SQL, (ticker, trade_date))
        row = cur.fetchone()
    return row[0] if row and row[0] is not None else None


def create_flag(
    conn,
    flag_type: str,
    ticker: str,
    trade_date: dt.date,
    *,
    analogue_date: Optional[dt.date] = None,
    corrected_regime: Optional[str] = None,
    note: Optional[str] = None,
) -> dict:
    """Insert a bt_audit_flags row and return the created flag dict.

    auto_regime is resolved from bt_daily_features at creation time.
    Raises ValueError on constraint violation (e.g., duplicate or
    missing required fields per CHECK constraints).
    """
    if flag_type not in ("regime_wrong", "not_a_true_analogue"):
        raise ValueError(f"Invalid flag_type: {flag_type!r}")
    if flag_type == "regime_wrong" and not corrected_regime:
        raise ValueError("corrected_regime is required for regime_wrong flags")
    if flag_type == "not_a_true_analogue" and analogue_date is None:
        raise ValueError("analogue_date is required for not_a_true_analogue flags")

    auto_regime = _lookup_auto_regime(conn, ticker, trade_date)

    with conn.cursor() as cur:
        cur.execute(_INSERT_SQL, (
            flag_type, ticker, trade_date, analogue_date,
            auto_regime, corrected_regime, note,
        ))
        row = cur.fetchone()
    return _row_to_dict(row)


def delete_flag(conn, flag_id: int) -> bool:
    """Delete a flag by ID. Returns True if deleted, False if not found."""
    with conn.cursor() as cur:
        cur.execute(_DELETE_SQL, (flag_id,))
        row = cur.fetchone()
    return row is not None


def promote_flag(conn, flag_id: int) -> dict:
    """Set promoted=True on a regime_wrong flag. Returns the updated flag."""
    with conn.cursor() as cur:
        cur.execute(_PROMOTE_SQL, (flag_id,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"Flag {flag_id} not found or not a regime_wrong flag")
    return _row_to_dict(row)


def demote_flag(conn, flag_id: int) -> dict:
    """Set promoted=False on a regime_wrong flag. Returns the updated flag."""
    with conn.cursor() as cur:
        cur.execute(_DEMOTE_SQL, (flag_id,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"Flag {flag_id} not found or not a regime_wrong flag")
    return _row_to_dict(row)


def list_flags_for_date(conn, ticker: str, trade_date: dt.date) -> list[dict]:
    """Return all flags relevant to a (ticker, trade_date):
    - regime_wrong flags where trade_date matches
    - not_a_true_analogue flags where trade_date OR analogue_date matches
    """
    with conn.cursor() as cur:
        cur.execute(_LIST_SQL, (ticker, trade_date, trade_date, trade_date))
        rows = cur.fetchall()
    return [_row_to_dict(r) for r in rows]
