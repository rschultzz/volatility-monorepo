"""Listed-strike snapping (CR-AO, decisions 1–3).

SPX lists the 5-point grid on non-monthly expiries only as expiry approaches
(CR-AO G0.2: ~20 % complete at 15 business days, ~90 % at 10, full at ≤ 7).
Rounding a target to the nearest 5 therefore names a strike that may not
exist on the signal day. The ground truth for what was tradable is the
EOD chain in `orats_oi_gamma` at the prior close: snap to the nearest strike
present there for that expiry.

One implementation, shared by the backtest harness, the capture scripts,
CR-AI Stage 2 and the live proposal leg pricing. Accepts either a psycopg
connection (`conn.execute(sql, params)`) or a SQLAlchemy connection
(`conn.exec_driver_sql(sql, params)`).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional


class StrikeNotListed(Exception):
    """No listed strike satisfies the request (expiry absent from the prior-close
    chain, or no strike on the required side of the anchor leg)."""


_PRIOR_CLOSE_SQL = (
    "SELECT max(trade_date) FROM orats_oi_gamma WHERE ticker = %s AND trade_date < %s"
)
_LISTED_SQL = (
    "SELECT DISTINCT strike FROM orats_oi_gamma "
    "WHERE ticker = %s AND trade_date = %s AND expir_date = %s"
)


def _run(conn, sql: str, params: tuple):
    if hasattr(conn, "exec_driver_sql"):          # SQLAlchemy Connection
        return conn.exec_driver_sql(sql, params)
    return conn.execute(sql, params)              # psycopg


def prior_close(conn, trade_date: date, ticker: str = "SPX") -> Optional[date]:
    """Last chain date strictly before trade_date (None if the chain has nothing earlier)."""
    row = _run(conn, _PRIOR_CLOSE_SQL, (ticker, trade_date)).fetchone()
    return row[0] if row else None


def listed_strikes(conn, expiry: date, trade_date: date, ticker: str = "SPX") -> tuple[Optional[date], list[float]]:
    """(prior_close_date, sorted strikes listed for `expiry` at that close)."""
    pc = prior_close(conn, trade_date, ticker)
    if pc is None:
        return None, []
    rows = _run(conn, _LISTED_SQL, (ticker, pc, expiry)).fetchall()
    return pc, sorted({float(r[0]) for r in rows})


def snap_to_candidates(target: float, candidates: Iterable[float], *, toward: Optional[float] = None) -> float:
    """Nearest candidate to target. Ties → the candidate nearer `toward`
    (spot / the magnet direction); with no `toward`, the lower strike."""
    cands = sorted(set(float(c) for c in candidates))
    if not cands:
        raise StrikeNotListed(f"no listed strike candidates for target {target}")
    best = min(abs(c - target) for c in cands)
    tied = [c for c in cands if abs(abs(c - target) - best) < 1e-9]
    if len(tied) == 1:
        return tied[0]
    if toward is None:
        return tied[0]
    return min(tied, key=lambda c: (abs(c - toward), c))


def snap_to_listed_strike(
    target: float,
    expiry: date,
    trade_date: date,
    conn,
    *,
    ticker: str = "SPX",
    toward: Optional[float] = None,
) -> float:
    """Decision 1: nearest strike listed for `expiry` at the prior close before
    `trade_date`. Raises StrikeNotListed when the expiry is absent from that chain."""
    pc, cands = listed_strikes(conn, expiry, trade_date, ticker)
    if not cands:
        raise StrikeNotListed(
            f"expiry {expiry} not in the {ticker} chain at prior close {pc} (trade_date {trade_date})"
        )
    return snap_to_candidates(target, cands, toward=toward)


@dataclass(frozen=True)
class SnappedVertical:
    anchor: float          # the snapped anchor leg (the leg placed at the target)
    other: float           # the snapped second leg, strictly on the offset side of anchor
    width_actual: float    # abs(other - anchor)
    width_nominal: float   # abs(offset_pts) — the structure's intent
    prior_close: Optional[date]


def snap_vertical_legs(
    target: float,
    offset_pts: float,
    expiry: date,
    trade_date: date,
    conn,
    *,
    ticker: str = "SPX",
    toward: Optional[float] = None,
) -> SnappedVertical:
    """Decision 3: snap the anchor leg to the listed grid, then the second leg
    to the listed strike nearest `anchor + offset_pts` that lies strictly on
    that side of the anchor. width_actual is what was actually traded;
    width_nominal keeps the 10-point intent. Raises StrikeNotListed when the
    expiry is absent or no strike exists on the required side."""
    pc, cands = listed_strikes(conn, expiry, trade_date, ticker)
    if not cands:
        raise StrikeNotListed(
            f"expiry {expiry} not in the {ticker} chain at prior close {pc} (trade_date {trade_date})"
        )
    anchor = snap_to_candidates(target, cands, toward=toward)
    side = [c for c in cands if (c > anchor if offset_pts > 0 else c < anchor)]
    if not side:
        raise StrikeNotListed(
            f"no listed strike {'above' if offset_pts > 0 else 'below'} {anchor} for expiry {expiry} at {pc}"
        )
    other = snap_to_candidates(anchor + offset_pts, side, toward=anchor)
    return SnappedVertical(
        anchor=anchor, other=other, width_actual=abs(other - anchor),
        width_nominal=abs(float(offset_pts)), prior_close=pc,
    )
