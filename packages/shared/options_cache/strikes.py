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

CR-AR (decisions 1–2): verticals are snapped as a *pair* with a hard width
cap — `snap_spread_to_listed` / `snap_vertical_pair` — never wider than
2 × the nominal width, narrower only when nothing at ≥ nominal exists within
the cap. Per-leg `snap_to_listed_strike` stays for single-leg callers only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional


class StrikeNotListed(Exception):
    """No listed strike satisfies the request (expiry absent from the prior-close
    chain, or no strike on the required side of the anchor leg)."""


class StructureNotListed(StrikeNotListed):
    """CR-AR decision 1: no listed pair of strikes forms the requested vertical
    within the width cap (expiry absent from the prior-close chain, or every
    listed pair is wider than 2 × nominal). Subclass of StrikeNotListed so
    callers that only know the per-leg error still stop cleanly."""


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


# ── CR-AR: pair snapping with width cap (decisions 1–2) ──────────────────────

@dataclass(frozen=True)
class SnappedSpread:
    """A vertical snapped as a pair. `anchor` is the leg placed at the target
    (the short leg in both harness structures: debit → k_high, credit → k_low);
    `other` is the wing. width_actual ≤ 2 × width_nominal always;
    narrower_than_nominal is True only when no pair at ≥ nominal existed
    within the cap (decision 2, recorded)."""
    k_low: float
    k_high: float
    anchor: float
    other: float
    width_actual: float
    width_nominal: float
    side: str
    narrower_than_nominal: bool = False
    prior_close: Optional[date] = None

    @property
    def widened(self) -> bool:
        return self.width_actual > self.width_nominal

    @property
    def at_cap(self) -> bool:
        return abs(self.width_actual - 2.0 * self.width_nominal) < 1e-9


def _anchor_of(pair: tuple[float, float], direction: int) -> float:
    """direction = sign(other − anchor): +1 → anchor is k_low, −1 → anchor is k_high."""
    return pair[0] if direction > 0 else pair[1]


def snap_spread_to_listed(
    target: float,
    width_nominal: float,
    side: str,
    direction,
    chain: Iterable[float],
    *,
    toward: Optional[float] = None,
    max_anchor_shift: Optional[float] = None,
) -> SnappedSpread:
    """CR-AR decisions 1–2 (+ Step 0 amendment A1). Choose the listed pair
    (k_low, k_high) for a vertical.

    target        — intended anchor strike (the leg placed at the drift target).
    width_nominal — the structure's intent (10 for the harness).
    side          — 'debit' | 'credit' (recorded; fixes the anchor for call
                    verticals: debit anchors k_high, credit anchors k_low).
    direction     — magnet direction / geometry of the wing relative to the
                    anchor: +1 → wing above the anchor (credit call spread),
                    −1 → wing below (debit call spread). None → derived from
                    `side` for call verticals (credit +1, debit −1).
    chain         — the strikes listed for the expiry at the prior close.
    toward        — spot (or any price) for tie-break (b): prefer the anchor
                    nearer it; None → the lower anchor.
    max_anchor_shift — amendment A1: the anchor may move at most this far from
                    `target` (default 2 × width_nominal, the same tolerance as
                    the width cap); a pair whose anchor is further away is not
                    the same trade and is never chosen.

    Candidates: every pair with k_high − k_low in [width_nominal, 2 × width_nominal]
    (decision 2's hard cap) whose anchor lies within max_anchor_shift of the
    target. Choose the pair minimising |anchor − target|; ties → (a) width
    closest to nominal, (b) anchor nearer `toward`, then the lower anchor. If
    no pair exists at ≥ nominal within the cap, fall back to the same rule over
    pairs narrower than nominal and flag it. Nothing at all → StructureNotListed.
    """
    if side not in ("debit", "credit"):
        raise ValueError(f"side must be 'debit' or 'credit', got {side!r}")
    w = float(width_nominal)
    if w <= 0:
        raise ValueError(f"width_nominal must be positive, got {width_nominal!r}")
    if direction is None:
        direction = +1 if side == "credit" else -1
    direction = 1 if float(direction) > 0 else -1
    shift_cap = 2.0 * w if max_anchor_shift is None else float(max_anchor_shift)

    cands = sorted(set(float(c) for c in chain))
    if not cands:
        raise StructureNotListed(f"no listed strikes for the expiry (target {target}, side {side})")

    pairs: list[tuple[float, float]] = []
    for i, lo in enumerate(cands):
        for hi in cands[i + 1:]:
            if hi - lo > 2.0 * w + 1e-9:
                break                      # cands sorted: wider from here on
            if abs(_anchor_of((lo, hi), direction) - float(target)) > shift_cap + 1e-9:
                continue                   # amendment A1: anchor too far from the target
            pairs.append((lo, hi))

    def rank(p: tuple[float, float]):
        a = _anchor_of(p, direction)
        width = p[1] - p[0]
        tb = abs(a - toward) if toward is not None else 0.0
        return (abs(a - float(target)), abs(width - w), tb, a)

    within = [p for p in pairs if p[1] - p[0] >= w - 1e-9]
    narrower = False
    if within:
        best = min(within, key=rank)
    else:
        below = [p for p in pairs if p[1] - p[0] < w - 1e-9]
        if not below:
            raise StructureNotListed(
                f"no listed pair within [{w:g}, {2 * w:g}] points with its anchor within {shift_cap:g} of "
                f"target {target} (side {side}; {len(cands)} strikes listed)"
            )
        best = min(below, key=rank)
        narrower = True

    anchor = _anchor_of(best, direction)
    other = best[1] if anchor == best[0] else best[0]
    return SnappedSpread(
        k_low=best[0], k_high=best[1], anchor=anchor, other=other,
        width_actual=best[1] - best[0], width_nominal=w, side=side,
        narrower_than_nominal=narrower,
    )


def snap_vertical_pair(
    target: float,
    width_nominal: float,
    side: str,
    expiry: date,
    trade_date: date,
    conn,
    *,
    ticker: str = "SPX",
    toward: Optional[float] = None,
    direction=None,
    max_anchor_shift: Optional[float] = None,
) -> SnappedSpread:
    """DB-backed `snap_spread_to_listed` against the prior-close chain for
    `expiry` (decision 1). Raises StructureNotListed when the expiry is absent
    from the chain or no pair exists within the cap."""
    pc, cands = listed_strikes(conn, expiry, trade_date, ticker)
    if not cands:
        raise StructureNotListed(
            f"expiry {expiry} not in the {ticker} chain at prior close {pc} (trade_date {trade_date})"
        )
    snapped = snap_spread_to_listed(target, width_nominal, side, direction, cands, toward=toward,
                                    max_anchor_shift=max_anchor_shift)
    return SnappedSpread(**{**snapped.__dict__, "prior_close": pc})
