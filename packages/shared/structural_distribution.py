"""Structural probability distributions for CR-G edge visualization.

Computes terminal and path-dependent probabilities from the K=20 historical
analogue set. All probabilities are TERMINAL (held-to-checkpoint assumption):
  - compute_terminal_prob_in_range: P(projected_session_close_tN lands in [lower, upper])
  - compute_session_excursion_metrics: DIAGNOSTIC path metrics (NOT edge math)

Normalization / projection (CR-G Step 2.5b fix):
  Analogues are from different price epochs (e.g. 2024-2025 SPX at 5000+ vs
  a 2023 anchor at 4200). Comparing absolute close prices to today's price
  bounds is semantically broken — the ranges are in different coordinate systems.

  Fix: project each analogue's normalized return onto today's price scale:
    normalized_return = (close_tN - session_open_t0) / implied_move_1d
    projected_close   = today_spot + normalized_return * today_implied_move

  where:
    close_tN         — the analogue's close at the requested timeframe
    session_open_t0  — the analogue's own RTH open on its trade_date
                       (first bar open; matches compute_outcome's first_open reference)
    implied_move_1d  — the analogue's own implied move on its trade_date
    today_spot       — current underlying price (caller's reference)
    today_implied_move — today's implied move (caller's reference)

  Analogues with NULL close, NULL session_open_t0, NULL implied_move_1d, or
  implied_move_1d <= 0 are excluded from BOTH numerator and denominator.

Input dict shape for compute_terminal_prob_in_range:
  Each analogue dict must have:
    'close'              — session close at the timeframe (caller selects t1/t5/t15)
    'anchor_spot'        — analogue's session_open_t0
    'anchor_implied_move' — analogue's implied_move_1d

Input dict shape for compute_session_excursion_metrics:
  Each analogue dict must have:
    'session_high'       — analogue's session high (in analogue's price scale)
    'session_low'        — analogue's session low  (in analogue's price scale)
    'anchor_spot'        — analogue's session_open_t0
    'anchor_implied_move' — analogue's implied_move_1d

Regime → range mapping from Step 0-A registry (locked):
  magnet-above:  [current_price, drift_target]
  magnet-below:  [drift_target, current_price]
  magnetic-pin:  [drift_target − tolerance, drift_target + tolerance]
    → tolerance not in regime_block; callers pass it explicitly as 0.25 * implied_move
  bounded:       [containment_zone.lower_price, containment_zone.upper_price]
  amplification / untethered / broken-magnet:  (None, None)  — no range, no edge zones

Spec deviation: not all outcome_status='computed' rows have a trade-thesis range.
broken-magnet rows have computed outcomes but no drift_target → treated as None-range
alongside amplification/untethered.
"""
from __future__ import annotations

from typing import Optional

from packages.shared.stats import wilson_ci


def _project_close(
    close: float,
    anchor_spot: float,
    anchor_implied_move: float,
    today_spot: float,
    today_implied_move: float,
) -> float:
    """Project an analogue's close price to today's price scale.

    normalized_return = (close - anchor_spot) / anchor_implied_move
    projected_close   = today_spot + normalized_return * today_implied_move
    """
    normalized_return = (close - anchor_spot) / anchor_implied_move
    return today_spot + normalized_return * today_implied_move


def compute_terminal_prob_in_range(
    analogues: list[dict],
    today_spot: float,
    today_implied_move: float,
    lower: Optional[float],
    upper: Optional[float],
) -> Optional[dict]:
    """Fraction of analogues whose projected close lands in [lower, upper].

    Each analogue dict must have: 'close', 'anchor_spot', 'anchor_implied_move'.
    Analogues with any of those fields None or anchor_implied_move <= 0 are
    excluded from both numerator and denominator (treated as missing data).

    The close is projected to today's scale via:
      normalized_return = (close - anchor_spot) / anchor_implied_move
      projected_close   = today_spot + normalized_return * today_implied_move

    Intervals are closed on both sides. One-sided ranges are supported:
      lower=None  → P(projected_close <= upper)
      upper=None  → P(projected_close >= lower)
      both=None   → returns None (no meaningful range)

    Returns:
        {"prob": float, "wilson_ci": (lo, hi), "n": int}
        None if both bounds are None or no valid analogues.
    """
    if lower is None and upper is None:
        return None

    valid_projected: list[float] = []
    for a in analogues:
        c   = a.get("close")
        asp = a.get("anchor_spot")
        aim = a.get("anchor_implied_move")
        if c is None or asp is None or aim is None:
            continue
        try:
            c_f, asp_f, aim_f = float(c), float(asp), float(aim)
        except (TypeError, ValueError):
            continue
        if aim_f <= 0:
            continue
        valid_projected.append(_project_close(c_f, asp_f, aim_f, today_spot, today_implied_move))

    n = len(valid_projected)
    if n == 0:
        return None

    def _in_range(v: float) -> bool:
        in_lo = (lower is None) or (v >= lower)
        in_hi = (upper is None) or (v <= upper)
        return in_lo and in_hi

    k = sum(1 for v in valid_projected if _in_range(v))
    ci_lo, ci_hi = wilson_ci(k, n)
    return {
        "prob": k / n,
        "wilson_ci": (ci_lo, ci_hi),
        "n": n,
    }


def compute_session_excursion_metrics(
    analogues: list[dict],
    today_spot: float,
    today_implied_move: float,
    lower: Optional[float],
    upper: Optional[float],
) -> Optional[dict]:
    """DIAGNOSTIC path-dependent metrics — NOT used for edge ratio math.

    Returns P(session intersects range) and P(session stayed entirely in range)
    across the analogue set. Both session_high and session_low are projected to
    today's price scale before the intersection/containment check.

    Each analogue dict must have: 'session_high', 'session_low',
    'anchor_spot', 'anchor_implied_move'.
    Analogues with any required field None or anchor_implied_move <= 0 are excluded.

    Session-level definitions (checked on projected values):
      intersected: proj_low <= upper  AND  proj_high >= lower
      stayed_in:   proj_low >= lower  AND  proj_high <= upper
    One-sided ranges treated as −∞ / +∞ on the open end.

    Returns:
        {intersected_prob, intersected_ci, stayed_in_prob, stayed_in_ci, n}
        None if both bounds are None or no valid sessions.
    """
    if lower is None and upper is None:
        return None

    n_intersected = 0
    n_stayed = 0
    n = 0

    for a in analogues:
        hi_s = a.get("session_high")
        lo_s = a.get("session_low")
        asp  = a.get("anchor_spot")
        aim  = a.get("anchor_implied_move")
        if hi_s is None or lo_s is None or asp is None or aim is None:
            continue
        try:
            hi_f, lo_f, asp_f, aim_f = float(hi_s), float(lo_s), float(asp), float(aim)
        except (TypeError, ValueError):
            continue
        if aim_f <= 0:
            continue

        proj_hi = _project_close(hi_f, asp_f, aim_f, today_spot, today_implied_move)
        proj_lo = _project_close(lo_f, asp_f, aim_f, today_spot, today_implied_move)
        n += 1

        if lower is None:
            intersected = True  # open lower end — any price to the left intersects
            stayed = proj_hi <= (upper if upper is not None else float("inf"))
        elif upper is None:
            intersected = proj_hi >= lower
            stayed = proj_lo >= lower
        else:
            intersected = proj_lo <= upper and proj_hi >= lower
            stayed = proj_lo >= lower and proj_hi <= upper

        if intersected:
            n_intersected += 1
        if stayed:
            n_stayed += 1

    if n == 0:
        return None

    ci_int_lo, ci_int_hi = wilson_ci(n_intersected, n)
    ci_stay_lo, ci_stay_hi = wilson_ci(n_stayed, n)

    return {
        "intersected_prob": n_intersected / n,
        "intersected_ci": (ci_int_lo, ci_int_hi),
        "stayed_in_prob": n_stayed / n,
        "stayed_in_ci": (ci_stay_lo, ci_stay_hi),
        "n": n,
    }


def get_trade_thesis_range(
    regime_block: dict,
    current_price: float,
    *,
    tolerance: Optional[float] = None,
) -> dict:
    """Returns trade-thesis range per the Step 0-A range registry.

    Output: {lower: Optional[float], upper: Optional[float], regime_kind: str}

    Range mappings (locked):
      magnet-above:  lower=current_price,  upper=drift_target
      magnet-below:  lower=drift_target,   upper=current_price
      magnetic-pin:  lower=drift_target-tolerance,  upper=drift_target+tolerance
      bounded:       lower=containment_zone.lower_price,  upper=containment_zone.upper_price
      amplification / untethered / broken-magnet: lower=None, upper=None

    magnetic-pin tolerance: not stored in regime_block (gex_landscape.py doesn't set it).
    Pass via the tolerance= keyword argument — typically 0.25 * implied_move at the
    time of classification. Also accepted from regime_block["tolerance"] if present
    (for forward compatibility if landscape ever starts providing it).

    Raises:
        ValueError if regime is magnet-* or pin but drift_target is missing,
        or bounded but containment_zone is missing, or magnetic-pin and no
        tolerance source is available. These are data integrity errors — the
        regime block is malformed and the caller must fix the upstream data.
    """
    regime_kind = regime_block.get("regime", "")

    _NO_RANGE_REGIMES = {"amplification", "untethered", "broken-magnet"}
    if regime_kind in _NO_RANGE_REGIMES:
        return {"lower": None, "upper": None, "regime_kind": regime_kind}

    if regime_kind in ("magnet-above", "magnet-below"):
        if "drift_target" not in regime_block:
            raise ValueError(
                f"regime_block missing 'drift_target' for regime={regime_kind!r}; "
                "check upstream landscape data"
            )
        dt_ = float(regime_block["drift_target"])
        if regime_kind == "magnet-above":
            return {"lower": current_price, "upper": dt_, "regime_kind": regime_kind}
        return {"lower": dt_, "upper": current_price, "regime_kind": regime_kind}

    if regime_kind == "magnetic-pin":
        if "drift_target" not in regime_block:
            raise ValueError(
                "regime_block missing 'drift_target' for regime='magnetic-pin'; "
                "check upstream landscape data"
            )
        dt_ = float(regime_block["drift_target"])
        tol = regime_block.get("tolerance") or tolerance
        if tol is None:
            raise ValueError(
                "magnetic-pin range requires tolerance; pass via regime_block['tolerance'] "
                "or tolerance=keyword argument (typically 0.25 * implied_move)"
            )
        tol = float(tol)
        return {
            "lower": dt_ - tol,
            "upper": dt_ + tol,
            "regime_kind": regime_kind,
        }

    if regime_kind == "bounded":
        if "containment_zone" not in regime_block:
            raise ValueError(
                "regime_block missing 'containment_zone' for regime='bounded'; "
                "check upstream landscape data"
            )
        cz = regime_block["containment_zone"]
        return {
            "lower": float(cz["lower_price"]),
            "upper": float(cz["upper_price"]),
            "regime_kind": regime_kind,
        }

    # Unknown / future regime — no range; don't raise since new regimes may be added
    return {"lower": None, "upper": None, "regime_kind": regime_kind}
