"""Structural probability distributions for CR-G edge visualization.

Computes terminal and path-dependent probabilities from the K=20 historical
analogue set. All probabilities are TERMINAL (held-to-checkpoint assumption):
  - compute_terminal_prob_in_range: P(session_close_tN lands in [lower, upper])
  - compute_session_excursion_metrics: DIAGNOSTIC path metrics (NOT edge math)

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


def compute_terminal_prob_in_range(
    close_values: list[Optional[float]],
    lower: Optional[float],
    upper: Optional[float],
) -> Optional[dict]:
    """Fraction of close_values landing in [lower, upper].

    Intervals are closed on both sides. One-sided ranges are supported:
      lower=None  → P(close <= upper)   (all values -inf..upper)
      upper=None  → P(close >= lower)   (all values lower..+inf)
      both=None   → returns None (no meaningful range)

    NULL (None) close values are excluded from both numerator and denominator —
    they represent missing corpus data, not out-of-range observations.

    Returns:
        {"prob": float, "wilson_ci": (lo, hi), "n": int}
        None if both bounds are None or no valid close values.
    """
    if lower is None and upper is None:
        return None

    valid = [v for v in close_values if v is not None]
    n = len(valid)
    if n == 0:
        return None

    def _in_range(v: float) -> bool:
        in_lo = (lower is None) or (v >= lower)
        in_hi = (upper is None) or (v <= upper)
        return in_lo and in_hi

    k = sum(1 for v in valid if _in_range(v))
    ci_lo, ci_hi = wilson_ci(k, n)
    return {
        "prob": k / n,
        "wilson_ci": (ci_lo, ci_hi),
        "n": n,
    }


def compute_session_excursion_metrics(
    ohlc_values: list[dict],
    lower: Optional[float],
    upper: Optional[float],
) -> Optional[dict]:
    """DIAGNOSTIC path-dependent metrics — NOT used for edge ratio math.

    Returns P(session intersects range) and P(session stayed entirely in range)
    across the analogue set. These expose path-dependency not captured by the
    terminal probability used in edge math.

    Session-level definitions:
      intersected: session_low <= upper  AND  session_high >= lower
        (the session's price range overlapped the trade-thesis zone at any point)
      stayed_in:   session_low >= lower  AND  session_high <= upper
        (the session remained entirely within the zone)
    One-sided ranges treated as −∞ / +∞ on the open end.

    OHLC dicts with any None value are excluded from the denominator —
    they represent corpus-end rows without complete bar data.

    Args:
        ohlc_values: list of dicts with keys session_high, session_low
                     (session_open/close also accepted but not used here)
        lower, upper: range bounds (same convention as compute_terminal_prob_in_range)

    Returns:
        {intersected_prob, intersected_ci, stayed_in_prob, stayed_in_ci, n}
        None if both bounds are None or no valid sessions.
    """
    if lower is None and upper is None:
        return None

    valid = [
        d for d in ohlc_values
        if d.get("session_high") is not None and d.get("session_low") is not None
    ]
    n = len(valid)
    if n == 0:
        return None

    n_intersected = 0
    n_stayed = 0
    for d in valid:
        lo_s = d["session_low"]
        hi_s = d["session_high"]

        if lower is None:
            intersected = hi_s >= float("-inf") and lo_s <= (upper or float("inf"))
        elif upper is None:
            intersected = hi_s >= lower
        else:
            intersected = lo_s <= upper and hi_s >= lower

        if lower is None:
            stayed = hi_s <= (upper or float("inf"))
        elif upper is None:
            stayed = lo_s >= lower
        else:
            stayed = lo_s >= lower and hi_s <= upper

        if intersected:
            n_intersected += 1
        if stayed:
            n_stayed += 1

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
