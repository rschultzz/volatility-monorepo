"""Post-touch direction qualification helpers (CR-025 / CR-I).

Implements direction-qualification rules for magnet-regime proposal selection
based on post-touch close distribution data from compute_structural_probability().

Regime asymmetry:
    magnet-above (magnet above spot):
        continuation direction → above  (price keeps climbing through magnet)
        reversion direction    → below  (price comes back down through magnet)
    magnet-below (magnet below spot):
        continuation direction → below  (price keeps falling through magnet)
        reversion direction    → above  (price comes back up through magnet)

Public API:
    dte_to_timeframe(dte)                              → 't1' | 't5' | 't15' | None
    credit_direction_qualifies(post_touch, regime, dte) → bool
    debit_direction_qualifies(post_touch, regime, dte)  → bool
"""
from __future__ import annotations

# Pattern labels that support each direction
_CREDIT_PATTERNS = frozenset({
    "touch-and-reject",       # price touched magnet then reversed
    "slow-revert",            # gradual reversion after touch
    "overshoot-then-revert",  # price overshot then came back
})
_DEBIT_PATTERNS = frozenset({
    "stepping-stone",   # price uses magnet as a stepping stone and continues
    "touch-and-pin",    # price pins at the magnet (supports iron fly / condor too)
})

# Wilson lower bound floor for direction qualification
_WILSON_FLOOR = 0.40


def dte_to_timeframe(dte: int | None) -> str | None:
    """Map a proposal DTE to the relevant post-touch timeframe key.

    Rule:  ≤3 → 't1',  4-9 → 't5',  ≥10 → 't15'.  Returns None when dte is None.

    Mirrors the dteToRow() rule in StructuralProbabilityBlock.jsx (Step 3);
    the timeframe a user's eye lands on when reviewing the bars matches the
    timeframe used for qualification of that trade's horizon.
    """
    if dte is None:
        return None
    if dte <= 3:
        return "t1"
    if dte <= 9:
        return "t5"
    return "t15"


def _reversion_key(regime: str) -> str:
    """Fraction key for the reversion direction given the regime.

    magnet-above → reversion is 'below' (price comes back down through magnet)
    magnet-below → reversion is 'above' (price comes back up through magnet)
    """
    return "below" if regime == "magnet-above" else "above"


def _continuation_key(regime: str) -> str:
    """Fraction key for the continuation direction given the regime.

    magnet-above → continuation is 'above' (price keeps going up past magnet)
    magnet-below → continuation is 'below' (price keeps going down past magnet)
    """
    return "above" if regime == "magnet-above" else "below"


def credit_direction_qualifies(
    post_touch: dict,
    regime: str,
    proposal_dte: int | None,
) -> bool:
    """Credit-fade qualifies when the pattern indicates reversion AND the
    Wilson lower bound on the reversion-direction fraction at the
    trade-DTE-relevant timeframe exceeds the qualification floor.

    Pattern gate:   touch-and-reject | slow-revert | overshoot-then-revert
    Wilson floor:   0.40 (strict greater-than)
    Regime mapping: magnet-above → reversion = below
                    magnet-below → reversion = above
    """
    if post_touch.get("pattern_label") not in _CREDIT_PATTERNS:
        return False
    timeframe = dte_to_timeframe(proposal_dte)
    if timeframe is None:
        return False
    rev_key = _reversion_key(regime)
    try:
        wilson_lo = post_touch["wilson_cis"][timeframe][rev_key][0]
    except (KeyError, TypeError, IndexError):
        return False
    return wilson_lo > _WILSON_FLOOR


def debit_direction_qualifies(
    post_touch: dict,
    regime: str,
    proposal_dte: int | None,
) -> bool:
    """Debit-to-target qualifies when the pattern indicates continuation or pin
    AND the Wilson lower bound on the continuation-direction fraction at the
    trade-DTE-relevant timeframe exceeds the qualification floor.

    Pattern gate:   stepping-stone | touch-and-pin
    Wilson floor:   0.40 (strict greater-than)
    Regime mapping: magnet-above → continuation = above
                    magnet-below → continuation = below
    """
    if post_touch.get("pattern_label") not in _DEBIT_PATTERNS:
        return False
    timeframe = dte_to_timeframe(proposal_dte)
    if timeframe is None:
        return False
    cont_key = _continuation_key(regime)
    try:
        wilson_lo = post_touch["wilson_cis"][timeframe][cont_key][0]
    except (KeyError, TypeError, IndexError):
        return False
    return wilson_lo > _WILSON_FLOOR
