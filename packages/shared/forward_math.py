"""Forward pricing math utilities shared across apps.

compute_spx_strike — convert a discounted-forward (ES-space) strike to
    the nearest SPX cash chain increment.
"""
from __future__ import annotations

from math import exp


def compute_spx_strike(strike_es: float, dte: int, r: float, q: float) -> int:
    """Convert an ES forward-space strike to the nearest SPX 5-point increment.

    ES strikes live in discounted-forward space:
        strike_es = strike_spx * exp((r - q) * t)

    Inverse:
        strike_spx = strike_es / exp((r - q) * t)

    where t = (dte + 1) / 252 — matching job_orats_eod.compute_discounted_level
    and the ingest convention.

    SPX standard chain increments are 5 points; result is rounded to nearest 5.

    Args:
        strike_es: Strike in ES forward space.
        dte: Calendar days to expiration (e.g. 15).
        r: Risk-free rate (annualized, e.g. 0.05 for 5 %).
        q: Continuous dividend yield (annualized, e.g. 0.015 for 1.5 %).

    Returns:
        SPX cash strike rounded to nearest 5-point increment (int).
    """
    t = (dte + 1) / 252.0
    carry = exp((r - q) * t)
    spx_raw = strike_es / carry
    return round(spx_raw / 5) * 5
