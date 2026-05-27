"""Implied probability distributions via Breeden-Litzenberger for CR-G edge visualization.

Recovers the risk-neutral density from call prices (Breeden-Litzenberger, 1978):
    q(K) = exp(r*T) * d²C/dK²

Edge math uses TERMINAL probability (same held-to-expiration assumption as the
structural layer):
    P_implied(lower ≤ S_T ≤ upper) = ∫[lower..upper] q(K) dK

Dense chains (≥ 8 strikes, max spacing ≤ 25 pts): non-uniform second derivative
directly on call prices.

Sparse chains (< 8 strikes OR max spacing > 25 pts): cubic spline on log(call_price)
vs. strike → interpolate to 1-pt fine grid → uniform second derivative.
Rationale: log-prices span many orders of magnitude; interpolating in log-space keeps
the cubic spline numerically stable across deep-OTM strikes.

Out-of-range extrapolation policy:
    When compute_implied_prob_in_range is called with a range extending beyond the
    PDF strike coverage, the nearest tail density is extended as a constant to the
    range boundary and a RuntimeWarning is emitted. The resulting integral is an
    approximation and should be treated as a floor/ceiling estimate, not a reliable
    probability. Callers should clip the range to the PDF coverage where possible.

option_chain dict keys:
    strike: float
    call_price: float   (mid-price; caller responsible for cleaning)
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline

_SPARSE_MIN_STRIKES = 8
_SPARSE_MAX_SPACING = 25.0  # pts between consecutive strikes
_FINE_GRID_STEP = 1.0       # interpolation step for sparse chains
_MIN_CALL_PRICE = 1e-8      # floor to prevent log(0)


def compute_implied_pdf(
    option_chain: list[dict],
    spot: float,
    risk_free_rate: float,
    time_to_expiration: float,
) -> dict[float, float]:
    """Breeden-Litzenberger risk-neutral PDF from call prices.

    For non-sparse chains: non-uniform second derivative applied directly.
    For sparse chains: cubic spline on log(C(K)) → fine 1-pt grid → uniform d²C/dK².

    All negative densities (numerical noise at boundary strikes) are clipped to 0.
    The returned PDF is NOT normalised — trapezoidal integration over the full
    strike range will yield a value < 1 since tails beyond coverage are missing.

    Args:
        option_chain: list of {"strike": float, "call_price": float} dicts.
            At least 3 strikes required for a meaningful PDF.
        spot: current underlying price (informational only; not used in computation)
        risk_free_rate: annualised continuously compounded risk-free rate
        time_to_expiration: years to expiration (fractional OK)

    Returns:
        {strike (float): prob_density (float)} for interior strikes only
        (boundary strikes are consumed by finite differences).
        Empty dict if < 3 valid strikes.
    """
    chain = sorted(option_chain, key=lambda d: d["strike"])
    strikes = np.array([float(d["strike"]) for d in chain])
    prices  = np.clip(
        np.array([float(d["call_price"]) for d in chain]),
        _MIN_CALL_PRICE,
        None,
    )

    if len(strikes) < 3:
        return {}

    disc = math.exp(risk_free_rate * time_to_expiration)
    spacings = np.diff(strikes)
    is_sparse = (
        len(strikes) < _SPARSE_MIN_STRIKES
        or float(spacings.max()) > _SPARSE_MAX_SPACING
    )

    if is_sparse:
        return _pdf_sparse(strikes, prices, disc)
    return _pdf_dense(strikes, prices, disc)


def _pdf_dense(
    strikes: np.ndarray,
    prices: np.ndarray,
    disc: float,
) -> dict[float, float]:
    """Non-uniform second derivative on original strike grid."""
    pdf: dict[float, float] = {}
    for i in range(1, len(strikes) - 1):
        h_l = strikes[i] - strikes[i - 1]
        h_r = strikes[i + 1] - strikes[i]
        # Lagrange second derivative for non-uniform spacing
        d2 = 2.0 * (
            prices[i - 1] / (h_l * (h_l + h_r))
            - prices[i] / (h_l * h_r)
            + prices[i + 1] / (h_r * (h_l + h_r))
        )
        pdf[float(strikes[i])] = max(0.0, d2 * disc)
    return pdf


def _pdf_sparse(
    strikes: np.ndarray,
    prices: np.ndarray,
    disc: float,
) -> dict[float, float]:
    """Cubic spline on log(C(K)) → fine grid → uniform d²C/dK²."""
    log_prices = np.log(prices)
    cs = CubicSpline(strikes, log_prices)

    fine = np.arange(float(strikes[0]), float(strikes[-1]) + _FINE_GRID_STEP * 0.5, _FINE_GRID_STEP)
    fine_prices = np.exp(cs(fine))
    fine_prices = np.clip(fine_prices, _MIN_CALL_PRICE, None)

    h = _FINE_GRID_STEP
    pdf: dict[float, float] = {}
    for i in range(1, len(fine) - 1):
        d2 = (fine_prices[i - 1] - 2.0 * fine_prices[i] + fine_prices[i + 1]) / (h * h)
        pdf[float(fine[i])] = max(0.0, d2 * disc)
    return pdf


def compute_implied_prob_in_range(
    pdf: dict[float, float],
    lower: Optional[float],
    upper: Optional[float],
) -> float:
    """Trapezoidal integration of PDF over [lower, upper].

    One-sided range convention (same as structural layer):
      lower=None → integrate from min available strike to upper
      upper=None → integrate from lower to max available strike

    Out-of-range extrapolation: if lower < min(pdf) or upper > max(pdf),
    the nearest-tail density is extended as a constant and a RuntimeWarning
    is emitted. See module docstring for extrapolation policy.

    Args:
        pdf: dict from compute_implied_pdf; must have ≥ 2 keys.
        lower, upper: range bounds (same one-sided convention as structural layer).

    Returns:
        float in [0, 1]; 0.0 if pdf is empty or lower == upper.
    """
    if not pdf:
        return 0.0

    ks = sorted(pdf.keys())
    pdf_min_k = ks[0]
    pdf_max_k = ks[-1]

    eff_lower = pdf_min_k if lower is None else lower
    eff_upper = pdf_max_k if upper is None else upper

    if eff_lower >= eff_upper:
        return 0.0

    # Warn on out-of-range extrapolation
    if lower is not None and lower < pdf_min_k:
        warnings.warn(
            f"compute_implied_prob_in_range: lower={lower:.1f} is below "
            f"min PDF strike={pdf_min_k:.1f}; extending with nearest-tail density. "
            "Result is approximate — clip range to PDF coverage where possible.",
            RuntimeWarning,
            stacklevel=2,
        )
    if upper is not None and upper > pdf_max_k:
        warnings.warn(
            f"compute_implied_prob_in_range: upper={upper:.1f} is above "
            f"max PDF strike={pdf_max_k:.1f}; extending with nearest-tail density. "
            "Result is approximate — clip range to PDF coverage where possible.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Build a fine integration grid covering [eff_lower, eff_upper]
    # Include any PDF interior points in this range; extrapolate tails with constant density.
    grid_ks = [k for k in ks if eff_lower <= k <= eff_upper]

    # Add boundary points if they fall inside or at the edge
    pts: list[float] = []
    if eff_lower not in grid_ks:
        pts.append(eff_lower)
    pts.extend(grid_ks)
    if eff_upper not in grid_ks:
        pts.append(eff_upper)
    pts = sorted(set(pts))

    if len(pts) < 2:
        return 0.0

    # Density at each grid point (interpolate/extrapolate)
    def _density(k: float) -> float:
        if k in pdf:
            return pdf[k]
        if k <= pdf_min_k:
            return pdf[pdf_min_k]  # constant tail
        if k >= pdf_max_k:
            return pdf[pdf_max_k]  # constant tail
        # Linear interpolation between nearest PDF keys
        lo_k = max(kk for kk in ks if kk <= k)
        hi_k = min(kk for kk in ks if kk >= k)
        if lo_k == hi_k:
            return pdf[lo_k]
        frac = (k - lo_k) / (hi_k - lo_k)
        return pdf[lo_k] + frac * (pdf[hi_k] - pdf[lo_k])

    densities = np.array([_density(k) for k in pts])
    xs = np.array(pts)
    return float(np.trapezoid(densities, xs))
