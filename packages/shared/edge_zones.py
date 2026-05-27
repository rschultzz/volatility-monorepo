"""Edge zone classification for CR-G ProposalEdgeChart.

Composes structural_distribution (terminal KNN prob) and implied_distribution
(Breeden-Litzenberger market-implied prob) into a list of classified price-axis
bands (zones) suitable for SVG chart rendering.

Conceptual model:
  1. Discretise the price axis into equal-width bins.
  2. Per bin: edge_ratio = structural_prob / implied_prob (terminal, held-to-expiration).
  3. Classify each bin's ratio into a band: strong-positive … strong-negative.
  4. Merge contiguous same-classification bins into zones for the chart layer.

Classification thresholds (calibration knobs — adjust here for the chart layer):

  ratio > 2.0           → 'strong-positive'    (green band)
  1.3 < ratio ≤ 2.0    → 'moderate-positive'  (light green)
  0.7 ≤ ratio ≤ 1.3    → 'neutral'            (no shading; structural ≈ market)
  0.5 ≤ ratio < 0.7    → 'moderate-negative'  (light red)
  ratio < 0.5          → 'strong-negative'    (red band)
  None / NaN / negative → 'unknown'            (no shading)

Neutral range [0.7, 1.3] is closed on both sides (including 1.3).
Moderate-positive starts strictly above 1.3 (ratio > 1.3).

analogues_ohlc keys:
  Must use 'session_close_t1', 'session_close_t5', 'session_close_t15' —
  the direct output of _rank_analogues_with_outcomes() in probability.py.

option_chain dict keys: 'strike' (float), 'call_price' (float).

Regime note: amplification / untethered / broken-magnet return [] immediately.
These regimes have no trade-thesis range and therefore no edge zone classification.
broken-magnet has outcome_status='computed' but no drift_target — treated same
as None-range regimes (spec deviation documented in structural_distribution.py).
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

from packages.shared.implied_distribution import (
    compute_implied_pdf,
    compute_implied_prob_in_range,
)
from packages.shared.structural_distribution import (
    compute_terminal_prob_in_range,
    get_trade_thesis_range,
)

_TIMEFRAME_TO_YEARS: dict[str, float] = {
    "t1":  1.0 / 252,
    "t5":  5.0 / 252,
    "t15": 15.0 / 252,
}

# Edge ratio classification thresholds (calibration knobs)
_STRONG_POSITIVE_THRESHOLD   = 2.0
_MODERATE_POSITIVE_THRESHOLD = 1.3
_MODERATE_NEGATIVE_THRESHOLD = 0.7
_STRONG_NEGATIVE_THRESHOLD   = 0.5


def classify_edge_ratio(ratio: Optional[float]) -> str:
    """Map a numerical edge ratio to a classification band.

    Edge ratio = structural_terminal_prob / market_implied_prob.
    Values > 1 indicate structural edge over the market's implied distribution.

    Boundary tie-breaking (documented per spec):
      - Neutral range is closed: [0.7, 1.3] — ratio=1.3 → 'neutral'.
      - Moderate-positive starts strictly above 1.3: 1.31 → 'moderate-positive'.
      - ratio=2.0 → 'moderate-positive' (≤ 2.0, not strong).
      - ratio=float('inf') → 'strong-positive' (structurally uncovered bin).
      - Negative ratios (shouldn't occur; defensive) → 'unknown'.
    """
    if ratio is None:
        return "unknown"
    try:
        r = float(ratio)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(r):
        return "unknown"
    if r < 0:
        warnings.warn(
            f"classify_edge_ratio: negative ratio {r:.4f}; returning 'unknown'. "
            "Check structural_prob and implied_prob signs.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "unknown"
    if r > _STRONG_POSITIVE_THRESHOLD:   # > 2.0 (inf also satisfies this)
        return "strong-positive"
    if r > _MODERATE_POSITIVE_THRESHOLD: # (1.3, 2.0]
        return "moderate-positive"
    if r >= _MODERATE_NEGATIVE_THRESHOLD: # [0.7, 1.3]
        return "neutral"
    if r >= _STRONG_NEGATIVE_THRESHOLD:  # [0.5, 0.7)
        return "moderate-negative"
    return "strong-negative"             # < 0.5  (includes 0.0)


def _auto_bin_size(option_chain: list[dict], implied_move: float) -> float:
    """Choose bin_size from chain spacing and implied_move.

    bin_size = max(1.0, min(min_strike_spacing, 0.25 * implied_move))

    Using min strike spacing (not median) ensures we match the chain's finest
    granularity — important if the chain has mixed-spacing regions.
    """
    if len(option_chain) >= 2:
        sorted_strikes = sorted(d["strike"] for d in option_chain)
        spacings = [sorted_strikes[i + 1] - sorted_strikes[i] for i in range(len(sorted_strikes) - 1)]
        strike_spacing = min(spacings)
    else:
        strike_spacing = float("inf")

    im_quarter = 0.25 * implied_move if implied_move > 0 else 5.0
    raw = min(strike_spacing, im_quarter)
    return max(1.0, raw)


def _price_axis_bins(
    spot: float,
    implied_move: float,
    price_range_sigma: float,
    bin_size: float,
) -> list[tuple[float, float]]:
    """Generate bin boundaries covering spot ± price_range_sigma * implied_move."""
    lower_raw = spot - price_range_sigma * implied_move
    upper_raw = spot + price_range_sigma * implied_move

    # Snap to bin_size boundaries (floor/ceil)
    lower_axis = math.floor(lower_raw / bin_size) * bin_size
    upper_axis = math.ceil(upper_raw / bin_size) * bin_size

    bins: list[tuple[float, float]] = []
    n_bins = round((upper_axis - lower_axis) / bin_size)
    for i in range(n_bins):
        b_lo = lower_axis + i * bin_size
        b_hi = lower_axis + (i + 1) * bin_size
        if b_lo < upper_axis - 1e-9:
            bins.append((b_lo, b_hi))
    return bins


def _make_zone(bins: list[dict]) -> dict:
    """Aggregate a contiguous run of same-classification bins into one zone."""
    ratios = [b["edge_ratio"] for b in bins if b["edge_ratio"] is not None]
    avg_ratio = sum(ratios) / len(ratios) if ratios else None
    min_n = min(b["structural_n"] for b in bins)
    mid = bins[len(bins) // 2]  # representative bin for tooltip
    return {
        "lower":            bins[0]["lower"],
        "upper":            bins[-1]["upper"],
        "classification":   bins[0]["classification"],
        "n_bins":           len(bins),
        "avg_edge_ratio":   avg_ratio,
        "min_structural_n": min_n,
        "representative":   {k: mid[k] for k in mid},
    }


def _group_bins_into_zones(bins: list[dict]) -> list[dict]:
    """Merge contiguous same-classification bins into zones."""
    if not bins:
        return []
    zones: list[dict] = []
    current: list[dict] = [bins[0]]
    for b in bins[1:]:
        if b["classification"] == current[-1]["classification"]:
            current.append(b)
        else:
            zones.append(_make_zone(current))
            current = [b]
    zones.append(_make_zone(current))
    return zones


def compute_edge_zones(
    spot: float,
    implied_move: float,
    option_chain: list[dict],
    analogues_ohlc: list[dict],
    timeframe: str,
    regime_block: dict,
    risk_free_rate: float = 0.05,
    time_to_expiration: Optional[float] = None,
    bin_size: Optional[float] = None,
    price_range_sigma: float = 1.5,
    *,
    tolerance: Optional[float] = None,
) -> list[dict]:
    """Compute classified edge zones across the price axis.

    Returns a list of zone dicts sorted by lower price ascending. Each zone covers
    a contiguous run of equal-classification price bins:

        {lower, upper, classification, n_bins, avg_edge_ratio, min_structural_n,
         representative: {lower, upper, structural_prob, structural_n, structural_ci,
                          implied_prob, edge_ratio, classification}}

    Returns [] for non-range regimes (amplification/untethered/broken-magnet).

    Args:
        spot:               Current underlying price.
        implied_move:       1-day implied move in points (from ORATS or feature vector).
        option_chain:       List of {'strike': float, 'call_price': float} dicts.
        analogues_ohlc:     Output of _rank_analogues_with_outcomes(); must include
                            'session_close_{timeframe}' keys for the requested timeframe.
        timeframe:          't1', 't5', or 't15' — which session checkpoint to use.
        regime_block:       GEX landscape regime dict (has 'regime', 'drift_target', etc.)
        risk_free_rate:     Annualised continuously compounded rate (default 0.05).
        time_to_expiration: Years to expiration for options pricing.
                            If None, derived from timeframe: t1→1/252, t5→5/252, t15→15/252.
        bin_size:           Price-axis bin width in points. If None, auto-chosen from
                            min(strike_spacing, 0.25*implied_move), floored at 1.0.
        price_range_sigma:  Axis coverage in implied_move units (default 1.5×).
        tolerance:          Required for magnetic-pin regime (0.25 * implied_move typically).
                            Forwarded to get_trade_thesis_range.

    Notes:
        - min_structural_n tracks the thinnest denominator in each zone; the chart
          should dim or annotate zones where min_structural_n < 5 (sparse data).
        - structural_prob = None (not 0.0) when all analogues had NULL close values for
          the timeframe; those bins get classification='unknown'.
        - structural_prob = 0.0 (zero analogues in bin) is a legitimate signal → 0/implied
          = 0.0 → 'strong-negative'. This is intentional.
        - implied_prob is suppressed to None when < 1e-10 (PDF gap); bin → 'unknown'.
    """
    # 1. Trade-thesis range (regime guard)
    try:
        trade_range = get_trade_thesis_range(regime_block, spot, tolerance=tolerance)
    except ValueError:
        return []

    if trade_range["lower"] is None and trade_range["upper"] is None:
        return []

    # 2. Bin size
    effective_bin_size = bin_size if bin_size is not None else _auto_bin_size(option_chain, implied_move)

    # 3. Price axis
    bin_boundaries = _price_axis_bins(spot, implied_move, price_range_sigma, effective_bin_size)
    if not bin_boundaries:
        return []

    # 4. Implied PDF (computed once, cached)
    tte = time_to_expiration
    if tte is None:
        tte = _TIMEFRAME_TO_YEARS.get(timeframe, _TIMEFRAME_TO_YEARS["t5"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        implied_pdf = compute_implied_pdf(option_chain, spot, risk_free_rate, tte)

    # 5. Close values for the requested timeframe
    close_key = f"session_close_{timeframe}"
    close_vals = [a.get(close_key) for a in analogues_ohlc]

    # 6. Per-bin computation
    bin_results: list[dict] = []
    for b_lo, b_hi in bin_boundaries:
        struct_result = compute_terminal_prob_in_range(close_vals, b_lo, b_hi)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            impl_p = compute_implied_prob_in_range(implied_pdf, b_lo, b_hi)

        implied_prob: Optional[float] = impl_p if impl_p > 1e-10 else None

        if struct_result is None:
            edge_ratio = None
        elif implied_prob is None:
            edge_ratio = None
        else:
            s_p = struct_result["prob"]
            edge_ratio = s_p / implied_prob if implied_prob > 0 else (
                float("inf") if s_p > 0 else None
            )

        bin_results.append({
            "lower":           b_lo,
            "upper":           b_hi,
            "structural_prob": struct_result["prob"] if struct_result is not None else None,
            "structural_n":    struct_result["n"]    if struct_result is not None else 0,
            "structural_ci":   struct_result["wilson_ci"] if struct_result is not None else None,
            "implied_prob":    implied_prob,
            "edge_ratio":      edge_ratio,
            "classification":  classify_edge_ratio(edge_ratio),
        })

    # 7. Group into zones
    return _group_bins_into_zones(bin_results)
