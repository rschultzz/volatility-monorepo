"""
GEX landscape — shared analytical module.

Pure analytical functions for the Gaussian-smoothed GEX landscape, extracted
from scripts/explore_gex_landscape.py (the Phase 0 prototype) so the same math
is callable from the CLI script, the EOD cron, the backfill script, and a
future endpoint. See specs/CR-007-gex-landscape-data-pipeline.md and the vault
notes [[gex-landscape]] / [[2026-05-20 - GEX Landscape Spot-Agnostic Storage]].

Storage is spot-agnostic: compute_landscape / find_walls / find_peaks_per_bucket
are pure functions of the OI/gamma data and are what the cron persists. The
spot-dependent classifiers (regime, per-bucket dominance, confluence, negative
zones) also live here but are applied at request time by callers — they are not
persisted (see the ADR).

compute_and_upsert_landscape is the cron/backfill entry point: it queries the
just-upserted orats_oi_gamma rows, computes the spot-agnostic artifacts, and
UPSERTs them into orats_gex_landscape on the caller-supplied connection.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def compute_landscape(
    df: pd.DataFrame,
    spot: float,
    range_pts: float = 200.0,
    step_pts: float = 1.0,
    spread_coef: float = 8.0,
) -> pd.DataFrame:
    """
    For each candidate spot price S in [spot ± range_pts]:
        GEX(S) = Σ_rows [ net_gex × exp(-(S - level)² / (2 × σ²)) ]
    where σ = spread_coef × sqrt(max(dte, 0.5)).

    DTE-scaled spread captures the physics: 0DTE gamma is concentrated
    (sharp narrow peaks), 45DTE gamma is diffuse (broad low humps).

    The same total is also decomposed into 4 DTE buckets for the stacked view.

    Returns columns:
        price, gex_total, gex_0dte, gex_near (1-7), gex_med (8-30), gex_struct (30+)
    All values in raw $ (divide by 1e9 for billions).
    """
    if df.empty:
        return pd.DataFrame(columns=["price", "gex_total", "gex_0dte",
                                     "gex_near", "gex_med", "gex_struct"])

    # Drop rows with missing critical fields
    df = df.dropna(subset=["discounted_level", "gex_call", "gex_put"]).copy()
    df["dte_eff"] = df["dte"].fillna(30.0).clip(lower=0.5)

    prices = np.arange(spot - range_pts, spot + range_pts + step_pts, step_pts)

    levels  = df["discounted_level"].to_numpy(dtype=float)
    dtes    = df["dte_eff"].to_numpy(dtype=float)
    # Match frontend convention: net = call - |put|
    nets    = df["gex_call"].to_numpy(dtype=float) - np.abs(df["gex_put"].to_numpy(dtype=float))
    spreads = spread_coef * np.sqrt(dtes)

    # Bucket each row by DTE for the stacked decomposition
    bucket = np.full(len(df), 3, dtype=int)  # 3 = structural (30+)
    bucket[dtes <= 30] = 2  # medium
    bucket[dtes <= 7]  = 1  # near
    bucket[dtes < 1.0] = 0  # 0DTE (after the 0.5 floor)

    M = len(prices)
    N = len(df)

    landscape_total  = np.zeros(M)
    landscape_0dte   = np.zeros(M)
    landscape_near   = np.zeros(M)
    landscape_med    = np.zeros(M)
    landscape_struct = np.zeros(M)

    P = prices[:, np.newaxis]  # (M, 1)

    # Process in chunks to keep peak memory bounded (~16MB per chunk at M=400)
    chunk = 2000
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        L = levels[start:end][np.newaxis, :]    # (1, c)
        S = spreads[start:end][np.newaxis, :]   # (1, c)
        W = nets[start:end][np.newaxis, :]      # (1, c)
        B = bucket[start:end]

        gaussians = np.exp(-((P - L) ** 2) / (2.0 * S * S))  # (M, c)
        weighted  = gaussians * W                            # (M, c)

        landscape_total += weighted.sum(axis=1)
        for b, sink in enumerate([landscape_0dte, landscape_near,
                                  landscape_med, landscape_struct]):
            mask = (B == b)
            if mask.any():
                sink += weighted[:, mask].sum(axis=1)

    return pd.DataFrame({
        "price":      prices,
        "gex_total":  landscape_total,
        "gex_0dte":   landscape_0dte,
        "gex_near":   landscape_near,
        "gex_med":    landscape_med,
        "gex_struct": landscape_struct,
    })


def find_walls(landscape: pd.DataFrame, min_prominence_pct: float = 0.10) -> pd.DataFrame:
    """
    Find peaks in |landscape|. Captures both positive walls (containment)
    and negative valleys (amplification). Sign attached.

    min_prominence_pct: peak must rise above its surroundings by at least
    this fraction of the day's max |GEX| to qualify.
    """
    if landscape.empty:
        return pd.DataFrame(columns=["price", "gex", "prominence", "sign"])

    gex = landscape["gex_total"].to_numpy()
    abs_gex = np.abs(gex)
    if abs_gex.max() <= 0:
        return pd.DataFrame(columns=["price", "gex", "prominence", "sign"])

    threshold = abs_gex.max() * min_prominence_pct
    peak_idx, props = find_peaks(abs_gex, prominence=threshold, distance=3)

    walls = pd.DataFrame({
        "price":      landscape["price"].iloc[peak_idx].to_numpy(),
        "gex":        landscape["gex_total"].iloc[peak_idx].to_numpy(),
        "prominence": props["prominences"],
    })
    walls["sign"] = np.sign(walls["gex"])
    return walls.sort_values("price").reset_index(drop=True)


def score_containment_zones(
    walls: pd.DataFrame,
    landscape: pd.DataFrame,
    spot: float,
) -> pd.DataFrame:
    """
    Score every adjacent pair of POSITIVE walls as a containment zone.
    Negative walls amplify (they don't constrain) so they're excluded as
    zone boundaries.

    Score = (min_strength × valley_depth) / width
    Stronger and deeper valley wins; wider is penalized.
    """
    pos_walls = walls[walls["sign"] > 0].reset_index(drop=True)
    if len(pos_walls) < 2:
        return pd.DataFrame()

    rows = []
    for i in range(len(pos_walls) - 1):
        lower = pos_walls.iloc[i]
        upper = pos_walls.iloc[i + 1]
        width = float(upper["price"] - lower["price"])
        if width <= 0:
            continue

        min_strength = float(min(lower["gex"], upper["gex"]))

        # Valley floor between the walls — deeper dip = stronger containment
        between = landscape[
            (landscape["price"] > lower["price"])
            & (landscape["price"] < upper["price"])
        ]
        valley_floor = float(between["gex_total"].min()) if len(between) else 0.0
        valley_depth = min_strength - valley_floor

        # Weighted equilibrium — stronger wall pulls equilibrium toward it
        eq = (
            (upper["gex"] * lower["price"] + lower["gex"] * upper["price"])
            / (lower["gex"] + upper["gex"])
        )

        rows.append({
            "lower_price":       float(lower["price"]),
            "upper_price":       float(upper["price"]),
            "lower_gex":         float(lower["gex"]),
            "upper_gex":         float(upper["gex"]),
            "width_pts":         width,
            "valley_floor":      valley_floor,
            "valley_depth":      valley_depth,
            "containment_score": (min_strength * valley_depth) / width,
            "equilibrium_price": float(eq),
            "contains_spot":     bool(lower["price"] <= spot <= upper["price"]),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("containment_score", ascending=False)
        .reset_index(drop=True)
    )


# ─── IV-aware distance classification ──────────────────────────────────────
#
# A "target" reported by the regime classifier (drift_target, confluence
# price, etc.) needs to be contextualized against today's expected daily
# range, or it can be massively misread. A magnet 128pt above on a day with
# implied move 40pt is NOT today's target — it's a structural pull that
# would take days/weeks to fully play out. The single-day target should be
# inside ~1.5× implied move.
#
# Buckets:
#   in-range       ≤ 1.0σ — likely to trade through during normal session
#   intraday-reach ≤ 1.5σ — plausible high/low for today (~85% probability)
#   stretch        ≤ 2.5σ — needs momentum or catalyst
#   multi-day      ≤ 4.0σ — structural pull, not today
#   far            > 4.0σ — not relevant for current session

DISTANCE_THRESHOLDS = [
    (1.0, "in-range"),
    (1.5, "intraday-reach"),
    (2.5, "stretch"),
    (4.0, "multi-day"),
    (float("inf"), "far"),
]


def classify_distance(distance_pts: float, implied_move_pts: float) -> dict:
    """
    Classify a distance from spot in terms of implied move units.

    Returns dict with:
        sigma:    distance / implied_move  (or None if no IV provided)
        class:    one of in-range, intraday-reach, stretch, multi-day, far, unknown
    """
    if implied_move_pts <= 0:
        return {"sigma": None, "class": "unknown"}
    sigma = abs(distance_pts) / implied_move_pts
    for threshold, cls in DISTANCE_THRESHOLDS:
        if sigma <= threshold:
            return {"sigma": float(sigma), "class": cls}
    return {"sigma": float(sigma), "class": "far"}


def compute_implied_move(spot: float, iv: float, dte: float = 1.0,
                         trading_days_per_year: int = 252) -> float:
    """1-sigma move in points: spot × iv × sqrt(dte / 252)."""
    if iv <= 0 or dte <= 0:
        return 0.0
    return float(spot * iv * (dte / trading_days_per_year) ** 0.5)


# ─── Regime classification ──────────────────────────────────────────────────
#
# Looks at where spot sits relative to the structural walls and amplification
# zones, and labels the day with one of seven regimes:
#
#   broken-magnet   — Spot crossed through the dominant wall since the last
#                     EOD reference. The wall is now BEHIND the movement, not
#                     ahead of it. Continued drift in the crossing direction
#                     is likely unless momentum reverses. Requires a
#                     prior_spot to detect.
#
#   magnetic-pin    — Spot is within near_dist_pts of the dominant wall AND
#                     no crossing detected. Strong gravitational anchor; pin
#                     trade.
#
#   magnet-above    — Dominant wall is above spot, no competitive wall in
#                     between. Upward drift expected toward the magnet.
#
#   magnet-below    — Dominant wall is below spot, no competitive wall in
#                     between. Downward drift expected toward the magnet.
#
#   bounded         — Competitive walls on BOTH sides of spot. Range-bound;
#                     condor candidate. (This is the canonical "between two
#                     walls" containment case.)
#
#   amplification   — Spot is near a significant negative GEX zone. Dealer
#                     hedging amplifies moves; breaking into the zone risks
#                     acceleration rather than mean reversion.
#
#   untethered      — Spot is far from any wall, or no walls detected. No
#                     structural guidance — falls back to other signals.
#
# The classifier runs its own peak detection at a more sensitive prominence
# (secondary_prominence) than the visualization, so it sees structure that's
# real but doesn't qualify for plotting when one strike dominates the scale.


def classify_regime(
    landscape: pd.DataFrame,
    spot: float,
    *,
    prior_spot: Optional[float] = None,
    implied_move: float = 0.0,
    near_dist_pts: float = 30.0,
    competitive_ratio: float = 0.5,
    secondary_prominence: float = 0.03,
    amplification_significance: float = 0.15,
    min_movement_pts: float = 10.0,
) -> dict:
    """
    Classify the day's GEX structure into a regime tag plus actionable detail.

    prior_spot:
        Previous-session reference spot (typically table_spot from
        orats_oi_gamma when --spot is used to override). Enables broken-magnet
        detection: if the dominant wall sits between prior_spot and current
        spot, the wall has been crossed since the last close, and the regime
        is broken-magnet rather than magnetic-pin.
    min_movement_pts:
        Only consider broken-magnet detection if |spot - prior_spot| exceeds
        this threshold. Avoids triggering on noise.
    near_dist_pts:
        Spot within this distance of the dominant wall → magnetic-pin regime
        (assuming the magnet hasn't been crossed).
    competitive_ratio:
        A wall is "competitive" if its abs(gex) >= ratio × dominant_strength.
        Lower this (e.g. 0.4) to detect more bounded days; raise it (e.g. 0.6)
        to require closer-to-equal walls.
    secondary_prominence:
        Re-runs find_walls() at this prominence to catch the secondary structure
        that gets hidden by find_walls()'s default plot-friendly threshold.
    amplification_significance:
        Negative zone is "significant" if abs(gex) >= ratio × dominant_strength.

    Returns dict with at minimum: {regime, notes, dominant_wall, spot, ...}
    Plus regime-specific extras: drift_target, drift_direction, containment_zone,
    amplification_zone, crossing_direction, prior_spot, spot_move.
    """
    # Use a sensitive prominence so the classifier can see secondary walls
    # even when one giant strike dominates the absolute scale.
    sensitive = find_walls(landscape, min_prominence_pct=secondary_prominence)

    pos_walls = sensitive[sensitive["sign"] > 0].reset_index(drop=True)
    neg_walls = sensitive[sensitive["sign"] < 0].reset_index(drop=True)

    # Reference strength for relative thresholds — use whichever side is bigger.
    # This matters when a bucket is negative-dominated (e.g. 0DTE on a chop day
    # has put-heavy strikes creating a negative zone with no positive walls).
    max_pos     = float(pos_walls["gex"].max())          if not pos_walls.empty else 0.0
    max_neg_abs = float(neg_walls["gex"].abs().max())    if not neg_walls.empty else 0.0
    reference_strength = max(max_pos, max_neg_abs)

    if reference_strength <= 0:
        return {
            "regime": "untethered",
            "spot": float(spot),
            "notes": ["No structural walls of either sign detected in landscape."],
        }

    # Check for significant negative zone near spot up-front, using
    # reference_strength as the relative threshold. This way a negative-only
    # bucket (e.g. 0DTE chop day) gets classified as amplification rather than
    # untethered.
    significant_neg = neg_walls[
        neg_walls["gex"].abs() >= amplification_significance * reference_strength
    ]
    nearest_neg = None
    if not significant_neg.empty:
        idx = (significant_neg["price"] - spot).abs().idxmin()
        cand = significant_neg.loc[idx]
        if abs(cand["price"] - spot) < near_dist_pts * 1.5:
            nearest_neg = {
                "price": float(cand["price"]),
                "gex": float(cand["gex"]),
                "distance": float(abs(cand["price"] - spot)),
            }

    # If there's no positive structure to anchor regular regime detection,
    # return amplification (if applicable) or untethered.
    if pos_walls.empty:
        if nearest_neg is not None:
            return {
                "regime": "amplification",
                "spot": float(spot),
                "amplification_zone": nearest_neg,
                "n_positive_walls": 0,
                "n_negative_walls": int(len(neg_walls)),
                "notes": [(
                    f"Net negative GEX region near spot — no positive structure "
                    f"to anchor a magnet. Spot {spot:.0f} within "
                    f"{nearest_neg['distance']:.0f}pt of negative zone at "
                    f"{nearest_neg['price']:.0f} ({nearest_neg['gex']/1e9:+.0f}B). "
                    f"Dealer hedging amplifies moves — expect chop or directional "
                    f"acceleration depending on flow."
                )],
            }
        return {
            "regime": "untethered",
            "spot": float(spot),
            "notes": ["No positive walls detected in landscape range."],
        }

    # Dominant wall = highest positive peak
    dom_idx = pos_walls["gex"].idxmax()
    dominant = pos_walls.loc[dom_idx]
    dom_strength = float(dominant["gex"])
    dom_price = float(dominant["price"])

    # Competitive walls relative to dominant
    competitive = pos_walls[pos_walls["gex"] >= competitive_ratio * dom_strength]
    competitive_above = competitive[competitive["price"] > spot].sort_values("price")
    competitive_below = competitive[competitive["price"] < spot].sort_values(
        "price", ascending=False
    )

    # Shared result skeleton
    result = {
        "spot":                  float(spot),
        "dominant_wall": {
            "price": dom_price,
            "gex":   dom_strength,
        },
        "spot_to_dominant_pts": float(dom_price - spot),
        "n_competitive_walls":  int(len(competitive)),
        "n_positive_walls":     int(len(pos_walls)),
        "n_negative_walls":     int(len(neg_walls)),
    }
    if prior_spot is not None:
        result["prior_spot"] = float(prior_spot)
        result["spot_move"]  = float(spot - prior_spot)

    # ── 0. Broken magnet: dominant wall crossed since prior reference ──
    # Detected BEFORE other regimes — a broken magnet is a different setup
    # than a static pin even when the static-pin criteria would also fire.
    if prior_spot is not None and abs(spot - prior_spot) >= min_movement_pts:
        crossed_up   = prior_spot < dom_price <= spot   # gap-up through magnet
        crossed_down = spot <= dom_price < prior_spot   # gap-down through magnet
        if crossed_up or crossed_down:
            direction       = "up" if crossed_up else "down"
            past_magnet_pts = float(abs(spot - dom_price))
            move_pts        = float(abs(spot - prior_spot))
            result["regime"] = "broken-magnet"
            result["crossing_direction"]    = direction
            result["distance_past_magnet"]  = past_magnet_pts
            result["notes"] = [
                (
                    f"Spot moved {prior_spot:.0f} → {spot:.0f} "
                    f"({'+' if spot > prior_spot else ''}{spot-prior_spot:.0f}pt), "
                    f"crossing the dominant wall at {dom_price:.0f} "
                    f"({dom_strength/1e9:.0f}B). The wall is now BEHIND the move — "
                    f"it acts as {'support from above' if crossed_up else 'resistance from below'} "
                    f"rather than as a magnet pulling price toward it. "
                    f"Continued drift {direction} likely if momentum persists; "
                    f"a return to test {dom_price:.0f} would be a "
                    f"{'pullback' if crossed_up else 'bounce'}."
                )
            ]
            return result

    # ── 1. Amplification: spot near significant negative GEX zone ──
    if nearest_neg is not None:
        result["regime"] = "amplification"
        result["amplification_zone"] = nearest_neg
        result["notes"] = [
            (
                f"Spot {spot:.0f} within {nearest_neg['distance']:.0f}pt of significant "
                f"negative GEX zone at {nearest_neg['price']:.0f} "
                f"({nearest_neg['gex']/1e9:+.0f}B). Dealer hedging amplifies moves into "
                f"this zone — break risks acceleration rather than mean reversion."
            )
        ]
        return result

    # ── 2. Bounded: competitive walls on both sides of spot ──
    if not competitive_above.empty and not competitive_below.empty:
        upper = competitive_above.iloc[0]   # nearest above
        lower = competitive_below.iloc[0]   # nearest below
        width = float(upper["price"] - lower["price"])
        eq = float(
            (upper["gex"] * lower["price"] + lower["gex"] * upper["price"])
            / (lower["gex"] + upper["gex"])
        )
        result["regime"] = "bounded"
        result["containment_zone"] = {
            "lower_price": float(lower["price"]),
            "upper_price": float(upper["price"]),
            "lower_gex":   float(lower["gex"]),
            "upper_gex":   float(upper["gex"]),
            "width_pts":   width,
            "equilibrium": eq,
        }
        result["notes"] = [
            (
                f"Bracketed by competitive walls at {lower['price']:.0f} "
                f"({lower['gex']/1e9:.0f}B) and {upper['price']:.0f} "
                f"({upper['gex']/1e9:.0f}B). Width {width:.0f}pt, weighted equilibrium "
                f"{eq:.0f}. Range-bound regime — condor / premium-sell candidate."
            )
        ]
        return result

    # ── 3. Magnetic-pin: spot very close to dominant wall ──
    if abs(dom_price - spot) < near_dist_pts:
        if spot > dom_price + 1:
            direction = "down"
        elif spot < dom_price - 1:
            direction = "up"
        else:
            direction = "pin"
        result["regime"] = "magnetic-pin"
        result["drift_target"]    = dom_price
        result["drift_direction"] = direction
        result["notes"] = [
            (
                f"Spot {spot:.0f} within {abs(dom_price-spot):.0f}pt of dominant wall at "
                f"{dom_price:.0f} ({dom_strength/1e9:.0f}B). Strong gravitational anchor — "
                f"expect pin at or near {dom_price:.0f}."
            )
        ]
        return result

    # ── 4 & 5. Magnet-above / magnet-below: dominant pulls from one side ──
    # Distinguished by which side of spot the dominant wall sits on.
    # Only enters these regimes if no competitive wall sits between spot and dominant.
    if dom_price > spot:
        between = pos_walls[
            (pos_walls["price"] > spot)
            & (pos_walls["price"] < dom_price)
            & (pos_walls["gex"] >= competitive_ratio * dom_strength)
        ]
        if between.empty:
            result["regime"]          = "magnet-above"
            result["drift_target"]    = dom_price
            result["drift_direction"] = "up"
            result["notes"] = [
                (
                    f"Spot {spot:.0f} is {dom_price-spot:.0f}pt below dominant wall at "
                    f"{dom_price:.0f} ({dom_strength/1e9:.0f}B). No competitive wall in "
                    f"between — upward drift expected unless price clears the magnet, "
                    f"after which structure above is diffuse."
                )
            ]
            return result
    else:  # dom_price < spot
        between = pos_walls[
            (pos_walls["price"] < spot)
            & (pos_walls["price"] > dom_price)
            & (pos_walls["gex"] >= competitive_ratio * dom_strength)
        ]
        if between.empty:
            result["regime"]          = "magnet-below"
            result["drift_target"]    = dom_price
            result["drift_direction"] = "down"
            result["notes"] = [
                (
                    f"Spot {spot:.0f} is {spot-dom_price:.0f}pt above dominant wall at "
                    f"{dom_price:.0f} ({dom_strength/1e9:.0f}B). No competitive wall in "
                    f"between — downward drift expected unless price clears the magnet, "
                    f"after which structure below is diffuse."
                )
            ]
            return result

    # ── 6. Untethered fallthrough (complex structure) ──
    result["regime"] = "untethered"
    result["notes"] = [
        (
            "Complex structure — dominant wall exists but spot's relationship to it "
            "doesn't fit a single regime. Look at the walls list manually."
        )
    ]
    return result


def _annotate_distance_class(regime_result: dict, implied_move: float) -> dict:
    """
    Post-process a regime result to add distance classification for any
    drift_target, dominant_wall, containment_zone, or amplification_zone.

    Adds 'target_classification' field with sigma + class for the primary
    target where applicable.
    """
    if implied_move <= 0:
        return regime_result

    spot = regime_result.get("spot", 0.0)

    # Classify the primary target (drift_target if present, else dominant wall)
    if "drift_target" in regime_result:
        dist = abs(regime_result["drift_target"] - spot)
        regime_result["target_classification"] = classify_distance(dist, implied_move)

    if "dominant_wall" in regime_result:
        dist = abs(regime_result["dominant_wall"]["price"] - spot)
        regime_result["dominant_wall_classification"] = classify_distance(dist, implied_move)

    if "containment_zone" in regime_result:
        z = regime_result["containment_zone"]
        regime_result["containment_zone"]["upper_classification"] = classify_distance(
            abs(z["upper_price"] - spot), implied_move
        )
        regime_result["containment_zone"]["lower_classification"] = classify_distance(
            abs(z["lower_price"] - spot), implied_move
        )

    if "amplification_zone" in regime_result:
        regime_result["amplification_zone"]["classification"] = classify_distance(
            regime_result["amplification_zone"]["distance"], implied_move
        )

    if "prior_spot" in regime_result and "dominant_wall" in regime_result:
        # broken-magnet — classify the wall distance from current spot
        regime_result["dominant_wall_classification"] = classify_distance(
            abs(regime_result["dominant_wall"]["price"] - spot), implied_move
        )

    return regime_result


# ─── Per-DTE-bucket classification ──────────────────────────────────────────
#
# The single-landscape regime classifier runs on summed-across-DTE GEX. But
# different timeframes can have very different structure, and the bucket that
# controls intraday price action varies by day:
#
#   - On a "weekly pin" day, 1-7 DTE dominates near spot.
#   - On a "0DTE chop" day, 0DTE has a negative zone right at spot — and the
#     longer-dated buckets wash it out in the aggregate.
#   - On a quiet structural day, 30+ DTE dominates because intraday OI is thin.
#
# Running classify_regime separately per bucket exposes these dynamics.
# Dominance is measured by integrated |GEX| within near_dist_pts of spot —
# whichever bucket has the most "material" in the immediate trading
# neighborhood is most likely to drive price.

# DTE bucket spec: (label, column, plot_color)
DTE_BUCKETS = [
    ("0DTE",     "gex_0dte",   "#ef4444"),
    ("1-7 DTE",  "gex_near",   "#f59e0b"),
    ("8-30 DTE", "gex_med",    "#10b981"),
    ("30+ DTE",  "gex_struct", "#3b82f6"),
]


def _bucket_landscape_view(landscape: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    """
    Return a landscape view where gex_total is the bucket's column.
    classify_regime operates on gex_total, so we substitute the bucket column
    in and leave the rest zeroed.
    """
    view = landscape[["price"]].copy()
    view["gex_total"]  = landscape[bucket_col]
    view["gex_0dte"]   = 0.0
    view["gex_near"]   = 0.0
    view["gex_med"]    = 0.0
    view["gex_struct"] = 0.0
    return view


def classify_per_bucket(
    landscape: pd.DataFrame,
    spot: float,
    *,
    prior_spot: Optional[float] = None,
    implied_move: float = 0.0,
    near_dist_pts: float = 30.0,
    bucket_specs: list = DTE_BUCKETS,
) -> dict:
    """
    Run classify_regime separately on each DTE bucket. Add a dominance_pct
    field measuring each bucket's relative |GEX| concentration within
    near_dist_pts of spot.

    Returns: {bucket_label: regime_dict_with_added_dominance_fields}
    """
    near_mask = (landscape["price"] - spot).abs() <= near_dist_pts

    results = {}
    for label, col, color in bucket_specs:
        view = _bucket_landscape_view(landscape, col)
        regime = classify_regime(view, spot, prior_spot=prior_spot,
                                 implied_move=implied_move)
        regime = _annotate_distance_class(regime, implied_move)

        # Local intensity = sum of |GEX| within near_dist_pts of spot
        local_abs = float(landscape.loc[near_mask, col].abs().sum())
        local_net = float(landscape.loc[near_mask, col].sum())
        local_max = float(landscape.loc[near_mask, col].abs().max() if near_mask.any() else 0.0)

        regime["bucket_label"]    = label
        regime["bucket_color"]    = color
        regime["local_intensity"] = local_abs
        regime["local_net"]       = local_net
        regime["local_max"]       = local_max

        results[label] = regime

    # Normalize to dominance percentages
    total_intensity = sum(r["local_intensity"] for r in results.values())
    if total_intensity > 0:
        for r in results.values():
            r["dominance_pct"] = (r["local_intensity"] / total_intensity) * 100.0
    else:
        for r in results.values():
            r["dominance_pct"] = 0.0

    return results


def summarize_per_bucket(per_bucket: dict) -> dict:
    """
    High-level cross-bucket summary:
      - which bucket dominates and by how much
      - whether buckets agree or disagree on regime
      - the "second story" — if dominant bucket is X but a meaningful #2 says
        something different, that's a divergence worth flagging
    """
    sorted_buckets = sorted(
        per_bucket.items(),
        key=lambda x: x[1]["dominance_pct"],
        reverse=True,
    )

    primary_label, primary = sorted_buckets[0]
    secondary_label, secondary = (sorted_buckets[1] if len(sorted_buckets) > 1
                                  else (None, None))

    regimes = [r["regime"] for r in per_bucket.values()]
    unique_regimes = set(regimes)

    # "Disagreement" is interesting only when buckets with meaningful dominance
    # (>15%) say different things. A 2% bucket dissenting from the dominant
    # one is noise; a 30% bucket dissenting is signal.
    meaningful = {
        r["regime"] for r in per_bucket.values()
        if r["dominance_pct"] >= 15.0
    }

    return {
        "primary_bucket":     primary_label,
        "primary_regime":     primary["regime"],
        "primary_dominance":  primary["dominance_pct"],
        "secondary_bucket":   secondary_label,
        "secondary_regime":   secondary["regime"] if secondary else None,
        "secondary_dominance": secondary["dominance_pct"] if secondary else 0.0,
        "unique_regimes":     sorted(unique_regimes),
        "meaningful_regimes": sorted(meaningful),
        "consensus":          len(meaningful) == 1,
    }


# ─── Peak confluence detection ──────────────────────────────────────────────
#
# Dominance by total mass favors broad-but-shallow buckets (e.g., 30+ DTE has
# 10x the dollar gamma of 1-7 DTE, even when 1-7 has a sharper, more tactical
# peak at the same strike). For predicting PIN behavior, sharpness matters
# more than mass — and even more important is when multiple buckets agree on
# the SAME price level. That's a confluence point.
#
# A confluence point is where 2+ DTE buckets have local maxima within ~10pt
# of each other. When the weekly (1-7 DTE) and monthly (8-30 DTE) both peak
# at the same strike, that's a much stronger pin signal than either alone.
# Multi-timeframe agreement = structural conviction.


def _compute_fwhm(gex_array: np.ndarray, prices: np.ndarray, peak_idx: int) -> float:
    """Full width at half maximum around a peak — measures how sharp the peak is.
    Smaller FWHM = sharper, more concentrated peak (stronger restoring force)."""
    if peak_idx <= 0 or peak_idx >= len(gex_array) - 1:
        return 0.0
    peak_val = float(gex_array[peak_idx])
    if peak_val <= 0:
        return 0.0
    half = peak_val / 2.0

    left = peak_idx
    while left > 0 and gex_array[left] > half:
        left -= 1
    right = peak_idx
    while right < len(gex_array) - 1 and gex_array[right] > half:
        right += 1

    return float(prices[right] - prices[left])


def find_peaks_per_bucket(
    landscape: pd.DataFrame,
    bucket_specs: list = DTE_BUCKETS,
    prominence_pct: float = 0.05,
) -> dict:
    """For each bucket, find positive local maxima with prominence + FWHM.

    Only POSITIVE peaks are returned — local maxima inside negative regions
    aren't pin levels, they're edge artifacts. Negative structure is handled
    separately by find_walls() (which captures both signs).
    """
    prices = landscape["price"].to_numpy()
    results = {}

    for label, col, color in bucket_specs:
        gex = landscape[col].to_numpy()
        if gex.max() <= 0:
            results[label] = []
            continue
        threshold = gex.max() * prominence_pct
        peak_idx, props = find_peaks(gex, prominence=threshold, distance=3)
        peaks = []
        for i, idx in enumerate(peak_idx):
            peak_val = float(gex[idx])
            # Skip negative-valued "peaks" — these are local maxima inside
            # negative GEX regions, not pin levels
            if peak_val <= 0:
                continue
            peaks.append({
                "price":      float(prices[idx]),
                "gex":        peak_val,
                "prominence": float(props["prominences"][i]),
                "fwhm":       _compute_fwhm(gex, prices, idx),
            })
        results[label] = peaks
    return results


def find_confluence_clusters(
    per_bucket_peaks: dict,
    max_cluster_width_pts: float = 10.0,
) -> list:
    """Cluster peaks across buckets that fall within max_cluster_width_pts.
    Each cluster is a list of dicts with {bucket, price, gex, prominence, fwhm}."""
    # Flatten with bucket labels
    all_peaks = []
    for label, peaks in per_bucket_peaks.items():
        for p in peaks:
            all_peaks.append({**p, "bucket": label})

    if not all_peaks:
        return []

    all_peaks.sort(key=lambda p: p["price"])

    clusters = []
    current = [all_peaks[0]]
    for p in all_peaks[1:]:
        # New peak within window of cluster's CURRENT extent (running average)
        cluster_center = sum(x["price"] for x in current) / len(current)
        if abs(p["price"] - cluster_center) <= max_cluster_width_pts:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
    clusters.append(current)
    return clusters


def classify_confluence_quality(max_gex: float) -> str:
    """
    Tag a confluence with a quality grade from its peak GEX strength.

    max_gex is the strongest peak in the cluster, in raw $ (divide by 1e9 for
    $B). Thresholds are calibrated against the labeled day set in
    tests/fixtures/confluence_calibration.json — top-cluster max_gex in $B:
        5/7  pin     — 712, 705
        5/6  target  — 612
        5/18 feature — 505
        5/20 feature — 125

    Peak strength, not peak width, separates the tiers. A real pin built on
    layered multi-strike hedging mass is broad, not sharp — see CR-011, which
    measured peak-width metrics (absolute FWHM, DTE-relative FWHM ratio, mass
    concentration) against the labeled set and found none of them monotonic.

    pin     — Strong magnet. Price locks in here if it reaches the level.
    target  — Moderate magnet. The destination of a directional move, but not
              a tight pin.
    feature — Weak level. A real feature of the field, but not a destination;
              price passes through it easily.
    """
    max_gex_b = max_gex / 1e9
    if max_gex_b >= 650:
        return "pin"
    if max_gex_b < 550:
        return "feature"
    return "target"


def score_confluence(cluster: list) -> dict:
    """Score a cluster:
        score = n_unique_buckets × max_gex / max(avg_fwhm, 5)

    Rewards multi-bucket agreement and high peak strength;
    penalizes diffuse peaks (large FWHM).
    """
    unique_buckets = sorted({p["bucket"] for p in cluster})
    n_buckets   = len(unique_buckets)
    max_gex     = max(p["gex"] for p in cluster)
    sum_gex     = sum(p["gex"] for p in cluster)
    fwhms       = [p["fwhm"] for p in cluster if p["fwhm"] > 0]
    avg_fwhm    = (sum(fwhms) / len(fwhms)) if fwhms else 0.0
    price_min   = min(p["price"] for p in cluster)
    price_max   = max(p["price"] for p in cluster)
    # Center = GEX-weighted average price (heavier peaks pull the center)
    if sum_gex > 0:
        center = sum(p["price"] * p["gex"] for p in cluster) / sum_gex
    else:
        center = (price_min + price_max) / 2.0
    score = (n_buckets * max_gex) / max(avg_fwhm, 5.0)

    return {
        "center_price":  float(center),
        "price_min":     float(price_min),
        "price_max":     float(price_max),
        "price_spread":  float(price_max - price_min),
        "n_buckets":     n_buckets,
        "buckets":       unique_buckets,
        "max_gex":       float(max_gex),
        "sum_gex":       float(sum_gex),
        "avg_fwhm":      float(avg_fwhm),
        "score":         float(score),
        "quality":       classify_confluence_quality(max_gex),
        "peaks":         cluster,
    }


def analyze_confluence(
    landscape: pd.DataFrame,
    *,
    bucket_specs: list = DTE_BUCKETS,
    max_cluster_width_pts: float = 10.0,
    prominence_pct: float = 0.05,
    min_buckets_for_confluence: int = 2,
) -> dict:
    """Full pipeline: find peaks per bucket → cluster → score → rank.

    Returns:
        {
            'confluences':       list of clusters with 2+ buckets (sorted by score),
            'single_bucket_peaks': list of single-bucket peaks (not confluence,
                                   but worth showing as secondary references),
            'all_clusters':      raw list before filtering, for debugging
        }
    """
    per_bucket_peaks = find_peaks_per_bucket(landscape, bucket_specs, prominence_pct)
    clusters = find_confluence_clusters(per_bucket_peaks, max_cluster_width_pts)
    scored = [score_confluence(c) for c in clusters]
    scored.sort(key=lambda s: s["score"], reverse=True)

    confluences = [s for s in scored if s["n_buckets"] >= min_buckets_for_confluence]
    singles     = [s for s in scored if s["n_buckets"] == 1]

    return {
        "confluences":        confluences,
        "single_bucket_peaks": singles,
        "all_clusters":       scored,
    }


def find_intraday_subtarget(
    confluences: list,
    walls: pd.DataFrame,
    spot: float,
    implied_move: float,
    *,
    max_sigma: float = 1.5,
    direction: Optional[str] = None,
) -> Optional[dict]:
    """
    When the primary regime target is structural (>2.5σ), find the nearest
    confluence or wall within intraday-reach (≤1.5σ) that's in the same
    direction. That's the realistic level to watch today.

    direction: 'up', 'down', or None. None means either side.

    Returns: dict with price, type, sigma_distance, label — or None if no
    actionable intraday level exists.
    """
    if implied_move <= 0:
        return None

    candidates = []

    # Confluences as candidates
    for c in confluences:
        dist = abs(c["center_price"] - spot)
        sigma = dist / implied_move
        if sigma > max_sigma:
            continue
        if direction == "up" and c["center_price"] < spot:
            continue
        if direction == "down" and c["center_price"] > spot:
            continue
        candidates.append({
            "price":          c["center_price"],
            "type":           f"confluence (★ × {c['n_buckets']})",
            "sigma_distance": sigma,
            "n_buckets":      c["n_buckets"],
            "max_gex":        c["max_gex"],
            "score":          c["score"],
        })

    # Walls (positive only — the structural pin candidates)
    if not walls.empty:
        pos = walls[walls["sign"] > 0]
        for _, w in pos.iterrows():
            dist = abs(w["price"] - spot)
            sigma = dist / implied_move
            if sigma > max_sigma:
                continue
            if direction == "up" and w["price"] < spot:
                continue
            if direction == "down" and w["price"] > spot:
                continue
            candidates.append({
                "price":          float(w["price"]),
                "type":           "wall",
                "sigma_distance": float(sigma),
                "max_gex":        float(w["gex"]),
                "score":          float(w["gex"]),
            })

    if not candidates:
        return None

    # Prefer highest-score candidate within reach
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[0]


def find_proximate_negative_zones(
    walls: pd.DataFrame,
    spot: float,
    implied_move: float,
    *,
    dom_strength: Optional[float] = None,
    max_sigma: float = 1.5,
    significance_ratio: float = 0.15,
) -> list:
    """
    Find significant negative GEX walls within max_sigma of spot.

    A negative zone is "significant" if abs(gex) >= significance_ratio × dom_strength
    (using the strongest positive wall as the reference; if none, uses max abs neg).

    Returns list of dicts sorted by abs(gex) descending — deepest neg zones first.
    """
    if implied_move <= 0 or walls is None or walls.empty:
        return []

    pos = walls[walls["sign"] > 0]
    neg = walls[walls["sign"] < 0].copy()
    if neg.empty:
        return []

    # Reference for significance
    if dom_strength is None or dom_strength <= 0:
        pos_max = float(pos["gex"].max()) if not pos.empty else 0.0
        neg_max = float(neg["gex"].abs().max())
        dom_strength = max(pos_max, neg_max)

    if dom_strength <= 0:
        return []

    threshold = significance_ratio * dom_strength

    zones = []
    for _, w in neg.iterrows():
        if abs(w["gex"]) < threshold:
            continue
        distance = abs(float(w["price"]) - spot)
        sigma = distance / implied_move
        if sigma > max_sigma:
            continue
        zones.append({
            "price":     float(w["price"]),
            "gex":       float(w["gex"]),
            "distance":  float(distance),
            "sigma":     float(sigma),
            "direction": "below" if w["price"] < spot else "above",
        })

    zones.sort(key=lambda z: abs(z["gex"]), reverse=True)
    return zones


# ─── Persistence (CR-007) ───────────────────────────────────────────────────
#
# compute_and_upsert_landscape is the cron/backfill entry point. It persists
# only the spot-agnostic artifacts of the field — the landscape grid, the
# walls, and the per-bucket peaks — all pure functions of the OI/gamma data.
# Spot-dependent classification (regime, confluence, neg zones) is left to
# request-time callers. See [[2026-05-20 - GEX Landscape Spot-Agnostic Storage]].

_LANDSCAPE_QUERY = """
    SELECT discounted_level, strike, expir_date, dte,
           stock_price, call_oi, put_oi, gamma, gex_call, gex_put
    FROM orats_oi_gamma
    WHERE ticker = %s
      AND trade_date = %s
      AND expir_date >= %s
      AND discounted_level IS NOT NULL
    ORDER BY expir_date, discounted_level
"""

_UPSERT_SQL = """
    INSERT INTO orats_gex_landscape
        (ticker, trade_date, landscape, walls, peaks_by_bucket,
         spread_coef, range_pts, step_pts, table_spot, version)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (ticker, trade_date) DO UPDATE SET
        landscape       = EXCLUDED.landscape,
        walls           = EXCLUDED.walls,
        peaks_by_bucket = EXCLUDED.peaks_by_bucket,
        spread_coef     = EXCLUDED.spread_coef,
        range_pts       = EXCLUDED.range_pts,
        step_pts        = EXCLUDED.step_pts,
        table_spot      = EXCLUDED.table_spot,
        version         = EXCLUDED.version,
        computed_at     = NOW()
"""


def _landscape_records(landscape: pd.DataFrame) -> list:
    """Serialize the landscape DataFrame to JSONB-ready dicts (native floats)."""
    return [
        {
            "price":      float(row.price),
            "gex_total":  float(row.gex_total),
            "gex_0dte":   float(row.gex_0dte),
            "gex_near":   float(row.gex_near),
            "gex_med":    float(row.gex_med),
            "gex_struct": float(row.gex_struct),
        }
        for row in landscape.itertuples(index=False)
    ]


def _walls_records(walls: pd.DataFrame) -> list:
    """Serialize the walls DataFrame to JSONB-ready dicts. sign is cast to int."""
    return [
        {
            "price":      float(row.price),
            "gex":        float(row.gex),
            "prominence": float(row.prominence),
            "sign":       int(row.sign),
        }
        for row in walls.itertuples(index=False)
    ]


def _peaks_records(peaks_by_bucket: dict) -> dict:
    """Serialize find_peaks_per_bucket output to JSONB-ready native floats."""
    return {
        label: [
            {
                "price":      float(p["price"]),
                "gex":        float(p["gex"]),
                "prominence": float(p["prominence"]),
                "fwhm":       float(p["fwhm"]),
            }
            for p in peaks
        ]
        for label, peaks in peaks_by_bucket.items()
    }


def compute_and_upsert_landscape(
    conn,
    ticker: str,
    trade_date: dt.date,
    *,
    spread_coef: float = 8.0,
    range_pts: float = 200.0,
    step_pts: float = 1.0,
    version: str,
) -> dict:
    """
    Compute the spot-agnostic GEX landscape for (ticker, trade_date) and UPSERT
    it into orats_gex_landscape.

    Reads the orats_oi_gamma rows already visible on `conn` (same filter the
    Phase 0 script uses: live expirations only, discounted_level not null),
    builds the Gaussian landscape grid, the walls, and the per-bucket peaks,
    and writes one orats_gex_landscape row.

    Runs entirely on the caller's connection and transaction — it issues no
    COMMIT. The EOD cron passes the same connection it used for the
    orats_oi_gamma upsert, so the landscape write and that upsert succeed or
    roll back together. The backfill script commits per date itself.

    stock_price from the first orats_oi_gamma row is stored as table_spot
    (reference metadata) and used as the landscape grid center; it is NOT an
    analytical spot — regime/confluence classification happens at request time
    against a caller-supplied spot (see the spot-agnostic-storage ADR).

    Raises ValueError if no orats_oi_gamma rows exist for (ticker, trade_date)
    or if their stock_price is NULL.

    Returns a summary dict: ticker, trade_date, n_landscape, n_walls,
    n_peaks_by_bucket, table_spot, version.
    """
    from psycopg.types.json import Jsonb

    with conn.cursor() as cur:
        cur.execute(_LANDSCAPE_QUERY, (ticker, trade_date, trade_date))
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]

    if not rows:
        raise ValueError(
            f"compute_and_upsert_landscape: no orats_oi_gamma rows for "
            f"({ticker!r}, {trade_date})"
        )

    df = pd.DataFrame(rows, columns=cols)

    raw_spot = df["stock_price"].iloc[0]
    if raw_spot is None or pd.isna(raw_spot):
        raise ValueError(
            f"compute_and_upsert_landscape: stock_price is NULL for "
            f"({ticker!r}, {trade_date})"
        )
    table_spot = float(raw_spot)

    landscape = compute_landscape(
        df, table_spot,
        range_pts=range_pts, step_pts=step_pts, spread_coef=spread_coef,
    )
    walls = find_walls(landscape)
    peaks_by_bucket = find_peaks_per_bucket(landscape)

    landscape_json = _landscape_records(landscape)
    walls_json = _walls_records(walls)
    peaks_json = _peaks_records(peaks_by_bucket)

    with conn.cursor() as cur:
        cur.execute(_UPSERT_SQL, (
            ticker, trade_date,
            Jsonb(landscape_json), Jsonb(walls_json), Jsonb(peaks_json),
            spread_coef, range_pts, step_pts, table_spot, version,
        ))

    return {
        "ticker":            ticker,
        "trade_date":        trade_date,
        "n_landscape":       len(landscape_json),
        "n_walls":           len(walls_json),
        "n_peaks_by_bucket": {k: len(v) for k, v in peaks_json.items()},
        "table_spot":        table_spot,
        "version":           version,
    }
