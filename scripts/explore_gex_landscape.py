#!/usr/bin/env python3
"""
explore_gex_landscape.py
========================

Phase 0 prototype for the GEX landscape visualization. NO DB writes, NO cron
changes, NO frontend touch. Reads orats_oi_gamma, computes the landscape via
DTE-scaled Gaussian spread, finds walls + containment zones, writes PNGs to
outputs/ for visual validation.

Goal: confirm the continuous landscape adds signal beyond the discrete GEX
lines currently rendered, BEFORE investing in the cron/endpoint/frontend build.

Pipeline
--------
orats_oi_gamma (per-strike, per-expiration, in $-space):
    gex_call = gamma × S² × call_oi × 100
    gex_put  = gamma × S² × put_oi  × 100
    discounted_level = strike × exp((r-q) × (dte+1)/252)

The script:
1.  Queries every row for trade_date where expir_date >= trade_date.
2.  For each candidate spot price S in [spot ± range_pts]:
        GEX(S) = Σ over rows [ net_gex × exp(-(S - level)² / (2 × σ²)) ]
    where σ = spread_coef × sqrt(max(dte, 0.5)).
    Net_gex = gex_call - |gex_put|  (matches frontend convention).
3.  Decomposes the landscape into 4 DTE buckets for the stacked view.
4.  scipy.signal.find_peaks on |landscape| → walls (positive + negative).
5.  For each adjacent pair of POSITIVE walls → containment zone with
    score = (min_strength × valley_depth) / width.

Env vars (reads .env from repo root):
    DATABASE_URL

Usage
-----
    # Single date:
    python scripts/explore_gex_landscape.py --date 2026-05-19

    # Multiple dates:
    python scripts/explore_gex_landscape.py --dates 2026-05-15,2026-05-16,2026-05-19

    # Tune the Gaussian spread coefficient (default 8.0):
    #   c=6  → tighter peaks (0DTE ~4.2pt, 30DTE ~33pt)
    #   c=8  → balanced     (0DTE ~5.7pt, 30DTE ~44pt)
    #   c=12 → broader      (0DTE ~8.5pt, 30DTE ~66pt)
    python scripts/explore_gex_landscape.py --date 2026-05-19 --spread-coef 6

    # Skip cache (force re-query):
    python scripts/explore_gex_landscape.py --date 2026-05-19 --no-cache

Outputs per date:
    outputs/landscape_<date>_compare.png  — side-by-side: GEX lines vs landscape
    outputs/landscape_<date>_stacked.png  — landscape decomposed by DTE bucket

The script prints structured stdout for each date so you can scan results
without opening every PNG. Walls and the active containment zone are printed.

Iteration loop
--------------
1. Run on 5-10 days you remember (a heavy-pin day, a trend day, an FOMC day, etc).
2. Look at the right panel of each *_compare.png. Is the wall structure obvious?
   Does the active zone bracket where price actually traded?
3. If walls are too "spiky" → bump spread_coef. If they're too washed out → lower it.
4. Once one coefficient looks good across diverse days → proceed to Phase 1 (cron).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ─── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
CACHE_DIR  = SCRIPT_DIR / ".cache"
OUTPUT_DIR = REPO_ROOT / "outputs"


# ─── Env loading (matches gamma_pin_study.py pattern) ───────────────────────

def load_env():
    for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        env_path = parent / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv  # type: ignore
                load_dotenv(env_path)
                return env_path
            except ImportError:
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
                return env_path
    return None


# ─── DB query ───────────────────────────────────────────────────────────────

def _make_engine():
    from sqlalchemy import create_engine
    from sqlalchemy.engine.url import make_url

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set")
    url = make_url(db_url)
    if url.get_backend_name() == "postgresql" and url.get_driver_name() in (None, "", "psycopg2"):
        url = url.set(drivername="postgresql+psycopg")
    return create_engine(url, pool_pre_ping=True)


def fetch_strike_data(trade_date: dt.date, ticker: str, use_cache: bool) -> pd.DataFrame:
    """
    Pull per-strike, per-expiration GEX rows for one trade_date.

    Matches the filter convention from _fetch_gex_grouped_by_level in
    apps/web/modules/Ironbeam/callbacks.py (expir_date >= trade_date) so we
    work with live contracts only — no already-expired snapshot residue.

    Returns columns: discounted_level, strike, expir_date, dte, stock_price,
                     call_oi, put_oi, gamma, gex_call, gex_put
    """
    from sqlalchemy import text

    cache_key = CACHE_DIR / f"oi_gamma_full_{ticker}_{trade_date.isoformat()}.parquet"
    if use_cache and cache_key.exists():
        try:
            return pd.read_parquet(cache_key)
        except Exception:
            pass  # fall through to re-query

    engine = _make_engine()
    sql = text("""
        SELECT
            discounted_level,
            strike,
            expir_date,
            dte,
            stock_price,
            call_oi,
            put_oi,
            gamma,
            gex_call,
            gex_put
        FROM orats_oi_gamma
        WHERE ticker = :tkr
          AND trade_date = :d
          AND expir_date >= :d
          AND discounted_level IS NOT NULL
        ORDER BY expir_date, discounted_level
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"tkr": ticker, "d": trade_date.isoformat()})

    if use_cache and not df.empty:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(cache_key)
        except Exception as e:
            print(f"  [cache] save failed: {e}", file=sys.stderr)

    return df


# ─── Landscape computation ──────────────────────────────────────────────────

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


def classify_confluence_quality(score: float, avg_fwhm: float) -> str:
    """
    Tag a confluence with a quality grade.

    Score is the raw (n_buckets × max_gex / fwhm) value in $/pt. Divide by 1e9
    for B/pt. Empirical thresholds based on observed days:
        5/7 pin-grade : score ≈ 84 B/pt, fwhm ≈ 25pt
        5/20 waypoint : score ≈  3 B/pt, fwhm ≈ 84pt

    pin-grade   — Sharp, multi-bucket, strong magnet. Price likely to stick.
    drift-grade — Moderate strength or breadth. Transitional / waypoint level.
    waypoint    — Soft level, easily passed through; just a feature of the
                  field, not a pin candidate.
    """
    score_b_per_pt = score / 1e9
    if score_b_per_pt >= 30 and avg_fwhm < 40:
        return "pin-grade"
    if score_b_per_pt < 5 or avg_fwhm > 80:
        return "waypoint"
    return "drift-grade"


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
        "quality":       classify_confluence_quality(score, avg_fwhm),
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


# ─── Plotting ───────────────────────────────────────────────────────────────

# Dark theme matching the dashboard
BG_OUTER = "#0b1220"
BG_INNER = "#1f2937"
GRID     = "#475569"
TEXT_HI  = "#f1f5f9"
TEXT_MID = "#cbd5e1"
TEXT_LO  = "#94a3b8"
SPOT_C   = "#fbbf24"  # amber
POS_C    = "#fbbf24"  # warm — positive GEX (contains)
NEG_C    = "#06b6d4"  # cool — negative GEX (amplifies)
ACTIVE_C = "#10b981"  # green — active containment zone


def _style_axis(ax):
    ax.set_facecolor(BG_INNER)
    ax.tick_params(colors=TEXT_LO)
    for s in ax.spines.values():
        s.set_color(GRID)


def plot_compare(
    landscape: pd.DataFrame,
    df_strikes: pd.DataFrame,
    walls: pd.DataFrame,
    zones: pd.DataFrame,
    spot: float,
    trade_date: dt.date,
    spread_coef: float,
    out_path: Path,
    regime: Optional[dict] = None,
):
    """
    Side-by-side: left = discrete GEX lines (the current viz, but unfiltered
    so you see everything the cron has), right = continuous landscape with
    walls and zones marked. Shared Y axis.

    If a regime dict is passed, a single-line summary is added as a subtitle.
    """
    fig, (ax_lines, ax_land) = plt.subplots(
        1, 2, figsize=(15, 10),
        gridspec_kw={"width_ratios": [1, 1.4], "wspace": 0.03},
        sharey=True,
    )
    fig.patch.set_facecolor(BG_OUTER)
    _style_axis(ax_lines)
    _style_axis(ax_land)

    y_min, y_max = landscape["price"].min(), landscape["price"].max()

    # ─── Left: discrete GEX lines, ALL strikes (not the filtered subset) ───
    # Aggregate per integer level (matches the cron's bucket=1)
    by_level = (
        df_strikes.assign(level=df_strikes["discounted_level"].round().astype(int))
        .groupby("level")
        .agg(
            call_gamma=("gex_call", "sum"),
            put_gamma=("gex_put", lambda x: -np.abs(x).sum()),
        )
        .reset_index()
    )
    by_level["net"] = by_level["call_gamma"] + by_level["put_gamma"]
    by_level = by_level[(by_level["level"] >= y_min) & (by_level["level"] <= y_max)]

    max_abs = float(by_level["net"].abs().max()) if len(by_level) else 1.0
    max_abs = max(max_abs, 1.0)

    for _, r in by_level.iterrows():
        net = float(r["net"])
        norm = min(1.0, abs(net) / max_abs)
        color = POS_C if net > 0 else NEG_C
        lw    = 0.4 + 3.6 * norm
        alpha = 0.15 + 0.70 * norm
        ax_lines.axhline(r["level"], color=color, linewidth=lw, alpha=alpha)

    ax_lines.axhline(spot, color=SPOT_C, linewidth=1.6, linestyle="--",
                     label=f"Spot {spot:.1f}", zorder=10)
    ax_lines.set_xlim(0, 1)
    ax_lines.set_xticks([])
    ax_lines.set_title(
        "Current viz · GEX lines\n(all strikes, unfiltered)",
        color=TEXT_MID, fontsize=11, pad=10,
    )
    ax_lines.set_ylabel("Price (discounted)", color=TEXT_MID)
    ax_lines.legend(loc="upper left", facecolor=BG_OUTER, edgecolor=GRID,
                    labelcolor=TEXT_MID, fontsize=9, framealpha=0.9)

    # ─── Right: landscape profile (proposed viz) ───
    price = landscape["price"].to_numpy()
    gex_b = landscape["gex_total"].to_numpy() / 1e9

    pos_mask = gex_b >= 0
    neg_mask = gex_b < 0

    # Use np.where to safely fill (preserve NaN where mask is false)
    pos_vals = np.where(pos_mask, gex_b, 0.0)
    neg_vals = np.where(neg_mask, gex_b, 0.0)

    ax_land.fill_betweenx(price, 0, pos_vals, color=POS_C, alpha=0.55,
                          label="positive GEX (contains)")
    ax_land.fill_betweenx(price, neg_vals, 0, color=NEG_C, alpha=0.55,
                          label="negative GEX (amplifies)")
    ax_land.plot(gex_b, price, color=TEXT_HI, linewidth=1.0, alpha=0.85)
    ax_land.axvline(0, color=GRID, linewidth=0.6, alpha=0.7)

    # Containment zone highlights — top 3, with active one outlined
    xlim_right = float(np.nanmax(np.abs(gex_b))) * 1.10 if len(gex_b) else 1.0
    xlim_right = max(xlim_right, 0.1)
    ax_land.set_xlim(-xlim_right, xlim_right)

    if not zones.empty:
        for i, z in zones.head(3).iterrows():
            is_active = bool(z["contains_spot"])
            color = ACTIVE_C if is_active else GRID
            alpha = 0.22 if is_active else 0.08
            ax_land.axhspan(z["lower_price"], z["upper_price"],
                            color=color, alpha=alpha, zorder=1)
            label = "ACTIVE" if is_active else f"#{i + 1}"
            ax_land.text(
                xlim_right * 0.97,
                (z["lower_price"] + z["upper_price"]) / 2,
                f"{label}  w={z['width_pts']:.0f}pt",
                ha="right", va="center",
                color=color, fontsize=8.5, fontweight="bold", alpha=0.95,
            )
            if is_active:
                ax_land.axhline(z["equilibrium_price"], color=ACTIVE_C,
                                linewidth=1.0, linestyle=":", alpha=0.85,
                                zorder=2)

    # Wall markers
    for _, w in walls.iterrows():
        gex_at_wall = float(w["gex"]) / 1e9
        marker_color = POS_C if w["sign"] > 0 else NEG_C
        ax_land.scatter([gex_at_wall], [w["price"]],
                        color=marker_color, edgecolor=BG_OUTER,
                        s=70, zorder=5, linewidths=1.2)

    ax_land.axhline(spot, color=SPOT_C, linewidth=1.6, linestyle="--", zorder=10)
    ax_land.set_title(
        f"Proposed viz · GEX landscape\n(continuous · walls · containment zones)",
        color=TEXT_MID, fontsize=11, pad=10,
    )
    ax_land.set_xlabel("GEX ($B)", color=TEXT_MID)
    ax_land.legend(loc="lower right", facecolor=BG_OUTER, edgecolor=GRID,
                   labelcolor=TEXT_MID, fontsize=9, framealpha=0.9)

    fig.suptitle(
        f"GEX landscape · SPX · {trade_date.isoformat()} · spread_coef={spread_coef}",
        color=TEXT_HI, fontsize=13, fontweight="bold", y=0.995,
    )

    if regime is not None:
        # Compact one-line summary under the suptitle, color-coded by regime
        regime_tag = regime.get("regime", "?")
        regime_colors = {
            "magnetic-pin":  ACTIVE_C,
            "magnet-above":  "#fbbf24",
            "magnet-below":  "#fbbf24",
            "bounded":       ACTIVE_C,
            "amplification": NEG_C,
            "broken-magnet": "#a78bfa",  # purple — to stand out as a momentum regime
            "untethered":    TEXT_LO,
        }
        col = regime_colors.get(regime_tag, TEXT_MID)

        # Build short summary string
        bits = [f"REGIME: {regime_tag}"]
        if "crossing_direction" in regime:
            bits.append(
                f"crossed wall at {regime['dominant_wall']['price']:.0f} "
                f"({regime.get('prior_spot', 0):.0f} → {regime['spot']:.0f}, "
                f"{regime['crossing_direction']})"
            )
        elif "drift_target" in regime and "drift_direction" in regime:
            bits.append(
                f"→ target {regime['drift_target']:.0f} "
                f"(drift {regime['drift_direction']} "
                f"{abs(regime['spot'] - regime['drift_target']):.0f}pt)"
            )
        if "containment_zone" in regime:
            z = regime["containment_zone"]
            bits.append(
                f"→ zone {z['lower_price']:.0f}–{z['upper_price']:.0f} "
                f"(eq {z['equilibrium']:.0f}, w {z['width_pts']:.0f}pt)"
            )
        if "amplification_zone" in regime:
            z = regime["amplification_zone"]
            bits.append(f"⚠ neg zone at {z['price']:.0f} ({z['distance']:.0f}pt away)")
        summary = "  ·  ".join(bits)

        fig.text(
            0.5, 0.955, summary,
            color=col, fontsize=11, fontweight="bold",
            ha="center", va="top",
        )

    plt.savefig(out_path, dpi=130, facecolor=BG_OUTER, bbox_inches="tight")
    plt.close(fig)


def plot_stacked(
    landscape: pd.DataFrame,
    spot: float,
    trade_date: dt.date,
    out_path: Path,
    per_bucket: Optional[dict] = None,
    confluences: Optional[list] = None,
):
    """
    Single panel: landscape decomposed by DTE bucket. Shows which expirations
    drive each wall. 0DTE = sharp/intraday, structural = broad/multi-week.

    If per_bucket is provided, legend labels include dominance % and regime
    tag so the eye-roll mapping "which timeframe controls today" is immediate.

    If confluences is provided, horizontal dashed lines mark levels where
    multiple buckets have peaks at the same price (confluence points are
    structurally stronger than any single bucket's peak).
    """
    fig, ax = plt.subplots(figsize=(9, 10))
    fig.patch.set_facecolor(BG_OUTER)
    _style_axis(ax)

    price = landscape["price"].to_numpy()

    layers = [
        ("0DTE",      landscape["gex_0dte"].to_numpy()   / 1e9, "#ef4444", "gex_0dte"),
        ("1-7 DTE",   landscape["gex_near"].to_numpy()   / 1e9, "#f59e0b", "gex_near"),
        ("8-30 DTE",  landscape["gex_med"].to_numpy()    / 1e9, "#10b981", "gex_med"),
        ("30+ DTE",   landscape["gex_struct"].to_numpy() / 1e9, "#3b82f6", "gex_struct"),
    ]

    # Identify dominant bucket if per_bucket info is present
    dominant_label = None
    if per_bucket is not None:
        dominant_label = max(per_bucket.items(),
                            key=lambda x: x[1]["dominance_pct"])[0]

    for name, vals, color, _col in layers:
        # If per_bucket is given, augment the label with dominance + regime
        if per_bucket is not None and name in per_bucket:
            r = per_bucket[name]
            marker = "●" if name == dominant_label else " "
            label = (
                f"{marker} {name:8s} {r['dominance_pct']:>4.0f}% · "
                f"{r['regime']}"
            )
            # Bolder line for dominant bucket
            lw = 2.6 if name == dominant_label else 1.4
            alpha_line = 1.0 if name == dominant_label else 0.75
        else:
            label = name
            lw = 1.6
            alpha_line = 0.9

        ax.plot(vals, price, color=color, linewidth=lw, label=label,
                alpha=alpha_line)
        ax.fill_betweenx(price, 0, vals, color=color, alpha=0.15)

    ax.plot(landscape["gex_total"].to_numpy() / 1e9, price,
            color=TEXT_HI, linewidth=1.0, linestyle="--",
            alpha=0.55, label="total")

    ax.axvline(0, color=GRID, linewidth=0.6, alpha=0.7)
    ax.axhline(spot, color=SPOT_C, linewidth=1.6, linestyle="--",
               label=f"Spot {spot:.1f}", zorder=10)

    # Confluence lines: horizontal markers at price levels where multiple
    # buckets agree on a peak. Width and opacity scale with n_buckets;
    # line style reflects quality grade.
    if confluences:
        xlim = ax.get_xlim()
        x_right = xlim[1]
        for i, c in enumerate(confluences[:4]):
            n = c["n_buckets"]
            # Color from confluence strength: 2 buckets = yellow, 3 = orange, 4 = bright green
            colors = {2: "#fbbf24", 3: "#fb923c", 4: ACTIVE_C}
            line_color = colors.get(n, ACTIVE_C)
            line_alpha = 0.35 + 0.20 * (n - 2)
            # Line style by quality: pin-grade solid, drift-grade dashed, waypoint dotted
            quality = c.get("quality", "waypoint")
            style_map = {"pin-grade": "-", "drift-grade": "--", "waypoint": ":"}
            line_style = style_map.get(quality, ":")
            line_width = {"pin-grade": 2.0, "drift-grade": 1.5, "waypoint": 1.0}.get(
                quality, 1.0
            )
            ax.axhline(c["center_price"], color=line_color,
                       linewidth=line_width, linestyle=line_style,
                       alpha=line_alpha, zorder=8)
            # Annotation on the right edge — small but readable
            quality_short = {"pin-grade": "PIN", "drift-grade": "DRIFT",
                             "waypoint": "soft"}.get(quality, "")
            ax.text(
                x_right * 0.985,
                c["center_price"],
                f"{c['center_price']:.0f}  {'★' * n}  {quality_short}",
                ha="right", va="center",
                color=line_color, fontsize=8.5, fontweight="bold",
                alpha=0.9,
                bbox=dict(facecolor=BG_OUTER, edgecolor="none", pad=2, alpha=0.7),
            )

    ax.set_xlabel("GEX ($B) — contribution by DTE bucket", color=TEXT_MID)
    ax.set_ylabel("Price (discounted)", color=TEXT_MID)

    ax.legend(loc="upper right", facecolor=BG_OUTER, edgecolor=GRID,
              labelcolor=TEXT_MID, fontsize=10, framealpha=0.9,
              prop={"family": "monospace"})

    fig.suptitle(
        f"GEX landscape by DTE bucket · SPX · {trade_date.isoformat()}",
        color=TEXT_HI, fontsize=13, fontweight="bold", y=0.995,
    )

    plt.savefig(out_path, dpi=130, facecolor=BG_OUTER, bbox_inches="tight")
    plt.close(fig)


# ─── Per-date orchestration ─────────────────────────────────────────────────

def run_one_date(trade_date: dt.date, args, out_dir: Path):
    print(f"\n=== {trade_date.isoformat()} ===")

    df = fetch_strike_data(trade_date, args.ticker, use_cache=not args.no_cache)
    if df.empty:
        print(f"  [skip] no orats_oi_gamma rows for ticker={args.ticker} trade_date={trade_date}")
        return

    table_spot = float(df["stock_price"].dropna().iloc[0])
    prior_spot: Optional[float] = None
    if args.spot is not None:
        spot = float(args.spot)
        prior_spot = table_spot   # used for broken-magnet detection
        spot_source = f"override (table reference: {table_spot:.2f})"
    else:
        spot = table_spot
        spot_source = f"orats_oi_gamma stock_price (SPX cash, prior EOD)"

    # Implied move resolution: --implied-move takes priority, then --iv
    implied_move = 0.0
    iv_source = ""
    if args.implied_move is not None:
        implied_move = float(args.implied_move)
        iv_source = f"--implied-move {implied_move:.1f}pt"
    elif args.iv is not None:
        implied_move = compute_implied_move(spot, float(args.iv))
        iv_source = f"--iv {args.iv:.4f} → {implied_move:.1f}pt (1σ daily)"
    n_expir = df["expir_date"].nunique()
    n_strikes = df["strike"].nunique()
    print(f"  spot:        {spot:.2f}  [{spot_source}]")
    if args.spot is None:
        print(f"  note:        levels are ES-forward; consider --spot <actual ES price>")
    if implied_move > 0:
        print(f"  implied:     ±{implied_move:.1f}pt 1σ  [{iv_source}]")
        print(f"               in-range:       {spot - implied_move:.0f} → {spot + implied_move:.0f}  (1σ)")
        print(f"               intraday reach: {spot - 1.5*implied_move:.0f} → {spot + 1.5*implied_move:.0f}  (1.5σ)")
    else:
        print(f"  implied:     not provided — targets won't be classified by realistic range")
        print(f"               (pass --implied-move or --iv to enable)")
    print(f"  rows:        {len(df):,} (strikes × expirations)")
    print(f"  expirations: {n_expir}")
    print(f"  strikes:     {n_strikes}")
    print(f"  dte range:   {int(df['dte'].min())} – {int(df['dte'].max())}")

    landscape = compute_landscape(
        df, spot,
        range_pts=args.range_pts,
        step_pts=args.step_pts,
        spread_coef=args.spread_coef,
    )

    walls = find_walls(landscape, min_prominence_pct=args.prominence)
    zones = score_containment_zones(walls, landscape, spot)
    regime = classify_regime(landscape, spot, prior_spot=prior_spot,
                             implied_move=implied_move)
    regime = _annotate_distance_class(regime, implied_move)
    per_bucket = classify_per_bucket(landscape, spot, prior_spot=prior_spot,
                                     implied_move=implied_move)
    bucket_summary = summarize_per_bucket(per_bucket)
    confluence_result = analyze_confluence(landscape)

    n_pos = int((walls["sign"] > 0).sum()) if not walls.empty else 0
    n_neg = int((walls["sign"] < 0).sum()) if not walls.empty else 0
    print(f"  walls:       {len(walls)}  (pos={n_pos}, neg={n_neg})")
    if not walls.empty:
        pos_walls = walls[walls["sign"] > 0].nlargest(5, "gex")
        for _, w in pos_walls.iterrows():
            print(f"    + {w['price']:>6.0f}   gex={w['gex']/1e9:+7.1f}B")
        neg_walls = walls[walls["sign"] < 0].nsmallest(3, "gex")
        for _, w in neg_walls.iterrows():
            print(f"    – {w['price']:>6.0f}   gex={w['gex']/1e9:+7.1f}B")

    print(f"  zones:       {len(zones)}")
    if not zones.empty:
        active = zones[zones["contains_spot"]]
        if not active.empty:
            z = active.iloc[0]
            print(
                f"  active:      {z['lower_price']:.0f} → {z['upper_price']:.0f}  "
                f"(width {z['width_pts']:.0f}pt, "
                f"eq {z['equilibrium_price']:.1f}, "
                f"spot−eq {spot - z['equilibrium_price']:+.1f})"
            )
        else:
            print(f"  active:      none — spot {spot:.1f} not inside any positive-wall pair")
        print("  top zones by containment score:")
        for i, z in zones.head(3).iterrows():
            marker = "*" if z["contains_spot"] else " "
            print(
                f"    {marker} #{i + 1}: {z['lower_price']:.0f}–{z['upper_price']:.0f}  "
                f"w={z['width_pts']:>3.0f}pt  "
                f"min_strength={min(z['lower_gex'], z['upper_gex'])/1e9:.1f}B  "
                f"score={z['containment_score']:.2e}"
            )

    # Regime classification — the high-level "what kind of day is this" summary
    print()
    print(f"  REGIME:      {regime['regime'].upper()}")
    for note in regime.get("notes", []):
        # Wrap long notes to 80 chars-ish for terminal readability
        words = note.split()
        line_chars = 0
        line_parts = []
        for w in words:
            if line_chars + len(w) > 72:
                print(f"               {' '.join(line_parts)}")
                line_parts = [w]
                line_chars = len(w)
            else:
                line_parts.append(w)
                line_chars += len(w) + 1
        if line_parts:
            print(f"               {' '.join(line_parts)}")
    if "drift_target" in regime:
        cls_info = regime.get("target_classification", {})
        cls_str = ""
        if cls_info.get("sigma") is not None:
            cls_str = f"  [{cls_info['class']}, {cls_info['sigma']:.1f}σ]"
        print(
            f"               target: {regime['drift_target']:.1f}  "
            f"(direction: {regime.get('drift_direction', '?')}, "
            f"distance: {abs(regime['spot'] - regime['drift_target']):.0f}pt)"
            f"{cls_str}"
        )
    if "containment_zone" in regime:
        z = regime["containment_zone"]
        print(
            f"               zone:   {z['lower_price']:.0f} → {z['upper_price']:.0f}  "
            f"(eq {z['equilibrium']:.1f}, width {z['width_pts']:.0f}pt)"
        )
    if "amplification_zone" in regime:
        z = regime["amplification_zone"]
        print(
            f"               neg:    {z['price']:.0f} "
            f"({z['gex']/1e9:+.0f}B, {z['distance']:.0f}pt away)"
        )
    if "crossing_direction" in regime:
        print(
            f"               crossed: {regime.get('prior_spot', 0):.0f} → "
            f"{regime['spot']:.0f} through wall at {regime['dominant_wall']['price']:.0f}  "
            f"(direction: {regime['crossing_direction']}, "
            f"distance past: {regime['distance_past_magnet']:.0f}pt)"
        )

    # Per-DTE-bucket breakdown — which timeframe is in control?
    print()
    print(f"  PER-DTE BREAKDOWN  (intensity = sum of |GEX| within ±30pt of spot):")
    for label, r in per_bucket.items():
        marker = "●" if label == bucket_summary["primary_bucket"] else " "
        # Compact one-line summary per bucket
        extras = []
        if "drift_target" in r:
            extras.append(f"target {r['drift_target']:.0f}")
        if "containment_zone" in r:
            z = r["containment_zone"]
            extras.append(f"zone {z['lower_price']:.0f}–{z['upper_price']:.0f}")
        if "amplification_zone" in r:
            z = r["amplification_zone"]
            extras.append(f"neg {z['price']:.0f}")
        if "crossing_direction" in r:
            extras.append(f"crossed {r['crossing_direction']}")
        extra_str = (" → " + ", ".join(extras)) if extras else ""

        print(
            f"   {marker} {label:8s}  "
            f"dom={r['dominance_pct']:>5.1f}%  "
            f"intensity={r['local_intensity']/1e9:>6.1f}B  "
            f"net={r['local_net']/1e9:>+6.1f}B  "
            f"regime={r['regime']}"
            f"{extra_str}"
        )

    print()
    print(
        f"  DOMINANT BUCKET: {bucket_summary['primary_bucket']} "
        f"({bucket_summary['primary_dominance']:.0f}%) — "
        f"regime: {bucket_summary['primary_regime']}"
    )
    if bucket_summary["secondary_dominance"] >= 15.0:
        print(
            f"  RUNNER-UP:       {bucket_summary['secondary_bucket']} "
            f"({bucket_summary['secondary_dominance']:.0f}%) — "
            f"regime: {bucket_summary['secondary_regime']}"
        )
    if not bucket_summary["consensus"] and len(bucket_summary["meaningful_regimes"]) > 1:
        print(
            f"  DIVERGENCE:      meaningful buckets disagree — "
            f"{' / '.join(bucket_summary['meaningful_regimes'])}"
        )
        print(
            f"                   (intraday signal may diverge from structural — "
            f"the dominant bucket usually controls; watch for regime shift if "
            f"price moves out of dominant bucket's range)"
        )

    # Confluence: peaks where multiple DTE buckets agree on the same price level
    confluences = confluence_result["confluences"]
    print()
    if confluences:
        print(f"  CONFLUENCE  (peaks aligned across buckets — confluence = stronger pin):")
        for i, c in enumerate(confluences[:5]):
            marker = "●" if i == 0 else " "
            stars = "★" * c["n_buckets"]
            buckets_str = ", ".join(b.replace(" DTE", "").replace("DTE", "0d")
                                    for b in c["buckets"])
            distance = c["center_price"] - spot
            # Add distance classification if implied_move available
            if implied_move > 0:
                cls = classify_distance(abs(distance), implied_move)
                cls_tag = f"  [{cls['class']}, {cls['sigma']:.1f}σ]"
            else:
                cls_tag = ""
            quality_tag = f"  [{c['quality']}]"
            print(
                f"   {marker} {c['center_price']:>6.0f}  {stars:<4s}  "
                f"({c['n_buckets']} buckets: {buckets_str})  "
                f"max={c['max_gex']/1e9:>5.0f}B  "
                f"fwhm={c['avg_fwhm']:>4.0f}pt  "
                f"score={c['score']/1e9:.1f}B/pt"
                f"{quality_tag}  "
                f"[{distance:+.0f}pt from spot]{cls_tag}"
            )
    else:
        print(f"  CONFLUENCE:   none — no peaks aligned across multiple buckets")
        print(f"                (a no-confluence day means no clear pin level — "
              f"contributing to chop/uncertainty)")

    # Intraday subtarget — when primary target is structural, point to a closer
    # actionable level within ~1.5σ that's in the same direction.
    if implied_move > 0 and "drift_target" in regime:
        target_cls = regime.get("target_classification", {}).get("class", "")
        if target_cls in ("stretch", "multi-day", "far"):
            direction = regime.get("drift_direction")
            subtarget = find_intraday_subtarget(
                confluences, walls, spot, implied_move,
                max_sigma=1.5, direction=direction,
            )
            print()
            if subtarget:
                print(
                    f"  INTRADAY ACTIONABLE:  target {subtarget['price']:.0f}  "
                    f"({subtarget['type']}, {subtarget['sigma_distance']:.1f}σ from spot)"
                )
                print(
                    f"                        the {regime['drift_target']:.0f} magnet is "
                    f"{regime['target_classification']['sigma']:.1f}σ away "
                    f"({target_cls}) — that's a structural pull, not today's range. "
                    f"Today, watch {subtarget['price']:.0f}."
                )
            else:
                print(
                    f"  INTRADAY ACTIONABLE:  no significant level within "
                    f"1.5σ ({1.5*implied_move:.0f}pt) in the direction of drift"
                )
                print(
                    f"                        the {regime['drift_target']:.0f} magnet is "
                    f"structural pull only — today's range likely stays within "
                    f"{spot-1.5*implied_move:.0f}–{spot+1.5*implied_move:.0f} unless catalyst"
                )

    # High-volatility zones: significant negative GEX walls within 1.5σ.
    # These aren't magnets but they ARE high-attention strikes that act as
    # accelerators or bounce candidates depending on arrival conditions.
    if implied_move > 0:
        dom_strength_for_neg = (
            regime["dominant_wall"]["gex"]
            if "dominant_wall" in regime else None
        )
        neg_zones = find_proximate_negative_zones(
            walls, spot, implied_move,
            dom_strength=dom_strength_for_neg,
        )
        if neg_zones:
            print()
            print(f"  HIGH-VOL ZONES  (negative GEX within 1.5σ — accelerators, not magnets):")
            for z in neg_zones[:3]:
                arrow = "↓" if z["direction"] == "below" else "↑"
                print(
                    f"   ⚠ {z['price']:>6.0f} ({z['gex']/1e9:+5.0f}B, "
                    f"{z['distance']:.0f}pt {z['direction']} spot, "
                    f"{z['sigma']:.1f}σ)  {arrow}"
                )
                # Conditional arrival framing
                if z["direction"] == "below":
                    print(f"               grind-down approach → acceleration through "
                          f"(trapdoor)")
                    print(f"               gap-down to here    → bounce candidate "
                          f"(put profit-taking)")
                else:
                    print(f"               grind-up approach   → acceleration through "
                          f"(squeeze)")
                    print(f"               gap-up to here      → exhaustion / fade "
                          f"candidate")

    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / f"landscape_{trade_date.isoformat()}_compare.png"
    plot_compare(landscape, df, walls, zones, spot, trade_date, args.spread_coef, p1,
                 regime=regime)
    print(f"  → {p1.relative_to(REPO_ROOT) if p1.is_relative_to(REPO_ROOT) else p1}")

    p2 = out_dir / f"landscape_{trade_date.isoformat()}_stacked.png"
    plot_stacked(landscape, spot, trade_date, p2, per_bucket=per_bucket,
                 confluences=confluence_result["confluences"])
    print(f"  → {p2.relative_to(REPO_ROOT) if p2.is_relative_to(REPO_ROOT) else p2}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--date", help="Single trade date YYYY-MM-DD")
    ap.add_argument("--dates", help="Comma-separated list of trade dates")
    ap.add_argument("--ticker", default="SPX")
    ap.add_argument("--spot", type=float, default=None,
                    help="Override the spot used for regime classification and "
                         "plot centering. Default reads stock_price from "
                         "orats_oi_gamma, which is SPX cash from the PRIOR EOD "
                         "(when ORATS pulled the chain) — usually NOT where the "
                         "session you're analyzing actually opens. GEX levels "
                         "are in ES-forward coordinates (discounted_level), so "
                         "pass the actual ES price at the moment you want to "
                         "analyze. E.g. --spot 7400 for 5/7 session open.")
    ap.add_argument("--implied-move", type=float, default=None,
                    help="Today's expected 1-sigma move in points (e.g. ATM "
                         "straddle width). Used to classify each target as "
                         "in-range/intraday-reach/stretch/multi-day/far. "
                         "Strongly recommended — without it, the model can "
                         "report 3-sigma 'targets' that are actually multi-day "
                         "structural pulls, not today's range.")
    ap.add_argument("--iv", type=float, default=None,
                    help="Alternative to --implied-move: ATM IV as decimal "
                         "(e.g. 0.107 for 10.7%%). Implied move computed as "
                         "spot × iv × sqrt(1/252).")
    ap.add_argument("--spread-coef", type=float, default=8.0,
                    help="Gaussian spread coef: σ = c × √DTE  (default 8.0)")
    ap.add_argument("--range-pts", type=float, default=200.0,
                    help="Price range ± from spot (default 200)")
    ap.add_argument("--step-pts", type=float, default=1.0,
                    help="Price step in landscape grid (default 1)")
    ap.add_argument("--prominence", type=float, default=0.10,
                    help="Min peak prominence as fraction of max |GEX| (default 0.10)")
    ap.add_argument("--no-cache", action="store_true",
                    help="Skip parquet cache; force DB re-query")
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR),
                    help=f"Output directory (default {OUTPUT_DIR})")
    args = ap.parse_args()

    if not args.date and not args.dates:
        ap.error("Provide --date or --dates")

    env_path = load_env()
    if env_path:
        print(f"[env] loaded {env_path}")

    if not os.environ.get("DATABASE_URL"):
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    dates: list[dt.date] = []
    if args.date:
        dates.append(dt.date.fromisoformat(args.date))
    if args.dates:
        dates.extend(dt.date.fromisoformat(d.strip()) for d in args.dates.split(",") if d.strip())

    out_dir = Path(args.out_dir)
    print(f"[config] spread_coef={args.spread_coef}  range_pts={args.range_pts}  "
          f"prominence={args.prominence}")

    for d in dates:
        try:
            run_one_date(d, args, out_dir)
        except Exception as e:
            print(f"  [error] {d}: {e}")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()