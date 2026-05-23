"""day_features — per-day signal-layer feature extraction (CR-013, v0.5).

Builds the structured feature vector that drives the Day Analogue Comparison
KNN endpoint. The feature vector summarizes the day's landscape + vol context
(regime tag, top-3 cluster slots σ-normalized by daily implied move, cluster
aggregates, per-bucket dominance, neg zones, implied move) so two days can be
compared by weighted Euclidean distance in feature space.

The module exposes:

    FEATURE_VERSION  — version tag stamped on every persisted row.
    FEATURE_NAMES    — ordered list of the 34 feature keys (29 active + 5
                        deferred vol-surface slots).
    EPSILON          — minimum stddev floor used by the similarity function
                        in the Analogues service to avoid division-by-zero
                        on near-constant features.
    extract_features(landscape_payload, spot, implied_move) -> dict
                        Pure function. Returns dict keyed by FEATURE_NAMES.
    compute_and_upsert_daily_features(conn, ticker, trade_date, *, version)
                        Cron + backfill helper. Loads the stored landscape,
                        materializes the spot-dependent classifier chain
                        (regime / per_bucket / confluences / neg_zones /
                        walls), runs extract_features, upserts the result
                        into bt_daily_features.

Design notes (CR-013 spec, Step 0 reconciliation):

* The stored ``orats_gex_landscape.landscape`` JSONB carries only the
  per-strike grid. ``regime``, ``bucket_summary``, ``confluences``,
  ``neg_zones`` are materialized at API serve time by
  ``build_gex_landscape_response()`` via the classifier chain in
  ``packages/shared/gex_landscape.py``. ``compute_and_upsert_daily_features``
  runs the *same* chain on the stored DataFrame, builds a payload-shaped
  dict, and feeds it to the pure ``extract_features``.

* The five "vol surface placeholder" features (``atm_iv_percentile``,
  ``skew_percentile``, ``smile_convexity``, ``term_structure_slope``,
  ``vol_risk_premium``) are populated as ``None`` in v0.5. The similarity
  function in the Analogues service is NULL-aware: features that are None
  on either side of a comparison are skipped, and the distance is rescaled
  by ``sqrt(n_total_features / n_active_features)``.

* ``broken-magnet`` (one of the seven regime tags emitted by
  ``classify_regime``) is not assigned a binary indicator in v0.5; it maps
  to all-zeros across ``is_*_day``. Adding ``is_broken_magnet_day`` is
  queued as a Phase-2 schema follow-up — none of the four CR-011
  calibration days are broken-magnet.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from typing import Any, Optional

# ─── Version + feature schema ──────────────────────────────────────────────

FEATURE_VERSION = "v0.5.0"

FEATURE_NAMES: list[str] = [
    # Regime / structure (6)
    "is_pin_day",
    "is_magnet_day",
    "is_bounded_day",
    "is_untethered_day",
    "is_amplification_day",
    "magnet_direction_signed",
    # Top 3 cluster slots (9)
    "cluster_1_max_gex",
    "cluster_1_quality_ordinal",
    "cluster_1_signed_distance_sigma",
    "cluster_2_max_gex",
    "cluster_2_quality_ordinal",
    "cluster_2_signed_distance_sigma",
    "cluster_3_max_gex",
    "cluster_3_quality_ordinal",
    "cluster_3_signed_distance_sigma",
    # Cluster aggregates (7)
    "n_clusters_total",
    "n_pin",
    "n_target",
    "n_feature",
    "n_clusters_above_spot",
    "n_clusters_below_spot",
    "top_cluster_fraction_of_total_max_gex",
    # Per-bucket dominance (4)
    "dominance_0DTE",
    "dominance_1_7",
    "dominance_8_30",
    "dominance_30plus",
    # Neg zones (3)
    "n_neg_zones",
    "nearest_neg_signed_distance_sigma",
    "total_neg_max_gex",
    # Vol regime (1 active)
    "implied_move_1d",
    # Vol surface placeholders (5, NULL in v0.5)
    "atm_iv_percentile",
    "skew_percentile",
    "smile_convexity",
    "term_structure_slope",
    "vol_risk_premium",
]

VOL_SURFACE_PLACEHOLDERS: tuple[str, ...] = (
    "atm_iv_percentile",
    "skew_percentile",
    "smile_convexity",
    "term_structure_slope",
    "vol_risk_premium",
)

EPSILON: float = 1e-6  # similarity-function stddev floor

# ─── Regime → indicator mapping ────────────────────────────────────────────

_REGIME_INDICATORS: dict[str, dict[str, Any]] = {
    "magnetic-pin":   {"is_pin_day": 1, "magnet_direction_signed": 0},
    "magnet-above":   {"is_magnet_day": 1, "magnet_direction_signed": 1},
    "magnet-below":   {"is_magnet_day": 1, "magnet_direction_signed": -1},
    "bounded":        {"is_bounded_day": 1, "magnet_direction_signed": 0},
    "amplification":  {"is_amplification_day": 1, "magnet_direction_signed": 0},
    "untethered":     {"is_untethered_day": 1, "magnet_direction_signed": 0},
    # broken-magnet → all zeros across indicators (see module docstring).
    "broken-magnet":  {"magnet_direction_signed": 0},
}

_QUALITY_ORDINAL: dict[str, int] = {"pin": 2, "target": 1, "feature": 0}

# Bucket label (as emitted by classify_per_bucket) → feature name.
_BUCKET_FEATURE: dict[str, str] = {
    "0DTE":     "dominance_0DTE",
    "1-7 DTE":  "dominance_1_7",
    "8-30 DTE": "dominance_8_30",
    "30+ DTE":  "dominance_30plus",
}


# ─── Pure feature extractor ────────────────────────────────────────────────

def extract_features(
    landscape_payload: dict,
    spot: float,
    implied_move: float,
) -> dict:
    """Build the 34-feature day vector.

    landscape_payload must carry the keys produced by
    ``build_gex_landscape_response`` (or its in-process equivalent
    materialized by ``compute_and_upsert_daily_features``):
    ``regime`` (dict with ``regime`` key), ``per_bucket`` (dict keyed by
    bucket label), ``confluences`` (list with ``center_price`` /
    ``max_gex`` / ``quality``), ``neg_zones`` (list with ``price`` /
    ``gex``).

    Pure: no DB, no time, no global state. Same inputs → same output.
    """
    out: dict[str, Any] = {name: 0 for name in FEATURE_NAMES}
    for name in VOL_SURFACE_PLACEHOLDERS:
        out[name] = None

    # ---- Regime indicators ------------------------------------------------
    regime_block = landscape_payload.get("regime") or {}
    regime_tag = regime_block.get("regime", "untethered")
    mapping = _REGIME_INDICATORS.get(regime_tag, {})
    for k, v in mapping.items():
        out[k] = v

    # ---- Top-3 cluster slots ---------------------------------------------
    confluences = list(landscape_payload.get("confluences") or [])
    confluences.sort(key=lambda c: c.get("max_gex", 0.0), reverse=True)

    for i in range(3):
        slot = i + 1
        if i < len(confluences):
            c = confluences[i]
            center = float(c.get("center_price", 0.0))
            max_gex_b = float(c.get("max_gex", 0.0)) / 1e9
            quality = c.get("quality", "feature")
            if implied_move and implied_move > 0:
                sd = (center - spot) / implied_move
            else:
                sd = 0.0
            out[f"cluster_{slot}_max_gex"] = max_gex_b
            out[f"cluster_{slot}_quality_ordinal"] = _QUALITY_ORDINAL.get(quality, 0)
            out[f"cluster_{slot}_signed_distance_sigma"] = sd
        # else: defaults (0/0/0) from the initial fill.

    # ---- Cluster aggregates ----------------------------------------------
    out["n_clusters_total"] = len(confluences)
    out["n_pin"] = sum(1 for c in confluences if c.get("quality") == "pin")
    out["n_target"] = sum(1 for c in confluences if c.get("quality") == "target")
    out["n_feature"] = sum(1 for c in confluences if c.get("quality") == "feature")
    out["n_clusters_above_spot"] = sum(
        1 for c in confluences if float(c.get("center_price", 0.0)) > spot
    )
    out["n_clusters_below_spot"] = sum(
        1 for c in confluences if float(c.get("center_price", 0.0)) < spot
    )
    total_max_gex_b = sum(float(c.get("max_gex", 0.0)) / 1e9 for c in confluences)
    if confluences and total_max_gex_b > 0:
        out["top_cluster_fraction_of_total_max_gex"] = (
            (float(confluences[0].get("max_gex", 0.0)) / 1e9) / total_max_gex_b
        )
    else:
        out["top_cluster_fraction_of_total_max_gex"] = 0.0

    # ---- Per-bucket dominance --------------------------------------------
    per_bucket = landscape_payload.get("per_bucket") or {}
    for bucket_label, feature_name in _BUCKET_FEATURE.items():
        block = per_bucket.get(bucket_label) or {}
        dominance = block.get("dominance_pct", 0.0)
        try:
            out[feature_name] = float(dominance) if dominance is not None else 0.0
        except (TypeError, ValueError):
            out[feature_name] = 0.0

    # ---- Neg zones -------------------------------------------------------
    neg_zones = list(landscape_payload.get("neg_zones") or [])
    out["n_neg_zones"] = len(neg_zones)
    if neg_zones and implied_move and implied_move > 0:
        # The /api/gex-landscape neg_zones are already sorted by abs(gex) desc.
        # "Nearest" here = closest to spot by distance, per the spec.
        nearest = min(neg_zones,
                      key=lambda z: abs(float(z.get("price", 0.0)) - spot))
        out["nearest_neg_signed_distance_sigma"] = (
            (float(nearest.get("price", 0.0)) - spot) / implied_move
        )
    else:
        out["nearest_neg_signed_distance_sigma"] = 0.0
    out["total_neg_max_gex"] = sum(
        abs(float(z.get("gex", 0.0))) for z in neg_zones
    ) / 1e9

    # ---- Vol regime ------------------------------------------------------
    out["implied_move_1d"] = float(implied_move) if implied_move else 0.0

    return out


# ─── Config hash ───────────────────────────────────────────────────────────

def compute_feature_config_hash(version: str = FEATURE_VERSION) -> str:
    """md5 over the feature schema + version. Bumps when FEATURE_NAMES or the
    version changes. Distinguishes implementation revisions within a version
    (e.g., a follow-up tweak to extract_features under the same v0.5.x line)."""
    payload = {
        "version": version,
        "feature_names": FEATURE_NAMES,
        "vol_surface_placeholders": list(VOL_SURFACE_PLACEHOLDERS),
        "regime_indicators": _REGIME_INDICATORS,
        "quality_ordinal": _QUALITY_ORDINAL,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.md5(blob).hexdigest()


# ─── DB-backed entry point ─────────────────────────────────────────────────

_UPSERT_SQL = """
    INSERT INTO bt_daily_features
        (ticker, trade_date, feature_vector, feature_version,
         feature_config_hash)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (ticker, trade_date, feature_version) DO UPDATE SET
        feature_vector      = EXCLUDED.feature_vector,
        feature_config_hash = EXCLUDED.feature_config_hash,
        computed_at         = NOW()
"""

_LANDSCAPE_ROW_SQL = """
    SELECT landscape, table_spot
    FROM orats_gex_landscape
    WHERE ticker = %s AND trade_date = %s
    LIMIT 1
"""

_IMPLIED_MOVE_SQL = """
    SELECT atmiv, dte
    FROM orats_monies_minute
    WHERE trade_date = %s
      AND ticker = %s
      AND atmiv IS NOT NULL
      AND dte > 0
    ORDER BY snapshot_pt DESC, dte ASC
    LIMIT 1
"""
# orats_monies_minute.trade_date is stored as TEXT (ISO YYYY-MM-DD); we
# pass trade_date.isoformat() rather than the date object so the cast
# happens client-side.


def _materialize_payload(landscape_rows: list, spot: float, implied_move: float) -> dict:
    """Run the spot-dependent classifier chain on a stored landscape grid,
    returning a payload-shaped dict with the keys ``extract_features``
    consumes: ``regime``, ``per_bucket``, ``confluences``, ``neg_zones``.

    Mirrors the chain in ``packages/shared/gex_landscape_api.py``'s
    ``build_gex_landscape_response``. Kept local to avoid a SQLAlchemy
    dependency in the cron path.
    """
    import pandas as pd
    from packages.shared.gex_landscape import (
        analyze_confluence,
        classify_per_bucket,
        classify_regime,
        find_proximate_negative_zones,
        find_walls,
        summarize_per_bucket,
    )

    if not landscape_rows:
        return {
            "regime": {"regime": "untethered"},
            "per_bucket": {},
            "bucket_summary": {},
            "confluences": [],
            "neg_zones": [],
        }

    landscape_df = pd.DataFrame(landscape_rows)
    walls = find_walls(landscape_df)
    regime = classify_regime(
        landscape_df, spot, prior_spot=spot, implied_move=implied_move,
    )
    per_bucket = classify_per_bucket(
        landscape_df, spot, prior_spot=spot, implied_move=implied_move,
    )
    bucket_summary = summarize_per_bucket(per_bucket)
    confluences = analyze_confluence(landscape_df)["confluences"]

    neg_zones: list = []
    if implied_move and implied_move > 0:
        dom_strength = (
            regime["dominant_wall"]["gex"]
            if "dominant_wall" in regime else None
        )
        neg_zones = find_proximate_negative_zones(
            walls, spot, implied_move, dom_strength=dom_strength,
        )

    return {
        "regime": regime,
        "per_bucket": per_bucket,
        "bucket_summary": bucket_summary,
        "confluences": confluences,
        "neg_zones": neg_zones,
    }


def compute_and_upsert_daily_features(
    conn,
    ticker: str,
    trade_date: dt.date,
    *,
    version: str = FEATURE_VERSION,
) -> dict:
    """Load the stored landscape, materialize the spot-dependent payload,
    run ``extract_features``, and UPSERT into ``bt_daily_features``.

    Runs entirely on the caller's connection and transaction — issues no
    COMMIT. The EOD cron passes the same connection used for the landscape
    upsert; the backfill script commits per date itself.

    Raises ``ValueError`` if no ``orats_gex_landscape`` row exists for
    ``(ticker, trade_date)`` or its ``table_spot`` is NULL.

    Returns a summary dict: ``ticker``, ``trade_date``, ``feature_version``,
    ``feature_config_hash``, ``spot``, ``implied_move``, ``n_features``.
    """
    from psycopg.types.json import Jsonb
    from packages.shared.gex_landscape import compute_implied_move

    with conn.cursor() as cur:
        cur.execute(_LANDSCAPE_ROW_SQL, (ticker, trade_date))
        row = cur.fetchone()
    if not row:
        raise ValueError(
            f"compute_and_upsert_daily_features: no orats_gex_landscape row "
            f"for ({ticker!r}, {trade_date})"
        )
    landscape_rows, table_spot = row
    if table_spot is None:
        raise ValueError(
            f"compute_and_upsert_daily_features: table_spot is NULL for "
            f"({ticker!r}, {trade_date})"
        )
    spot = float(table_spot)

    # Implied move — ATM IV at last snapshot, smallest dte>0, computed
    # against spot via compute_implied_move(spot, iv, dte=1.0).
    with conn.cursor() as cur:
        cur.execute(_IMPLIED_MOVE_SQL, (trade_date.isoformat(), ticker))
        iv_row = cur.fetchone()
    if iv_row and iv_row[0] is not None:
        try:
            iv_f = float(iv_row[0])
            implied_move = compute_implied_move(spot, iv_f, dte=1.0)
        except (TypeError, ValueError):
            implied_move = 0.0
    else:
        implied_move = 0.0

    payload = _materialize_payload(landscape_rows, spot, implied_move)
    features = extract_features(payload, spot, implied_move)
    config_hash = compute_feature_config_hash(version)

    with conn.cursor() as cur:
        cur.execute(_UPSERT_SQL, (
            ticker, trade_date,
            Jsonb(features), version, config_hash,
        ))

    return {
        "ticker": ticker,
        "trade_date": trade_date,
        "feature_version": version,
        "feature_config_hash": config_hash,
        "spot": spot,
        "implied_move": implied_move,
        "n_features": len(features),
    }
