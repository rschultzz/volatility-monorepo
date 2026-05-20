"""
GEX landscape delivery-layer builder (CR-008).

build_gex_landscape_response reads a stored orats_gex_landscape row, recomputes
the spot-agnostic walls + per-bucket peaks fresh from the stored landscape
field, then runs the spot-dependent classifier chain from
packages.shared.gex_landscape against a caller-supplied (spot, iv/implied_move).

This is the analytical surface behind GET /api/gex-landscape. It mirrors the
pure-builder pattern of packages/shared/options_cache/pricing.py: no Flask
coupling — the callback in apps/web/modules/Ironbeam/callbacks.py is a thin
wrapper that opens engine.connect() and JSON-serializes the result.

Why recompute walls/peaks instead of reading the stored arrays: the cron's
stored walls/peaks_by_bucket are extracted from a table_spot-centered grid and
can be edge-clipped (see specs/CR-008-... and the CR-007 wrap-up). The stored
`landscape` field is the analytical source of truth; walls/peaks are re-derived
from it here. CR-008 also widened the stored grid to range_pts=300.

conn is a SQLAlchemy Connection (see the spec amendment) — callbacks.py owns a
module-level SQLAlchemy engine, not a psycopg connection.

See specs/CR-008-gex-landscape-delivery-layer.md.
"""
from __future__ import annotations

import datetime as dt
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from packages.shared.gex_landscape import (
    _annotate_distance_class,
    _peaks_records,
    _walls_records,
    analyze_confluence,
    classify_distance,
    classify_per_bucket,
    classify_regime,
    compute_implied_move,
    find_intraday_subtarget,
    find_peaks_per_bucket,
    find_proximate_negative_zones,
    find_walls,
    summarize_per_bucket,
)

logger = logging.getLogger(__name__)

# classify_distance buckets that the Phase 0 script treats as "structural" —
# only then does it look for a closer intraday subtarget. Mirrored here so the
# endpoint's intraday_subtarget field matches the script's behavior exactly.
_STRUCTURAL_TARGET_CLASSES = ("stretch", "multi-day", "far")

_ROW_QUERY = text("""
    SELECT ticker, trade_date, landscape, walls, peaks_by_bucket,
           spread_coef, range_pts, step_pts, table_spot, version, computed_at
    FROM orats_gex_landscape
    WHERE ticker = :ticker AND trade_date = :trade_date
""")


def _coerce_date(value) -> dt.date:
    """Coerce a date / datetime / ISO string into a date. Raises on bad input."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value).strip())


def _to_native(obj):
    """Recursively coerce numpy/pandas scalars to JSON-native Python.

    HTTP-boundary hardening: the analytical module casts to float()/int()
    internally, but this guarantees the payload survives jsonify even if a
    numpy scalar or non-finite float slips through.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return _to_native(obj.item())
    if isinstance(obj, np.ndarray):
        return [_to_native(v) for v in obj.tolist()]
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, float):
        # JSON has no NaN/Inf — null them so the response stays strict JSON.
        return obj if math.isfinite(obj) else None
    return obj


def build_gex_landscape_response(
    conn,
    ticker: str,
    trade_date,
    spot,
    *,
    iv: Optional[float] = None,
    implied_move: Optional[float] = None,
) -> tuple[dict, int]:
    """
    Build the GET /api/gex-landscape response payload.

    Returns (payload, http_status):
      200 — success
      400 — bad params (missing/non-numeric spot, malformed date, iv and
            implied_move both supplied, negative iv/implied_move)
      404 — no orats_gex_landscape row for (ticker, trade_date)
      500 — unexpected error

    conn is a SQLAlchemy Connection. iv and implied_move are mutually
    exclusive; iv (decimal, e.g. 0.107) is converted to a 1-sigma daily move
    via compute_implied_move. If neither is given, distance classifications
    come back class="unknown" and intraday_subtarget / neg_zones are omitted.

    walls and peaks_by_bucket in the response are recomputed fresh from the
    stored landscape field — the stored extracted arrays are cron diagnostics.
    """
    # ── param validation ────────────────────────────────────────────────
    if iv is not None and implied_move is not None:
        return ({"error": "iv and implied_move are mutually exclusive"}, 400)

    try:
        td = _coerce_date(trade_date)
    except (ValueError, TypeError):
        return ({"error": f"invalid date: {trade_date!r} (expected YYYY-MM-DD)"}, 400)

    try:
        spot_f = float(spot)
    except (TypeError, ValueError):
        return ({"error": "spot is required and must be a number"}, 400)
    if not math.isfinite(spot_f) or spot_f <= 0:
        return ({"error": "spot must be a positive, finite number"}, 400)

    iv_f: Optional[float] = None
    if implied_move is not None:
        try:
            move = float(implied_move)
        except (TypeError, ValueError):
            return ({"error": "implied_move must be a number"}, 400)
        if not math.isfinite(move) or move < 0:
            return ({"error": "implied_move must be non-negative"}, 400)
    elif iv is not None:
        try:
            iv_f = float(iv)
        except (TypeError, ValueError):
            return ({"error": "iv must be a number"}, 400)
        if not math.isfinite(iv_f) or iv_f < 0:
            return ({"error": "iv must be non-negative"}, 400)
        move = compute_implied_move(spot_f, iv_f)
    else:
        move = 0.0

    # ── read the stored row + run the analytical chain ──────────────────
    try:
        row = conn.execute(
            _ROW_QUERY, {"ticker": ticker, "trade_date": td}
        ).mappings().first()

        if row is None:
            return (
                {"error": f"no gex_landscape row for ({ticker}, {td.isoformat()})"},
                404,
            )

        landscape_records = row["landscape"]
        if not landscape_records:
            return ({"error": "stored landscape row has no landscape data"}, 500)

        landscape = pd.DataFrame(landscape_records)

        table_spot = (
            float(row["table_spot"]) if row["table_spot"] is not None else None
        )

        # Recompute walls + per-bucket peaks fresh from the stored landscape
        # field (NOT the stored extracted arrays — see the module docstring).
        walls = find_walls(landscape)
        peaks_by_bucket = find_peaks_per_bucket(landscape)

        # Spot-dependent classifier chain. prior_spot = table_spot mirrors the
        # Phase 0 script's behavior when run with an explicit --spot override:
        # it enables broken-magnet detection against the prior-EOD reference.
        regime = classify_regime(
            landscape, spot_f, prior_spot=table_spot, implied_move=move,
        )
        regime = _annotate_distance_class(regime, move)

        per_bucket = classify_per_bucket(
            landscape, spot_f, prior_spot=table_spot, implied_move=move,
        )
        bucket_summary = summarize_per_bucket(per_bucket)

        confluences = analyze_confluence(landscape)["confluences"]
        for c in confluences:
            c["distance_classification"] = classify_distance(
                abs(c["center_price"] - spot_f), move,
            )

        # Intraday subtarget — only when the primary drift target is itself
        # structural (>1.5 sigma away); matches the script's gating.
        intraday_subtarget = None
        if move > 0 and "drift_target" in regime:
            target_cls = regime.get("target_classification", {}).get("class", "")
            if target_cls in _STRUCTURAL_TARGET_CLASSES:
                intraday_subtarget = find_intraday_subtarget(
                    confluences, walls, spot_f, move,
                    max_sigma=1.5, direction=regime.get("drift_direction"),
                )

        # Proximate negative zones — omitted entirely when no implied move.
        neg_zones: list = []
        if move > 0:
            dom_strength = (
                regime["dominant_wall"]["gex"]
                if "dominant_wall" in regime else None
            )
            neg_zones = find_proximate_negative_zones(
                walls, spot_f, move, dom_strength=dom_strength,
            )

        payload = {
            "ticker": ticker,
            "trade_date": td.isoformat(),
            "spot": spot_f,
            "iv": iv_f,
            "implied_move": move if move > 0 else None,
            "table_spot": table_spot,
            "spread_coef": float(row["spread_coef"]),
            "range_pts": float(row["range_pts"]),
            "step_pts": float(row["step_pts"]),
            "version": row["version"],
            "computed_at": (
                row["computed_at"].isoformat() if row["computed_at"] else None
            ),
            "landscape": landscape_records,            # stored, passed through
            "walls": _walls_records(walls),            # recomputed
            "peaks_by_bucket": _peaks_records(peaks_by_bucket),  # recomputed
            "regime": regime,
            "per_bucket": per_bucket,
            "bucket_summary": bucket_summary,
            "confluences": confluences,
            "intraday_subtarget": intraday_subtarget,
            "neg_zones": neg_zones,
        }
        return (_to_native(payload), 200)

    except Exception as e:  # noqa: BLE001 — boundary: never leak a 500 traceback
        logger.exception("gex-landscape builder failed for (%s, %s)", ticker, td)
        return ({"error": f"internal error: {e}"}, 500)
