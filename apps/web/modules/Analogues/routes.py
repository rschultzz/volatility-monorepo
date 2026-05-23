"""Analogues routes — Flask wiring for /api/analogues (CR-013, v0.5;
label-gate removed in CR-014).

Endpoint:
    GET /api/analogues?date=YYYY-MM-DD&spot=<float>&implied_move=<float>
        [&k=5][&ticker=SPX][&feature_version=v0.5.0]

Returns the K nearest historical days from bt_daily_features ranked by
σ-normalized weighted Euclidean distance over the day's feature vector.
Candidates are any day with a stored feature vector at the requested
feature_version — no human-label requirement (CR-014). When labels do
exist for a returned analogue's trade_date, they ride along as the
`labeled_signals` enrichment on that analogue's response object.

Empty corpus → analogues: []. The anchor day is excluded from its own
neighbor list.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Any, Optional
from zoneinfo import ZoneInfo

import psycopg
from flask import jsonify, request

_UTC = ZoneInfo("UTC")
_PT = ZoneInfo("America/Los_Angeles")

from packages.shared.day_features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    _materialize_payload,
    extract_features,
)
from packages.shared.gex_landscape import compute_implied_move
from packages.shared.audit_overrides import get_effective_regime

from apps.web.modules.Bars.service import fetch_rth_open

from .service import feature_stats, rank_analogues, feature_distance_breakdown


_K_DEFAULT = 5
_K_MAX = 20


def _normalize_db_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql+"):
        url = "postgresql://" + url.split("://", 1)[1]
    if "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return url


def _conn():
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(_normalize_db_url(raw))


def _parse_date(s: str) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def _parse_float(s, *, allow_none: bool = False) -> Optional[float]:
    if s is None or s == "":
        return None if allow_none else None
    try:
        f = float(s)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _load_feature_row(conn, ticker: str, trade_date: dt.date, version: str
                     ) -> Optional[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT feature_vector
            FROM bt_daily_features
            WHERE ticker=%s AND trade_date=%s AND feature_version=%s
            """,
            (ticker, trade_date, version),
        )
        row = cur.fetchone()
    return row[0] if row else None


def _materialize_anchor_features(conn, ticker: str, trade_date: dt.date,
                                 spot: float, implied_move: float
                                 ) -> Optional[dict]:
    """On-the-fly compute when bt_daily_features doesn't have a row for the
    anchor date — typically because cron hasn't run yet for today."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT landscape FROM orats_gex_landscape
            WHERE ticker=%s AND trade_date=%s
            """,
            (ticker, trade_date),
        )
        row = cur.fetchone()
    if not row:
        return None
    landscape_rows = row[0]
    payload = _materialize_payload(landscape_rows, spot, implied_move)
    return extract_features(payload, spot, implied_move)


def _load_candidates(conn, ticker: str, version: str) -> list[tuple]:
    """All candidate days for similarity ranking: every day with a stored
    feature vector at the requested feature_version (CR-014).

    CR-013 originally joined on bt_signals and required label IS NOT NULL.
    CR-014 dropped that gate — the analogues panel is a structural day
    comparison (auto-classified features + price outcomes), not a
    label-based backtest. Labels still ride along as optional enrichment
    via _fetch_labeled_signals when they exist.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.trade_date, f.feature_vector
            FROM bt_daily_features f
            WHERE f.ticker = %s
              AND f.feature_version = %s
            """,
            (ticker, version),
        )
        rows = cur.fetchall()
    return [(d.isoformat(), v) for (d, v) in rows]


def _fetch_labeled_signals(conn, trade_date_iso: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT signal_id, label, label_note, outcome, realized_pts,
                   direction, strategy_key
            FROM bt_signals
            WHERE trade_date = %s AND label IS NOT NULL
            ORDER BY signal_id
            """,
            (dt.date.fromisoformat(trade_date_iso),),
        )
        rows = cur.fetchall()
    return [
        {
            "signal_id": r[0],
            "label": r[1],
            "label_note": r[2],
            "outcome": r[3],
            "realized_pts": float(r[4]) if r[4] is not None else None,
            "direction": r[5],
            "strategy_key": r[6],
        }
        for r in rows
    ]


def _fetch_session_outcomes(conn, trade_date_iso: str) -> dict:
    """EOD outcomes from ironbeam_es_1m_bars for the trade_date.

    The bars table stores `datetime` as naive UTC; this filter converts
    to PT before comparing the calendar date so a single PT session
    (~390 bars, 06:30→13:00 PT) doesn't span two UTC days. Returns
    empty dict if no bars for the date.
    """
    trade_date = dt.date.fromisoformat(trade_date_iso)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT datetime, open, high, low, close
            FROM ironbeam_es_1m_bars
            WHERE (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date = %s
            ORDER BY datetime ASC
            """,
            (trade_date,),
        )
        rows = cur.fetchall()
    if not rows:
        return {}
    open_px = float(rows[0][1]) if rows[0][1] is not None else None
    close_px = float(rows[-1][4]) if rows[-1][4] is not None else None
    highs = [float(r[2]) for r in rows if r[2] is not None]
    lows = [float(r[3]) for r in rows if r[3] is not None]
    hi = max(highs) if highs else None
    lo = min(lows) if lows else None

    def _utc_naive_to_pt_iso(d):
        if d is None:
            return None
        return d.replace(tzinfo=_UTC).astimezone(_PT).isoformat()

    out: dict[str, Any] = {
        "session_start": _utc_naive_to_pt_iso(rows[0][0]),
        "session_end": _utc_naive_to_pt_iso(rows[-1][0]),
    }
    if open_px is not None and close_px is not None:
        out["eod_return_pts"] = close_px - open_px
    if hi is not None and lo is not None:
        out["intraday_range_pts"] = hi - lo
    if open_px is not None and hi is not None:
        out["mfe_above_open_pts"] = hi - open_px
    if open_px is not None and lo is not None:
        out["mfe_below_open_pts"] = lo - open_px
    return out


def _fetch_landscape_summary(conn, ticker: str, trade_date_iso: str,
                             implied_move: float) -> dict:
    """Cheap regime + dominant_bucket + top_cluster lookup, materialized
    from the stored landscape grid against that day's own table_spot."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT landscape, table_spot FROM orats_gex_landscape
            WHERE ticker=%s AND trade_date=%s
            """,
            (ticker, dt.date.fromisoformat(trade_date_iso)),
        )
        row = cur.fetchone()
    if not row or row[1] is None:
        return {}
    spot = float(row[1])
    payload = _materialize_payload(row[0], spot, implied_move)
    conf = payload.get("confluences") or []
    top = max(conf, key=lambda c: c.get("max_gex", 0.0)) if conf else None
    bucket_summary = payload.get("bucket_summary") or {}
    return {
        "regime": (payload.get("regime") or {}).get("regime"),
        "dominant_bucket": bucket_summary.get("primary_bucket"),
        "top_cluster": (
            {
                "center_price": top["center_price"],
                "quality": top["quality"],
                "max_gex": top["max_gex"],
            }
            if top else None
        ),
    }


def _resolve_implied_move(conn, ticker: str, trade_date: dt.date,
                          spot: float) -> float:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT atmiv FROM orats_monies_minute
            WHERE trade_date=%s AND ticker=%s AND atmiv IS NOT NULL AND dte>0
            ORDER BY snapshot_pt DESC, dte ASC LIMIT 1
            """,
            (trade_date.isoformat(), ticker),
        )
        row = cur.fetchone()
    if not row or row[0] is None:
        return 0.0
    try:
        return compute_implied_move(spot, float(row[0]), dte=1.0)
    except (TypeError, ValueError):
        return 0.0


def _fetch_excluded_analogues(conn, ticker: str, anchor_date: dt.date) -> set[str]:
    """Return ISO date strings of analogues flagged as 'not_a_true_analogue'
    for this specific anchor date."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT analogue_date
            FROM bt_audit_flags
            WHERE ticker = %s
              AND trade_date = %s
              AND flag_type = 'not_a_true_analogue'
            """,
            (ticker, anchor_date),
        )
        rows = cur.fetchall()
    return {r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]) for r in rows if r[0]}


def _latest_feature_version(conn, ticker: str) -> str:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT feature_version
            FROM bt_daily_features
            WHERE ticker = %s
            ORDER BY computed_at DESC
            LIMIT 1
            """,
            (ticker,),
        )
        row = cur.fetchone()
    return row[0] if row else FEATURE_VERSION


def register_analogues_routes(server) -> None:
    """Wire /api/analogues onto the Flask server."""
    if "analogues_get" in server.view_functions:
        return

    def analogues_get():
        # ── params ────────────────────────────────────────────────────────
        date_s = (request.args.get("date") or "").strip()
        anchor_date = _parse_date(date_s)
        if not anchor_date:
            return jsonify({"ok": False,
                            "error": "date is required (YYYY-MM-DD)"}), 400

        spot_param = _parse_float(request.args.get("spot"))
        implied_move_param = _parse_float(request.args.get("implied_move"))
        ticker = (request.args.get("ticker") or "SPX").strip() or "SPX"

        try:
            k = int(request.args.get("k") or _K_DEFAULT)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "k must be an integer"}), 400
        if k < 1:
            return jsonify({"ok": False, "error": "k must be >= 1"}), 400
        if k > _K_MAX:
            k = _K_MAX

        # ── DB session ────────────────────────────────────────────────────
        try:
            conn = _conn()
        except Exception as e:
            return jsonify({"ok": False, "error": f"db connect failed: {e}"}), 500

        try:
            # ── Spot resolution (CR-016) ──────────────────────────────────
            spot = spot_param
            spot_source = "param"
            if spot is None or spot <= 0:
                bars_open = fetch_rth_open(conn, anchor_date)
                if bars_open is not None and bars_open > 0:
                    spot = bars_open
                    spot_source = "bars"
                else:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT table_spot FROM orats_gex_landscape "
                            "WHERE ticker=%s AND trade_date=%s LIMIT 1",
                            (ticker, anchor_date),
                        )
                        row = cur.fetchone()
                    if row and row[0] is not None:
                        spot = float(row[0])
                        spot_source = "landscape"
                    else:
                        spot = 5000.0
                        spot_source = "default"

            implied_move = implied_move_param
            if implied_move is None or implied_move < 0:
                implied_move = _resolve_implied_move(conn, ticker, anchor_date, spot)

            version = (request.args.get("feature_version") or "").strip()
            if not version:
                version = _latest_feature_version(conn, ticker)

            # ── anchor vector ─────────────────────────────────────────────
            anchor_vec = _load_feature_row(conn, ticker, anchor_date, version)
            if anchor_vec is None:
                anchor_vec = _materialize_anchor_features(
                    conn, ticker, anchor_date, spot, implied_move,
                )
            if anchor_vec is None:
                return jsonify({
                    "ok": False,
                    "error": (
                        f"no landscape or feature row for ({ticker}, "
                        f"{anchor_date.isoformat()}) — backfill required"
                    ),
                }), 404

            # ── candidates (excluding flagged not_a_true_analogue) ────────
            excluded_dates = _fetch_excluded_analogues(conn, ticker, anchor_date)
            candidates = _load_candidates(conn, ticker, version)
            candidates = [(d, v) for (d, v) in candidates if d not in excluded_dates]

            stats = feature_stats(v for (_, v) in candidates) if candidates else {}
            ranked = rank_analogues(
                anchor_vec, candidates, k,
                exclude_date=anchor_date.isoformat(), stats=stats,
            )

            # ── enrich top-K ──────────────────────────────────────────────
            analogues = []
            cand_by_date = dict(candidates)
            for trade_date_iso, distance in ranked:
                vec = cand_by_date.get(trade_date_iso, {})
                signals = _fetch_labeled_signals(conn, trade_date_iso)
                outcomes = _fetch_session_outcomes(conn, trade_date_iso)
                ls_summary = _fetch_landscape_summary(
                    conn, ticker, trade_date_iso,
                    implied_move=float(vec.get("implied_move_1d", 0.0) or 0.0),
                )
                # Feature-distance breakdown (CR-016)
                distances = feature_distance_breakdown(anchor_vec, vec, stats)
                # Effective regime (CR-016)
                analogue_date = dt.date.fromisoformat(trade_date_iso)
                effective_regime = get_effective_regime(conn, ticker, analogue_date)
                analogues.append({
                    "trade_date": trade_date_iso,
                    "similarity_score": float(distance),
                    "feature_vector": vec,
                    "regime": effective_regime or ls_summary.get("regime"),
                    "auto_regime": ls_summary.get("regime"),
                    "labeled_signals": signals,
                    "outcomes": outcomes,
                    "landscape_summary": ls_summary,
                    "feature_distances": distances,
                })

            return jsonify({
                "ok": True,
                "anchor": {
                    "trade_date": anchor_date.isoformat(),
                    "ticker": ticker,
                    "feature_version": version,
                    "feature_vector": anchor_vec,
                    "spot": spot,
                    "spot_source": spot_source,
                    "implied_move": implied_move,
                },
                "k": k,
                "n_candidates": len(candidates),
                "analogues": analogues,
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        finally:
            try:
                conn.close()
            except Exception:
                pass

    server.add_url_rule(
        "/api/analogues",
        endpoint="analogues_get",
        view_func=analogues_get,
        methods=["GET"],
    )
