"""TodaySetup routes — Flask wiring for /api/setup/proposals (CR-015/016).

Endpoint:
    GET /api/setup/proposals
        ?date=YYYY-MM-DD
        [&spot=<float>]         optional; resolved from ironbeam_es_1m_bars when absent
        [&implied_move=<float>] optional; resolved from orats_monies_minute when absent
        [&ticker=SPX]
        [&anchor_strategy=cluster_centered]

Returns JSON with ok, context (includes spot_source), and proposals list.
CR-016: spot is now optional. When absent, the first RTH bar's open from
ironbeam_es_1m_bars is used. spot_source in the response indicates the
resolution path: "param" | "bars" | "landscape" | "default".

Promoted regime_wrong audit overrides are applied for template selection.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Optional
from zoneinfo import ZoneInfo

import psycopg
from flask import jsonify, request

from packages.shared.day_features import _materialize_payload
from packages.shared.gex_landscape import compute_implied_move
from packages.shared.audit_overrides import get_effective_regime

from apps.web.modules.Bars.service import fetch_rth_open
from .service import build_proposals_response


_PT = ZoneInfo("America/Los_Angeles")


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
        return None
    try:
        f = float(s)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


def _load_landscape(conn, ticker: str, trade_date: dt.date) -> Optional[tuple]:
    """Return (landscape_rows, table_spot) or None."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT landscape, table_spot
            FROM orats_gex_landscape
            WHERE ticker=%s AND trade_date=%s
            """,
            (ticker, trade_date),
        )
        row = cur.fetchone()
    return row or None


def _resolve_implied_move(conn, ticker: str, trade_date: dt.date, spot: float) -> float:
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


def _build_context(
    trade_date: dt.date,
    ticker: str,
    spot: float,
    implied_move: float,
    landscape_payload: dict,
) -> dict:
    regime_block = landscape_payload.get("regime") or {}
    regime = regime_block.get("regime")
    bucket_summary = landscape_payload.get("bucket_summary") or {}
    confluences = landscape_payload.get("confluences") or []
    top = max(confluences, key=lambda c: c.get("max_gex", 0.0)) if confluences else None
    return {
        "date": trade_date.isoformat(),
        "ticker": ticker,
        "spot": spot,
        "implied_move": implied_move,
        "regime": regime,
        "dominant_bucket": bucket_summary.get("primary_bucket"),
        "top_cluster": (
            {
                "center_price": top["center_price"],
                "quality": top.get("quality"),
                "max_gex": top.get("max_gex"),
            }
            if top else None
        ),
        "clusters": [
            {
                "center_price": c["center_price"],
                "quality": c.get("quality"),
                "max_gex": c.get("max_gex"),
                "avg_fwhm": c.get("avg_fwhm"),
                "bucket": c.get("bucket"),
            }
            for c in confluences
        ],
    }


def register_today_setup_routes(server) -> None:
    """Wire /api/setup/proposals onto the Flask server."""
    if "today_setup_proposals" in server.view_functions:
        return

    def today_setup_proposals():
        # ── params ────────────────────────────────────────────────────────
        date_s = (request.args.get("date") or "").strip()
        trade_date = _parse_date(date_s)
        if not trade_date:
            return jsonify({"ok": False, "error": "date is required (YYYY-MM-DD)"}), 400

        spot_param = _parse_float(request.args.get("spot"))
        implied_move_param = _parse_float(request.args.get("implied_move"))
        ticker = (request.args.get("ticker") or "SPX").strip() or "SPX"
        anchor_strategy = (
            (request.args.get("anchor_strategy") or "cluster_centered").strip()
            or "cluster_centered"
        )

        # ── DB session ────────────────────────────────────────────────────
        try:
            conn = _conn()
        except Exception as e:
            return jsonify({"ok": False, "error": f"db connect failed: {e}"}), 500

        try:
            row = _load_landscape(conn, ticker, trade_date)
            if not row:
                return jsonify({
                    "ok": False,
                    "error": (
                        f"no landscape for ({ticker}, {trade_date.isoformat()}) — "
                        "backfill required"
                    ),
                }), 404

            landscape_rows, table_spot = row

            # ── Spot resolution (CR-016) ──────────────────────────────────
            spot = spot_param
            spot_source = "param"
            if spot is None or spot <= 0:
                bars_open = fetch_rth_open(conn, trade_date)
                if bars_open is not None and bars_open > 0:
                    spot = bars_open
                    spot_source = "bars"
                elif table_spot is not None:
                    spot = float(table_spot)
                    spot_source = "landscape"
                else:
                    spot = 5000.0
                    spot_source = "default"

            implied_move = implied_move_param
            if not implied_move:
                implied_move = _resolve_implied_move(conn, ticker, trade_date, spot)

            payload = _materialize_payload(landscape_rows, spot, implied_move)

            # ── Effective-regime override (CR-016) ────────────────────────
            effective_regime = get_effective_regime(conn, ticker, trade_date)
            if effective_regime and effective_regime != payload.get("regime", {}).get("regime"):
                # Patch the payload's regime so template selection uses the override.
                payload = dict(payload)
                payload["regime"] = dict(payload.get("regime") or {})
                payload["regime"]["regime"] = effective_regime

            context = _build_context(trade_date, ticker, spot, implied_move, payload)
            context["spot_source"] = spot_source
            response = build_proposals_response(
                payload, spot, implied_move, context, anchor_strategy
            )
            return jsonify(response)

        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        finally:
            try:
                conn.close()
            except Exception:
                pass

    server.add_url_rule(
        "/api/setup/proposals",
        endpoint="today_setup_proposals",
        view_func=today_setup_proposals,
        methods=["GET"],
    )
