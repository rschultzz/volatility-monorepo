"""Proposals routes — POST /api/proposals/pl-data (CR-G Step 4).

Composes BSM pricing, structural KNN probability, market-implied probability,
and edge zone classification into a single JSON response for frontend chart
rendering.

POST /api/proposals/pl-data
    Body (JSON):
        trade_date   — "YYYY-MM-DD" (required)
        ticker       — e.g. "SPX" (optional, default "SPX")
        legs         — non-empty list of leg objects (required):
                         {strike, expiration, flag ('c'/'p'),
                          side ('long'/'short'), qty (int, default 1)}
        regime_block — GEX landscape regime dict with at least {regime} key (required)
        timeframe    — "t1", "t5", or "t15" (required)

    Response 200:
        {ok, trade_date, ticker, evaluation_time, current_spot, implied_move,
         legs, pl_curve, iv_curve, trade_thesis, edge_zones, greeks,
         key_levels, warnings}

    Response 400: validation failure → {ok: false, errors: [...]}
    Response 503: DB connect failed → {ok: false, error: "..."}
    Response 500: unexpected error  → {ok: false, error: "..."}
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import warnings
from typing import Optional
from zoneinfo import ZoneInfo

import psycopg
from flask import jsonify, request

from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.edge_zones import compute_edge_zones
from packages.shared.forward_math import compute_spx_strike
from packages.shared.implied_distribution import (
    compute_implied_pdf,
    compute_implied_prob_in_range,
)
from packages.shared.options_cache.pricing import (
    build_real_strike_band,
    price_proposal_legs,
)
from packages.shared.pricing.engine import compute_position_greeks
from packages.shared.probability import _rank_analogues_with_outcomes
from packages.shared.structural_distribution import (
    compute_terminal_prob_in_range,
    get_trade_thesis_range,
)

from .service import (
    build_bsm_chain,
    build_entry_time,
    build_evaluation_time,
    build_grid_bounds,
    calibrate_leg_iv,
    compute_initial_cost,
    compute_key_levels,
    compute_pl_curve,
    compute_pl_curves,
)

_VALID_TIMEFRAMES = {"t1", "t5", "t15"}
_VALID_FLAGS      = {"c", "p"}
_VALID_SIDES      = {"long", "short"}

# TTE in years for each timeframe (used for implied PDF + edge zones)
_TTE_BY_TIMEFRAME: dict[str, float] = {
    "t1":  1.0 / 252,
    "t5":  5.0 / 252,
    "t15": 15.0 / 252,
}

log = logging.getLogger(__name__)


# ── DB helpers ────────────────────────────────────────────────────────────────

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


def _fetch_anchor_data(
    conn,
    ticker: str,
    trade_date: dt.date,
    feature_version: str,
) -> tuple[float, float, dict]:
    """Load (spot, implied_move, feature_vector) for ticker/trade_date.

    spot         — session_open_t0 from bt_daily_outcomes_active
    implied_move — feature_vector['implied_move_1d']
    feature_vector — full dict from bt_daily_features_active

    Raises ValueError if data is missing (caller returns 400).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT feature_vector FROM bt_daily_features_active
            WHERE ticker = %s AND trade_date = %s AND feature_version = %s
            """,
            (ticker, trade_date, feature_version),
        )
        fv_row = cur.fetchone()

    if not fv_row:
        raise ValueError(
            f"no feature vector for ({ticker}, {trade_date}, {feature_version!r}); "
            "ensure the daily feature backfill has run for this date"
        )
    fv: dict = fv_row[0]

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT session_open_t0 FROM bt_daily_outcomes_active
            WHERE ticker = %s AND trade_date = %s AND feature_version = %s
            """,
            (ticker, trade_date, feature_version),
        )
        out_row = cur.fetchone()

    if not out_row or out_row[0] is None:
        raise ValueError(
            f"no session_open_t0 for ({ticker}, {trade_date}); "
            "CR-G Step 2.5a backfill (session_open_t0 column) is required"
        )

    spot = float(out_row[0])
    raw_im = fv.get("implied_move_1d") if isinstance(fv, dict) else None
    try:
        implied_move = float(raw_im) if raw_im is not None else 0.0
    except (TypeError, ValueError):
        implied_move = 0.0

    return spot, implied_move, fv


def _fetch_smile_row(
    conn,
    ticker: str,
    trade_date: dt.date,
    expir_date: dt.date,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (atmiv, risk_free_rate, yield_rate) closest to expir_date's DTE.

    Queries orats_monies_minute for the latest snapshot on trade_date,
    ranking by |dte - target_dte| to select the nearest available expiration.
    Returns (None, None, None) if no data exists for this date.
    """
    dte_target = (expir_date - trade_date).days
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT atmiv, risk_free_rate, yield_rate
            FROM orats_monies_minute
            WHERE ticker = %s
              AND trade_date = %s
              AND atmiv IS NOT NULL
              AND dte > 0
            ORDER BY ABS(dte - %s) ASC, snapshot_pt DESC
            LIMIT 1
            """,
            (ticker, trade_date.isoformat(), dte_target),
        )
        row = cur.fetchone()

    if not row:
        return None, None, None
    atmiv   = float(row[0]) if row[0] is not None else None
    rfr     = float(row[1]) if row[1] is not None else 0.05
    yield_r = float(row[2]) if row[2] is not None else 0.0
    return atmiv, rfr, yield_r


# ── Validation ────────────────────────────────────────────────────────────────

def _parse_date(s) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None


def _validate_body(body: dict) -> tuple[list[str], dict]:
    """Validate POST body. Returns (errors, normalized).

    errors  — list of human-readable strings; empty means body is valid.
    normalized — cleaned fields ready for use (only meaningful when errors empty).
    """
    errors: list[str] = []

    # trade_date
    trade_date = _parse_date(body.get("trade_date"))
    if trade_date is None:
        errors.append("trade_date is required (YYYY-MM-DD)")

    # ticker
    ticker = str(body.get("ticker") or "SPX").strip() or "SPX"

    # timeframe
    timeframe = body.get("timeframe")
    if timeframe not in _VALID_TIMEFRAMES:
        errors.append(
            f"timeframe must be one of {sorted(_VALID_TIMEFRAMES)}, got {timeframe!r}"
        )

    # regime_block
    regime_block = body.get("regime_block")
    if not isinstance(regime_block, dict):
        errors.append("regime_block is required and must be an object")
    elif not regime_block.get("regime"):
        errors.append("regime_block.regime is required and must be a non-empty string")

    # legs
    raw_legs = body.get("legs")
    if not isinstance(raw_legs, list) or len(raw_legs) == 0:
        errors.append("legs is required and must be a non-empty list")
        raw_legs = []

    norm_legs: list[dict] = []
    for i, leg in enumerate(raw_legs):
        if not isinstance(leg, dict):
            errors.append(f"legs[{i}] must be an object")
            continue
        leg_errs: list[str] = []

        try:
            strike = float(leg["strike"])
        except (KeyError, TypeError, ValueError):
            leg_errs.append("strike (number)")
            strike = 0.0

        expir = _parse_date(leg.get("expiration"))
        if expir is None:
            leg_errs.append("expiration (YYYY-MM-DD)")

        flag = leg.get("flag")
        if flag not in _VALID_FLAGS:
            leg_errs.append(f"flag (one of {sorted(_VALID_FLAGS)})")

        side = leg.get("side")
        if side not in _VALID_SIDES:
            leg_errs.append(f"side (one of {sorted(_VALID_SIDES)})")

        qty = leg.get("qty", 1)
        try:
            qty = int(qty)
            if qty <= 0:
                raise ValueError("qty must be positive")
        except (TypeError, ValueError):
            leg_errs.append("qty (positive integer)")
            qty = 1

        if leg_errs:
            errors.append(f"legs[{i}] invalid field(s): {', '.join(leg_errs)}")
        else:
            norm_legs.append({
                "strike":     strike,
                "expiration": expir,
                "flag":       flag,
                "side":       side,
                "qty":        qty,
            })

    return errors, {
        "trade_date":   trade_date,
        "ticker":       ticker,
        "timeframe":    timeframe,
        "regime_block": regime_block,
        "legs":         norm_legs,
    }


# ── Route registration ────────────────────────────────────────────────────────

def register_proposals_routes(server) -> None:
    """Wire POST /api/proposals/pl-data onto the Flask server."""
    if "proposals_pl_data" in server.view_functions:
        return

    def proposals_pl_data():  # noqa: C901 (complex but linear pipeline)
        body = request.get_json(silent=True) or {}

        # ── 1. Validation ─────────────────────────────────────────────────
        errors, norm = _validate_body(body)
        if errors:
            return jsonify({"ok": False, "errors": errors}), 400

        trade_date:   dt.date = norm["trade_date"]
        ticker:       str     = norm["ticker"]
        timeframe:    str     = norm["timeframe"]
        regime_block: dict    = norm["regime_block"]
        raw_legs:     list    = norm["legs"]

        # Shortest expiration drives evaluation_time (at-expiry payoff curve).
        # entry_time is the pricing moment: 07:00 PT (10:00 ET) on trade_date.
        # entry_pt_naive is the same instant as a naive PT datetime, required by
        # fetch_option_bars which rejects tz-aware inputs.
        expir_dates    = sorted(leg["expiration"] for leg in raw_legs)
        shortest_expir = expir_dates[0]
        evaluation_time = build_evaluation_time(shortest_expir)
        entry_time      = build_entry_time(trade_date)
        _PT_TZ = ZoneInfo("America/Los_Angeles")
        entry_pt_naive  = entry_time.astimezone(_PT_TZ).replace(tzinfo=None)

        # ── 2. DB connect ─────────────────────────────────────────────────
        try:
            conn = _conn()
        except Exception as e:
            return jsonify({"ok": False, "error": f"db connect failed: {e}"}), 503

        warn_msgs: list[str] = []

        try:
            # ── 3. Anchor data (spot, implied_move, feature_vector) ───────
            try:
                spot, implied_move, feature_vector = _fetch_anchor_data(
                    conn, ticker, trade_date, CANONICAL_FEATURE_VERSION
                )
            except ValueError as e:
                return jsonify({"ok": False, "error": str(e)}), 400

            # ── 4. Smile (atmiv + risk-free rate + yield rate) ────────────
            atmiv, risk_free_rate, yield_rate = _fetch_smile_row(
                conn, ticker, trade_date, shortest_expir
            )
            if atmiv is None:
                warn_msgs.append(
                    f"no ORATS smile data for ({ticker}, {trade_date}); "
                    "using fallback atmiv=0.15"
                )
                atmiv = 0.15
            if risk_free_rate is None:
                risk_free_rate = 0.05
            if yield_rate is None:
                yield_rate = 0.0

            market_state = {"risk_free_rate": risk_free_rate}

            # ── 5. Build legs with IV + expiration datetime ────────────────
            # Expiration date → 16:00 ET datetime for pricing engine.
            # IVs start as atmiv; calibrated per-leg in step 6b if real mids.
            legs_with_iv: list[dict] = [
                {
                    **leg,
                    "expiration": build_evaluation_time(leg["expiration"]),
                    "iv":         atmiv,
                }
                for leg in raw_legs
            ]

            # ── 6. Real entry cost via ORATS mids ─────────────────────────
            # price_proposal_legs fetches real bid/ask mids from the options
            # cache (writing through on miss).  Falls back to BSM at entry_time
            # if all legs are unavailable so the route never hard-fails.
            real_pricing = price_proposal_legs(
                raw_legs,
                trade_date=trade_date,
                entry_pt=entry_pt_naive,
                r=risk_free_rate,
                q=yield_rate,
            )
            warn_msgs.extend(real_pricing["warnings"])

            if real_pricing["net_debit"] is not None:
                initial_cost = real_pricing["net_debit"]
            else:
                # BSM fallback at entry_time ensures T > 0 (zero-debit fix).
                initial_cost = compute_initial_cost(
                    legs_with_iv, spot, entry_time, market_state
                )
                if real_pricing["warnings"]:
                    warn_msgs.append(
                        "real entry pricing unavailable; using BSM fallback"
                    )

            # ── 6b. Calibrate per-leg BSM vol to reproduce real entry mid ─
            # Replaces atmiv with the per-leg implied vol anchored to the real
            # mid so that the P/L curve passes through the real entry price.
            real_legs_idx = {
                i: rleg for i, rleg in enumerate(real_pricing.get("legs", []))
            }
            for i, leg in enumerate(legs_with_iv):
                rleg = real_legs_idx.get(i, {})
                real_mid = rleg.get("mid")
                if real_mid is None or real_mid <= 0:
                    continue
                T_entry = max(
                    0.0,
                    (leg["expiration"] - entry_time).total_seconds() / (365.25 * 24 * 3600),
                )
                calib_iv = calibrate_leg_iv(
                    spot=spot,
                    strike=float(leg["strike"]),
                    T=T_entry,
                    r=risk_free_rate,
                    flag=leg["flag"],
                    real_mid=abs(real_mid),   # use abs — calibrate on the price, sign via side
                    iv_guess=atmiv,
                )
                leg["iv"] = calib_iv

            # ── 7. Price grid bounds (shared by P/L curve + edge zones) ───
            # Computed once so both consumers cover the same asymmetric range.
            grid_lo, grid_hi = build_grid_bounds(spot, implied_move, regime_block)

            # ── 8. Multi-horizon P/L curves (expiry + t1/t5/t15) ─────────
            # BSM P/L curves calibrated to real entry mids.  The at-expiry
            # curve is intrinsic − real_debit; horizon curves are smooth BSM
            # surfaces that pass near the real entry debit at spot.
            pl_curves = compute_pl_curves(
                legs_with_iv, spot, implied_move,
                evaluation_time, entry_time, market_state, initial_cost,
                regime_block=regime_block,
            )
            # pl_curve (singular) = the at-expiry curve for backwards compat.
            pl_curve = next(
                (c for c in pl_curves if c["label"] == "expiry"),
                pl_curves[0] if pl_curves else {"prices": [], "pnl": []},
            )

            # ── 8b. IV curve (flat atmiv across price grid) ────────────────
            iv_curve = {
                "prices": pl_curve["prices"],
                "iv":     [atmiv] * len(pl_curve["prices"]),
            }

            # ── 9. Real strike band for implied PDF (Breeden-Litzenberger) ──
            # Fetches real per-strike call mids around the thesis range from
            # the options cache. Falls back to a synthetic BSM chain if the
            # band returns fewer than 2 strikes (total cache miss scenario).
            tte = _TTE_BY_TIMEFRAME[timeframe]
            option_chain = build_real_strike_band(
                spot, implied_move,
                expiration_date=shortest_expir,
                entry_pt=entry_pt_naive,
            )
            if len(option_chain) < 2:
                warn_msgs.append(
                    "real strike band unavailable (< 2 strikes fetched); "
                    "falling back to BSM chain (no skew)"
                )
                option_chain = build_bsm_chain(spot, atmiv, risk_free_rate, tte)

            # ── 10. Analogues (K=20 KNN) ───────────────────────────────────
            analogues = _rank_analogues_with_outcomes(
                feature_vector, conn, 20, CANONICAL_FEATURE_VERSION,
                ticker=ticker,
                exclude_date=trade_date.isoformat(),
            )

            # ── 11. Trade thesis range ─────────────────────────────────────
            pin_tolerance = 0.25 * implied_move if implied_move > 0 else None
            try:
                trade_range = get_trade_thesis_range(
                    regime_block, spot, tolerance=pin_tolerance
                )
            except ValueError as e:
                warn_msgs.append(f"trade thesis range: {e}")
                trade_range = {
                    "lower":       None,
                    "upper":       None,
                    "regime_kind": regime_block.get("regime", ""),
                }

            # ── 12. Structural probability in range ────────────────────────
            close_key = f"session_close_{timeframe}"
            analogue_records = [
                {
                    "close":               a.get(close_key),
                    "anchor_spot":         a.get("session_open_t0"),
                    "anchor_implied_move": a.get("implied_move_1d"),
                }
                for a in analogues
            ]
            struct_result = compute_terminal_prob_in_range(
                analogue_records, spot, implied_move,
                trade_range["lower"], trade_range["upper"],
            )

            # ── 13. Implied probability in range ───────────────────────────
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                impl_pdf = compute_implied_pdf(option_chain, spot, risk_free_rate, tte)
                impl_p = compute_implied_prob_in_range(
                    impl_pdf, trade_range["lower"], trade_range["upper"]
                )
            impl_prob = impl_p if impl_p is not None and impl_p > 1e-10 else None

            # ── 14. Edge zones ─────────────────────────────────────────────
            edge_zones = compute_edge_zones(
                spot, implied_move, option_chain, analogues,
                timeframe, regime_block,
                risk_free_rate=risk_free_rate,
                time_to_expiration=tte,
                price_bounds=(grid_lo, grid_hi),
                tolerance=pin_tolerance,
            )

            # ── 15. Greeks at spot ─────────────────────────────────────────
            greeks = compute_position_greeks(
                legs_with_iv, spot, evaluation_time, market_state
            )

            # ── 16. Key levels ─────────────────────────────────────────────
            key_levels = compute_key_levels(pl_curve["pnl"], pl_curve["prices"])

            # ── 17. Assemble trade thesis block ────────────────────────────
            struct_prob = struct_result["prob"] if struct_result else None
            edge_ratio = (
                struct_prob / impl_prob
                if struct_prob is not None and impl_prob and impl_prob > 0
                else None
            )
            trade_thesis = {
                "lower":          trade_range["lower"],
                "upper":          trade_range["upper"],
                "regime_kind":    trade_range["regime_kind"],
                "structural_prob": struct_prob,
                "structural_ci":   (
                    list(struct_result["wilson_ci"]) if struct_result else None
                ),
                "structural_n":    struct_result["n"] if struct_result else 0,
                "implied_prob":    impl_prob,
                "edge_ratio":      edge_ratio,
            }

            # ── 18. Per-leg response (echo + IV + real prices + strike_spx) ─
            # Merge real pricing from price_proposal_legs into the leg output.
            # Index-aligned: real_pricing["legs"] has the same order as raw_legs.
            real_legs_by_idx = {
                i: rleg for i, rleg in enumerate(real_pricing.get("legs", []))
            }
            legs_out = []
            for i, (leg, raw) in enumerate(zip(legs_with_iv, raw_legs)):
                rleg = real_legs_by_idx.get(i, {})
                dte_i = (raw["expiration"] - trade_date).days
                spx_strike_i = rleg.get("spx_strike") or compute_spx_strike(
                    raw["strike"], dte_i, risk_free_rate, yield_rate
                )
                real_mid = rleg.get("mid")
                if real_mid is not None:
                    initial_val = round(
                        (1.0 if raw["side"] == "long" else -1.0) * raw.get("qty", 1) * real_mid,
                        4,
                    )
                else:
                    initial_val = round(
                        compute_initial_cost([leg], spot, entry_time, market_state), 4
                    )
                legs_out.append({
                    "strike":        raw["strike"],
                    "strike_spx":    spx_strike_i,
                    "opra":          rleg.get("opra"),
                    "expiration":    raw["expiration"].isoformat(),
                    "flag":          raw["flag"],
                    "side":          raw["side"],
                    "qty":           raw.get("qty", 1),
                    "iv":            atmiv,
                    "bid":           rleg.get("bid"),
                    "ask":           rleg.get("ask"),
                    "mid":           real_mid,
                    "initial_value": initial_val,
                })

            return jsonify({
                "ok":             True,
                "trade_date":     trade_date.isoformat(),
                "ticker":         ticker,
                "evaluation_time": evaluation_time.isoformat(),
                "entry_time":     entry_time.isoformat(),
                "current_spot":   spot,
                "implied_move":   implied_move,
                "legs":           legs_out,
                "pl_curve":       pl_curve,
                "pl_curves":      pl_curves,
                "iv_curve":       iv_curve,
                "trade_thesis":   trade_thesis,
                "edge_zones":     edge_zones,
                "greeks":         greeks,
                "key_levels":     key_levels,
                "warnings":       warn_msgs,
            })

        except Exception as e:
            log.exception(
                "proposals_pl_data unhandled error for (%s, %s)", ticker, trade_date
            )
            return jsonify({"ok": False, "error": str(e)}), 500
        finally:
            try:
                conn.close()
            except Exception:
                pass

    server.add_url_rule(
        "/api/proposals/pl-data",
        endpoint="proposals_pl_data",
        view_func=proposals_pl_data,
        methods=["POST"],
    )
