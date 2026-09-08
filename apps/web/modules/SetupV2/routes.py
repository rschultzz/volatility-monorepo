"""SetupV2 routes — Flask wiring for the Setup v2 tab (CR-AW).

Endpoints:
    GET /setup-v2, /setup-v2/          the v2 page (react_today_setup/dist/setup-v2.html)
    GET /api/setup-v2/card?date=YYYY-MM-DD[&ticker=SPX]

The card is led by the max price to pay for the proposed debit spread, from
what the spread was worth at the analogue closes (decision 2), then the band
evidence (decision 4), the analogue facts, the management lines, the stamp
(decisions 5–7) and a KNN-factors tab (decision 6).

Analogue set (decision 2, Step 0 fact 4): the card reuses the v1 route's input
helpers — landscape row, bars-open spot, `_resolve_implied_move`, carry rates,
effective regime, `build_proposals_response` — so the walk-forward analogue set
is the same one `/api/setup/proposals` shows. `/today-setup` and its module are
untouched; this module only imports from them.

All DB I/O lives here; the pure helpers are in service.py.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import psycopg
from flask import jsonify, request, send_from_directory

from packages.shared.audit_overrides import get_effective_regime
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.day_features import (
    _OPEN_STRADDLE_SQL,
    _materialize_payload,
    extract_features,
)
from packages.shared.gex_landscape import compute_implied_move
from packages.shared.knn_config import CANONICAL_KNN_CONFIG_VERSION, get_knn_config
from packages.shared.options_cache import repository as options_repo
from packages.shared.options_cache.pricing import price_proposal_legs
from packages.shared.probability import (
    _rank_analogues_with_outcomes,
    analogue_fair_value,
    compute_structural_probability,
)

from apps.web.modules.Bars.service import fetch_rth_open
from apps.web.modules.TodaySetup.routes import (
    _build_context,
    _fetch_carry_rates,
    _load_landscape,
    _normalize_db_url,
    _parse_date,
    _resolve_implied_move,
)
from apps.web.modules.TodaySetup.service import build_proposals_response

from .service import (
    ENTRY_MINUTE_PT,
    FEE_PER_CONTRACT_PER_LEG,
    QUOTE_WINDOW_PT,
    band_for_sigma,
    expected_pnl,
    fee_points,
    horizon_mix,
    knn_factor_rows,
    match_quality,
    net_debit_by_minute,
    next_reference_run,
    pick_reference_cr_id,
    reference_cell_dict,
    seed_for_date,
    select_reference_rows,
    verdict,
)

_PT = ZoneInfo("America/Los_Angeles")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
V2_DIST_DIR = (REPO_ROOT / "react_today_setup" / "dist").resolve()
V2_INDEX = "setup-v2.html"

_DEBIT_TEMPLATE_ID = "debit_spread_to_target"
_STRUCTURE_LEGS = 2


def _conn():
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(_normalize_db_url(raw))


# ── DB helpers ────────────────────────────────────────────────────────────────


def _fetch_open_straddle_im(conn, ticker: str, trade_date: dt.date, table_spot: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    """(implied move, atmiv) the harness banded the reference cells with (A3):
    first `orats_monies_minute` row at/after 06:33 PT, smallest dte > 0,
    `compute_implied_move(table_spot, atmiv, dte=1)`. (None, None) when absent."""
    if table_spot is None:
        return None, None
    floor_ts = dt.datetime.combine(trade_date, dt.time(6, 33, 0))
    with conn.cursor() as cur:
        cur.execute(_OPEN_STRADDLE_SQL, (trade_date.isoformat(), ticker, floor_ts))
        row = cur.fetchone()
    if not row or row[0] is None:
        return None, None
    try:
        iv = float(row[0])
        im = compute_implied_move(float(table_spot), iv, dte=1.0)
    except (TypeError, ValueError):
        return None, None
    return (im if im else None), iv


def _fetch_analogue_outcomes(conn, ticker: str, dates: list[dt.date]) -> list[dict]:
    """Outcome rows for the analogue dates: horizon and the signed close
    distance from the wall (Step 0: close − drift_target at horizon end)."""
    if not dates:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT trade_date, outcome_status, horizon_sessions,
                   final_close_distance_from_target, reached_touch, reached_close,
                   horizon_end_date, session_close_t15
            FROM bt_daily_outcomes_active
            WHERE ticker = %s AND feature_version = %s AND trade_date = ANY(%s)
            """,
            (ticker, CANONICAL_FEATURE_VERSION, dates),
        )
        rows = cur.fetchall()
    return [
        {
            "trade_date":       r[0].isoformat(),
            "outcome_status":   r[1],
            "horizon_sessions": r[2],
            "final_close_distance_from_target": float(r[3]) if r[3] is not None else None,
            "reached_touch":    r[4],
            "reached_close":    r[5],
            "horizon_end_date": r[6],
            "session_close_t15": float(r[7]) if r[7] is not None else None,
        }
        for r in rows
    ]


_RTH_CLOSE_SQL = """
    SELECT d, close FROM (
        SELECT (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date AS d, close,
               row_number() OVER (
                   PARTITION BY (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date
                   ORDER BY datetime DESC) AS rn
        FROM ironbeam_es_1m_bars
        WHERE (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date = ANY(%s)
          AND (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::time
              BETWEEN '06:30:00' AND '13:00:00'
    ) x WHERE rn = 1
"""


def _fetch_rth_closes(conn, dates: list[dt.date]) -> dict[dt.date, float]:
    """ES RTH session close (last 06:30–13:00 PT bar) per date — the same
    session definition the outcome runner's daily bars use (Bars/service.py)."""
    dates = sorted({d for d in dates if d is not None})
    if not dates:
        return {}
    with conn.cursor() as cur:
        cur.execute(_RTH_CLOSE_SQL, (dates,))
        return {r[0]: float(r[1]) for r in cur.fetchall() if r[1] is not None}


def t15_distances_below_target(outcomes: list[dict], closes_at_horizon: dict) -> tuple[list[float], dict]:
    """CR-AW A2: per computed analogue, the wall re-derived from the stored
    outcome — ``target = close_at_horizon − final_close_distance_from_target``
    (checked in Step 0b: equals ``pick_drift_target(walls)`` to the point on
    all 377 magnet-above rows) — and ``d15 = target − session_close_t15``,
    points below the wall at T+15 sessions. Analogues without a T+15 close or
    without a horizon close are excluded and counted.

    Returns (d15 list, {"n_computed", "n_valued", "n_no_t15_close",
    "n_no_horizon_close"}).
    """
    d15: list[float] = []
    n_no_t15 = n_no_close = 0
    for o in outcomes:
        if o.get("outcome_status") != "computed":
            continue
        ch = closes_at_horizon.get(o.get("horizon_end_date"))
        fcd = o.get("final_close_distance_from_target")
        c15 = o.get("session_close_t15")
        if ch is None or fcd is None:
            n_no_close += 1
            continue
        if c15 is None:
            n_no_t15 += 1
            continue
        target = float(ch) - float(fcd)
        d15.append(round(target - float(c15), 4))
    return d15, {
        "n_computed":         sum(1 for o in outcomes if o.get("outcome_status") == "computed"),
        "n_valued":           len(d15),
        "n_no_t15_close":     n_no_t15,
        "n_no_horizon_close": n_no_close,
    }


def _fetch_reference_rows(conn) -> tuple[Optional[str], list[dict]]:
    """Latest REF-% cr_id (else CR-AR) and its debit / close / pooled rows."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT cr_id, max(created_at) FROM bt_edge_backtest_results GROUP BY cr_id"
        )
        ids = [(r[0], r[1]) for r in cur.fetchall()]
    cr_id = pick_reference_cr_id(ids)
    if cr_id is None:
        return None, []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT run_id, cr_id, structure_type, outcome_type, distance_band,
                   post_touch_pattern, partition, threshold, n_dates, n_filled,
                   n_settled, mean_pnl, win_rate, wilson_lo, wilson_hi,
                   baseline_mean, beat_baseline, created_at, mean_width_actual
            FROM bt_edge_backtest_results
            WHERE cr_id = %s AND structure_type = 'debit' AND outcome_type = 'close'
              AND post_touch_pattern IS NULL
            """,
            (cr_id,),
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return cr_id, rows


def _fetch_feature_corpus(conn, ticker: str, before: dt.date) -> list[tuple[str, str, dict]]:
    """(trade_date iso, regime_at_classification, feature_vector) for every
    active corpus row before `before` — the KNN pool and the magnet-above
    percentile corpus (decision 6)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT trade_date, regime_at_classification, feature_vector
            FROM bt_daily_features_active
            WHERE ticker = %s AND feature_version = %s AND trade_date < %s
            """,
            (ticker, CANONICAL_FEATURE_VERSION, before),
        )
        rows = cur.fetchall()
    return [(r[0].isoformat(), r[1], r[2] or {}) for r in rows]


def _quote_by_minute(legs_out: list[dict], trade_date: dt.date, width: Optional[float]) -> list[dict]:
    """Cached minute bars for both legs over QUOTE_WINDOW_PT → spread price per
    minute (A8). Cache read only — a miss is an empty strip, never a fetch."""
    long_ = next((l for l in legs_out if l["side"] == "long" and l.get("opra")), None)
    short = next((l for l in legs_out if l["side"] == "short" and l.get("opra")), None)
    if not long_ or not short:
        return []
    start = dt.datetime.combine(trade_date, QUOTE_WINDOW_PT[0])
    end = dt.datetime.combine(trade_date, QUOTE_WINDOW_PT[1])
    try:
        bars = {
            "long":  options_repo.get_bars_for_contract(long_["opra"], start, end),
            "short": options_repo.get_bars_for_contract(short["opra"], start, end),
        }
    except Exception as e:  # pragma: no cover - the strip is optional
        log.warning("quote_by_minute unavailable for %s: %s", trade_date, e)
        return []
    return net_debit_by_minute(bars, width)


# ── Card assembly ─────────────────────────────────────────────────────────────


def _proposal_legs_for_pricing(proposal: dict, trade_date: dt.date) -> tuple[list[dict], dt.date]:
    """pl-data leg shape for price_proposal_legs; expiry = trade_date +
    expiry_dte_target calendar days (the v1 card convention, Step 0 fact 5)."""
    dte = int(proposal.get("expiry_dte_target") or 0)
    expiration = trade_date + dt.timedelta(days=dte)
    legs = [
        {
            "strike":     float(l["strike"]),
            "expiration": expiration,
            "flag":       "c" if l.get("type") == "call" else "p",
            "side":       l["side"],
            "qty":        int(l.get("quantity") or 1),
        }
        for l in proposal.get("legs", [])
    ]
    return legs, expiration


def build_card(conn, ticker: str, trade_date: dt.date) -> tuple[dict, int]:
    """Assemble the card payload. Returns (payload, http_status)."""
    warnings: list[str] = []

    row = _load_landscape(conn, ticker, trade_date)
    if not row:
        return {
            "ok": False,
            "error": f"no landscape for ({ticker}, {trade_date.isoformat()}) — backfill required",
        }, 404
    landscape_rows, table_spot = row

    # ── v1 inputs (decision 2: the same analogue set as /api/setup/proposals) ─
    spot_source = "bars"
    spot = fetch_rth_open(conn, trade_date)
    if spot is None or spot <= 0:
        if table_spot is not None:
            spot, spot_source = float(table_spot), "landscape"
        else:
            spot, spot_source = 5000.0, "default"
    implied_move = _resolve_implied_move(conn, ticker, trade_date, spot)
    risk_free_rate, yield_rate = _fetch_carry_rates(conn, ticker, trade_date)

    payload = _materialize_payload(landscape_rows, spot, implied_move)
    effective_regime = get_effective_regime(conn, ticker, trade_date)
    if effective_regime and effective_regime != payload.get("regime", {}).get("regime"):
        payload = dict(payload)
        payload["regime"] = dict(payload.get("regime") or {})
        payload["regime"]["regime"] = effective_regime

    context = _build_context(trade_date, ticker, spot, implied_move, payload)
    context["spot_source"] = spot_source
    regime_block = payload.get("regime") or {}
    regime = regime_block.get("regime")
    drift_target = regime_block.get("drift_target")

    today_features = extract_features(payload, spot, implied_move)

    structural_probability: Optional[dict] = None
    try:
        structural_probability = compute_structural_probability(
            today_features, conn, k=200, ticker=ticker,
            exclude_date=trade_date.isoformat(), before_date=trade_date.isoformat(),
            regime_kind=effective_regime,
        )
    except Exception as e:  # pragma: no cover - surfaced as a warning
        log.warning("structural_probability failed for (%s, %s): %s", ticker, trade_date, e)
        warnings.append(f"structural probability unavailable: {e}")

    analogue_rows = _rank_analogues_with_outcomes(
        today_features, conn, 200, CANONICAL_FEATURE_VERSION,
        ticker=ticker, exclude_date=trade_date.isoformat(), before_date=trade_date.isoformat(),
    )
    analogue_dates = [dt.date.fromisoformat(r["trade_date"]) for r in analogue_rows]
    outcomes = _fetch_analogue_outcomes(conn, ticker, analogue_dates)
    computed = [o for o in outcomes if o["outcome_status"] == "computed"]

    # ── The proposed structure (v1 debit proposal) ──────────────────────────
    proposals = build_proposals_response(
        payload, spot, implied_move, context, "cluster_centered",
        risk_free_rate=risk_free_rate, yield_rate=yield_rate,
    )["proposals"]
    debit = next((p for p in proposals if p.get("template_id") == _DEBIT_TEMPLATE_ID), None)

    structure: dict = {"template_id": _DEBIT_TEMPLATE_ID, "listed": False, "listed_reason": None, "legs": []}
    quote: dict = {"net_debit": None, "quote_minute": None, "quote_valid": None, "stale_quote": None,
                   "market_implied": None, "entry_minute_pt": ENTRY_MINUTE_PT.strftime("%H:%M"),
                   "by_minute": [], "window_pt": [QUOTE_WINDOW_PT[0].strftime("%H:%M"), QUOTE_WINDOW_PT[1].strftime("%H:%M")],
                   "warnings": []}
    width: Optional[float] = None
    if debit is None:
        structure["listed_reason"] = "no_magnet_proposal"
        warnings.append(f"no {_DEBIT_TEMPLATE_ID} proposal for regime {regime!r}")
    else:
        raw_legs, expiration = _proposal_legs_for_pricing(debit, trade_date)
        entry_pt = dt.datetime.combine(trade_date, ENTRY_MINUTE_PT)
        live = trade_date == dt.datetime.now(_PT).date()
        try:
            priced = price_proposal_legs(
                raw_legs, trade_date=trade_date, entry_pt=entry_pt,
                r=risk_free_rate, q=yield_rate, live=live,
            )
        except Exception as e:  # pragma: no cover - surfaced, never a hard fail
            log.exception("price_proposal_legs failed for (%s, %s)", ticker, trade_date)
            priced = {"legs": [], "net_debit": None, "width_actual": None, "width_nominal": None,
                      "listed": True, "listed_reason": None, "stale_quote": False,
                      "warnings": [f"pricing failed: {e}"]}
        quote["warnings"] = list(priced.get("warnings") or [])
        listed = bool(priced.get("listed", True))
        width_actual = priced.get("width_actual")
        width_nominal = priced.get("width_nominal")
        width = float(width_actual) if width_actual else (float(width_nominal) if width_nominal else 10.0)
        legs_out = []
        for l, raw in zip(priced.get("legs") or [], raw_legs):
            legs_out.append({
                "side":         raw["side"],
                "type":         "call" if raw["flag"] == "c" else "put",
                "strike_es":    raw["strike"],
                "strike_spx":   l.get("spx_strike"),
                "strike_spx_raw": l.get("spx_strike_raw"),
                "listed":       l.get("listed", True),
                "opra":         l.get("opra"),
                "bid":          _fnum(l.get("bid")),
                "ask":          _fnum(l.get("ask")),
                "mid":          _fnum(l.get("mid")),
                "delta":        _fnum(l.get("delta")),
                "quote_valid":  l.get("quote_valid"),
                "stale_quote":  l.get("stale_quote", False),
                "quote_minute": l.get("quote_minute"),
            })
        short = next((l for l in legs_out if l["side"] == "short"), None)
        long_ = next((l for l in legs_out if l["side"] == "long"), None)
        structure = {
            "template_id":   _DEBIT_TEMPLATE_ID,
            "listed":        listed,
            "listed_reason": priced.get("listed_reason") if not listed else None,
            "legs":          legs_out,
            "short_strike_spx": short["strike_spx"] if short else None,
            "long_strike_spx":  long_["strike_spx"] if long_ else None,
            "width_nominal": _fnum(width_nominal),
            "width_actual":  _fnum(width_actual),
            "expiry":        expiration.isoformat(),
            "dte_calendar":  (expiration - trade_date).days,
            "expiry_dte_target": debit.get("expiry_dte_target"),
            "direction":     "call" if raw_legs and raw_legs[0]["flag"] == "c" else "put",
            "rationale":     debit.get("rationale"),
            "source":        debit.get("source"),
        }
        net_debit = _fnum(priced.get("net_debit"))
        quote_minutes = [l.get("quote_minute") for l in legs_out if l.get("quote_minute")]
        quote.update({
            "net_debit":     net_debit,
            "quote_minute":  max(quote_minutes) if quote_minutes else None,
            "quote_valid":   all(l.get("quote_valid") for l in legs_out) if legs_out else None,
            "stale_quote":   bool(priced.get("stale_quote", False)),
            "market_implied": round(net_debit / width, 4) if (net_debit is not None and width) else None,
            "by_minute":     _quote_by_minute(legs_out, trade_date, width) if listed else [],
            "window_pt":     [QUOTE_WINDOW_PT[0].strftime("%H:%M"), QUOTE_WINDOW_PT[1].strftime("%H:%M")],
        })

    # ── Fair value at T+15 sessions (decision 2 / 3, amendment A2) ──────────
    fv_width = width or 10.0
    closes_at_horizon = _fetch_rth_closes(conn, [o.get("horizon_end_date") for o in computed])
    d15, counts = t15_distances_below_target(computed, closes_at_horizon)
    fair = analogue_fair_value(d15, fv_width, n_boot=1000, seed=seed_for_date(trade_date))
    fair["basis"] = "t15"
    fair["valuation_horizon_sessions"] = 15
    fair.update(counts)
    fair["horizon_mix"] = horizon_mix(computed)   # outcome horizons of the set, for reference only
    fee_pts = fee_points(_STRUCTURE_LEGS)
    pnl = expected_pnl(fair, quote["net_debit"], fee_pts)
    pnl["fee_per_contract_per_leg"] = FEE_PER_CONTRACT_PER_LEG
    the_verdict = verdict(quote["net_debit"], fair.get("max_price"), listed=structure["listed"])

    # ── Band (decision 4, A3) ───────────────────────────────────────────────
    im_open, atmiv_open = _fetch_open_straddle_im(conn, ticker, trade_date, table_spot)
    sigma = None
    if drift_target is not None and im_open and table_spot is not None:
        sigma = round((float(drift_target) - float(table_spot)) / im_open, 4)
    today_band = band_for_sigma(sigma)
    cr_id, ref_rows = _fetch_reference_rows(conn)
    cells = select_reference_rows(ref_rows)
    rerun_at = max((r["created_at"] for r in ref_rows if r.get("created_at")), default=None)
    rerun_date = rerun_at.date() if isinstance(rerun_at, dt.datetime) else rerun_at
    band = {
        "today":            today_band,
        "sigma":            sigma,
        "im_open_straddle": round(im_open, 4) if im_open else None,
        "atmiv_open":       round(atmiv_open, 6) if atmiv_open else None,
        "table_spot":       float(table_spot) if table_spot is not None else None,
        "thresholds":       {"near_max": 1.5, "far_min": 2.0},
        "cr_id":            cr_id,
        "rows":             [reference_cell_dict(cells.get(b)) for b in ("near", "mid", "far") if cells.get(b)],
        "all":              reference_cell_dict(cells.get("all")),
        "today_cell":       reference_cell_dict(cells.get(today_band)) if today_band else None,
    }

    # ── Analogue facts ──────────────────────────────────────────────────────
    sp = structural_probability or {}
    k = len(analogue_rows)
    analogues = {
        "k":                  k,
        "k_with_outcomes":    len(computed),
        "touch_rate":         sp.get("touch_rate"),
        "touch_ci":           [sp.get("touch_ci_lower"), sp.get("touch_ci_upper")],
        "mean_days_to_reach": sp.get("mean_days_to_reach"),
        "close_at_wall_rate": sp.get("close_rate"),
        "full_payout_rate":   fair.get("full_payout_rate"),
        "any_payout_rate":    fair.get("any_payout_rate"),
        "horizon_mix":        fair["horizon_mix"],
        "date_range":         [min(analogue_dates).isoformat(), max(analogue_dates).isoformat()] if analogue_dates else None,
        "similarity_ceiling": get_knn_config().get("distance_ceiling"),
        "note":               sp.get("note"),
    }

    # ── KNN factors tab (decision 6) ────────────────────────────────────────
    knn_cfg = get_knn_config()
    corpus = _fetch_feature_corpus(conn, ticker, trade_date)
    by_date = {d: v for (d, _, v) in corpus}
    magnet_above = [v for (_, reg, v) in corpus if reg == "magnet-above"]
    analogue_vectors = [by_date[r["trade_date"]] for r in analogue_rows if r["trade_date"] in by_date]
    factors = knn_factor_rows(today_features, knn_cfg.get("feature_weights"), magnet_above, analogue_vectors)
    knn = {
        "config_version":    CANONICAL_KNN_CONFIG_VERSION,
        "distance_ceiling":  knn_cfg.get("distance_ceiling"),
        "z_diff_cap":        knn_cfg.get("z_diff_cap"),
        "half_life_months":  knn_cfg.get("half_life_months"),
        "corpus_n":          len(magnet_above),
        "corpus_scope":      f"magnet-above days before {trade_date.isoformat()}",
        "analogue_n":        len(analogue_vectors),
        "factors":           factors,
        "match_quality":     match_quality(factors),
    }

    # ── Manage / stamp (decisions 5, 7) ─────────────────────────────────────
    today_cell = band["today_cell"]
    manage = {
        "enter_under_max": fair.get("max_price"),
        "hold_to_close_mean_pnl": today_cell["mean_pnl"] if today_cell else None,
        "hold_to_close_baseline": today_cell["baseline_mean"] if today_cell else None,
        "dte_calendar": structure.get("dte_calendar"),
        "expiry": structure.get("expiry"),
        "watches": [
            {"key": "wall_half_life", "label": "Wall watch", "status": "untested", "value": None,
             "note": "wall gamma half-life by expiry — needs the wall table (CR-AT)"},
            {"key": "wall_trend", "label": "Wall trend", "status": "untested", "value": None,
             "note": "needs the wall table (CR-AT)"},
            {"key": "vol_state", "label": "Vol state", "status": "untested",
             "value": _vol_state_value(factors),
             "note": "IV rank and VRP are not populated — no action either way until they read"},
        ],
    }
    stamp = {
        "cr_id":        cr_id,
        "run_id":       str(ref_rows[0]["run_id"]) if ref_rows and ref_rows[0].get("run_id") else None,
        "cell":         f"{today_band} · debit · hold to close" if today_band else None,
        "n":            today_cell["n"] if today_cell else None,
        "rerun_date":   rerun_date.isoformat() if rerun_date else None,
        "next_run":     next_reference_run(rerun_date),
        "fees_included": True,
        "fee_per_contract_per_leg": FEE_PER_CONTRACT_PER_LEG,
        "fee_pts":      fee_pts,
        "quote_minute_pt": ENTRY_MINUTE_PT.strftime("%H:%M"),
        "feature_version": CANONICAL_FEATURE_VERSION,
        "knn_config_version": CANONICAL_KNN_CONFIG_VERSION,
    }

    return {
        "ok":         True,
        "date":       trade_date.isoformat(),
        "ticker":     ticker,
        "context":    context,
        "wall": {
            "price_es":     float(drift_target) if drift_target is not None else None,
            "gex_b":        (debit or {}).get("source", {}).get("dominant_wall_gex_b") if debit else None,
            "sigma":        sigma,
            "band":         today_band,
            "regime":       regime,
            "above_spot":   (float(drift_target) > spot) if drift_target is not None else None,
        },
        "structure":  structure,
        "quote":      quote,
        "fair_value": fair,
        "pnl":        pnl,
        "verdict":    the_verdict,
        "band":       band,
        "analogues":  analogues,
        "structural_probability": structural_probability,
        "knn":        knn,
        "manage":     manage,
        "stamp":      stamp,
        "warnings":   warnings,
    }, 200


def _vol_state_value(factors: list[dict]) -> Optional[dict]:
    for r in factors:
        if r["key"] == "implied_move_1d":
            return {"implied_move_percentile": r.get("percentile"), "implied_move": r.get("today")}
    return None


def _fnum(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── Route registration ────────────────────────────────────────────────────────


def _v2_build_ready() -> bool:
    return V2_DIST_DIR.exists() and (V2_DIST_DIR / V2_INDEX).exists()


def register_setup_v2_routes(server) -> None:
    """Wire /setup-v2 and /api/setup-v2/card onto the Flask server."""
    if "setup_v2_card" in server.view_functions:
        return

    def setup_v2_index():
        if not _v2_build_ready():
            return ("Setup v2 build not found. Run: cd react_today_setup && npm run build", 503)
        return send_from_directory(str(V2_DIST_DIR), V2_INDEX)

    def setup_v2_card():
        date_s = (request.args.get("date") or "").strip()
        trade_date = _parse_date(date_s)
        if not trade_date:
            return jsonify({"ok": False, "error": "date is required (YYYY-MM-DD)"}), 400
        ticker = (request.args.get("ticker") or "SPX").strip() or "SPX"
        try:
            conn = _conn()
        except Exception as e:
            return jsonify({"ok": False, "error": f"db connect failed: {e}"}), 500
        try:
            payload, status = build_card(conn, ticker, trade_date)
            return jsonify(payload), status
        except Exception as e:
            log.exception("setup_v2_card failed for (%s, %s)", ticker, trade_date)
            return jsonify({"ok": False, "error": str(e)}), 500
        finally:
            try:
                conn.close()
            except Exception:
                pass

    server.add_url_rule("/setup-v2", endpoint="setup_v2_index", view_func=setup_v2_index, methods=["GET"])
    server.add_url_rule("/setup-v2/", endpoint="setup_v2_index_slash", view_func=setup_v2_index, methods=["GET"])
    server.add_url_rule("/api/setup-v2/card", endpoint="setup_v2_card", view_func=setup_v2_card, methods=["GET"])
