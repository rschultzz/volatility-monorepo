"""TodaySetup service — pure functions for /api/setup/proposals (CR-015).

Stateless: wraps generate_proposals for the endpoint. All DB I/O lives in
routes.py. This module is responsible only for building the response shape.

Public entry points:
    build_proposals_response(landscape_payload, spot, implied_move, context)
        → dict  (the full JSON-serialisable response body)
    apply_direction_qualification(proposals, structural_probability)
        → list[dict]  (CR-AV: proposals unchanged; post_touch annotated with
                       advisory_only / advisory — never filters or promotes)
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from packages.shared.forward_math import compute_spx_strike
from packages.shared.post_touch_qualification import dte_to_timeframe
from packages.shared.strategy_templates import generate_proposals, Leg, TradeProposal

# Template IDs for the two magnet-regime spread variants
_CREDIT_TEMPLATE_ID = "directional_spread_to_target"   # short at target, long further OTM
_DEBIT_TEMPLATE_ID  = "debit_spread_to_target"         # long inside target, short at target

_MAGNET_REGIMES = frozenset({"magnet-above", "magnet-below"})


# CR-AV decision 1: the credit-fade template is no longer emitted as a live
# proposal (refuted CR-AH, held through CR-AM/AN/AP/AR). The template class and
# generate_proposals are unchanged — the harness and its scripts still use them.
_LIVE_EXCLUDED_TEMPLATE_IDS = frozenset({_CREDIT_TEMPLATE_ID})


def _continuation_key(regime_kind: str) -> str:
    """magnet-above → 'above' (price keeps going through the magnet);
    magnet-below → 'below'."""
    return "above" if regime_kind == "magnet-above" else "below"


def apply_direction_qualification(
    proposals: list[dict],
    structural_probability: dict,
) -> list[dict]:
    """CR-AV decision 2: advisory only — never filters or promotes.

    Returns `proposals` unchanged (no `confidence_badge`, no drop, no
    promotion) and annotates `structural_probability["post_touch"]` in place:

        advisory_only: True                       (decision 4; the card reads it)
        advisory: {pattern_label, n, n_pooled, timeframe, direction,
                   fraction, wilson_lo, wilson_hi}   (label · n · "t5 above 61%")

    `filter_mode`, `fractions`, `wilson_cis`, `pattern_label`, `same_bucket_n`
    and `total_touchers` stay in the payload for audit; nothing here decides
    display from `filter_mode`. Evidence: CR-AL (label does not predict P&L),
    CR-AM (no reversal) — the gate is demoted to an advisory block.
    """
    post_touch = structural_probability.get("post_touch")
    if post_touch is None:
        return proposals

    regime_kind = structural_probability.get("regime_kind", "") or ""
    magnet_props = [
        p for p in proposals
        if p.get("source", {}).get("type") == "regime_target"
    ]
    proposal_dte = magnet_props[0].get("expiry_dte_target") if magnet_props else None
    timeframe = dte_to_timeframe(proposal_dte)
    direction = _continuation_key(regime_kind) if regime_kind in _MAGNET_REGIMES else None

    fraction = wilson_lo = wilson_hi = None
    if timeframe and direction:
        try:
            fraction = (post_touch.get("fractions") or {})[timeframe][direction]
        except (KeyError, TypeError):
            fraction = None
        try:
            ci = (post_touch.get("wilson_cis") or {})[timeframe][direction]
            wilson_lo, wilson_hi = ci[0], ci[1]
        except (KeyError, TypeError, IndexError):
            wilson_lo = wilson_hi = None

    n = post_touch.get("same_bucket_n")
    if n is None:
        n = post_touch.get("total_touchers")

    post_touch["advisory_only"] = True
    post_touch["advisory"] = {
        "pattern_label": post_touch.get("pattern_label"),
        "n": n,
        "n_pooled": post_touch.get("total_touchers"),
        "timeframe": timeframe,
        "direction": direction,
        "fraction": fraction,
        "wilson_lo": wilson_lo,
        "wilson_hi": wilson_hi,
    }
    return proposals


def _leg_to_dict(leg: Leg, *, strike_spx: Optional[int] = None) -> dict:
    d = {
        "side": leg.side,
        "type": leg.type,
        "strike": leg.strike,
        "quantity": leg.quantity,
    }
    if strike_spx is not None:
        d["strike_spx"] = strike_spx
    return d


def _proposal_to_dict(
    p: TradeProposal,
    *,
    risk_free_rate: Optional[float] = None,
    yield_rate: Optional[float] = None,
) -> dict:
    """Serialise a TradeProposal to a JSON-ready dict.

    When risk_free_rate and yield_rate are provided, each leg gains a
    strike_spx field: the ES-space strike converted to the nearest
    SPX 5-point chain increment via compute_spx_strike.
    """
    legs_out = []
    for leg in p.legs:
        spx = None
        if risk_free_rate is not None and yield_rate is not None:
            spx = compute_spx_strike(
                leg.strike, p.expiry_dte_target, risk_free_rate, yield_rate
            )
        legs_out.append(_leg_to_dict(leg, strike_spx=spx))

    d = {
        "template_id": p.template_id,
        "template_kind": p.template_kind,
        "anchor_strategy": p.anchor_strategy,
        "rationale": p.rationale,
        "legs": legs_out,
        "expiry_dte_target": p.expiry_dte_target,
        "expiry_dte_bucket": p.expiry_dte_bucket,
        "source": p.source,
    }
    if p.wing_distance_recipe:
        d["wing_distance_recipe"] = p.wing_distance_recipe
    return d


def build_proposals_response(
    landscape_payload: dict,
    spot: float,
    implied_move: float,
    context: dict,
    anchor_strategy: str = "cluster_centered",
    *,
    risk_free_rate: Optional[float] = None,
    yield_rate: Optional[float] = None,
) -> dict:
    """Build the full /api/setup/proposals response dict.

    Args:
        landscape_payload: Output of _materialize_payload.
        spot: Reference spot price for the day.
        implied_move: 1-day 1σ implied move in points.
        context: Pre-built context block (date, ticker, regime, etc.).
        anchor_strategy: Key into ANCHOR_STRATEGIES registry.
        risk_free_rate: Annualised risk-free rate from orats_monies_minute;
            used to populate strike_spx on each leg. None → omit strike_spx.
        yield_rate: Annualised continuous dividend yield; same source/usage.

    Returns:
        JSON-serialisable dict with "ok", "context", and "proposals" keys.
    """
    proposals = generate_proposals(
        landscape_payload, spot, implied_move, anchor_strategy
    )
    # CR-AV decision 1: credit-fade is not a live proposal; no placeholder.
    proposals = [p for p in proposals if p.template_id not in _LIVE_EXCLUDED_TEMPLATE_IDS]
    return {
        "ok": True,
        "context": context,
        "proposals": [
            _proposal_to_dict(p, risk_free_rate=risk_free_rate, yield_rate=yield_rate)
            for p in proposals
        ],
    }
