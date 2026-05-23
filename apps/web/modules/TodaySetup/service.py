"""TodaySetup service — pure functions for /api/setup/proposals (CR-015).

Stateless: wraps generate_proposals for the endpoint. All DB I/O lives in
routes.py. This module is responsible only for building the response shape.

Public entry points:
    build_proposals_response(landscape_payload, spot, implied_move, context)
        → dict  (the full JSON-serialisable response body)
"""
from __future__ import annotations

from dataclasses import asdict

from packages.shared.strategy_templates import generate_proposals, Leg, TradeProposal


def _leg_to_dict(leg: Leg) -> dict:
    return {
        "side": leg.side,
        "type": leg.type,
        "strike": leg.strike,
        "quantity": leg.quantity,
    }


def _proposal_to_dict(p: TradeProposal) -> dict:
    d = {
        "template_id": p.template_id,
        "template_kind": p.template_kind,
        "anchor_strategy": p.anchor_strategy,
        "rationale": p.rationale,
        "legs": [_leg_to_dict(leg) for leg in p.legs],
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
) -> dict:
    """Build the full /api/setup/proposals response dict.

    Args:
        landscape_payload: Output of _materialize_payload.
        spot: Reference spot price for the day.
        implied_move: 1-day 1σ implied move in points.
        context: Pre-built context block (date, ticker, regime, etc.).
        anchor_strategy: Key into ANCHOR_STRATEGIES registry.

    Returns:
        JSON-serialisable dict with "ok", "context", and "proposals" keys.
    """
    proposals = generate_proposals(
        landscape_payload, spot, implied_move, anchor_strategy
    )
    return {
        "ok": True,
        "context": context,
        "proposals": [_proposal_to_dict(p) for p in proposals],
    }
