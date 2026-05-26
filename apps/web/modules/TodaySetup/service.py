"""TodaySetup service — pure functions for /api/setup/proposals (CR-015).

Stateless: wraps generate_proposals for the endpoint. All DB I/O lives in
routes.py. This module is responsible only for building the response shape.

Public entry points:
    build_proposals_response(landscape_payload, spot, implied_move, context)
        → dict  (the full JSON-serialisable response body)
    apply_direction_qualification(proposals, structural_probability)
        → list[dict]  (filtered + badged proposals based on post-touch data)
"""
from __future__ import annotations

from dataclasses import asdict

from packages.shared.post_touch_qualification import (
    credit_direction_qualifies,
    debit_direction_qualifies,
)
from packages.shared.strategy_templates import generate_proposals, Leg, TradeProposal

# Template IDs for the two magnet-regime spread variants
_CREDIT_TEMPLATE_ID = "directional_spread_to_target"   # short at target, long further OTM
_DEBIT_TEMPLATE_ID  = "debit_spread_to_target"         # long inside target, short at target

_MAGNET_REGIMES = frozenset({"magnet-above", "magnet-below"})


def _add_badge(proposal: dict, badge: str) -> dict:
    """Return a copy of proposal dict with confidence_badge set."""
    return {**proposal, "confidence_badge": badge}


def apply_direction_qualification(
    proposals: list[dict],
    structural_probability: dict,
) -> list[dict]:
    """Filter and badge magnet-regime proposals using post-touch direction data.

    Decision flow:
        filter_mode=insufficient / zero_dte_corpus_insufficient
            → badge all proposals; pass through unchanged
        regime not in magnet regimes
            → pass through unchanged (pin/bounded/no-trade unaffected)
        magnet regime + strict/pooled-fallback post_touch:
            credit_qualifies only  → keep credit proposal, badge "credit-fade supported"
            debit_qualifies only   → keep debit proposal,  badge "debit-to-target supported"
            both or neither        → keep both, badge "mixed pattern — no clear direction"

    Args:
        proposals:              List of serialised proposal dicts (from _proposal_to_dict).
        structural_probability: Full SP dict from compute_structural_probability().
    Returns:
        Modified proposals list (same length or shorter for single-direction cases).
    """
    post_touch = structural_probability.get("post_touch")
    if post_touch is None:
        return proposals  # no post-touch data → legacy pass-through

    filter_mode = post_touch.get("filter_mode")

    # ── Thin-corpus branches: badge and pass through ──────────────────────────
    if filter_mode == "insufficient":
        badge = "low-confidence — post-touch sample insufficient"
        return [_add_badge(p, badge) for p in proposals]

    if filter_mode == "zero_dte_corpus_insufficient":
        badge = "0DTE corpus insufficient"
        return [_add_badge(p, badge) for p in proposals]

    # ── Direction-selection for magnet regimes only ───────────────────────────
    regime_kind = structural_probability.get("regime_kind", "")
    if regime_kind not in _MAGNET_REGIMES:
        return proposals  # magnetic-pin, bounded, etc — no direction selection

    # Derive proposal DTE from the first regime_target proposal (both spread
    # templates share the same DTE since they key off the same drift cluster).
    magnet_props = [
        p for p in proposals
        if p.get("source", {}).get("type") == "regime_target"
    ]
    proposal_dte = magnet_props[0].get("expiry_dte_target") if magnet_props else None

    credit_ok = credit_direction_qualifies(post_touch, regime_kind, proposal_dte)
    debit_ok  = debit_direction_qualifies(post_touch, regime_kind, proposal_dte)

    # Partition by template ID; preserve non-magnet proposals (pin, condor, etc.)
    credit_props = [p for p in proposals if p.get("template_id") == _CREDIT_TEMPLATE_ID]
    debit_props  = [p for p in proposals if p.get("template_id") == _DEBIT_TEMPLATE_ID]
    other_props  = [
        p for p in proposals
        if p.get("template_id") not in (_CREDIT_TEMPLATE_ID, _DEBIT_TEMPLATE_ID)
    ]

    if credit_ok and not debit_ok:
        direction_props = [_add_badge(p, "credit-fade supported") for p in credit_props]
    elif debit_ok and not credit_ok:
        direction_props = [_add_badge(p, "debit-to-target supported") for p in debit_props]
    else:
        # Both qualify (rare) or neither qualifies (mixed) → emit both
        badge = "mixed pattern — no clear direction"
        direction_props = (
            [_add_badge(p, badge) for p in credit_props]
            + [_add_badge(p, badge) for p in debit_props]
        )

    return other_props + direction_props


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
