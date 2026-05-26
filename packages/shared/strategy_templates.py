"""Strategy templates for trade proposal generation (CR-015).

Six templates map (regime, cluster) → concrete trade proposals.  Templates
are pure functions of the landscape payload — no DB I/O.

Public entry point:
    generate_proposals(landscape_payload, spot, implied_move)
        → list[TradeProposal]

Template registry:
    TEMPLATES — ordered list of all six Template instances

DTE target mapping:
    DTE_TARGET_BY_BUCKET — bucket label → representative DTE target
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from packages.shared.strike_anchors import ANCHOR_STRATEGIES


# ── DTE target lookup ─────────────────────────────────────────────────────────

DTE_TARGET_BY_BUCKET: dict[str, int] = {
    "0DTE": 1,
    "1-7":  3,
    "8-30": 15,
    "30+":  45,
}


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Leg:
    side: Literal["long", "short"]
    type: Literal["call", "put"]
    strike: float
    quantity: int = 1


@dataclass
class TradeProposal:
    template_id: str
    template_kind: Literal["butterfly", "spread", "condor", "no_trade"]
    anchor_strategy: str
    rationale: str
    legs: list[Leg]
    expiry_dte_target: int
    expiry_dte_bucket: str
    source: dict
    wing_distance_recipe: str = ""  # butterfly-only; empty for other kinds


# ── Template protocol ─────────────────────────────────────────────────────────

@runtime_checkable
class Template(Protocol):
    template_id: str
    template_kind: str

    def applies_to(
        self,
        regime_block: dict,
        cluster: dict | None,
        clusters_all: list[dict],
        spot: float,
        implied_move: float,
    ) -> bool: ...

    def propose(
        self,
        regime_block: dict,
        cluster: dict | None,
        clusters_all: list[dict],
        spot: float,
        implied_move: float,
        anchor_strategy: str = "cluster_centered",
    ) -> TradeProposal: ...


# ── Helpers ───────────────────────────────────────────────────────────────────

_PIN_REGIMES = frozenset({"magnetic-pin", "pinned"})
_MAGNET_REGIMES = frozenset({"magnet-above", "magnet-below"})
_NOTRADE_REGIMES = frozenset({"feature", "untethered", "amplification", "broken-magnet"})


def _bucket_dte(cluster: dict) -> tuple[str, int]:
    bucket = cluster.get("bucket") or "8-30"
    return bucket, DTE_TARGET_BY_BUCKET.get(bucket, 15)


def _cluster_source(cluster: dict, regime_block: dict) -> dict:
    return {
        "type": "cluster",
        "cluster_center": cluster.get("center_price"),
        "cluster_quality": cluster.get("quality"),
        "cluster_max_gex": cluster.get("max_gex"),
        "cluster_avg_fwhm": cluster.get("avg_fwhm"),
        "regime": regime_block.get("regime"),
    }


# ── Butterfly templates ───────────────────────────────────────────────────────

class _PinButterflyTemplate:
    template_kind = "butterfly"

    def __init__(self, template_id: str, wing_distance_recipe: str, label: str):
        self.template_id = template_id
        self._recipe = wing_distance_recipe
        self._label = label  # human-readable label for rationale

    def applies_to(
        self,
        regime_block: dict,
        cluster: dict | None,
        clusters_all: list[dict],
        spot: float,
        implied_move: float,
    ) -> bool:
        if cluster is None:
            return False
        regime = regime_block.get("regime", "")
        if regime not in _PIN_REGIMES:
            return False
        return cluster.get("quality") == "pin"

    def propose(
        self,
        regime_block: dict,
        cluster: dict | None,
        clusters_all: list[dict],
        spot: float,
        implied_move: float,
        anchor_strategy: str = "cluster_centered",
    ) -> TradeProposal:
        strategy = ANCHOR_STRATEGIES[anchor_strategy]
        lo, body, hi = strategy.butterfly_strikes(cluster, self._recipe, implied_move)
        bucket, dte_target = _bucket_dte(cluster)
        half_wing = hi - body
        return TradeProposal(
            template_id=self.template_id,
            template_kind="butterfly",
            anchor_strategy=anchor_strategy,
            rationale=(
                f"Pin cluster at {body:.0f} ({self._label} wings ±{half_wing:.0f}pt). "
                f"Body at cluster center; wings span {self._recipe.replace('_', '-')}."
            ),
            legs=[
                Leg(side="long",  type="call", strike=lo),
                Leg(side="short", type="call", strike=body, quantity=2),
                Leg(side="long",  type="call", strike=hi),
            ],
            expiry_dte_target=dte_target,
            expiry_dte_bucket=bucket,
            source=_cluster_source(cluster, regime_block),
            wing_distance_recipe=self._recipe,
        )


# ── Directional spread ────────────────────────────────────────────────────────

class _DirectionalSpreadTemplate:
    template_id = "directional_spread_to_target"
    template_kind = "spread"
    _WIDTH_PTS = 10.0

    def applies_to(
        self,
        regime_block: dict,
        cluster: dict | None,
        clusters_all: list[dict],
        spot: float,
        implied_move: float,
    ) -> bool:
        # Cluster-based iteration: skip when called with a cluster — this
        # template is regime-driven, not cluster-driven.
        if cluster is not None:
            return False
        regime = regime_block.get("regime", "")
        return regime in _MAGNET_REGIMES and "drift_target" in regime_block

    def propose(
        self,
        regime_block: dict,
        cluster: dict | None,
        clusters_all: list[dict],
        spot: float,
        implied_move: float,
        anchor_strategy: str = "cluster_centered",
    ) -> TradeProposal:
        strategy = ANCHOR_STRATEGIES[anchor_strategy]
        drift_target = float(regime_block["drift_target"])
        short, long_, direction = strategy.spread_strikes(drift_target, spot, self._WIDTH_PTS)
        regime = regime_block.get("regime")
        dom_gex = (regime_block.get("dominant_wall") or {}).get("gex", 0.0)
        dom_gex_b = dom_gex / 1e9 if dom_gex else 0.0

        # Pick dominant bucket from the first cluster that is in the drift direction
        # (or fall back to 8-30 DTE which is the standard single-day trading bucket).
        drift_cluster = next(
            (
                c for c in clusters_all
                if abs(c.get("center_price", 0.0) - drift_target) < 30
            ),
            None,
        )
        bucket, dte_target = (
            _bucket_dte(drift_cluster) if drift_cluster else ("8-30", 15)
        )

        return TradeProposal(
            template_id=self.template_id,
            template_kind="spread",
            anchor_strategy=anchor_strategy,
            rationale=(
                f"Drift target {drift_target:.0f} {'above' if direction == 'call' else 'below'} "
                f"spot {spot:.0f} ({drift_target - spot:+.0f}pt). "
                f"Short {direction} at target, long {self._WIDTH_PTS:.0f}pt further OTM. "
                f"Dominant wall: {dom_gex_b:.0f}B."
            ),
            legs=[
                Leg(side="short", type=direction, strike=short),
                Leg(side="long",  type=direction, strike=long_),
            ],
            expiry_dte_target=dte_target,
            expiry_dte_bucket=bucket,
            source={
                "type": "regime_target",
                "drift_target": drift_target,
                "drift_direction": regime_block.get("drift_direction"),
                "regime": regime,
                "dominant_wall_gex_b": round(dom_gex_b, 1),
            },
        )


# ── Debit spread to target ────────────────────────────────────────────────────

class _DebitToTargetTemplate:
    """Debit spread toward the drift target.

    For magnet-above (calls):  long call 10pt inside target, short call at target.
    For magnet-below (puts):   long put 10pt inside target, short put at target.

    Emitted alongside _DirectionalSpreadTemplate (credit) for every magnet-regime
    payload; post-touch direction qualification in service.py filters which survives.
    CR-F will later add capped/uncapped/hedged debit variants selected by pattern_label.
    """
    template_id = "debit_spread_to_target"
    template_kind = "spread"
    _WIDTH_PTS = 10.0

    def applies_to(
        self,
        regime_block: dict,
        cluster: dict | None,
        clusters_all: list[dict],
        spot: float,
        implied_move: float,
    ) -> bool:
        if cluster is not None:
            return False
        regime = regime_block.get("regime", "")
        return regime in _MAGNET_REGIMES and "drift_target" in regime_block

    def propose(
        self,
        regime_block: dict,
        cluster: dict | None,
        clusters_all: list[dict],
        spot: float,
        implied_move: float,
        anchor_strategy: str = "cluster_centered",
    ) -> TradeProposal:
        drift_target = float(regime_block["drift_target"])
        regime = regime_block.get("regime")
        dom_gex = (regime_block.get("dominant_wall") or {}).get("gex", 0.0)
        dom_gex_b = dom_gex / 1e9 if dom_gex else 0.0

        drift_cluster = next(
            (
                c for c in clusters_all
                if abs(c.get("center_price", 0.0) - drift_target) < 30
            ),
            None,
        )
        bucket, dte_target = (
            _bucket_dte(drift_cluster) if drift_cluster else ("8-30", 15)
        )

        if regime == "magnet-above":
            direction = "call"
            long_strike  = drift_target - self._WIDTH_PTS  # 10pt inside target (toward spot)
            short_strike = drift_target
        else:  # magnet-below
            direction = "put"
            long_strike  = drift_target + self._WIDTH_PTS  # 10pt inside target (toward spot)
            short_strike = drift_target

        return TradeProposal(
            template_id=self.template_id,
            template_kind="spread",
            anchor_strategy=anchor_strategy,
            rationale=(
                f"Debit {direction} spread to {drift_target:.0f} "
                f"({drift_target - spot:+.0f}pt from spot {spot:.0f}). "
                f"Long {direction} {self._WIDTH_PTS:.0f}pt inside target; "
                f"short {direction} at target. Pays if price reaches "
                f"{drift_target:.0f}. Dominant wall: {dom_gex_b:.0f}B."
            ),
            legs=[
                Leg(side="long",  type=direction, strike=long_strike),
                Leg(side="short", type=direction, strike=short_strike),
            ],
            expiry_dte_target=dte_target,
            expiry_dte_bucket=bucket,
            source={
                "type": "regime_target",
                "drift_target": drift_target,
                "drift_direction": regime_block.get("drift_direction"),
                "regime": regime,
                "dominant_wall_gex_b": round(dom_gex_b, 1),
            },
        )


# ── Bounded iron condor ───────────────────────────────────────────────────────

class _BoundedCondorTemplate:
    template_id = "bounded_iron_condor"
    template_kind = "condor"
    _WING_WIDTH_PTS = 10.0

    def applies_to(
        self,
        regime_block: dict,
        cluster: dict | None,
        clusters_all: list[dict],
        spot: float,
        implied_move: float,
    ) -> bool:
        if cluster is not None:
            return False
        return (
            regime_block.get("regime") == "bounded"
            and "containment_zone" in regime_block
        )

    def propose(
        self,
        regime_block: dict,
        cluster: dict | None,
        clusters_all: list[dict],
        spot: float,
        implied_move: float,
        anchor_strategy: str = "cluster_centered",
    ) -> TradeProposal:
        strategy = ANCHOR_STRATEGIES[anchor_strategy]
        cz = regime_block["containment_zone"]
        lower_price = float(cz["lower_price"])
        upper_price = float(cz["upper_price"])
        lower_gex_b = float(cz.get("lower_gex", 0.0)) / 1e9
        upper_gex_b = float(cz.get("upper_gex", 0.0)) / 1e9
        lp, sp, sc, lc = strategy.condor_strikes(lower_price, upper_price, self._WING_WIDTH_PTS)
        width = upper_price - lower_price

        return TradeProposal(
            template_id=self.template_id,
            template_kind="condor",
            anchor_strategy=anchor_strategy,
            rationale=(
                f"Bounded between {lower_price:.0f} ({lower_gex_b:.0f}B) and "
                f"{upper_price:.0f} ({upper_gex_b:.0f}B), width {width:.0f}pt. "
                f"Short put at lower wall, short call at upper wall; "
                f"longs {self._WING_WIDTH_PTS:.0f}pt outside each short."
            ),
            legs=[
                Leg(side="long",  type="put",  strike=lp),
                Leg(side="short", type="put",  strike=sp),
                Leg(side="short", type="call", strike=sc),
                Leg(side="long",  type="call", strike=lc),
            ],
            expiry_dte_target=15,
            expiry_dte_bucket="8-30",
            source={
                "type": "containment_zone",
                "lower_price": lower_price,
                "upper_price": upper_price,
                "lower_gex_b": round(lower_gex_b, 1),
                "upper_gex_b": round(upper_gex_b, 1),
                "regime": regime_block.get("regime"),
            },
        )


# ── Feature no-trade ──────────────────────────────────────────────────────────

class _FeatureNoTradeTemplate:
    template_id = "feature_no_trade"
    template_kind = "no_trade"

    def applies_to(
        self,
        regime_block: dict,
        cluster: dict | None,
        clusters_all: list[dict],
        spot: float,
        implied_move: float,
    ) -> bool:
        if cluster is not None:
            return False
        return regime_block.get("regime") in _NOTRADE_REGIMES

    def propose(
        self,
        regime_block: dict,
        cluster: dict | None,
        clusters_all: list[dict],
        spot: float,
        implied_move: float,
        anchor_strategy: str = "cluster_centered",
    ) -> TradeProposal:
        regime = regime_block.get("regime", "unknown")
        return TradeProposal(
            template_id=self.template_id,
            template_kind="no_trade",
            anchor_strategy=anchor_strategy,
            rationale=(
                f"Today's landscape is {regime!r} regime — no qualifying pin or "
                f"directional target. GEX structure is diffuse or negative-zone driven. "
                f"No structural trade setup available; observe for developing structure."
            ),
            legs=[],
            expiry_dte_target=0,
            expiry_dte_bucket="",
            source={"type": "regime", "regime": regime},
        )


# ── Template registry ─────────────────────────────────────────────────────────

TEMPLATES: list[Template] = [
    _PinButterflyTemplate(
        "pin_butterfly_tight", "half_fwhm", "tight (half-FWHM)"
    ),
    _PinButterflyTemplate(
        "pin_butterfly_medium", "full_fwhm", "medium (full-FWHM)"
    ),
    _PinButterflyTemplate(
        "pin_butterfly_wide", "sigma_1x", "wide (1σ)"
    ),
    _DirectionalSpreadTemplate(),
    _DebitToTargetTemplate(),
    _BoundedCondorTemplate(),
    _FeatureNoTradeTemplate(),
]


# ── Public entry point ────────────────────────────────────────────────────────

def generate_proposals(
    landscape_payload: dict,
    spot: float,
    implied_move: float,
    anchor_strategy: str = "cluster_centered",
) -> list[TradeProposal]:
    """Map a landscape payload to a list of trade proposals.

    Iterates every (template, cluster) combination plus a no-cluster pass
    for regime-driven templates. Pure function — no DB I/O.

    Args:
        landscape_payload: Output of _materialize_payload — must contain
            "regime" (dict) and "confluences" (list of cluster dicts).
        spot: Current or reference spot price.
        implied_move: 1-day 1σ implied move in points.
        anchor_strategy: Key into ANCHOR_STRATEGIES registry.
    """
    regime_block = landscape_payload.get("regime") or {}
    confluences = landscape_payload.get("confluences") or []
    proposals: list[TradeProposal] = []

    for template in TEMPLATES:
        # Cluster-driven templates: call applies_to for each cluster.
        # Regime-driven templates: applies_to returns False for any non-None cluster.
        for cluster in [*confluences, None]:
            if template.applies_to(regime_block, cluster, confluences, spot, implied_move):
                proposals.append(
                    template.propose(
                        regime_block, cluster, confluences, spot, implied_move,
                        anchor_strategy,
                    )
                )

    return proposals
