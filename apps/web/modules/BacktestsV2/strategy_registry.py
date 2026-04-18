from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class V2StrategySpec:
    key: str
    label: str
    description: str
    badges: tuple[str, ...]
    defaults: Dict[str, Any]
    runner: Callable[..., Dict[str, Any]]


def _base_defaults() -> Dict[str, Any]:
    return {
        "startDate": None,
        "endDate": None,
        "minLevelGexBn": 50,
        "zoneMergeDistancePts": 10,
        "minCleanMovePoints": 20,
        "targetProximityPts": 5,
        "maxZoneBreachPts": 5,
        "pivotStrengthBars": 3,
        "levelFamily": "primary",
        "maxResults": 2500,
        "consolidationWindowMinutes": 15,
        "shortPutSkewIncreasePct": 80,
        "shortCallSkewMaxPct": 30,
        "entryWithinTopPts": 2,
        "entrySearchWindowMinutes": 30,
        "initialStopPts": 6,
        "trailActivateProfitPts": 10,
        "trailingStopPts": 6,
        "takeProfitPts": 20,
        "maxPriorDownUpRatio": 2.0,
        "maxStartPctOfRange": 0.20,
        "maxMoveLossPct": 0.75,
        "minMinutesAfterOpen": 15,
    }


def build_strategy_registry(scan_runner: Callable[..., Dict[str, Any]]) -> Dict[str, V2StrategySpec]:
    base = _base_defaults()

    return {
        "up_move_short": V2StrategySpec(
            key="up_move_short",
            label="Up Move → Short",
            description=(
                "Current V2 production workflow: detect valid directional moves, "
                "evaluate the up-move short setup near target, then simulate short entry, "
                "initial stop, trailing stop activation, and take-profit."
            ),
            badges=(
                "RTH only",
                "Same day only",
                "Transitive GEX zones",
                "Last pivot inside source zone",
                "Up short setup near target",
                "Simulated short entry + exits",
            ),
            defaults=base,
            runner=scan_runner,
        ),
        "down_move_scan": V2StrategySpec(
            key="down_move_scan",
            label="Down Move Library Entry",
            description=(
                "Library-backed directional view for down moves. "
                "This entry currently filters the shared scan engine to down-move rows "
                "while preserving the existing core logic."
            ),
            badges=(
                "Library-backed strategy",
                "Directional filtering",
                "Down-move view",
            ),
            defaults=base,
            runner=scan_runner,
        ),
    }


def serialize_strategy(spec: V2StrategySpec, *, defaults_override: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "key": spec.key,
        "label": spec.label,
        "description": spec.description,
        "badges": list(spec.badges),
        "defaults": dict(defaults_override or spec.defaults),
    }
