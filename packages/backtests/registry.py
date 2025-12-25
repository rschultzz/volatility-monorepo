# packages/backtests/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from packages.backtests.gex_fade import GexFadeParams, run_gex_fade_backtest


@dataclass(frozen=True)
class ParamSpec:
    """
    UI + coercion spec for a single strategy parameter.

    - scale: multiply UI value by scale before passing to strategy (useful for Bn -> raw units)
    - kind:
        - "number" -> dcc.Input(type="number")
        - "bool"   -> dcc.RadioItems(True/False)
        - "select" -> dcc.Dropdown
    """
    name: str
    label: str
    kind: str  # "number" | "bool" | "select"
    default: Any
    group: str = "Parameters"
    help: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[Dict[str, Any]]] = None
    scale: float = 1.0
    cast: str = "float"  # "float" | "int" | "bool" | "str"


@dataclass(frozen=True)
class StrategySpec:
    key: str
    label: str
    description: str
    params: List[ParamSpec]
    run: Callable[[pd.DataFrame, Dict[str, Any], Dict[str, Any]], Tuple[pd.DataFrame, Dict[str, Any]]]


def _coerce_value(v: Any, spec: ParamSpec) -> Any:
    if v is None or v == "":
        v = spec.default

    if spec.cast == "bool":
        return bool(v)

    if spec.cast == "int":
        try:
            return int(v)
        except Exception:
            return int(spec.default)

    if spec.cast == "str":
        return str(v)

    # float (default)
    try:
        return float(v)
    except Exception:
        return float(spec.default)


def coerce_params(raw: Dict[str, Any], specs: List[ParamSpec]) -> Dict[str, Any]:
    """
    Convert UI values -> strategy-ready values (casts + scaling).
    """
    out: Dict[str, Any] = {}
    for s in specs:
        val = _coerce_value(raw.get(s.name), s)
        # Apply scaling to numeric types only
        if s.cast in ("float", "int") and s.scale not in (None, 1.0):
            val = float(val) * float(s.scale)
            if s.cast == "int":
                val = int(round(val))
        out[s.name] = val
    return out


def _run_gex_fade(df: pd.DataFrame, raw_params: Dict[str, Any], ctx: Dict[str, Any]):
    spec = get_strategies()["gex_fade"]
    params_dict = coerce_params(raw_params, spec.params)
    params = GexFadeParams(**params_dict)

    # Future-proofing: if strategy later supports SS compare, pass it through
    if ctx.get("compare_to_expected_ss") and hasattr(params, "compare_to_expected_ss"):
        try:
            setattr(params, "compare_to_expected_ss", True)
        except Exception:
            pass

    return run_gex_fade_backtest(df, params)


def get_strategies() -> Dict[str, StrategySpec]:
    """
    Registry of strategies. Add new strategies here without touching app.py.
    """
    gex_fade_params = [
        ParamSpec(
            name="entry_proximity_max",
            label="Entry proximity max (pts below wall)",
            kind="number",
            cast="float",
            default=2.0,
            min=0.0,
            step=0.25,
            group="Entry / Filters",
        ),
        ParamSpec(
            name="gex_wall_min",
            label="Min |wall GEX| (Bn)",
            kind="number",
            cast="float",
            default=50.0,   # UI in billions
            min=0.0,
            step=1.0,
            scale=1e9,      # convert Bn -> raw
            group="Entry / Filters",
            help="Example: 50 = 50B",
        ),
        ParamSpec(
            name="gex_net_min",
            label="Min net GEX (Bn)",
            kind="number",
            cast="float",
            default=0.0,    # UI in billions
            step=1.0,
            scale=1e9,
            group="Entry / Filters",
            help="Use >0 to require 'call-heavy' days",
        ),
        ParamSpec(
            name="min_bar_index",
            label="Min bar index",
            kind="number",
            cast="int",
            default=30,
            min=0,
            step=1,
            group="Time / Session",
        ),
        ParamSpec(
            name="max_bar_index",
            label="Max bar index",
            kind="number",
            cast="int",
            default=350,
            min=0,
            step=1,
            group="Time / Session",
        ),
        ParamSpec(
            name="require_rth",
            label="Require RTH",
            kind="bool",
            cast="bool",
            default=True,
            group="Time / Session",
        ),
        ParamSpec(
            name="min_abs_skew",
            label="Min |put skew| (pp)",
            kind="number",
            cast="float",
            default=0.0,
            min=0.0,
            step=0.1,
            group="Skew Filter",
        ),
        ParamSpec(
            name="stop_loss_points",
            label="Stop loss (pts)",
            kind="number",
            cast="float",
            default=2.0,
            min=0.0,
            step=0.25,
            group="Risk",
        ),
        ParamSpec(
            name="target_rr",
            label="Target R multiple",
            kind="number",
            cast="float",
            default=2.0,
            min=0.0,
            step=0.25,
            group="Risk",
        ),
        ParamSpec(
            name="max_bars_in_trade",
            label="Max bars in trade",
            kind="number",
            cast="int",
            default=60,
            min=1,
            step=1,
            group="Risk",
        ),
        ParamSpec(
            name="max_trades_per_day",
            label="Max trades per day",
            kind="number",
            cast="int",
            default=8,
            min=0,
            step=1,
            group="Risk",
        ),
    ]

    return {
        "gex_fade": StrategySpec(
            key="gex_fade",
            label="GEX Fade (short at call wall)",
            description="Short near overhead GEX wall when filters pass; exits via stop/target/max-bars.",
            params=gex_fade_params,
            run=_run_gex_fade,
        )
    }
