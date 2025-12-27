# apps/web/modules/Ironbeam/indicators/aggressor_flow.py

from __future__ import annotations

from typing import Any, Dict, List

from .base import IndicatorPlugin, IndicatorSpec


class AggressorFlow(IndicatorPlugin):
    id = "aggressor_flow"
    name = "Aggressor Flow"
    kind = "panel"

    def default_config(self) -> Dict[str, Any]:
        return {
            "ema_len": 840,
            "resample": "1s",
            "session": "RTH",
            "pos_color": "#60a5fa",
            "neg_color": "#ef4444",
            "opacity": 1.0,
        }

    def schema(self) -> Dict[str, Any]:
        # NOTE: For HTML number inputs, `step` controls which values are considered valid.
        # With min=10 and step=10, values like 16 are invalid (red) and revert.
        # We use step=1 so any integer >= 10 is allowed.
        return {
            "ema_len": {"type": "int", "min": 10, "max": 5000, "step": 1, "label": "EMA length"},
            "resample": {"type": "select", "options": ["1s", "5s", "15s", "1m"], "label": "Resample"},
            "session": {"type": "select", "options": ["RTH", "FULL"], "label": "Session"},
            "pos_color": {"type": "text", "label": "Buy color (hex)"},
            "neg_color": {"type": "text", "label": "Sell color (hex)"},
            "opacity": {"type": "float", "min": 0.05, "max": 1.0, "step": 0.05, "label": "Opacity"},
        }

    def required_datasets(self) -> List[str]:
        return ["flow"]

    def build(self, ctx: Dict[str, Any], cfg: Dict[str, Any]) -> IndicatorSpec:
        # Step 7 will move the flow plot generation into this plugin.
        return IndicatorSpec(kind="panel", figure=None, height=260)
