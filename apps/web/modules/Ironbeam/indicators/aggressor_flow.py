# apps/web/modules/Ironbeam/indicators/aggressor_flow.py

from __future__ import annotations

import os
from typing import Any, Dict, List

from .base import IndicatorPlugin, IndicatorSpec


class AggressorFlow(IndicatorPlugin):
    """Aggressor Flow (buy/sell pressure) panel indicator.

    NOTE: The figure itself is still built in callbacks.py today (build_aggressor_flow_figure),
    but this plugin defines the settings schema so the UI can render controls cleanly.
    """

    id = "aggressor_flow"
    name = "Aggressor Flow"
    kind = "panel"

    def default_config(self) -> Dict[str, Any]:
        return {
            "ema_len": int(os.getenv("IRONBEAM_FLOW_EMA_LEN", "840")),
            "resample": str(os.getenv("IRONBEAM_FLOW_RESAMPLE", "1s")),
            "session": str(os.getenv("IRONBEAM_FLOW_SESSION", "RTH")),
            "hist_alpha": float(os.getenv("IRONBEAM_FLOW_HIST_ALPHA", "0.30")),
            "panel_height": int(os.getenv("IRONBEAM_FLOW_PANEL_HEIGHT", "260")),
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "ema_len": {"type": "int", "min": 1, "max": 5000, "step": 1, "label": "EMA length"},
            "resample": {"type": "select", "options": ["1s", "5s", "15s", "1m"], "label": "Resample"},
            "session": {"type": "select", "options": ["RTH", "FULL"], "label": "Session"},
            "hist_alpha": {"type": "float", "min": 0.05, "max": 1.0, "step": 0.05, "label": "Histogram opacity"},
            "panel_height": {"type": "int", "min": 140, "max": 520, "step": 10, "label": "Panel height (px)"},
        }

    def required_datasets(self) -> List[str]:
        return ["flow"]

    def build(self, ctx: Dict[str, Any], cfg: Dict[str, Any]) -> IndicatorSpec:
        # Height is used by callbacks when rendering the panel container.
        height = 260
        try:
            if isinstance(cfg, dict) and cfg.get("panel_height") is not None:
                height = int(cfg.get("panel_height"))
        except Exception:
            height = 260
        return IndicatorSpec(kind="panel", figure=None, height=height)
