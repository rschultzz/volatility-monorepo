# apps/web/modules/Ironbeam/indicators/registry.py

from __future__ import annotations

from typing import Dict, List

from .aggressor_flow import AggressorFlow
from .base import IndicatorPlugin

PLUGINS: List[IndicatorPlugin] = [
    AggressorFlow(),
]

PLUGIN_MAP: Dict[str, IndicatorPlugin] = {p.id: p for p in PLUGINS}


def options():
    """Dash-friendly options list."""
    return [{"label": p.name, "value": p.id} for p in PLUGINS]
