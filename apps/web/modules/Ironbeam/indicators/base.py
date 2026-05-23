# apps/web/modules/Ironbeam/indicators/base.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

Kind = Literal["overlay", "panel"]

@dataclass
class IndicatorSpec:
    """
    What an indicator produces.
    - overlay: add traces to the main price figure
    - panel: return a full figure for its own dcc.Graph
    """
    kind: Kind
    traces: Optional[list] = None          # plotly traces (overlay)
    figure: Optional[dict] = None          # plotly figure dict (panel)
    height: int = 260                      # default panel height


class IndicatorPlugin:
    """
    Minimal plugin contract.
    Step 6 will use this for UI + rendering.
    """
    id: str
    name: str
    kind: Kind  # "overlay" or "panel"

    def default_config(self) -> Dict[str, Any]:
        return {}

    def schema(self) -> Dict[str, Any]:
        """
        UI schema for auto-building settings controls later.
        """
        return {}

    def required_datasets(self) -> List[str]:
        """
        Optional: declare which datasets this indicator needs so we can fetch once.
        """
        return []

    def build(self, ctx: Dict[str, Any], cfg: Dict[str, Any]) -> IndicatorSpec:
        raise NotImplementedError
