"""Strike anchor strategies for trade proposal generation (CR-015).

An anchor strategy determines HOW strikes are placed around a structural
anchor point (cluster center, drift_target, containment zone wall).

v1a ships exactly one implementation: ClusterCenteredAnchor.
The registry (ANCHOR_STRATEGIES) maps name → instance so templates can
look up strategies by string without importing concrete classes.

Wing distance recipes for butterflies:
    "half_fwhm"  — body ± cluster["avg_fwhm"] / 2
    "full_fwhm"  — body ± cluster["avg_fwhm"]
    "sigma_1x"   — body ± implied_move  (1σ daily implied move)
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AnchorStrategy(Protocol):
    name: str

    def butterfly_strikes(
        self,
        cluster: dict,
        wing_distance_recipe: str,
        implied_move: float,
    ) -> tuple[float, float, float]:
        """Return (lower_wing, body, upper_wing)."""
        ...

    def spread_strikes(
        self,
        short_strike: float,
        spot: float,
        width_pts: float,
    ) -> tuple[float, float, str]:
        """Return (short, long, direction) where direction is 'call' or 'put'."""
        ...

    def condor_strikes(
        self,
        lower_price: float,
        upper_price: float,
        wing_width_pts: float,
    ) -> tuple[float, float, float, float]:
        """Return (long_put, short_put, short_call, long_call)."""
        ...


class ClusterCenteredAnchor:
    """Place strikes centred on the structural anchor point.

    For butterflies: body = cluster center_price, wings computed from recipe.
    For spreads: short = drift_target price passed in, long offset by width.
    For condors: shorts at containment zone walls, longs beyond by wing_width.
    """

    name = "cluster_centered"

    _VALID_RECIPES = frozenset({"half_fwhm", "full_fwhm", "sigma_1x"})

    def butterfly_strikes(
        self,
        cluster: dict,
        wing_distance_recipe: str,
        implied_move: float,
    ) -> tuple[float, float, float]:
        if wing_distance_recipe not in self._VALID_RECIPES:
            raise ValueError(
                f"Unknown wing_distance_recipe {wing_distance_recipe!r}. "
                f"Valid: {sorted(self._VALID_RECIPES)}"
            )
        body = float(cluster["center_price"])
        fwhm = float(cluster.get("avg_fwhm") or 0.0)
        if wing_distance_recipe == "half_fwhm":
            half = fwhm / 2.0
        elif wing_distance_recipe == "full_fwhm":
            half = fwhm
        else:  # sigma_1x
            half = float(implied_move)
        return (body - half, body, body + half)

    def spread_strikes(
        self,
        short_strike: float,
        spot: float,
        width_pts: float,
    ) -> tuple[float, float, str]:
        direction = "call" if short_strike > spot else "put"
        if direction == "call":
            long_strike = short_strike + width_pts
        else:
            long_strike = short_strike - width_pts
        return (short_strike, long_strike, direction)

    def condor_strikes(
        self,
        lower_price: float,
        upper_price: float,
        wing_width_pts: float,
    ) -> tuple[float, float, float, float]:
        short_put = lower_price
        short_call = upper_price
        long_put = short_put - wing_width_pts
        long_call = short_call + wing_width_pts
        return (long_put, short_put, short_call, long_call)


ANCHOR_STRATEGIES: dict[str, AnchorStrategy] = {
    ClusterCenteredAnchor.name: ClusterCenteredAnchor(),
}
