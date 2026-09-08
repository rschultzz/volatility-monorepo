"""Shared trading-cost constants (CR-AS decision 8, persisted in CR-AU amendment A3).

FEE_PER_CONTRACT_PER_LEG — brokerage commission in dollars per option contract,
charged on each leg each time it is traded (open or close). Confirmed 2026-09-07
(Schwab, $0.65). A two-leg vertical opened and closed is 2 legs × 2 sides = 4
contract-sides; a four-leg condor is 8. In SPX points divide by the 100 multiplier.

The 0DTE condor module still carries its own FEE_PER_CONTRACT_SIDE (1.30, the
CR-AS default before the fee was confirmed); pointing it here is a follow-up.
"""
from __future__ import annotations

FEE_PER_CONTRACT_PER_LEG: float = 0.65
CONTRACT_MULTIPLIER: float = 100.0


def round_trip_fee_pts(n_legs: int, fee_per_contract_per_leg: float = FEE_PER_CONTRACT_PER_LEG) -> float:
    """Commission to open and close an `n_legs` structure, in index points per spread
    (one contract per leg): n_legs × 2 sides × fee / multiplier."""
    if n_legs <= 0:
        raise ValueError(f"n_legs must be positive, got {n_legs}")
    return n_legs * 2 * fee_per_contract_per_leg / CONTRACT_MULTIPLIER
