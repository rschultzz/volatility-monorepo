"""CR-AU amendment A3 — shared fee constant and per-structure round-trip fee."""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[3])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest

from packages.shared.config import CONTRACT_MULTIPLIER, FEE_PER_CONTRACT_PER_LEG, round_trip_fee_pts


def test_fee_constant_is_the_confirmed_schwab_figure():
    assert FEE_PER_CONTRACT_PER_LEG == 0.65 and CONTRACT_MULTIPLIER == 100.0


def test_vertical_round_trip_is_four_contract_sides():
    # 2 legs × 2 sides × $0.65 = $2.60 = 0.026 SPX points
    assert round_trip_fee_pts(2) == pytest.approx(0.026)


def test_condor_round_trip_is_eight_contract_sides():
    assert round_trip_fee_pts(4) == pytest.approx(0.052)
    assert round_trip_fee_pts(4, 1.30) == pytest.approx(0.104)


def test_rejects_non_positive_leg_count():
    with pytest.raises(ValueError):
        round_trip_fee_pts(0)
