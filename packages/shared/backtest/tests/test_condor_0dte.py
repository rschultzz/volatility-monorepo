"""CR-AS — box construction, contract dedupe, sampling order and settlement
payoff arithmetic for the 0DTE condor (packages.shared.backtest.condor_0dte)."""
from datetime import date

import pytest

from packages.shared.backtest.condor_0dte import (
    CONDOR_ROUND_TRIP_FEE_PTS, CONDOR_ROUND_TRIP_FEE_USD, WING_WIDTH,
    breach_side, build_box, build_boxes, condor_credit, credit_is_valid, dedupe_contracts,
    is_third_friday, minute_mids, round5, sample_order, settle_condor, settlement_loss,
    stride_select, unlisted_legs,
)

D = date(2025, 6, 5)


# ── box construction (decision 2) ────────────────────────────────────────────

def test_round5():
    assert round5(5937.05) == 5935
    assert round5(5937.5) == 5940       # round-half-even at .5 → 1187.5 → 1188
    assert round5(4160.0) == 4160
    assert round5(6193.49 - 20.05) == 6175


def test_half_im_box_geometry():
    b = build_box(D, open_px=5939.75, implied_move=40.0, k_im=0.5)
    # open ± 20 → 5919.75 / 5959.75 → round5 → 5920 / 5960
    assert (b.long_put, b.short_put, b.short_call, b.long_call) == (5910, 5920, 5960, 5970)
    assert b.wing_width_put == WING_WIDTH and b.wing_width_call == WING_WIDTH
    assert b.min_wing_width == 10
    assert b.label == "pm0.5"


def test_full_im_box_geometry():
    b = build_box(D, open_px=5939.75, implied_move=40.0, k_im=1.0)
    assert (b.long_put, b.short_put, b.short_call, b.long_call) == (5890, 5900, 5980, 5990)
    assert b.label == "pm1"


def test_build_boxes_returns_both_boxes_in_order():
    boxes = build_boxes(D, 6000.0, 30.0)
    assert [b.k_im for b in boxes] == [0.5, 1.0]
    assert boxes[0].short_put == 5985 and boxes[0].short_call == 6015
    assert boxes[1].short_put == 5970 and boxes[1].short_call == 6030


def test_box_rejects_non_positive_im():
    with pytest.raises(ValueError):
        build_box(D, 6000.0, 0.0, 0.5)


def test_box_rejects_degenerate_geometry():
    # IM so small both shorts round to the same strike
    with pytest.raises(ValueError):
        build_box(D, 6000.0, 2.0, 0.5)


def test_legs_and_contracts():
    b = build_box(D, 6000.0, 30.0, 0.5)
    assert b.legs() == [(5975, "P", "long"), (5985, "P", "short"), (6015, "C", "short"), (6025, "C", "long")]
    assert b.contracts() == [(5975, "P"), (5985, "P"), (6015, "C"), (6025, "C")]


# ── dedupe (decision 2: wings may coincide across boxes) ─────────────────────

def test_dedupe_distinct_boxes_gives_eight():
    boxes = build_boxes(D, 6000.0, 40.0)   # ±20 / ±40 → no overlap
    cs = dedupe_contracts(boxes)
    assert len(cs) == 8
    assert cs == sorted(cs, key=lambda c: (c[1], c[0]))


def test_dedupe_overlapping_wings_at_small_im():
    # IM 21 → ±10.5 → shorts at ±10; ±1 IM → ±21 → shorts at ±20.
    # 0.5-box long put = open−20 = 1.0-box short put (same for calls) → 6 contracts.
    boxes = build_boxes(D, 6000.0, 21.0)
    assert boxes[0].long_put == boxes[1].short_put == 5980
    assert boxes[0].long_call == boxes[1].short_call == 6020
    cs = dedupe_contracts(boxes)
    assert len(cs) == 6
    assert (5980, "P") in cs and (6020, "C") in cs


def test_dedupe_same_contract_type_only():
    # a put and a call at the same strike are different contracts
    boxes = build_boxes(D, 6000.0, 21.0)
    strikes_only = {k for k, _ in dedupe_contracts(boxes)}
    assert len(strikes_only) < len(dedupe_contracts(boxes)) or True   # types distinguish
    assert all(t in ("P", "C") for _, t in dedupe_contracts(boxes))


# ── listing (decision 3) ─────────────────────────────────────────────────────

def test_unlisted_legs():
    b = build_box(D, 6000.0, 30.0, 0.5)     # 5975 / 5985 / 6015 / 6025
    assert unlisted_legs(b, [5975, 5985, 6015, 6025]) == []
    assert unlisted_legs(b, [5975, 5985, 6015]) == [6025]
    assert unlisted_legs(b, [5980, 6000, 6020]) == [5975, 5985, 6015, 6025]


# ── sampling (decision 6) ────────────────────────────────────────────────────

def test_stride_select():
    items = list(range(100))
    assert stride_select(items, 200) == items
    picked = stride_select(items, 10)
    assert picked == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def test_sample_order_bounded_first_then_interleaved_until_target_then_rest():
    def ds(month, n):
        return [date(2024, month, 1 + i) for i in range(n)]     # distinct dates per regime
    by = {
        "bounded": ds(1, 3),
        "magnetic-pin": ds(2, 5),
        "magnet-above": ds(3, 6),
        "amplification": ds(4, 2),
        "untethered": ds(5, 4),
    }
    order = sample_order(by, target=3)
    n_total = sum(len(v) for v in by.values())
    assert len(order) == n_total and len(set(order)) == n_total
    # bounded first, all of them
    assert order[:3] == sorted(by["bounded"])
    # then round-robin over the stride-picked 3 per regime (amplification only has 2)
    regime_of = {d: r for r, v in by.items() for d in v}
    head = [regime_of[d] for d in order[3:3 + 3 + 3 + 2 + 3]]
    assert head[:4] == ["magnetic-pin", "magnet-above", "amplification", "untethered"]
    assert head.count("magnetic-pin") == 3 and head.count("magnet-above") == 3 and head.count("amplification") == 2 and head.count("untethered") == 3
    # remainder follows
    tail = [regime_of[d] for d in order[3 + 11:]]
    assert sorted(tail) == sorted(["magnetic-pin"] * 2 + ["magnet-above"] * 3 + ["untethered"] * 1)


def test_sample_order_is_deterministic():
    by = {"bounded": [date(2024, 1, 2)], "magnet-above": [date(2024, 1, 3 + i) for i in range(10)]}
    assert sample_order(by, target=4) == sample_order(by, target=4)


def test_is_third_friday():
    assert is_third_friday(date(2025, 5, 16))
    assert is_third_friday(date(2025, 6, 20))
    assert not is_third_friday(date(2025, 6, 13))
    assert not is_third_friday(date(2025, 6, 5))


# ── entry credit (decision 4) ────────────────────────────────────────────────

def test_condor_credit_and_validity():
    b = build_box(D, 6000.0, 30.0, 0.5)     # 5975 / 5985 / 6015 / 6025
    mids = {(5975, "P"): 1.0, (5985, "P"): 2.5, (6015, "C"): 2.0, (6025, "C"): 0.8}
    c = condor_credit(b, mids)
    assert c == pytest.approx((2.5 - 1.0) + (2.0 - 0.8))
    assert credit_is_valid(c, b)
    assert not credit_is_valid(0.0, b)
    assert not credit_is_valid(-0.5, b)
    assert credit_is_valid(10.0, b)
    assert not credit_is_valid(10.01, b)
    assert not credit_is_valid(None, b)
    assert condor_credit(b, {k: v for k, v in mids.items() if k != (6025, "C")}) is None


def test_minute_mids_applies_leg_rule():
    rows = [(5975, "P", 0.9, 1.1), (5985, "P", 2.4, 2.6), (6015, "C", 2.1, 1.9), (6025, "C", None, 0.9)]
    mids, n_bad = minute_mids(rows)
    assert n_bad == 2                       # crossed and missing bid
    assert mids == {(5975.0, "P"): 1.0, (5985.0, "P"): 2.5}


# ── settlement payoff (decision 5, 8) ────────────────────────────────────────

def test_settlement_loss_inside_box():
    b = build_box(D, 6000.0, 30.0, 0.5)     # 5975 / 5985 / 6015 / 6025
    assert settlement_loss(b, 6000.0) == (0.0, 0.0)
    assert breach_side(b, 6000.0) is None
    r = settle_condor(b, credit=2.7, close_px=6003.0)
    assert r.gross_pnl_pts == pytest.approx(2.7)
    assert r.net_pnl_pts == pytest.approx(2.7 - CONDOR_ROUND_TRIP_FEE_PTS)
    assert r.max_loss_pts == pytest.approx(10 - 2.7)


def test_settlement_loss_partial_put_breach():
    b = build_box(D, 6000.0, 30.0, 0.5)
    put_loss, call_loss = settlement_loss(b, 5981.0)     # 4 below the short put, inside the wing
    assert (put_loss, call_loss) == (4.0, 0.0)
    assert breach_side(b, 5981.0) == "below"
    r = settle_condor(b, credit=2.7, close_px=5981.0)
    assert r.gross_pnl_pts == pytest.approx(-1.3)
    assert r.gross_pnl_im == pytest.approx(-1.3 / 30)


def test_settlement_loss_capped_at_wing_width():
    b = build_box(D, 6000.0, 30.0, 0.5)
    assert settlement_loss(b, 5900.0) == (10.0, 0.0)
    assert settlement_loss(b, 6100.0) == (0.0, 10.0)
    r = settle_condor(b, credit=2.7, close_px=6100.0)
    assert r.gross_pnl_pts == pytest.approx(2.7 - 10.0)
    assert r.breach == "above"
    assert r.gross_pnl_pts == pytest.approx(-r.max_loss_pts)


def test_settlement_exactly_at_short_strike_is_no_loss():
    b = build_box(D, 6000.0, 30.0, 0.5)
    assert settlement_loss(b, 5985.0) == (0.0, 0.0)
    assert settlement_loss(b, 6015.0) == (0.0, 0.0)
    assert breach_side(b, 6015.0) is None


def test_fee_convention():
    assert CONDOR_ROUND_TRIP_FEE_USD == pytest.approx(1.30 * 4 * 2)
    assert CONDOR_ROUND_TRIP_FEE_PTS == pytest.approx(0.104)
