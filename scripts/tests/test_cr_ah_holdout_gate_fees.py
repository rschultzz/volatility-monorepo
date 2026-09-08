"""CR-AU decisions 4 and 5 — holdout read gate (n only without the flag; raises
on a missing file; P&L only with the file AND n >= 60) and commission-net P&L."""
import os
import sys
from datetime import date, datetime
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("BACKFILL_DATABASE_URL", "postgresql://test:test@localhost:1/test")   # import-time guard only; no connection is made

import pytest

from packages.shared.config import FEE_PER_CONTRACT_PER_LEG
from packages.shared.strategy_templates import Leg
import scripts.cr_ah_step4_analysis as H

TD = date(2026, 6, 16)


def _trade(partition="holdout", band="near", settlement=5112.0, structure="debit", net_credit=-3.0):
    legs = ([Leg(side="long", type="call", strike=5090.0), Leg(side="short", type="call", strike=5100.0)]
            if structure == "debit" else
            [Leg(side="short", type="call", strike=5100.0), Leg(side="long", type="call", strike=5110.0)])
    return H.TradeData(
        trade_date=TD, structure=structure, band=band, partition=partition, sigma=1.0, drift_target=5100.0,
        structural_prob=0.6, spread_width=10.0, legs=legs, short_strike=5100.0, expiry_date=date(2026, 7, 8),
        pattern_label=None, reversion_wilson_lo=None, continuation_wilson_lo=None,
        entry_scan=[H.MinuteScan(snapshot_pt=datetime(2026, 6, 16, 6, 31), net_credit=net_credit, edge=0.3)],
        baseline_net_credit=net_credit, touch_resolution="no_touch", touch_datetime_pt=None, touch_pos_val=None,
        settlement_price=settlement,
    )


# ── decision 5: fees ─────────────────────────────────────────────────────────

def test_vertical_fee_is_four_contract_sides_in_points():
    assert H.VERTICAL_FEE_PTS == pytest.approx(2 * 2 * FEE_PER_CONTRACT_PER_LEG / 100) == pytest.approx(0.026)


def test_compute_pnl_net_is_gross_minus_fee_and_gross_unchanged():
    r = H.compute_pnl(_trade(partition="train"), 0.0)
    assert r["close_pnl"] == pytest.approx(-3.0 + 10.0)          # debit paid 3, settles 12 above the short → full width
    assert r["close_pnl_net"] == pytest.approx(r["close_pnl"] - 0.026)
    assert r["baseline_close_pnl_net"] == pytest.approx(r["baseline_close_pnl"] - 0.026)
    assert r["touch_exit_pnl_net"] is None and r["fee_pts"] == pytest.approx(0.026)
    r2 = H.compute_pnl(_trade(partition="train"), 0.0, fee_pts=0.0)
    assert r2["close_pnl_net"] == r2["close_pnl"]


def test_cell_stats_carry_mean_pnl_net():
    trades = [_trade(partition="train"), _trade(partition="train", settlement=5095.0)]
    r = H.fmt_stats(H.aggregate([(t, H.compute_pnl(t, 0.0)) for t in trades]))
    assert r["mean_pnl"] == pytest.approx((7.0 + 2.0) / 2)
    assert r["mean_pnl_net"] == pytest.approx(r["mean_pnl"] - 0.026)
    assert H.fmt_stats(H.CellStats())["mean_pnl_net"] is None


# ── decision 4: holdout read gate ────────────────────────────────────────────

def test_gate_without_flag_is_locked_and_prints_n_only():
    g = H.holdout_read_gate(15, None)
    assert not g.unlocked and g.line == "holdout: n=15, unread (threshold 60)"
    g2 = H.holdout_read_gate(200, None)                    # n alone never unlocks
    assert not g2.unlocked and "unread" in g2.line


def test_gate_with_flag_but_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        H.holdout_read_gate(100, str(tmp_path / "does-not-exist.md"))


def test_gate_unlocks_only_with_file_and_n_at_least_60(tmp_path):
    prereg = tmp_path / "preregistration.md"; prereg.write_text("# holdout read\n")
    assert not H.holdout_read_gate(59, str(prereg)).unlocked
    assert "unread (threshold 60)" in H.holdout_read_gate(59, str(prereg)).line
    g = H.holdout_read_gate(60, str(prereg))
    assert g.unlocked and "READ UNLOCKED" in g.line


def test_by_band_prints_holdout_n_only_when_locked(capsys):
    data = [_trade(partition="train"), _trade(partition="holdout"), _trade(partition="holdout", settlement=None)]
    H.print_by_band(data, 0.0, "debit", "all splits", holdout_unlocked=False)
    out = capsys.readouterr().out
    train_line = next(l for l in out.splitlines() if l.strip().startswith("near    train"))
    ho_lines = [l for l in out.splitlines() if "holdout" in l]
    assert "7.00" in train_line and "6.97" in train_line                    # gross and net for train
    assert ho_lines and all("unread — n only; dates matured=1" in l for l in ho_lines)
    assert all("7.00" not in l and "6.97" not in l for l in ho_lines)      # no holdout P&L
    assert "near    holdout     2  (unread" in out


def test_by_band_prints_holdout_pnl_only_when_unlocked(capsys):
    data = [_trade(partition="train"), _trade(partition="holdout")]
    H.print_by_band(data, 0.0, "debit", "all splits", holdout_unlocked=True)
    out = capsys.readouterr().out
    ho = next(l for l in out.splitlines() if l.strip().startswith("near    holdout"))
    assert "7.00" in ho and "6.97" in ho and "unread" not in ho


def test_summary_a_b_print_gate_line_when_locked_and_pnl_when_unlocked(tmp_path):
    debit = [_trade(partition="train")] + [_trade(partition="holdout") for _ in range(3)]
    credit = [_trade(partition="train", structure="credit", net_credit=3.0)] + \
             [_trade(partition="holdout", structure="credit", net_credit=3.0) for _ in range(3)]
    locked = H.build_summary(debit, credit, 0.0, 0.0, holdout_gate=H.holdout_read_gate(15, None))
    assert locked.count("holdout: n=15, unread (threshold 60)") == 2
    assert "Debit near-band holdout (n=" not in locked and "Credit near-band holdout (n=" not in locked
    prereg = tmp_path / "prereg.md"; prereg.write_text("x")
    unlocked = H.build_summary(debit, credit, 0.0, 0.0, holdout_gate=H.holdout_read_gate(60, str(prereg)))
    assert "Debit near-band holdout (n=3)" in unlocked and "unread" not in unlocked
    legacy = H.build_summary(debit, credit, 0.0, 0.0)                  # no gate passed → pre-CR-AU behaviour
    assert "Debit near-band holdout (n=3)" in legacy


def test_cli_accepts_new_flags():
    a = H._parse_args(["--selection-only", "--holdout-read", "x.md"])
    assert a.selection_only and a.holdout_read == "x.md"
    assert H._parse_args([]).holdout_read is None and not H._parse_args([]).selection_only
