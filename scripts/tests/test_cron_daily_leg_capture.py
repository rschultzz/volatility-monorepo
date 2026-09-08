"""CR-AU decision 1 — leg builder for magnet / non-magnet days, condor box,
dedupe against fetched windows, window-closed guard (pure paths only)."""
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from packages.shared.options_cache.models import FetchedWindow
from scripts.cron_daily_leg_capture import (
    MAX_WAIT_MIN, MIN_LAG_MIN, POLL_SECONDS, Leg, build_plan, capture_window, dedupe_legs, format_plan, legs_for,
    plan_condor, plan_debit, poll_until, window_is_closed,
)

TD = date(2026, 9, 4)
EXPIRY = date(2026, 9, 25)          # 15 business days after 2026-09-04 (Labor Day skipped)
SNAP = datetime(2026, 9, 4, 6, 33)
DENSE = [float(k) for k in range(7600, 7900, 5)]


def _plan(regime="magnet-above", target=7800.0, chain=DENSE, spx_open=7742.9, im=68.22, es_open=7742.25, snap=SNAP):
    return build_plan(TD, regime=regime, implied_move=im, im_source="feature_vector.implied_move_1d",
                      spx_open=spx_open, spx_open_snapshot=snap, es_open=es_open, target=target, spot=7751.95,
                      debit_expiry=EXPIRY, chain=chain)


def test_capture_window_is_0630_to_0645_pt():
    w0, w1 = capture_window(TD)
    assert (w0.hour, w0.minute) == (6, 30) and (w1.hour, w1.minute) == (6, 45) and w0.date() == TD


def test_magnet_day_has_debit_pair_and_condor_box():
    p = _plan()
    assert p.skip is None
    assert p.debit == {"short": 7800.0, "long": 7790.0, "width_actual": 10.0, "narrower_than_nominal": False, "expiry": EXPIRY}
    # ±0.5 IM on the SPX 06:33 spot: 7742.9 ± 34.11 → 7708.79 / 7777.01 → round5 → 7710 / 7775; wings 10 out
    assert p.condor["strikes"] == (7700.0, 7710.0, 7775.0, 7785.0) and p.condor["expiry"] == TD
    assert [(l.structure, l.role, l.opra) for l in p.legs] == [
        ("debit", "short", "SPX260925C07800000"),
        ("debit", "long", "SPX260925C07790000"),
        ("condor", "long", "SPX260904P07700000"),
        ("condor", "short", "SPX260904P07710000"),
        ("condor", "short", "SPX260904C07775000"),
        ("condor", "long", "SPX260904C07785000"),
    ]


def test_non_magnet_day_has_condor_only():
    p = _plan(regime="amplification", target=None, chain=[], spx_open=7635.27, im=51.99, es_open=7650.5)
    assert p.debit is None
    assert p.condor["strikes"] == (7600.0, 7610.0, 7660.0, 7670.0)
    assert len(p.legs) == 4 and {l.structure for l in p.legs} == {"condor"}
    assert "debit: none" in format_plan(p)


def test_debit_pair_snaps_to_listed_pair_with_width_cap():
    # 25-point grid near the target: 7775/7800 is the only pair within [10, 20]? No — 25 > 20, so unlistable
    sparse = [7750.0, 7775.0, 7800.0, 7825.0]
    p = _plan(chain=sparse)
    assert "unlistable" in p.debit
    assert len(p.legs) == 4 and {l.structure for l in p.legs} == {"condor"}     # condor still captured
    # 20-point pair exists → chosen at the cap, recorded width
    p2 = _plan(chain=[7780.0, 7800.0, 7820.0])
    assert p2.debit["short"] == 7800.0 and p2.debit["long"] == 7780.0 and p2.debit["width_actual"] == 20.0


def test_plan_debit_none_for_non_magnet_and_unlistable_without_target():
    assert plan_debit("magnetic-pin", 7800.0, EXPIRY, DENSE, 7750.0) is None
    assert "unlistable" in plan_debit("magnet-above", None, EXPIRY, DENSE, 7750.0)


def test_condor_degenerate_box_is_unlistable():
    c = plan_condor(TD, 7745.0, 1.0)          # IM 1 → both shorts round to 7745
    assert "unlistable" in c and legs_for(None, c) == []


def test_skip_reasons():
    assert _plan(snap=None, spx_open=None).skip == "no_spx_open_snapshot"
    assert _plan(snap=datetime(2026, 9, 4, 6, 41)).skip == "no_spx_open_snapshot"      # past the 06:40 ceiling
    assert _plan(es_open=7900.0).skip == "bad_spx_open"                                # basis +157 > 100
    assert _plan(im=None).skip == "no_implied_move"
    assert _plan(es_open=None).skip is None                                            # ES open unknown → basis unchecked
    assert "SKIP no_spx_open_snapshot" in format_plan(_plan(snap=None, spx_open=None))


def test_dedupe_against_fetched_windows():
    p = _plan()
    w0, w1 = p.window
    full = FetchedWindow(opra_symbol=p.legs[0].opra, window_start_pt=datetime(2026, 9, 4, 6, 30),
                         window_end_pt=datetime(2026, 9, 4, 13, 0), row_count=390, source="historical_backfill")
    partial = FetchedWindow(opra_symbol=p.legs[2].opra, window_start_pt=w0, window_end_pt=w0 + timedelta(minutes=7),
                            row_count=8, source="historical_backfill")
    to_fetch, covered = dedupe_legs(p.legs, p.window, {p.legs[0].opra: [full], p.legs[2].opra: [partial]})
    assert [l.opra for l in covered] == [p.legs[0].opra]
    assert p.legs[2] in to_fetch and len(to_fetch) == 5
    assert "[cached]" in format_plan(p, covered={p.legs[0].opra})


def test_window_closed_guard():
    assert not window_is_closed(datetime(2026, 9, 4, 6, 35), TD)                       # decision 1's 06:35 — too early
    assert not window_is_closed(datetime(2026, 9, 4, 6, 49), TD)
    assert window_is_closed(datetime(2026, 9, 4, 6, 45 + MIN_LAG_MIN), TD)
    assert window_is_closed(datetime(2026, 9, 5, 9, 0), TD)                            # a past date is always closed


# ── decision 1 (amended 2026-09-07): poll for the 06:33 PT snapshot, bounded ──

class _Clock:
    """Fake PT clock: sleep_fn advances it, so the poll's timeout is driven by the test."""
    def __init__(self, start):
        self.now = start; self.sleeps = []
    def now_fn(self):
        return self.now
    def sleep_fn(self, s):
        self.sleeps.append(s); self.now += timedelta(seconds=s)


def test_poll_returns_immediately_when_snapshot_present():
    clk = _Clock(datetime(2026, 9, 4, 6, 50))
    v, waited, attempts = poll_until(lambda: (7742.9, SNAP), max_wait_min=MAX_WAIT_MIN, now_fn=clk.now_fn, sleep_fn=clk.sleep_fn, log=lambda m: None)
    assert v == (7742.9, SNAP) and attempts == 1 and waited == 0 and clk.sleeps == []


def test_poll_keeps_probing_until_snapshot_appears():
    clk = _Clock(datetime(2026, 9, 4, 5, 50))          # DST-drift case: cron fired an hour early
    seen = []
    def probe():
        seen.append(clk.now)
        return (7742.9, SNAP) if clk.now >= datetime(2026, 9, 4, 6, 34) else None
    v, waited, attempts = poll_until(probe, max_wait_min=MAX_WAIT_MIN, poll_s=POLL_SECONDS, now_fn=clk.now_fn, sleep_fn=clk.sleep_fn, log=lambda m: None)
    assert v == (7742.9, SNAP)
    assert attempts == 45 and waited == 44.0                  # 05:50 → 06:34 at 60 s
    assert all(s == POLL_SECONDS for s in clk.sleeps)
    assert MAX_WAIT_MIN == 120 and POLL_SECONDS == 60


def test_poll_times_out_after_120_minutes_and_returns_none():
    clk = _Clock(datetime(2026, 9, 7, 6, 50))          # Labor Day: no monies snapshot ever appears
    logs = []
    v, waited, attempts = poll_until(lambda: None, max_wait_min=MAX_WAIT_MIN, poll_s=POLL_SECONDS, now_fn=clk.now_fn, sleep_fn=clk.sleep_fn, log=logs.append)
    assert v is None
    assert waited == 120.0 and attempts == 121               # probes at t = 0, 1, …, 120 min
    assert clk.now == datetime(2026, 9, 7, 8, 50)
    assert sum(clk.sleeps) == 120 * 60 and len(logs) == 120


def test_poll_with_zero_budget_probes_once():
    clk = _Clock(datetime(2026, 9, 5, 9, 0))           # a past --date or --dry-run: check once, never sleep
    v, waited, attempts = poll_until(lambda: None, max_wait_min=0, now_fn=clk.now_fn, sleep_fn=clk.sleep_fn, log=lambda m: None)
    assert v is None and attempts == 1 and clk.sleeps == []


def test_poll_last_sleep_is_clipped_to_the_budget():
    clk = _Clock(datetime(2026, 9, 4, 6, 0))
    v, waited, attempts = poll_until(lambda: None, max_wait_min=2.5, poll_s=60, now_fn=clk.now_fn, sleep_fn=clk.sleep_fn, log=lambda m: None)
    assert v is None and clk.sleeps == [60, 60, 30] and waited == 2.5 and attempts == 4
