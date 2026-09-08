"""CR-AU decision 2 — matured-trade capture planner and dedupe (pure paths)."""
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from packages.shared.options_cache.models import FetchedWindow
from scripts.cr_aa_sweep_pending_outcomes import (
    TOUCH_WINDOW_MIN, capture_enabled, plan_matured_windows, windows_to_fetch,
)

TD = date(2026, 6, 16)
EXPIRY = date(2026, 7, 8)


def test_settlement_window_due_once_expiry_has_closed():
    w = plan_matured_windows(TD, EXPIRY, "no_touch", None, latest_session=date(2026, 7, 8))
    assert w == [{"label": "settlement", "start": datetime(2026, 7, 8, 12, 50), "end": datetime(2026, 7, 8, 13, 0), "due": True}]


def test_settlement_deferred_while_expiry_is_in_the_future():
    # a 1-7 DTE bucket matures 5 sessions in; the 15-bday expiry is still ahead
    w = plan_matured_windows(TD, EXPIRY, "no_touch", None, latest_session=date(2026, 6, 23))
    assert [x["due"] for x in w] == [False]


def test_touch_window_is_touch_plus_90_for_actionable_touches_only():
    tp = datetime(2026, 6, 18, 9, 17)
    w = plan_matured_windows(TD, EXPIRY, "rth_touch", tp, latest_session=date(2026, 7, 8))
    assert w[1] == {"label": "touch (rth_touch)", "start": tp, "end": tp + timedelta(minutes=TOUCH_WINDOW_MIN), "due": True}
    assert TOUCH_WINDOW_MIN == 90
    gap = datetime(2026, 6, 19, 6, 30)
    w2 = plan_matured_windows(TD, EXPIRY, "gap_touch", gap, latest_session=date(2026, 6, 19))
    assert w2[1]["label"] == "touch (gap_touch)" and w2[1]["due"] is True and w2[0]["due"] is False
    for res in ("afterhours_touch_retraced", "no_touch"):
        assert len(plan_matured_windows(TD, EXPIRY, res, None, date(2026, 7, 8))) == 1


def test_touch_window_deferred_until_its_day_has_closed():
    tp = datetime(2026, 6, 18, 12, 0)
    w = plan_matured_windows(TD, EXPIRY, "rth_touch", tp, latest_session=date(2026, 6, 17))
    assert w[1]["due"] is False


def test_windows_to_fetch_dedupes_against_fetched_windows_and_skips_deferred():
    opras = ["SPX260708C06200000", "SPX260708C06190000"]
    windows = plan_matured_windows(TD, EXPIRY, "rth_touch", datetime(2026, 6, 18, 9, 17), latest_session=date(2026, 7, 8))
    full_settle = FetchedWindow(opra_symbol=opras[0], window_start_pt=datetime(2026, 7, 8, 12, 50),
                                window_end_pt=datetime(2026, 7, 8, 13, 0), row_count=11, source="historical_backfill")
    partial_touch = FetchedWindow(opra_symbol=opras[1], window_start_pt=datetime(2026, 6, 18, 9, 17),
                                  window_end_pt=datetime(2026, 6, 18, 9, 40), row_count=24, source="historical_backfill")
    todo, covered = windows_to_fetch(opras, windows, {opras[0]: [full_settle], opras[1]: [partial_touch]})
    assert covered == 1
    assert [(o, w["label"]) for o, w in todo] == [
        (opras[1], "settlement"), (opras[0], "touch (rth_touch)"), (opras[1], "touch (rth_touch)"),
    ]
    # nothing due → nothing to fetch, nothing counted as covered
    deferred = plan_matured_windows(TD, EXPIRY, "no_touch", None, latest_session=date(2026, 6, 20))
    assert windows_to_fetch(opras, deferred, {}) == ([], 0)


def test_capture_requires_orats_token():
    assert capture_enabled({"ORATS_API_KEY": "abc"}) is True
    assert capture_enabled({"ORATS_API_KEY": "  "}) is False
    assert capture_enabled({}) is False
