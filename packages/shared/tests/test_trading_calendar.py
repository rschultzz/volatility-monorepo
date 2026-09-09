"""CR-BD decision 1 / G1 — NYSE trading-day helper: Labor Day 2026, Juneteenth,
July 4 observed, Thanksgiving, Christmas; the 2027-12-31 expiry guard."""
import sys
from datetime import date
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[3])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest

from packages.shared.trading_calendar import (
    CALENDAR_VALID_THROUGH, NYSE_HOLIDAYS, is_trading_day, next_trading_day, nth_trading_day, prev_trading_day,
)


def test_labor_day_2026_is_skipped_both_ways():
    # the bug: Friday 09-04's data was stamped onto Monday 09-07 (closed); Tuesday 09-08 got nothing
    assert next_trading_day(date(2026, 9, 4)) == date(2026, 9, 8)
    assert prev_trading_day(date(2026, 9, 8)) == date(2026, 9, 4)
    assert not is_trading_day(date(2026, 9, 7))


def test_juneteenth_2026_and_the_orphaned_monday():
    assert next_trading_day(date(2026, 6, 18)) == date(2026, 6, 22)     # Fri 06-19 closed, weekend
    assert prev_trading_day(date(2026, 6, 22)) == date(2026, 6, 18)


def test_independence_day_observed_2026():
    assert not is_trading_day(date(2026, 7, 3))                          # July 4 is a Saturday → Friday observed
    assert next_trading_day(date(2026, 7, 2)) == date(2026, 7, 6)
    assert prev_trading_day(date(2026, 7, 6)) == date(2026, 7, 2)


def test_thanksgiving_2026_and_the_half_day_after():
    assert not is_trading_day(date(2026, 11, 26))
    assert is_trading_day(date(2026, 11, 27))                            # early close is still a session
    assert next_trading_day(date(2026, 11, 25)) == date(2026, 11, 27)


def test_christmas_2026_and_new_year_2027():
    assert not is_trading_day(date(2026, 12, 25))
    assert next_trading_day(date(2026, 12, 24)) == date(2026, 12, 28)
    assert next_trading_day(date(2026, 12, 31)) == date(2027, 1, 4)      # Fri 01-01 closed
    assert not is_trading_day(date(2027, 1, 1))


def test_weekends_and_plain_weekdays():
    assert next_trading_day(date(2026, 9, 11)) == date(2026, 9, 14)     # Fri → Mon
    assert prev_trading_day(date(2026, 9, 14)) == date(2026, 9, 11)
    assert next_trading_day(date(2026, 9, 8)) == date(2026, 9, 9)


def test_nth_trading_day_matches_harness_expiry_rule():
    assert nth_trading_day(date(2026, 9, 4), 15) == date(2026, 9, 28)   # what CR-AU's G2 debit pair used
    with pytest.raises(ValueError):
        nth_trading_day(date(2026, 9, 4), 0)


def test_holiday_list_is_all_weekdays_and_ten_per_year():
    for h in NYSE_HOLIDAYS:
        assert h.weekday() < 5, h
    per_year = {y: sum(1 for h in NYSE_HOLIDAYS if h.year == y) for y in range(2023, 2028)}
    assert per_year == {2023: 10, 2024: 10, 2025: 11, 2026: 10, 2027: 10}   # 2025 has the day of mourning


def test_calendar_not_expired():
    """Fails after CALENDAR_VALID_THROUGH so the hardcoded list gets extended (decision 1)."""
    assert date.today() <= CALENDAR_VALID_THROUGH, (
        f"NYSE_HOLIDAYS is only maintained through {CALENDAR_VALID_THROUGH}; add the next year's closures "
        f"to packages/shared/trading_calendar.py and advance CALENDAR_VALID_THROUGH")
    assert max(NYSE_HOLIDAYS).year == CALENDAR_VALID_THROUGH.year
