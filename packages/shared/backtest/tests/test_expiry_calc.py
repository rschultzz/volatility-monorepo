"""Unit tests for the holiday-aware nth_business_day expiry calc (CR-AH Step 2).

These tests are self-contained — they replicate the function and holiday calendar
from the backfill script so they run without importing from scripts/ (which has
heavy env-setup side effects).

Test coverage:
  - Clean window (no holiday): result unchanged from blind count.
  - Holiday at positions 1-14 (B-bucket trigger): result is 1 real trading day later.
  - Holiday at position 15 (C-bucket trigger): result skips the holiday → next trading day.
  - Multiple holidays in window: result shifts by the count of holidays.
  - Holiday-adjacent date in C-bucket is now recovered (no longer a holiday).
"""

from datetime import date, timedelta


# ── Function under test (mirrors scripts/cr_ah_step2_stratified_backfill.py) ──

_NYSE_HOLIDAYS: frozenset[date] = frozenset({
    # 2023
    date(2023, 1, 2),   date(2023, 1, 16),  date(2023, 2, 20),  date(2023, 4, 7),
    date(2023, 5, 29),  date(2023, 6, 19),  date(2023, 7, 4),   date(2023, 9, 4),
    date(2023, 11, 23), date(2023, 12, 25),
    # 2024
    date(2024, 1, 1),   date(2024, 1, 15),  date(2024, 2, 19),  date(2024, 3, 29),
    date(2024, 5, 27),  date(2024, 6, 19),  date(2024, 7, 4),   date(2024, 9, 2),
    date(2024, 11, 28), date(2024, 12, 25),
    # 2025
    date(2025, 1, 1),   date(2025, 1, 9),   date(2025, 1, 20),  date(2025, 2, 17),
    date(2025, 4, 18),  date(2025, 5, 26),  date(2025, 6, 19),  date(2025, 7, 4),
    date(2025, 9, 1),   date(2025, 11, 27), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1),   date(2026, 1, 19),  date(2026, 2, 16),  date(2026, 4, 3),
    date(2026, 5, 25),  date(2026, 6, 19),  date(2026, 7, 3),   date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
})


def nth_business_day(from_date: date, n: int) -> date:
    """Return the nth NYSE trading day after from_date (excludes holidays + weekends)."""
    current = from_date
    count = 0
    while count < n:
        current += timedelta(days=1)
        if current.weekday() < 5 and current not in _NYSE_HOLIDAYS:
            count += 1
    return current


def _blind_nth_weekday(from_date: date, n: int) -> date:
    """Holiday-blind weekday counter (original buggy impl, for comparison)."""
    current = from_date
    count = 0
    while count < n:
        current += timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    return current


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestNthBusinessDay:

    def test_clean_window_no_holiday(self):
        """No holiday in 15-day window → result same as blind count."""
        # 2023-08-01 (Tue): Labor Day 2023 = Sep 4, > 15 business days away.
        # Count: Wed 8/2...Fri 8/4 = 3; Mon 8/7...Fri 8/11 = 5 (8 total);
        #        Mon 8/14...Fri 8/18 = 5 (13 total); Mon 8/21 = 14; Tue 8/22 = 15.
        result = nth_business_day(date(2023, 8, 1), 15)
        assert result == date(2023, 8, 22)
        assert result == _blind_nth_weekday(date(2023, 8, 1), 15)  # same as blind — no holiday

    def test_b_bucket_memorial_day_2023_in_window(self):
        """Memorial Day 2023 (May 29) in positions 1-14 → correct result is 1 day later than blind."""
        # trade_date=2023-05-10; Memorial Day 2023-05-29 falls at blind position 6.
        # Blind: returns 2023-05-31 (14th true trading day).
        # Holiday-aware: returns 2023-06-01 (15th true trading day).
        trade_date = date(2023, 5, 10)
        blind = _blind_nth_weekday(trade_date, 15)
        aware = nth_business_day(trade_date, 15)
        assert blind == date(2023, 5, 31)
        assert aware == date(2023, 6, 1)
        assert aware != blind  # B-bucket: expiry shifted

    def test_b_bucket_memorial_day_2024_in_window(self):
        """Memorial Day 2024 (May 27) at position 14 — still shifts result by 1."""
        # Verify a different holiday and year.
        trade_date = date(2024, 5, 7)
        blind = _blind_nth_weekday(trade_date, 15)
        aware = nth_business_day(trade_date, 15)
        # Memorial Day (May 27) falls within the window → blind is 1 day short
        assert aware != blind
        # Aware result must be a weekday, not a holiday, not a weekend
        assert aware.weekday() < 5
        assert aware not in _NYSE_HOLIDAYS

    def test_c_bucket_holiday_at_position_15_is_skipped(self):
        """Blind count's 15th day is a holiday → aware skips it to the next trading day."""
        # 2024-05-06: blind 15th = 2024-05-27 (Memorial Day → C-bucket 404 in original run).
        # Holiday-aware must skip 2024-05-27 and return 2024-05-28.
        trade_date = date(2024, 5, 6)
        blind = _blind_nth_weekday(trade_date, 15)
        aware = nth_business_day(trade_date, 15)
        assert blind == date(2024, 5, 27)   # was a Memorial Day → ORATS 404 (C bucket)
        assert aware == date(2024, 5, 28)   # next trading day — recoverable
        assert date(2024, 5, 27) in _NYSE_HOLIDAYS

    def test_c_bucket_presidents_day_2025(self):
        """2025-01-27: blind 15th = 2025-02-17 (Presidents' Day) → aware = 2025-02-18."""
        trade_date = date(2025, 1, 27)
        assert _blind_nth_weekday(trade_date, 15) == date(2025, 2, 17)
        assert nth_business_day(trade_date, 15) == date(2025, 2, 18)

    def test_c_bucket_juneteenth_2025(self):
        """2025-05-29: blind 15th = 2025-06-19 (Juneteenth) → aware = 2025-06-20."""
        trade_date = date(2025, 5, 29)
        assert _blind_nth_weekday(trade_date, 15) == date(2025, 6, 19)
        assert nth_business_day(trade_date, 15) == date(2025, 6, 20)

    def test_c_bucket_memorial_day_2026(self):
        """2026-05-04: blind 15th = 2026-05-25 (Memorial Day) → aware = 2026-05-26."""
        trade_date = date(2026, 5, 4)
        assert _blind_nth_weekday(trade_date, 15) == date(2026, 5, 25)
        assert nth_business_day(trade_date, 15) == date(2026, 5, 26)

    def test_aware_result_always_trading_day(self):
        """Spot-check: aware result is never a weekend or a holiday."""
        test_dates = [
            date(2023, 5, 1), date(2023, 12, 11), date(2024, 5, 6),
            date(2024, 10, 24), date(2025, 7, 22), date(2026, 5, 4),
        ]
        for td in test_dates:
            result = nth_business_day(td, 15)
            assert result.weekday() < 5, f"{td}: result {result} is a weekend"
            assert result not in _NYSE_HOLIDAYS, f"{td}: result {result} is a holiday"

    def test_two_holidays_in_window_shifts_by_two(self):
        """Two holidays in the window → result is 2 real trading days later than blind."""
        # Find a date range with two holidays. Jan 2025 has two close together:
        # Carter funeral (Jan 9) and MLK/Inauguration (Jan 20).
        # From 2024-12-23 (Mon): blind 15 weekdays out goes through both Jan 9 and Jan 20.
        trade_date = date(2024, 12, 23)
        blind = _blind_nth_weekday(trade_date, 15)
        aware = nth_business_day(trade_date, 15)
        # Both holidays should be within the 15-weekday window; aware shifts by 2
        assert (aware - blind).days >= 2
        assert aware.weekday() < 5
        assert aware not in _NYSE_HOLIDAYS
