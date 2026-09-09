"""NYSE trading-day calendar (CR-BD decision 1).

One helper for every cron and backfill that has to step from one session to
the next. Weekend-only stepping (`next_business_day`) stamped Friday's ORATS
data onto Labor Day 2026 and left Tuesday with no row at all; this module knows
the market holidays.

Source: hardcoded NYSE full-closure dates 2023–2027 (no calendar package is in
the cron or web requirements; ten dates a year did not justify a dependency on
five services). 2023–2026 are the harness's `_NYSE_HOLIDAYS` verbatim, including
2025-01-09 (national day of mourning). Extend NYSE_HOLIDAYS and
CALENDAR_VALID_THROUGH together; `test_calendar_not_expired` fails once today is
past CALENDAR_VALID_THROUGH so the list cannot silently run out.

Early closes (day after Thanksgiving, Christmas Eve, July 3) are trading days.
"""
from __future__ import annotations

from datetime import date, timedelta

CALENDAR_VALID_THROUGH = date(2027, 12, 31)

NYSE_HOLIDAYS: frozenset[date] = frozenset({
    # 2023
    date(2023, 1, 2),  date(2023, 1, 16), date(2023, 2, 20), date(2023, 4, 7),
    date(2023, 5, 29), date(2023, 6, 19), date(2023, 7, 4),  date(2023, 9, 4),
    date(2023, 11, 23), date(2023, 12, 25),
    # 2024
    date(2024, 1, 1),  date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
    date(2024, 5, 27), date(2024, 6, 19), date(2024, 7, 4),  date(2024, 9, 2),
    date(2024, 11, 28), date(2024, 12, 25),
    # 2025 (incl. 2025-01-09 national day of mourning)
    date(2025, 1, 1),  date(2025, 1, 9),  date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4),
    date(2025, 9, 1),  date(2025, 11, 27), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1),  date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3),  date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    # 2027 (Juneteenth Sat → Fri 06-18; Independence Day Sun → Mon 07-05; Christmas Sat → Fri 12-24)
    date(2027, 1, 1),  date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5),  date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
})


def is_trading_day(d: date) -> bool:
    """Weekday and not an NYSE full-closure date."""
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


def next_trading_day(d: date) -> date:
    """First NYSE session strictly after `d`."""
    nd = d + timedelta(days=1)
    while not is_trading_day(nd):
        nd += timedelta(days=1)
    return nd


def prev_trading_day(d: date) -> date:
    """Last NYSE session strictly before `d`."""
    pd_ = d - timedelta(days=1)
    while not is_trading_day(pd_):
        pd_ -= timedelta(days=1)
    return pd_


def nth_trading_day(d: date, n: int) -> date:
    """The n-th NYSE session strictly after `d` (n ≥ 1); same rule as the harness's nth_business_day."""
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}")
    cur = d
    for _ in range(n):
        cur = next_trading_day(cur)
    return cur
